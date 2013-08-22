"""
Created on Jun 20, 2013
    
    Used for the training phase. It builds a model for the images in 
    train_data_path (JSON file).
    It is also used for checking how many canditates per mitotic point
    we have (that is, the quality of the segmentation), by using the 
    appropriate functionality in FeatureGetter.
    
@author: Bibiana and Adria 
"""

import data_io
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from FeatureGetter import FeatureGetter
from ImageSaver import ImageSaver
import os
from Utils import Utils
from WndchrmWorker import WndchrmWorkerTrain
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

class Trainer:
    def __init__(self, load=False, loadWndchrm=False):
        self.load = load
        self.loadWndchrm = loadWndchrm

    """
       Used for wndchrm. It moves all the old files to a new place, so they do not get overwritten.
    """    
    def prepareEnvironment(self):
        # People want to save time
        trainingPathPositive = os.path.join(data_io.get_training_folder(), data_io.get_positive_folder())
        trainingPathOldPositive = os.path.join(data_io.get_training_old_folder(), data_io.get_positive_folder())
        Utils.shift(data_io.get_training_old_folder(), trainingPathOldPositive, data_io.get_positive_folder(), trainingPathPositive)
        trainingPathNegative = os.path.join(data_io.get_training_folder(), data_io.get_negative_folder())
        trainingPathOldNegative = os.path.join(data_io.get_training_old_folder(), data_io.get_negative_folder())
        Utils.shift(data_io.get_training_old_folder(), trainingPathOldNegative, data_io.get_negative_folder(), trainingPathNegative)
        os.mkdir(trainingPathPositive)
        os.mkdir(trainingPathNegative)
        if not self.load:
            Utils.shift('.', data_io.get_savez_name(), data_io.get_savez_name(), data_io.get_savez_name())
        if not self.loadWndchrm:
            Utils.shift('.', data_io.get_wndchrm_dataset(), data_io.get_wndchrm_dataset(), data_io.get_wndchrm_dataset())
                
    def run(self):
        print "Preparing the environment"
        self.prepareEnvironment()
        print "Reading in the training data"
        imageCollections = data_io.get_train_df()
        wndchrmWorker = WndchrmWorkerTrain()
        print "Getting features"
        if not self.loadWndchrm: #Last wndchrm set of features
            featureGetter = FeatureGetter()
            fileName = data_io.get_savez_name()
            if not self.load: #Last features calculated from candidates
                (namesObservations, coordinates, train) = Utils.calculateFeatures(fileName, featureGetter, imageCollections)
            else:
                (namesObservations, coordinates, train) = Utils.loadFeatures(fileName)
            print "Getting target vector"
            (indexes, target, obs) = featureGetter.getTargetVector(coordinates, namesObservations, train)
            print "Saving images"
            imageSaver = ImageSaver(coordinates[indexes], namesObservations[indexes],
                                    imageCollections, featureGetter.patchSize, target[indexes])
            imageSaver.saveImages()
            print "Executing wndchrm algorithm and extracting features"
            (train, target) = wndchrmWorker.executeWndchrm()
        else:
            (train, target) = wndchrmWorker.loadWndchrmFeatures()
        print "Training the model"
        model = RandomForestClassifier(n_estimators=500, verbose=2, n_jobs=1, min_samples_split=30, random_state=1, compute_importances=True)
        model.fit(train, target)
        print model.feature_importances_
        print "Saving the classifier"
        data_io.save_model(model)

    def runWithoutWndchrm(self):
        print "Reading in the training data"
        imageCollections = data_io.get_train_df()
        print "Getting features"
        featureGetter = FeatureGetter()
        fileName = data_io.get_savez_name()
        if not self.load: #Last features calculated from candidates
            (namesObservations, coordinates, train) = Utils.calculateFeatures(fileName, featureGetter, imageCollections)
        else:
            (namesObservations, coordinates, train) = Utils.loadFeatures(fileName)
        print "Getting target vector"
        (indexes, target, obs) = featureGetter.getTargetVector(coordinates, namesObservations, train)
        print "Training the model"
        classifier = RandomForestClassifier(n_estimators=500, verbose=2, n_jobs=1, min_samples_split=10, random_state=1, compute_importances=True)
        #classifier = KNeighborsClassifier(n_neighbors=50)
        model = Pipeline([('scaling', MinMaxScaler()), ('classifying', classifier)])
        model.fit(obs[indexes], target[indexes])
        print "Saving the classifier"
        data_io.save_model(model)
    
    """
        Checks the quality of the segmentation.
    """
    def checkCandidates(self):
        imageCollections = data_io.get_train_df()
        featureGetter = FeatureGetter()
        (namesObservations, coordinates, train) = featureGetter.getTransformedDatasetChecking(imageCollections)
        imageNames = namesObservations
        currentImage = imageNames[0]
        csvArray = Utils.readcsv(imageNames[0])
        mitoticPointsDetected = 0
        totalMitoticPoints = len(csvArray)
        finalTrain = []
        for i in range(len(coordinates)):
            if imageNames[i] != currentImage:
                csvArray = Utils.readcsv(imageNames[i])
                totalMitoticPoints += len(csvArray)
                currentImage = imageNames[i]
            for point in csvArray:
                if ((point[0]-coordinates[i][0]) ** 2 + (point[1]-coordinates[i][1]) ** 2)< 30**2:
                    mitoticPointsDetected += 1
                    csvArray.remove(point)
                    finalTrain.append(train[i])
                    break
        finalTrain = np.array(finalTrain)
        allArea = finalTrain[:,0]
        allPerimeter = finalTrain[:,1]
        allRoundness = finalTrain[:,2]
        totalObservations = len(coordinates)
        print "Minimum Area: %f" % np.min(allArea)
        print "Minimum Perimeter: %f" % np.min(allPerimeter)
        print "Minimum Roundness: %f" % np.min(allRoundness)
        print "Maximum Area: %f" % np.max(allArea)
        print "Maximum Perimeter: %f" % np.max(allPerimeter)
        print "Maximum Roundness: %f" % np.max(allRoundness)
        print "Total number of candidates: %d" % (totalObservations)
        print "Total number of mitotic points: %d" %(totalMitoticPoints)
        print "Mitotic points detected: %d" %(mitoticPointsDetected)
        print "Mitotic points missed: %d" %(totalMitoticPoints-mitoticPointsDetected)
        
if __name__ == "__main__":
    tr = Trainer(load=True, loadWndchrm=True)
    tr.checkCandidates()
