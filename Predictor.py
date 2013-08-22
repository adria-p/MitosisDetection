"""
Created on Jun 20, 2013
    
    Used for the prediction phase. It calculates the predictions of a stored model 
    for the images in valid_data_path (JSON file).
    
@author: Bibiana and Adria 
"""

import data_io
from FeatureGetter import FeatureGetter
import os
from ImageSaver import ImageSaver
from WndchrmWorker import WndchrmWorkerPredict
from Utils import Utils
import numpy as np

class Predictor:
    def __init__(self, load=False, loadWndchrm=False):
        self.load = load
        self.loadWndchrm = loadWndchrm
    """
       Used for wndchrm. It moves all the old files to a new place, so they do not get overwritten.
   """         
    def prepareEnvironment(self):
        # People want to save time
        testingPath = os.path.join(data_io.get_testing_folder(), data_io.get_test_folder())
        testingPathOld = os.path.join(data_io.get_testing_old_folder(), data_io.get_test_folder())
        Utils.shift(data_io.get_testing_old_folder(), testingPathOld, data_io.get_test_folder(), testingPath)
        os.mkdir(testingPath)
        if not self.load:
            Utils.shift('.', data_io.get_savez_name_test(), data_io.get_savez_name_test(), data_io.get_savez_name_test())
        if not self.loadWndchrm:
            Utils.shift('.', data_io.get_wndchrm_dataset_test(), data_io.get_wndchrm_dataset_test(), data_io.get_wndchrm_dataset_test())
  
    def run(self):
        print "Preparing the environment"
        self.prepareEnvironment()
        print "Loading the classifier"
        classifier = data_io.load_model()
        imageCollections = data_io.get_valid_df()
        featureGetter = FeatureGetter()
        wndchrmWorker = WndchrmWorkerPredict()
        print "Getting the features"
        if not self.loadWndchrm: #Last wndchrm set of features
            fileName = data_io.get_savez_name_test()
            if not self.load: #Last features calculated from candidates
                (namesObservations, coordinates, _) = Utils.calculateFeatures(fileName, featureGetter, imageCollections)
            else:
                (namesObservations, coordinates, _) = Utils.loadFeatures(fileName)
            print "Saving images"
            imageSaver = ImageSaver(coordinates, namesObservations, imageCollections, featureGetter.patchSize)
            imageSaver.saveImages()
            print "Executing wndchrm algorithm"
            valid = wndchrmWorker.executeWndchrm(namesObservations)
        else:
            (valid, namesObservations) = wndchrmWorker.loadWndchrmFeatures()
        print "Making predictions"
        predictions = classifier.predict(valid)
        predictions = predictions.reshape(len(predictions), 1)
        print "Writing predictions to file"
        data_io.write_submission(namesObservations, coordinates, predictions)
        data_io.write_submission_nice(namesObservations, coordinates, predictions)
        print "Calculating final results"
        return Predictor.finalResults(namesObservations, predictions, coordinates)

    def runWithoutWndchrm(self):
        print "Loading the classifier"
        classifier = data_io.load_model()
        imageCollections = data_io.get_valid_df()
        featureGetter = FeatureGetter()
        print "Getting the features"
        fileName = data_io.get_savez_name_test()
        if not self.load: #Last features calculated from candidates
            (namesObservations, coordinates, valid) = Utils.calculateFeatures(fileName, featureGetter, imageCollections)
        else:
            (namesObservations, coordinates, valid) = Utils.loadFeatures(fileName)
        print "Making predictions"
        #valid = normalize(valid, axis=0) #askdfhashdf
        predictions = classifier.predict(valid)
        predictions = predictions.reshape(len(predictions), 1)
        print "Writing predictions to file"
        data_io.write_submission(namesObservations, coordinates, predictions)
        data_io.write_submission_nice(namesObservations, coordinates, predictions)
        print "Calculating final results"
        return Predictor.finalResults(namesObservations, predictions, coordinates)

    @staticmethod
    def getCurrentNumber(name):
        return int(name.split(os.sep)[1])
    
    """
        Given some predictions on some images, gets the results comparing with the ground truth.
    """
    @staticmethod    
    def finalResults(imageNames, predictions, coordinates):
        currentImage = imageNames[0]
        falseNegatives = 0
        falsePositives = 0
        truePositives = 0
        arrayTP = np.zeros(12)
        arrayFP = np.zeros(12)
        arrayFN = np.zeros(12)
        csvArray = Utils.readcsv(imageNames[0])
        for i in range(len(predictions)):
            if imageNames[i] != currentImage:
                csvArray = Utils.readcsv(imageNames[i])
                currentImage = imageNames[i]
            
            result = False
            for point in csvArray:
                if ((point[0]-coordinates[i][0]) ** 2 + (point[1]-coordinates[i][1]) ** 2)< 50**2:
                    result = True
                    break
            if predictions[i] == 1:
                if result:
                    truePositives += 1
                    arrayTP[Predictor.getCurrentNumber(currentImage)-1] += 1
                else:
                    falsePositives += 1
                    arrayFP[Predictor.getCurrentNumber(currentImage)-1] += 1
            if predictions[i] == 0:
                if result:
                    falseNegatives += 1
                    arrayFN[Predictor.getCurrentNumber(currentImage)-1] += 1
        precision = 0 if truePositives+falsePositives == 0 else (truePositives+0.0)/(truePositives+falsePositives+0.0)
        recall = 0 if truePositives+falseNegatives == 0 else (truePositives+0.0)/(truePositives+falseNegatives+0.0)
        fmeasure = 0 if recall+precision == 0 else 2*(precision*recall)/(recall+precision)
        print "TP: " + str(truePositives)
        print "FP: " + str(falsePositives)
        print "FN: " + str(falseNegatives)
        print "Precision: "+ str(precision)
        print "Recall: " + str(recall)
        print "FMeasure: " + str(fmeasure)
        return [truePositives, falsePositives, falseNegatives, precision, recall, fmeasure, (arrayTP, arrayFP, arrayFN)]
        
if __name__ == "__main__":
    pr = Predictor()
    pr.run()
