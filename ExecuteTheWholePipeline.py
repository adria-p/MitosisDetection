"""
Created on Jun 21, 2013
    
    Sometimes you just want to "press a button" and go for a walk.
    Let the machine work for you. This file does it.
    
    Three classes to choose: The first one just runs one time the pipeline, 
    the second one is just for having a feeling of the parameters, and the 
    last one is the one that performs the cross validation and gets the test
    score as it should be gotten.
    
    NOTE (very important): In order to run the Cross-validation class, it is
    needed to have run at least once the first class, since the cross-validation
    class needs to have the features already saved in the disk. 
    
    More detailed description at the beginning of every class.
    
@author: Bibiana and Adria 
"""
from Trainer import Trainer
from Predictor import Predictor
import data_io
from Utils import Utils
from FeatureGetter import FeatureGetter
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

""" 
    From the images in your train_data_path (JSON file) it trains.
    Afterwards, from the images in you valid_data_path it predicts.
    If the features are already calculated, set load to True to
    load them from disk.
"""
class PipeLineExecutor:
    def run(self):
        tr = Trainer(load=False, loadWndchrm=False)
        tr.run()
        pr = Predictor(load=False, loadWndchrm=False)
        return pr.run()

    def runWithoutWndchrm(self):
        tr = Trainer(load=False, loadWndchrm=False)
        tr.runWithoutWndchrm()
        pr = Predictor(load=False, loadWndchrm=False)
        pr.runWithoutWndchrm()
    """
        If we are running the Cross-validation class, we need to put 
        the testing and training files together, so that we split them
        as we wish.
    """
    def mergeFiles(self, trainFeaturesFile, testFeaturesFile):
        (namesObservationsTr, coordinatesTr, train) = Utils.loadFeatures(trainFeaturesFile)
        (namesObservationsTe, coordinatesTe, test) = Utils.loadFeatures(testFeaturesFile)
        namesObservations = np.concatenate((namesObservationsTr,namesObservationsTe))
        coordinates = np.concatenate((coordinatesTr, coordinatesTe))
        dataset = np.concatenate((train, test))
        namesObservations = np.reshape(namesObservations, (namesObservations.shape[0],1))
        return (namesObservations, coordinates, dataset)            
    
    """ 
        To play around with the features, this function helps the user
        in selecting which ones to avoid.
    """
    def filterIndexes(self, length):
        avoidHaralick = False
        avoidZernike = False
        avoidTamura = False
        avoidTransformations = False
        avoidGeneralImageFeatures = True
        avoidNoBackground = False
        avoidBackground = False
        onlyRGB = False
        onlyBRVL = False
        onlyHEDAV = False
        avoidedList = []
        if avoidGeneralImageFeatures:
            avoidedList.extend(range(4,244))
        if avoidHaralick:
            preList = range(244,length)
            avoidedList.extend([item for item in preList if ((item-244)%423)%105 in range(7,59)])
        if avoidZernike:
            preList = range(244,length)
            avoidedList.extend([item for item in preList if ((item-244)%423)%105 in range(59,84)])
        if avoidTamura:
            preList = range(244,length)
            avoidedList.extend([item for item in preList if (item-244)%423 in range(420,423)])
        if avoidBackground:
            preList = range(244,length)
            avoidedList.extend([item for item in preList if ((item-244)%423) in range(105,423)])
        if avoidNoBackground:
            preList = range(244,length)
            avoidedList.extend([item for item in preList if ((item-244)%423) in range(105)])
        if avoidTransformations:
            preList = range(244,length)
            avoidedList.extend([item for item in preList if ((item-244)%423)%105 in range(84,105)])
        if onlyRGB:
            avoidedList.extend(range(667, length))
        if onlyBRVL:
            avoidedList.extend(range(244,667))
            avoidedList.extend(range(1090, length))
        if onlyHEDAV:
            avoidedList.extend(range(244,1090))
        mask = np.ones(length)
        mask[avoidedList] = 0
        mask = np.where(mask == 1)[0]
        return mask

    """
        Split the data in k parts, with shuffled indexes.
    """
    def getShuffledSplits(self, data, indexes, k):
        shuffledData = data[indexes,:]
        splittedData = np.split(shuffledData[:len(shuffledData)-(len(shuffledData)%k)], k)
        if len(shuffledData)%k != 0:
            splittedData[-1] = np.concatenate((splittedData[-1], shuffledData[len(shuffledData)-(len(shuffledData)%k)+1:]))
        return splittedData
    
    """
        Get the splitted data together without the validation split.
    """
    def getTrainData(self, splittedData, i):
        toReturn = np.delete(splittedData, i, 0)
        toReturn = tuple(tuple(x) for x in toReturn) 
        toReturn = np.concatenate(toReturn)
        return toReturn
        
    """
        Split the data by patient.
    """
    def getSplits(self, names, coords, dataset):
        finalNames = []
        finalCoords = []
        finalDataset = []
        currentDataset = []
        currentCoords = []
        currentNames = []
        currentName = names[0][0].split('.')[0][:-3]     
        for i in range(len(names)):
            if currentName!= names[i][0].split('.')[0][:-3]:
                finalNames.append(np.array(currentNames))
                finalCoords.append(np.array(currentCoords))
                finalDataset.append(np.array(currentDataset))
                currentDataset = []
                currentCoords = []
                currentNames = []
                currentName = names[i][0].split('.')[0][:-3]
            currentNames.append(names[i])
            currentCoords.append(coords[i])
            currentDataset.append(dataset[i])
        finalNames.append(np.array(currentNames))
        finalCoords.append(np.array(currentCoords))
        finalDataset.append(np.array(currentDataset))
        return (np.array(finalNames), np.array(finalCoords), np.array(finalDataset))

    """
        Split the dataset in half.
    """
    def getNewSplits(self, splittedNamesObs, splittedCoords, splittedData):
        finalNames = []
        finalNames.append(np.concatenate((splittedNamesObs[0], splittedNamesObs[1], splittedNamesObs[2], splittedNamesObs[3], splittedNamesObs[4], splittedNamesObs[5])))
        finalNames.append(np.concatenate((splittedNamesObs[6], splittedNamesObs[7], splittedNamesObs[8], splittedNamesObs[9], splittedNamesObs[10], splittedNamesObs[11])))
        finalCoords = []
        finalCoords.append(np.concatenate((splittedCoords[0], splittedCoords[1], splittedCoords[2], splittedCoords[3], splittedCoords[4], splittedCoords[5])))
        finalCoords.append(np.concatenate((splittedCoords[6], splittedCoords[7], splittedCoords[8], splittedCoords[9], splittedCoords[10], splittedCoords[11])))
        finalDataset = []
        finalDataset.append(np.concatenate((splittedData[0], splittedData[1], splittedData[2], splittedData[3], splittedData[4], splittedData[5])))
        finalDataset.append(np.concatenate((splittedData[6], splittedData[7], splittedData[8], splittedData[9], splittedData[10], splittedData[11])))
        return (np.array(finalNames), np.array(finalCoords), np.array(finalDataset))

""" 
    Looking patients separately and many more options. 
    Testing purposes only. Do not obtain the test score from this!
"""
class PipeLineExecutorManualSplit(PipeLineExecutor):    
    def run(self, k=3, patientSplit=True, useOnlyRF=True, breakin2=True):
        featureGetter = FeatureGetter()
        overallTP = 0
        overallFP = 0
        overallFN = 0
        fileNameTrain = data_io.get_savez_name()
        fileNameTest = data_io.get_savez_name_test()
        print "Merging files..."
        (namesObservations, coordinates, dataset) = self.mergeFiles(fileNameTrain, fileNameTest)

        dataset = dataset[:,self.filterIndexes(len(dataset[0]))]        
        print "Shuffling and splitting the data"
        indexesChanged = np.arange(len(dataset))
        np.random.shuffle(indexesChanged)
        if patientSplit:
            k = 12
            (splittedNamesObs, splittedCoords, splittedData) = self.getSplits(namesObservations, coordinates, dataset)
            if breakin2:
                k = 2
                (splittedNamesObs, splittedCoords, splittedData) = self.getNewSplits(splittedNamesObs, splittedCoords, splittedData)
        else:
            splittedNamesObs = self.getShuffledSplits(namesObservations, indexesChanged, k)
            splittedCoords = self.getShuffledSplits(coordinates, indexesChanged, k)
            splittedData = self.getShuffledSplits(dataset, indexesChanged, k)
        
        del(dataset)
        del(coordinates)
        del(namesObservations)
        del(indexesChanged)
        
        overallArrayTP = np.zeros(12)
        overallArrayFP = np.zeros(12)
        overallArrayFN = np.zeros(12)

        for i in range(k-1,-1,-1):#i is the index of the validation
            print "Doing cross-validation for i=%d" %i    
            namesObservationsTest = splittedNamesObs[i]
            coordinatesTest = splittedCoords[i]
            datasetTest = splittedData[i]
            namesObservationsTest = np.reshape(namesObservationsTest, namesObservationsTest.shape[0])
            namesObservationsTrain = self.getTrainData(splittedNamesObs,i)
            coordinatesTrain = self.getTrainData(splittedCoords,i)
            datasetTrain = self.getTrainData(splittedData, i)
            namesObservationsTrain = np.reshape(namesObservationsTrain, namesObservationsTrain.shape[0])
            print "Getting target vector"
    
            (indexes, target, obs) = featureGetter.getTargetVector(coordinatesTrain, namesObservationsTrain, datasetTrain)
        
            print "Selecting features"
            classifier = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=1, min_samples_split=1, random_state=1, compute_importances=True)
            model = Pipeline([('scaling', MinMaxScaler()), ('classifying', classifier)])
            model.fit(obs[indexes], target[indexes])
            if not useOnlyRF:
                importances = classifier.feature_importances_
                filterImportances = np.where(importances > 0.0001)[0]
                print len(filterImportances)
                #namesObservationsTrain = np.reshape(namesObservationsTrain, namesObservationsTrain.shape[0]) 
                print "Training model"
                #classifier = RandomForestClassifier(n_estimators=500, verbose=2, n_jobs=1, min_samples_split=100, random_state=1, compute_importances=True)
                #classifier = KNeighborsClassifier()
                classifier = LinearSVC(verbose=1)
                #classifier = MLPClassifier(verbose=1)
                model = Pipeline([('scaling', MinMaxScaler()), ('classifying', classifier)])
                model.fit(obs[indexes][:,filterImportances], target[indexes])
            print "Making predictions"
            if not useOnlyRF:
                predictions = model.predict(datasetTest[:,filterImportances])
            else:
                predictions = model.predict(datasetTest)
            predictions = predictions.reshape(len(predictions), 1)
            print "Calculating final results"
            [truePositives, falsePositives, falseNegatives, _, _, _, (arrayTP, arrayFP, arrayFN)] = Predictor.finalResults(namesObservationsTest, predictions, coordinatesTest)
            print arrayTP
            print arrayFP
            print arrayFN
            
            overallArrayTP += arrayTP
            overallArrayFP += arrayFP
            overallArrayFN += arrayFN
            overallTP += truePositives
            overallFP += falsePositives
            overallFN += falseNegatives
            del(datasetTrain)
            del(datasetTest)
            del(coordinatesTrain)
            del(coordinatesTest)
            del(namesObservationsTrain)
            del(namesObservationsTest)
        
        precision = 0 if overallTP+overallFP == 0 else (overallTP+0.0)/(overallTP+overallFP+0.0)
        recall = 0 if overallTP+overallFN == 0 else (overallTP+0.0)/(overallTP+overallFN+0.0)
        fmeasure = 0 if recall+precision == 0 else 2*(precision*recall)/(recall+precision)
        
        print "Overall results for k=%d" %k
        print overallTP
        print overallFP
        print overallFN
        print precision
        print recall
        print fmeasure
        
        for i in range(len(overallArrayTP)):
            "Results for patient number %d:"% (i+1)
            overallTP = overallArrayTP[i]
            overallFP = overallArrayFP[i]
            overallFN = overallArrayFN[i]
            precision = 0 if overallTP+overallFP == 0 else (overallTP+0.0)/(overallTP+overallFP+0.0)
            recall = 0 if overallTP+overallFN == 0 else (overallTP+0.0)/(overallTP+overallFN+0.0)
            fmeasure = 0 if recall+precision == 0 else 2*(precision*recall)/(recall+precision)
            print precision
            print recall
            print fmeasure

"""
    Obtain the test score with this one. You can configure the number of splits of the cross-validation, 
    and, if you want to use other methods apart from random forests, set useOnlyRF to False.
"""
class PipeLineExecutorCrossVal(PipeLineExecutor):     
    def run(self, k=3, useOnlyRF=True):
        featureGetter = FeatureGetter()
        fileNameTrain = data_io.get_savez_name()
        fileNameTest = data_io.get_savez_name_test()
        print "Merging files..."
        (namesObservations, coordinates, dataset) = self.mergeFiles(fileNameTrain, fileNameTest)
        dataset = dataset[:,self.filterIndexes(len(dataset[0]))]        
        print "Shuffling and splitting the data"
        indexesChanged = np.arange(len(dataset))
        np.random.shuffle(indexesChanged)
        splittedNamesObs = self.getShuffledSplits(namesObservations, indexesChanged, k+1)
        splittedCoords = self.getShuffledSplits(coordinates, indexesChanged, k+1)
        splittedData = self.getShuffledSplits(dataset, indexesChanged, k+1)
        
        """Leave the last split for testing"""
        testNamesObs = splittedNamesObs[k]
        testCoords = splittedCoords[k]
        testDataset = splittedData[k]
        
        splittedNamesObs = splittedNamesObs[:k]
        splittedCoords = splittedCoords[:k]
        splittedData = splittedData[:k]
        
        del(dataset)
        del(coordinates)
        del(namesObservations)
        del(indexesChanged)

        bestModel = None
        bestFmeasure = 0
        
        for i in range(k-1,-1,-1):#i is the index of the validation
            print "Doing cross-validation for i=%d" %i    
            namesObservationsValid = splittedNamesObs[i]
            coordinatesValid = splittedCoords[i]
            datasetValid = splittedData[i]
            namesObservationsValid = np.reshape(namesObservationsValid, namesObservationsValid.shape[0])
            namesObservationsTrain = self.getTrainData(splittedNamesObs,i)
            coordinatesTrain = self.getTrainData(splittedCoords,i)
            datasetTrain = self.getTrainData(splittedData, i)
            namesObservationsTrain = np.reshape(namesObservationsTrain, namesObservationsTrain.shape[0])
            print "Getting target vector"
            (indexes, target, obs) = featureGetter.getTargetVector(coordinatesTrain, namesObservationsTrain, datasetTrain)
            print "Selecting features"
            classifier = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=1, min_samples_split=1, random_state=1, compute_importances=True)
            model = Pipeline([('scaling', MinMaxScaler()), ('classifying', classifier)])
            model.fit(obs[indexes], target[indexes])
            if not useOnlyRF:
                importances = classifier.feature_importances_
                filterImportances = np.where(importances > 0.0001)[0]
                print len(filterImportances)
                #namesObservationsTrain = np.reshape(namesObservationsTrain, namesObservationsTrain.shape[0]) 
                print "Training model"
                #classifier = RandomForestClassifier(n_estimators=500, verbose=2, n_jobs=1, min_samples_split=100, random_state=1, compute_importances=True)
                #classifier = KNeighborsClassifier()
                classifier = LinearSVC(verbose=1)
                #classifier = MLPClassifier(verbose=1)
                model = Pipeline([('scaling', MinMaxScaler()), ('classifying', classifier)])
                model.fit(obs[indexes][:,filterImportances], target[indexes])
            print "Making predictions"
            if not useOnlyRF:
                predictions = model.predict(datasetValid[:,filterImportances])
            else:
                predictions = model.predict(datasetValid)
            predictions = predictions.reshape(len(predictions), 1)
            print "Calculating validation results"
            [_, _, _, _, _, fmeasure, _] = Predictor.finalResults(namesObservationsValid, predictions, coordinatesValid)
            if fmeasure > bestFmeasure:
                bestFmeasure = fmeasure
                bestModel = model
            del(datasetTrain)
            del(datasetValid)
            del(coordinatesTrain)
            del(coordinatesValid)
            del(namesObservationsTrain)
            del(namesObservationsValid)
        
        print "Calculating final results"
        predictions = bestModel.predict(testDataset)
        print "The final score is: "
        testNamesObs = np.reshape(testNamesObs, testNamesObs.shape[0])
        Predictor.finalResults(testNamesObs, predictions, testCoords)
        

if __name__ == "__main__":
    plwcv = PipeLineExecutorCrossVal()
    plwcv.run(k=10)