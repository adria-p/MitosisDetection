"""
Created on Jun 20, 2013
    
    Misc methods. Mostly I/O and string formatting.
    
@author: Bibiana and Adria
"""
import os
import csv
import numpy as np
import shutil

class Utils:
    """
        Cross OS file name formatting.
    """
    @staticmethod
    def getPrettyName(name):
        name = os.path.join(*name.split("\\"))
        return os.path.join(*name.split("/"))

    """
        Read csv and check for the coordinates of the mitotic points.
    """
    @staticmethod
    def readcsv(image):
        csvPath = (image.split(".")[0]) + ".csv"
        csvPath = Utils.getPrettyName(csvPath)
        if not os.path.exists(csvPath):
            return []
        csvFile = open(csvPath, "r")
        csvReader = csv.reader(csvFile)
        points = []
        for row in csvReader:
            points.append([int(x) for x in row])
        return points

    @staticmethod
    def getFolderName(name):
        newName = Utils.getPrettyName(name)
        newName = newName.split(os.path.sep)
        return "_".join(newName)

    @staticmethod
    def loadFeatures(fileName):
        outfile = open(fileName, "rb")
        npzfile = np.load(outfile)
        namesObservations = npzfile['namesObservations']
        coordinates = npzfile['coordinates']
        dataset = npzfile['dataset']
        return (namesObservations, coordinates, dataset)

    """
        Get the name of where to move an old folder to, so
        that the old one does not get overwritten (for wndchrm use)
    """
    @staticmethod
    def shift(folder, dest, splitting, check):
        if os.path.exists(check):
            files = os.listdir(folder)
            filesIntList = [int(filename.split(splitting)[1]) for filename in files if splitting in filename and filename != splitting]
            newFileNumber = 1 if len(filesIntList) == 0 else max(filesIntList) + 1
            newFileName = dest + str(newFileNumber)
            shutil.move(check, newFileName)

    """
        It stores the features in a file after extracting them.
    """
    @staticmethod
    def calculateFeatures(fileName, featureGetter, imageCollections):
        (namesObservations, coordinates, dataset) = featureGetter.getTransformedDataset(imageCollections)
        outfile = open(fileName, "wb")
        np.savez(outfile, namesObservations=namesObservations, coordinates=coordinates, dataset=dataset)
        return (namesObservations, coordinates, dataset)
