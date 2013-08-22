"""
Created on Jun 20, 2013

    For Wndchrm utility, we need to have all the patches saved into separate images.
    This class does exactly this.

@author: Bibiana and Adria
"""
from ImageWorker import RGBImageWorker
from Utils import Utils
import os
import data_io
from skimage import io

class ImageSaver:
    def __init__(self, coordinates, namesObservations, imageCollections, patchSize, targetVector=None):
        self.coordinates = coordinates
        self.namesObservations = namesObservations
        self.imageCollections = imageCollections
        self.patchSize = patchSize
        self.targetVector = targetVector
        self.length = len(self.coordinates)
        self.rows = len(self.imageCollections[0][0])
        self.columns = len(self.imageCollections[0][0][0])

    def getImageFromName(self, name):
        for imageCollection in self.imageCollections:
            index = 0
            newName = Utils.getPrettyName(name)
            for image in imageCollection.files:
                if Utils.getPrettyName(image) == newName:
                    return imageCollection[index]
                index += 1
        return None
    
    def saveImages(self):
        for index in range(self.length):
            image = self.getImageFromName(self.namesObservations[index])
            imageWorker = RGBImageWorker(image, self.rows, self.columns)
            patch = imageWorker.getPatch(self.coordinates[index], self.patchSize)
            reducedFileName = "%s_%d.tif" % (Utils.getFolderName(self.namesObservations[index]), index)
            if self.targetVector is None:
                fileName = os.path.join(data_io.get_testing_folder(), data_io.get_test_folder(), reducedFileName)
            elif self.targetVector[index] == 1:
                fileName = os.path.join(data_io.get_training_folder(), data_io.get_positive_folder(), reducedFileName)
            else:
                fileName = os.path.join(data_io.get_training_folder(), data_io.get_negative_folder(), reducedFileName)
            io.imsave(fileName, patch.image)