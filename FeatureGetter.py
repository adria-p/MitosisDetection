"""
Created on Jun 21, 2013
    
    This class is useful for three purposes: firstly, as its name indicates,
    it calculates all the features from a certain collection of images, 
    using ImageWorker.
    Secondly, it is also used to check the number of candidates extracted
    per mitotic point (that is, checking the quality of the filters),
    implemented in the method getTransformedDatasetChecking.
    Finally, it is also used to provide the target vector of a dataset.
    
@author: Bibiana and Adria 
"""

import numpy as np
import copy
from ImageWorker import RGBImageWorker
from Utils import Utils
from Balancer import DummyBalancer, NearestNeighborBalancer, KMeansBalancer, PostKMeansBalancer,\
    CircularBalancer, RandomBalancer
from SmoteWorker import SmoteWorker
import matplotlib.pyplot as plt

class FeatureGetter:
    def __init__(self, patchsize=30):
        self.patchSize = patchsize
        self.tolerance = 30 ** 2
    """
        Checking whether the point is in the corresponding radius or not.
    """
    def isInAdmissibleRadius(self, point, observation):
        return ((point[0] - observation[0]) ** 2 + (point[1] - observation[1]) ** 2) < self.tolerance
    
    """
        Count if we missed any mitotic point in an image.
    """
    def checkMissedCount(self, imageName, centers):
        csvArray = Utils.readcsv(imageName)
        totalCount = len(csvArray)
        totalMitoticPoints = len(csvArray)
        missedCount = 0
        for i in csvArray:
            found = False
            for center in centers:
                if self.isInAdmissibleRadius(i, center):
                    found = True
                    break
            if not found:
                print "missed! %s" % (str(i))
                missedCount += 1
        return (missedCount, totalCount, totalMitoticPoints)

    """
        Plot the patch for which we are calculating the features.
    """
    def plotPatch(self, patch, mask):
        plt.imshow(patch.image)
        plt.show()
        finalImage = patch.image
        finalImage[~mask] = 1 
        plt.imshow(finalImage)
        plt.show()
    
    """
        Calculate the features of a collection of images.
    """
    def getTransformedDataset(self, imageCollections, onlyPatchzone=True):
        dataset = []
        namesObservations = []
        coordinates = []
        for imageCollection in imageCollections:
            imageCount = 0
            for image in imageCollection:
                im = image
                imageName = imageCollection.files[imageCount]
                imageWorker = RGBImageWorker(im, convert=True)
                (binaryImageWorkerCenters, HEDAVWorker, BRVLWorker, binaryImageWorker) = imageWorker.getBinaryImage()
                workerPack = [imageWorker, BRVLWorker, HEDAVWorker]
                generalStatistics = imageWorker.getGeneralStatistics(hara=True, zern=False)
                centers = binaryImageWorkerCenters.getCenters()
                print len(centers)
                for center in centers:
                    print "center"
                    binaryPatch = binaryImageWorker.getPatch(center, self.patchSize)
                    binaryStatistics = binaryPatch.getGeneralStatistics()
                    namesObservations.append(imageName)
                    coordinates.append([center[0], center[1]])
                    observation = binaryStatistics
                    observation.extend(copy.deepcopy(generalStatistics))
                    if onlyPatchzone:
                        for worker in workerPack:
                            patch = worker.getPatch(center, self.patchSize)
                            observation.extend(patch.getGeneralStatistics(hara=True, zern=True, tamura=False, only1D=patch.image[binaryPatch.image]))
                            observation.extend(patch.getGeneralStatistics(hara=True, zern=True, tamura=True))
                    else:
                        for patchsize in range(1, self.patchSize, 10):
                            for worker in workerPack:
                                patch = worker.getPatch(center, patchsize)
                                observation.extend(patch.getGeneralStatistics())
                        patch = imageWorker.getPatch(center, self.patchSize)
                        observation.extend(patch.getGeneralStatistics(hara=True, zern=True, tamura=True))
                    print len(observation)
                    dataset.append(observation)
                print imageName
                imageCount += 1
        dataset = np.array(dataset)
        return (namesObservations, coordinates, dataset)
    
    """
        Check the quality of the segmentation filters on an image collection.
    """
    def getTransformedDatasetChecking(self, imageCollections):
        dataset = []
        namesObservations = []
        coordinates = []
        totalMitoticPoints = 0
        for imageCollection in imageCollections:
            imageCount = 0
            missedCount = 0
            totalCount = 0
            for image in imageCollection:
                im = image
                imageName = imageCollection.files[imageCount]
                if not "10" in imageName:
                    imageCount += 1
                    continue
                imageWorker = RGBImageWorker(im, convert=True)
                (binaryImageWorkerCenters, _, _, binaryImageWorker) = imageWorker.getBinaryImage()
                centers = binaryImageWorkerCenters.getCenters()
                count = 0
                (incMissedCount, incTotalCount, incTotalMitoticPoints) = self.checkMissedCount(imageName, centers)
                totalCount += incTotalCount
                totalMitoticPoints += incTotalMitoticPoints
                missedCount += incMissedCount
                for center in centers:
                    binaryPatch = binaryImageWorker.getPatch(center, self.patchSize)
                    binaryStatistics = binaryPatch.getGeneralStatistics()
                    count += 1
                    namesObservations.append(imageName)
                    coordinates.append([center[0], center[1]])
                    observation = binaryStatistics
                    dataset.append(observation)
                print "Candidates: %d" % (count)
                print "Missed this patient: %d" % (missedCount)
                if totalCount != 0:
                    print "Missed ratio this patient: %f" % ((missedCount + 0.0) / (totalCount + 0.0))
                print "Total number of candidates: %d" % (len(dataset))
                print "Total number of mitotic points: %d" % totalMitoticPoints
                print "Total ratio mitotic/candidate: %f" % ((0.0 + totalMitoticPoints) / (0.0 + len(dataset)))
                print imageName
                imageCount += 1
        return (np.array(namesObservations), np.array(coordinates), np.array(dataset))

    """
        Given the ground truth, calculate the target vector a dataset.
    """
    def getTargetVector(self, coordinates, names, observations, balancingMode=5, overSampling=100):
        if balancingMode == 0:  # No balancing
            balancer = DummyBalancer()
        elif balancingMode == 1:
            balancer = NearestNeighborBalancer(observations)
        elif balancingMode == 2:
            balancer = KMeansBalancer(observations)
        elif balancingMode == 3:
            balancer = PostKMeansBalancer(observations)
        elif balancingMode == 4:
            balancer = CircularBalancer(observations)
        elif balancingMode == 5:
            balancer = RandomBalancer(observations)
        else:
            raise ValueError("Incorrect balancing mode.")
        target = np.zeros(len(coordinates))
        currentImage = ""
        pointsArray = 0
        indexesPicked = []
        indexesToPick = []
        for obsNum in range(len(coordinates)):
            if names[obsNum] != currentImage:
                indexesPicked.extend(balancer.balance(indexesToPick, pointsArray))
                currentImage = names[obsNum]
                csvArray = Utils.readcsv(currentImage)
                indexesToPick = []
                pointsArray = 0
            for point in csvArray:
                if self.isInAdmissibleRadius(point, coordinates[obsNum]):
                    target[obsNum] = 1
                    indexesPicked.append(obsNum)
                    pointsArray += 1
                    break
            if target[obsNum] == 0:
                indexesToPick.append(obsNum)
        if overSampling != 0:
            newValuesAdded = SmoteWorker.run(observations[np.where(target == 1)[0]], overSampling)
            target = np.concatenate((target, np.ones(len(newValuesAdded))))
            indexesPicked.extend(range(len(coordinates), len(coordinates) + len(newValuesAdded)))
            newObservations = np.concatenate((observations, newValuesAdded))
        else:
            newObservations = observations
        target = np.array(target)
        balancer.observations = newObservations
        indexesPicked = balancer.postBalance(indexesPicked, target)
        return (indexesPicked, target, newObservations)
