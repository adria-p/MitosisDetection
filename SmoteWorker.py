"""
Created on 1 Jul 2013
    
    Implementation of the SMOTE algorithm. Based on a proposal of addition to the scikit-learn.
    
@author: Bibiana and Adria
"""
import numpy as np
from random import choice
from sklearn.neighbors import NearestNeighbors
import random

class SmoteWorker:
    @staticmethod
    def run(positiveSamplesInput, addedPercentage=100, neighborNum=9):
        """
        Returns (addedPercentage/100) * numMinoritySamples synthetic minority samples.
    
        Parameters
        ----------
        positiveSamplesInput : array-like, shape = [numMinoritySamples, numFeatures]
            Holds the minority samples
        addedPercentage : percetange of new synthetic samples: 
            numSyntheticSamples = addedPercentage/100 * numMinoritySamples. Can be < 100.
        neighborNum : int. Number of nearest neighbours. 
    
        Returns
        -------
        S : array, shape = [(addedPercentage/100) * numMinoritySamples, numFeatures]
        """    
        numMinoritySamples, numFeatures = positiveSamplesInput.shape
       
        if addedPercentage < 100:
            positiveSamples = positiveSamplesInput.tolist()
            random.shuffle(positiveSamples)
            positiveSamples = np.array(positiveSamples)
            addedPercentage = (addedPercentage+0.0)/100.0
            numSyntheticSamples = int(addedPercentage * numMinoritySamples)
            addedPercentage = 1
            numMinoritySamples = numSyntheticSamples
        else:
            if (addedPercentage % 100) != 0:
                raise ValueError("addedPercentage must be < 100 or multiple of 100")
            addedPercentage = int((addedPercentage+0.0)/100.0)
            numSyntheticSamples = addedPercentage * numMinoritySamples
            positiveSamples = positiveSamplesInput
            
        S = np.zeros(shape=(numSyntheticSamples, numFeatures))
        #Learn nearest neighbours
        neigh = NearestNeighbors(n_neighbors = neighborNum)
        neigh.fit(positiveSamples)
        #Calculate synthetic samples
        for i in xrange(numMinoritySamples):
            nn = neigh.kneighbors(positiveSamples[i], return_distance=False)
            for n in xrange(addedPercentage):
                nnIndex = choice(nn[0])
                #NOTE: nn includes positiveSamples[i], we don't want to select it 
                while nnIndex == i:
                    nnIndex = choice(nn[0])
                    
                dif = positiveSamples[nnIndex] - positiveSamples[i]
                gap = np.random.random()
                S[n + i * addedPercentage, :] = positiveSamples[i,:] + gap * dif[:]
        return S