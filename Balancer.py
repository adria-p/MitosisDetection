"""
Created on Jun 20, 2013

    Different ways of undersampling. There is the base class called Balancer,
    implemented by 7 classes. Explanation for each one above each class.

@author: Bibiana and Adria
"""
from sklearn.cluster.k_means_ import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import spectral_clustering
import random

class Balancer:
    def __init__(self, observations=None, numClusters=5):
        self.numClusters = numClusters
        self.observations = observations

    def balance(self, indexesToPick, pointsArray):
        return indexesToPick
    
    def postBalance(self, indexesPicked, target):
        return indexesPicked

"""
    Dummy balancer, it does not balance at all.
"""
class DummyBalancer(Balancer):
    pass

"""
    For every mitotic point, this balancer takes the closest point to it.
"""
class NearestNeighborBalancer(Balancer):
    def balance(self, indexesToPick, pointsArray):
        indexesPicked = indexesToPick[0:2]
        for i in range(pointsArray):
            baseObs = self.observations[indexesPicked[-i]]
            arrayDistances = [np.linalg.norm(baseObs - self.observations[x]) for x in indexesToPick]
            if len(arrayDistances) != 0:
                minimumIndex = np.argmin(arrayDistances)
                indexesPicked.append(indexesToPick[minimumIndex])
        return indexesPicked

""" 
    Base class for cluster balancers. It clusters the negative points,
    and picks random points from every cluster.
"""
class GeneralClusterBalancer(Balancer):
    def ClusterBalance(self, indexesToPick, stopCount, kmeansFlag=True):
        print "ClusterBalancing..."
        indexesPicked = []
        obs1 = self.observations[indexesToPick]
        obs = normalize(obs1, axis=0)
        if len(indexesToPick) != 0:
            if kmeansFlag:
                if(len(indexesToPick) < self.numClusters):
                    cluster = KMeans(init='k-means++', n_clusters=len(obs), n_init=10)
                else:
                    cluster = KMeans(init='k-means++', n_clusters=self.numClusters, n_init=10)
            else:
                if(len(indexesToPick) < self.numClusters):
                    cluster = spectral_clustering(n_clusters=len(obs), n_init=10)
                else:
                    cluster = spectral_clustering(n_clusters=self.numClusters, n_init=10)
            cluster.fit(obs)
            labels = cluster.labels_
            whenToStop = max(2, stopCount)
            count = 0
            while count != whenToStop:
                cluster_list = range(self.numClusters)
                index = 0
                for j in labels:
                    if j in cluster_list:
                        indexesPicked.append(indexesToPick[index])
                        cluster_list.remove(j)
                        count += 1
                        if count == whenToStop:
                            break
                        labels[index] = -1
                        if len(cluster_list) == 0:
                            break
                    index += 1
        return indexesPicked

"""
    This balancer picks a predefined amount of negative points randomly, 
    (that amount is a ratio of the number of positive points). It is much faster
    than k-means balancer, and the results are not so bad. 
"""    
class RandomBalancer(Balancer):
    def postBalance(self, indexesPicked, target, addedPercentage=7):
        newIndexesPicked = np.where(target == 1)[0]
        numberOfPositiveExamples = len(newIndexesPicked)
        indexesToPick = np.where(target == 0)[0].tolist()
        random.seed(1)
        random.shuffle(indexesToPick)
        indexesToPick = np.array(indexesToPick)
        newIndexesToPick = indexesToPick[0:numberOfPositiveExamples+int(numberOfPositiveExamples*addedPercentage)]
        return np.concatenate((newIndexesPicked, newIndexesToPick))

"""
    Use the GeneralClusterBalancer with k-means, per image basis.
"""
class KMeansBalancer(GeneralClusterBalancer):    
    def balance(self, indexesToPick, pointsArray):
        random.shuffle(indexesToPick)
        return self.ClusterBalance(indexesToPick, pointsArray)

"""
    Use the GeneralClusterBalancer with k-means, 
    applied on all negative points at once.
"""
class PostKMeansBalancer(GeneralClusterBalancer):
    def postBalance(self, indexesPicked, target):
        newIndexesPicked = np.where(target == 1)[0]
        numberOfPositiveExamples = len(newIndexesPicked)
        indexesToPick = np.where(target == 0)[0]
        random.seed(97)
        random.shuffle(indexesToPick)
        whenToStop = numberOfPositiveExamples+(self.numClusters-(numberOfPositiveExamples%self.numClusters))
        return np.concatenate((newIndexesPicked, self.ClusterBalance(indexesToPick, whenToStop)))

"""
    Use the GeneralClusterBalancer with spectral clustering, 
    applied on all negative points at once.
"""
class CircularBalancer(GeneralClusterBalancer):
    def postBalance(self, indexesPicked, target):
        newIndexesPicked = np.where(target == 1)[0]
        numberOfPositiveExamples = len(newIndexesPicked)
        indexesToPick = np.where(target == 0)[0]
        random.seed(97)
        random.shuffle(indexesToPick)
        whenToStop = numberOfPositiveExamples+(self.numClusters-(numberOfPositiveExamples%self.numClusters))
        return np.concatenate((newIndexesPicked, self.ClusterBalance(indexesToPick, whenToStop, kmeansFlag=False)))     