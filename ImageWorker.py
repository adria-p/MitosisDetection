"""
Created on Jun 20, 2013

    This class is the core of two important parts of the project:
    firstly, the feature calculation is done here. And secondly, the 
    image segmentation is done here.  
    
    More details on every class/method.  

@author: Bibiana and Adria
"""
from scipy import ndimage, fftpack
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from math import pi
from skimage.util import dtype
from scipy.ndimage.filters import gaussian_laplace
from skimage.color import rgb2hed
from mahotas.features import texture, zernike_moments
import pywt
from scipy.stats.stats import skew, kurtosis
from Tamura import Tamura
from scipy.stats.mstats_basic import mquantiles

"""
    Constant for L resizing
"""
rangeL = (1.0/0.95047)*116.- 16.

"""
    Constants for HEDAB resizing
"""
imageTest = np.array([[[255,0,0]]], dtype=np.uint8)
minH = rgb2hed(imageTest)[0][0][0]
imageTest = np.array([[[0,255,255]]], dtype=np.uint8)
maxH = rgb2hed(imageTest)[0][0][0]
rangeH = maxH-minH
imageTest = np.array([[[0,255,0]]], dtype=np.uint8)
minE = rgb2hed(imageTest)[0][0][1]
imageTest = np.array([[[255,0,255]]], dtype=np.uint8)
maxE = rgb2hed(imageTest)[0][0][1]
rangeE = maxE-minE
imageTest = np.array([[[0,0,255]]], dtype=np.uint8)
minDAB = rgb2hed(imageTest)[0][0][2]
imageTest = np.array([[[255,255,0]]], dtype=np.uint8)
maxDAB = rgb2hed(imageTest)[0][0][2]
rangeDAB = maxDAB-minDAB

"""
    Base class for operating with images. It forces the user to implement
    the getPatch method on the classes that inherit from it. It also has
    some general purpose methods for ploting and calculating limits.
"""
class ImageWorker:
    def __init__(self, image, rows=None, columns=None, convert=False):
        if convert:
            self.image = dtype.img_as_float(image)
        else:
            self.image = image
        self.rows = len(image) if rows is None else rows
        self.columns = len(image[0]) if columns is None else columns

    def showOnlyPatches(self, centers, size):
        finalImage = np.zeros(self.image.shape, dtype=np.uint8)
        for center in centers:
            (beginningi, beginningj, finali, finalj) = self.getLimits(center, size)
            finalImage[beginningi:finali, beginningj:finalj, :] = self.image[beginningi:finali, beginningj:finalj, :]
        plt.imshow(finalImage)
        plt.show()

    def getLimits(self, center, size):
        beginningi = max(center[0] - (size - 1) / 2, 0)
        beginningj = max(center[1] - (size - 1) / 2, 0)
        finali = min(beginningi + size, self.rows-1)
        finalj = min(beginningj + size, self.columns-1)
        return (beginningi, beginningj, finali, finalj)

    def getPatch(self, center, size):
        raise NotImplementedError("Should have implemented this")

"""
    3D image (RGB, HSV, LAB, etc.) worker. It gets the needed features, and it implements
    the segmentation operation with getBinaryImage(), which gets the binary image that contains
    True for those pixels that were below all the thresholds and True for the ones which were not
    below all thresholds.
"""
class RGBImageWorker(ImageWorker):
    """
        The only1D is used when we are calculating the features of only those pixels
        that pass all the filters.
    """
    def getGeneralStatistics(self, hara=False, zern=False, tamura=False, only1D=None):
        generalStatistics = []
        if self.rows == 1 and self.columns == 1:
            for index in range(3):
                generalStatistics.append(self.image[0,0,index])
            return generalStatistics
        if not only1D is None:
            im = only1D
            generalStatistics.extend(self._calculateStatistics(im, haralick=hara, zernike=zern))
            fourierTransform = np.abs(fftpack.fft2(im)) #fourierTransform
            generalStatistics.extend(self._calculateStatistics(fourierTransform))
            waveletTransform = pywt.dwt2(im, 'sym5')[0]
            generalStatistics.extend(self._calculateStatistics(waveletTransform))
            waveletFourierTransform = pywt.dwt2(fourierTransform, 'sym5')[0]
            generalStatistics.extend(self._calculateStatistics(waveletFourierTransform)) 
            if tamura:
                generalStatistics.extend(self.get3Dstatistics(tamura=True))  
            return generalStatistics
        for index in range(3):
            im = self.image[:, :, index]
            generalStatistics.extend(self._calculateStatistics(im, haralick=hara, zernike=zern))
            fourierTransform = np.abs(fftpack.fft2(im)) #fourierTransform
            generalStatistics.extend(self._calculateStatistics(fourierTransform))
            waveletTransform = pywt.dwt2(im, 'sym5')[0]
            generalStatistics.extend(self._calculateStatistics(waveletTransform))
            waveletFourierTransform = pywt.dwt2(fourierTransform, 'sym5')[0]
            generalStatistics.extend(self._calculateStatistics(waveletFourierTransform))  
        if tamura:
            generalStatistics.extend(self.get3Dstatistics(tamura=True))          
        return generalStatistics

    """
        Features calculated with only one component (for example, R from RGB).
    """
    def _calculateStatistics(self, img, haralick=False, zernike=False):
        result = []
        #3-bin histogram
        result.extend(mquantiles(img))
        #First four moments
        result.extend([img.mean(), img.var(), skew(img,axis=None), kurtosis(img,axis=None)]) 
        #Haralick features
        if haralick:
            integerImage = dtype.img_as_ubyte(img)
            result.extend(texture.haralick(integerImage).flatten())
        #Zernike moments
        if zernike:
            result.extend(zernike_moments(img, int(self.rows)/2 + 1))
        return result
    """
        Features calculated with the whole image at once.
    """
    def get3Dstatistics(self, tamura=False):
        result = []
        #Tamura features
        if tamura:
            #result.append(Tamura.coarseness(self.image)) it may not work!
            result.append(Tamura.contrast(self.image))
            result.extend(Tamura.directionality(self.image))
        return result
    """
        Plot the histogram of a given channel.
    """
    def plotHistogram(self, img):
        hist, bin_edges = np.histogram(img, bins=60)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, hist, lw=2)
        plt.show()

    """
        Filter an image based on a percentile.
    """
    def filterImage(self, im, threshold):
        if self.ploting:
            self.plotHistogram(im)
        return im < np.percentile(im,threshold)
    
    """
        V from HSV color space
    """
    def getV(self):
        preV = np.asanyarray(self.image)
        preV = dtype.img_as_float(preV)
        return preV.max(-1)
    
    """
        Y from XYZ color space
    """
    def getY(self):
        arr = np.asanyarray(self.image)
        arr = dtype.img_as_float(arr)
        mask = arr > 0.04045
        arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
        arr[~mask] /= 12.92
        Y = arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722
        return Y
    
    """
        It returns L from LAB color space and L from LUV color space
    """
    def getL(self):
        Y = self.getY()
        Y = Y / 0.95047
        mask = Y > 0.008856
        Y2 = np.power(Y, 1. / 3.)
        Y[mask] = Y2[mask]
        Y[~mask] = 7.787 * Y[~mask] + 16. / 116.
        L = (116. * Y) - 16.
        L2 = (116. * Y2) - 16.
        return (L, L2)
    """
        It returns the thresholded (segmented) image, as well as
        the HED image and BRVL image.
    """
    def getBinaryImage(self):
        self.ploting = False
        HEDAB = rgb2hed(self.image)
        R = self.image[:, :, 0]
        G = self.image[:, :, 1]
        B = self.image[:, :, 2]
        H = HEDAB[:,:,0]    
        E = HEDAB[:,:,1]
        DAB = HEDAB[:,:,2]
        BR = B*2/((1+R+G)*(1+B+R+G)) #Blue-ratio image
        V = self.getV() #From HSV
        (L, L2) = self.getL() #From CIELAB and CIELUV
        BRSmoothed = ndimage.gaussian_filter(BR,1)
        LSmoothed = ndimage.gaussian_filter(L,1)
        VSmoothed = ndimage.gaussian_filter(V,1)
        HSmoothed = ndimage.gaussian_filter(H,1)
        ESmoothed = ndimage.gaussian_filter(E,1)
        RSmoothed = ndimage.gaussian_filter(R,1)
        DABSmoothed = ndimage.gaussian_filter(DAB,1)
        imLLog = self.filterImage(gaussian_laplace(LSmoothed,9), 85) == False
        imVLog = self.filterImage(gaussian_laplace(VSmoothed, 9), 85) == False
        imELog = self.filterImage(gaussian_laplace(ESmoothed,9), 84) == False 
        imRLog = self.filterImage(gaussian_laplace(RSmoothed, 9), 84) == False
        imDABLog = self.filterImage(gaussian_laplace(DABSmoothed,9), 50)
        imHLog = self.filterImage(gaussian_laplace(HSmoothed,9), 8)
        imLog = self.filterImage(gaussian_laplace(BRSmoothed,9), 9)
        imR = self.filterImage(R, 2.5)
        imB = self.filterImage(B, 10.5)
        imV = self.filterImage(V, 6.5)
        imL = self.filterImage(L, 2.5)
        imL2 = self.filterImage(L2, 2.5)
        imE = self.filterImage(E, 18)
        imH = self.filterImage(H, 95) == False
        imDAB = self.filterImage(DAB, 55) == False
        imBR = self.filterImage(BR, 63) == False
        binaryImg = imR & imV & imB & imL & imL2 & imE & imH & imDAB & imLog & imBR & imLLog & imVLog & imELog & imHLog & imRLog & imDABLog
        openImg = ndimage.binary_opening(binaryImg, iterations=2)
        closedImg = ndimage.binary_closing(openImg, iterations=8)
        if self.ploting:
            plt.imshow(self.image)
            plt.show()
            plt.imshow(imR)
            plt.show()
            plt.imshow(imV)
            plt.show()
            plt.imshow(imB)
            plt.show()
            plt.imshow(imL)
            plt.show()
            plt.imshow(closedImg)
            plt.show()
        
        BRVL = np.zeros(self.image.shape)
        BRVL[:,:,0] = BR
        BRVL[:,:,1] = V
        BRVL[:,:,2] = L/rangeL
        #ResizeHEDAB, from 0 to 1.
        HEDAB[:,:,0] = (H - minH)/rangeH
        HEDAB[:,:,1] = (E - minE)/rangeE
        HEDAB[:,:,2] = (DAB - minDAB)/rangeDAB
        
        return (BinaryImageWorker(closedImg, self.rows, self.columns), 
                RGBImageWorker(HEDAB, self.rows, self.columns), 
                RGBImageWorker(BRVL, self.rows, self.columns), BinaryImageWorker(binaryImg, self.rows, self.columns))

    def getPatch(self, center, size):
        (beginningi, beginningj, finali, finalj) = self.getLimits(center, size)
        subImageWorker = RGBImageWorker(self.image[beginningi:finali, beginningj:finalj, :], finali - beginningi, finalj - beginningj)
        return subImageWorker

"""
    It gets the binary characteristics of an image, such as area, peremiter or roundness.
    It also gets the centers of all the blobs from the binary thresholded image.
"""
class BinaryImageWorker(ImageWorker):
    def getPatch(self, center, size):
        (beginningi, beginningj, finali, finalj) = self.getLimits(center, size)
        subImageWorker = BinaryImageWorker(self.image[beginningi:finali, beginningj:finalj], finali - beginningi, finalj - beginningj)
        return subImageWorker

    def getGeneralStatistics(self):
        area = np.sum(self.image)
        perimeterArray = [len(x) for x in measure.find_contours(self.image, 0.5)]
        perimeter = max(perimeterArray) if len(perimeterArray) != 0 else 0
        roundness = 4 * area * pi / (perimeter * perimeter) if perimeter != 0 else 0
        finalStatistics = [area, perimeter, roundness, len(self.getCenters())]
        return finalStatistics

    def getCenters(self):
        lbl = ndimage.label(self.image)
        centers = ndimage.measurements.center_of_mass(self.image, lbl[0], range(1, lbl[1] + 1))
        centers = np.round(centers)
        return centers