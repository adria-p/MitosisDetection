"""
Created on Jun 21, 2013

    Wndchrm wrapper.

@author: kosklain
"""
import data_io
import numpy as np
import subprocess
import os

class WndchrmWorker:
    def loadWndchrmFeatures(self):
        raise NotImplementedError("Should have implemented this")
    def executeWndchrm(self):
        raise NotImplementedError("Should have implemented this")
    def parseWndchrmOutput(self):
        raise NotImplementedError("Should have implemented this")
    
class WndchrmWorkerTrain(WndchrmWorker):
    def loadWndchrmFeatures(self):
        outfile = open(data_io.get_wndchrm_dataset(), "rb")
        npzfile = np.load(outfile)
        train = npzfile['train']
        target = npzfile['target']
        return (train, target)

    def executeWndchrm(self):
        command = ["wndchrm", "train", data_io.get_training_folder(), data_io.get_wndchrm_datafit()]
        subprocess.call(" ".join(command), shell=True)
        (train, target) = self.parseWndchrmOutput()
        outfile = open(data_io.get_wndchrm_dataset(), "wb")
        np.savez(outfile, train=train, target=target)
        return (train, target)

    def parseWndchrmOutput(self):
        output = open(data_io.get_wndchrm_datafit(),"r")
        train = []
        target = []
        line = output.readline()
        while not "positiveSamples" in line:
            line = output.readline()
        line = output.readline()
        while line != "":
            train.append([np.float64(num) for num in line.split(" ")])
            line = output.readline()
            toAppend = 1 if "positiveSamples" in line else 0
            target.append(toAppend)
            line = output.readline()
        train = np.array(train)
        target = np.array(target)
        return (train, target)
    
class WndchrmWorkerPredict(WndchrmWorker):
    def loadWndchrmFeatures(self):
        outfile = open(data_io.get_wndchrm_dataset_test(), "rb")
        npzfile = np.load(outfile)
        valid = npzfile['valid']
        return valid
    
    def executeWndchrm(self, namesObservations):
        #wndchrm classify -Ttestset.fit dataset.fit folder
        testingFolder = os.path.join(data_io.get_testing_folder(), data_io.get_test_folder())
        command = ["wndchrm", "classify", "-T%s" %(data_io.get_wndchrm_datafit_test()) , 
                   data_io.get_wndchrm_datafit(), testingFolder]
        subprocess.call(" ".join(command), shell=True)
        valid = self.parseWndchrmOutput()
        outfile = open(data_io.get_wndchrm_dataset_test(), "wb")
        np.savez(outfile, valid=valid, namesObservations=namesObservations)
        return valid

    def parseWndchrmOutput(self):
        output = open("T%s" %(data_io.get_wndchrm_datafit_test()),"r")
        valid = []
        line = output.readline()
        while len(line) != 1:
            line = output.readline()
        line = output.readline()
        while line != "":
            valid.append([np.float64(num) for num in line.split(" ")])
            line = output.readline()
            line = output.readline()
        valid = np.array(valid)
        return valid