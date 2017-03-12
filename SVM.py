import os
import scipy.io as sio
import numpy as np
import gui
import mathtool as mt

from svm import *
from svmutil import *
from gui import DrawROC


class SVM:
    def __init__(self,trainSample,trainLabel,cost):
        assert isinstance(cost, float)
        assert isinstance(trainSample, np.ndarray)
        assert 2 == trainSample.ndim
        assert isinstance(trainLabel, np.ndarray)
        assert 1 == trainLabel.ndim
        
        self.__cost = cost
        self.__trainSample = trainSample
        self.__trainLabel = trainLabel
        self.__gamma = None

        self.__MakeGram()
        self.__Optimize()
        
    def __KernelFunc(self,x1,x2):
        assert isinstance(x1,np.ndarray)
        assert isinstance(x2,np.ndarray)
        assert x1.size == x2.size
        assert x1.ndim == x1.ndim
        
        return 1 + np.sum(x1 * x2, axis = x1.ndim - 1)
        
    def __MakeGram(self):

        sample = self.__trainSample.size
        self.__gram = np.zeros((sample,sample),float)
        for j in range(sample):
            self.__gram[j,j] = self.__KernelFunc(self.__trainSample[j], self.__trainSample[j])
            for i in range(j + 1,sample):
                self.__gram[j,i] = self.__KernelFunc(self.__trainSample[j], self.__trainSample[i])
                self.__gram[i,j] = self.__gram[j,i]
    
    def __Optimize(self):
        H = np.zeros(self.__gram.shape,float)
        sample = H.shape[0]
        for j in range(sample):
            for i in range(j + 1,sample):
                H[j,i] = self.__gram[j,i] * self.__trainLabel[j] * self.__trainLabel[i] / self.__cost
                H[i,j] = H[j,i]
        b = np.ones(sample,float)
        assert mt.DischargeCalculation(H, b)
        self.__gamma = b
        
    def Calc(self,newSample):
        assert isinstance(newSample, np.ndarray)
        assert 1 == newSample.ndim
        assert None != self.__gamma
        
        dim = newSample.size
        k = self.__gamma \
          * self.__trainLabel \
          * self.__KernelFunc(newSample, self.__trainSample)\
          / self.__cost
        return np.sum(k) 
           
if "__main__" == __name__:
    
    if not os.path.exists("Train.mat"):
        print("Train Score Matrix NOT FOUND.");exit()
    if not os.path.exists("Test.mat"):
        print("Train Score Matrix NOT FOUND.");exit()

    trainScore = sio.loadmat("Train.mat")["trainScore"]
    trainLabel = sio.loadmat("Train.mat")["trainLabel"][0]
    problem = svm_problem(trainLabel,trainScore.tolist())
    param = svm_parameter('')

    trained = svm_train(problem,param)
    
    testScore = sio.loadmat("Test.mat")["testScore"]
    testLabel = sio.loadmat("Test.mat")["testLabel"][0]
    label,acc,score = svm_predict(testLabel,testScore.tolist(),trained)
    
    score = np.sum(score,axis=1)
    gui.DrawDET(score,testLabel)
    gui.DrawROC(score,testLabel)
    
    print("Done.")