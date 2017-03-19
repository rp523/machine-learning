import os
import scipy.io as sio
import numpy as np
import gui
import mathtool as mt
from matplotlib import pyplot as plt

from svm import *
from svmutil import *


class SVM:
    
    def __init__(self,trainSample,trainLabel,cost):
        assert isinstance(cost, float)
        assert isinstance(trainSample, np.ndarray)
        assert 2 == trainSample.ndim
        assert isinstance(trainLabel, np.ndarray)
        assert 1 == trainLabel.ndim
        
        self.__epsilon = 0.001
        self.__cost = cost
        self.__trainSample = trainSample
        self.__trainLabel = trainLabel
        self.__sample = self.__trainLabel.size
        self.__gamma = np.zeros(self.__sample,float)
        self.__gram = None

        self.__MakeGram()
        self.__Optimize()
        
    def __KernelFunc(self,x1,x2):
        assert isinstance(x1,np.ndarray)
        assert isinstance(x2,np.ndarray)
        assert x1.size == x2.size
        assert x1.ndim == x1.ndim
        
        return np.exp(-np.sum((x1 - x2)**2))
        #return 1 + np.sum(x1 * x2, axis = x1.ndim - 1)
        
    def __MakeGram(self):

        self.__gram = np.zeros((self.__sample,self.__sample),float)
        for y in range(self.__sample):
            for x in range(y,self.__sample):
                self.__gram[y,x] = self.__KernelFunc(self.__trainSample[y], self.__trainSample[x])
                self.__gram[x,y] = self.__gram[y,x] # 対称行列のはずなので、コピーするだけ
    
    def __CalcTrainSample(self,index):
        return np.dot(self.__gamma * self.__trainLabel,self.__gram[index])
    
    def __MeetsKKT(self,index):
        if 0.0 >= self.__gamma[index]:
            return (1.0 <= self.__CalcTrainSample(index) * self.__trainLabel[index])
        if self.__cost <= self.__gamma[index]:
            return (1.0 >= self.__CalcTrainSample(index) * self.__trainLabel[index])
        else:
            return (1.0 == self.__CalcTrainSample(index) * self.__trainLabel[index])
            
    def __SMO(self,index1,index2):
        
        gamma1OLD = self.__gamma[index1]
        gamma2OLD = self.__gamma[index2]

        E1 = self.__CalcTrainSample(index1) - self.__trainLabel[index1]
        E2 = self.__CalcTrainSample(index2) - self.__trainLabel[index2]
        K11 = self.__gram[index1,index1]
        K22 = self.__gram[index2,index2]
        K12 = self.__gram[index1,index2]
        
        # index2のラグランジュ係数を更新
        self.__gamma[index2] += self.__trainLabel[index2] * (E1 - E2) / (K11 + K22 - 2.0 * K12)

        # 不等式制約を破らないよう、値のとる範囲を制限
        if self.__trainLabel[index2] != self.__trainLabel[index1]:
            L = max(0.0        , gamma2OLD - gamma1OLD              )
            H = min(self.__cost, gamma2OLD - gamma1OLD + self.__cost)
        else:
            L = max(0.0        , gamma2OLD + gamma1OLD - self.__cost)
            H = min(self.__cost, gamma2OLD + gamma1OLD              )
        if L > self.__gamma[index2]:
            self.__gamma[index2] = L
        elif H < self.__gamma[index2]:
            self.__gamma[index2] = H
            
        # index1のラグランジュ係数を更新
        self.__gamma[index1] += (self.__trainLabel[index1] * self.__trainLabel[index2]) * (gamma2OLD - self.__gamma[index2])
        
        # 値の変更があった場合にtrueを返す
        return ((self.__gamma[index1] != gamma1OLD) or (self.__gamma[index2] != gamma2OLD))
    
    def __Optimize(self):
        for y in range(self.__sample):
            if not self.__MeetsKKT(y):
                for x in range(y + 1, self.__sample):
                    if not self.__MeetsKKT(x):
                        self.__SMO(y,x)
        
        while 1:
            updated = False
            for y in range(self.__sample):
                for x in range(y + 1, self.__sample):
                    updated = self.__SMO(y,x)
            if False == updated:
                break

        print(np.sum(self.__gamma == 0.0) / self.__gamma.size)
        exit()
        
    def Calc(self,testScore):
        assert isinstance(testScore, np.ndarray)
        assert None != self.__gamma
        
        testScoreTrans = np.transpose(testScore)
        k = self.__gamma \
          * self.__trainLabel \
          * self.__KernelFunc(testScore, self.__trainSample)\
          / self.__cost
        return np.sum(k) 
           
if "__main__" == __name__:
    
    if not os.path.exists("Train.mat"):
        print("Train Score Matrix NOT FOUND.");exit()
    if not os.path.exists("Test.mat"):
        print("Train Score Matrix NOT FOUND.");exit()

    trainScore = sio.loadmat("Train.mat")["trainScore"]
    trainLabel = sio.loadmat("Train.mat")["trainLabel"][0]
    testScore = sio.loadmat("Test.mat")["testScore"]
    testLabel = sio.loadmat("Test.mat")["testLabel"][0]
    
    svm = SVM(trainScore,trainLabel,0.5)
    score = SVM.Calc(testScore)
    '''
    #LIBSVM
    problem = svm_problem(trainLabel,trainScore.tolist())
    param = svm_parameter('')
    trained = svm_train(problem,param)
    label,acc,score = svm_predict(testLabel,testScore.tolist(),trained)
    score = np.sum(score,axis=1)
    '''
    #gui.DrawDET(score,testLabel)
    gui.DrawROC(score,testLabel)
    
    print("Done.")