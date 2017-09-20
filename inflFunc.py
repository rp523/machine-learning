#coding: utf-8
from common.origLib import *
from adaBoost import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from common.mathtool import MakeLU, SolveLU

class CInfluenceParam(CParam):
    def __init__(self):
        setDicts = dicts()
        setDicts["learner"] = selparam("RealAdaBoost")
        setDicts["evalTarget"] = selparam("largeNeg", "smallPos")
        setDicts["removeMax"] = 1
        super().__init__(setDicts) 

class CInfluence:
    def __init__(self, inParam = None):
        assert(isinstance(inParam, CInfluenceParam))
        if None != inParam:
            self.__param = inParam
        else:
            self.__param = CInfluenceParam()

        hessian, self.__learnMat, self.__evalMat, self.__evalLabel, self.__evalScore = self.__Prepare()
        self.__hessianLU = MakeLU(hessian)
    
    def RefineLearningSample(self):
        if self.__param["evalTarget"]["largeNeg"]:
            negMat     = self.__evalMat[  self.__evalLabel == -1]
            negScore   = self.__evalScore[self.__evalLabel == -1]
            evalTarget = negMat[np.argmax(negScore)]
        elif self.__param["evalTarget"]["smallPos"]:
            posMat     = self.__evalMat[  self.__evalLabel == 1]
            posScore   = self.__evalScore[self.__evalLabel == 1]
            evalTarget = posMat[np.argmin(posScore)]
        
        lH = SolveLU(self.__hessianLU, evalTarget)
        
        learnSample = self.__learnMat.shape[0]
        print("searching harmful learning-sample...")
        lHl = np.empty(learnSample)
        for s in tqdm(range(learnSample)):
            learnVec = self.__learnMat[s]
            lHl[s] = np.sum(learnVec * lH)
        if self.__param["evalTarget"]["largeNeg"]:
            return np.argsort(lHl)[self.__param["removeMax"]:]
        elif self.__param["evalTarget"]["smallPos"]:
            return np.argsort(lHl)[self.__param["removeMax"]:]
        
    def __Prepare(self):
        if self.__param["learner"]["RealAdaBoost"]:
            adaBoost = CAdaBoost()

            learnFtrMat = adaBoost.Load(type = "learnFeature")
            relia = adaBoost.Load(type = "reliability")
            id = adaBoost.Load(type = "boostOrder")
            learnScore = adaBoost.Load(type = "learnScore")
            learnLabel = adaBoost.Load(type = "learnLabel")
            learnWeight = np.exp(- learnLabel * learnScore)

            assert(learnFtrMat.shape[0] == learnScore.size)
            assert((learnFtrMat >= 0.0).all())
            assert((learnFtrMat <= 1.0).all())
            assert(learnWeight.ndim == 1)
            
            learnSample = learnFtrMat.shape[0]
            featureNum = relia.shape[0]
            adaBin = relia.shape[1]
            thetaN = featureNum * adaBin
            
            # スコアをAdaBoostのBinで量子化
            learnBinMat = (learnFtrMat * adaBin).astype(np.int)
            learnBinMat[learnBinMat >= adaBin] = adaBin - 1
            
            base = np.arange(featureNum) * adaBin

            hessian = np.zeros((thetaN, thetaN))
            learnDiffL = np.empty((learnSample, thetaN))
            print("making hessian...")
            for s in tqdm(range(learnSample)):
                oneLine = np.zeros(thetaN)
                oneLine[base + learnBinMat[s]] = 1
                learnDiffL[s] = oneLine * learnWeight[s] * (- learnLabel[s])

                oneMat = np.dot(oneLine.reshape(-1, 1), oneLine.reshape(1, -1))
                hessian = hessian + oneMat
            
            assert((hessian == hessian.T).all())
            #plt.imshow(hessian, cmap='hot', interpolation='nearest')
            #plt.show()
            #exit()
            
            print("making evalVec...")
            evalFtrMat = adaBoost.Load(type = "evalFeature")
            evalScore = adaBoost.Load(type = "evalScore")
            evalLabel = adaBoost.Load(type = "evalLabel")
            evalWeight = np.exp(- evalLabel * evalScore)
            evalSample = evalFtrMat.shape[0]

            # スコアをAdaBoostのBinで量子化
            evalBinMat = (evalFtrMat * adaBin).astype(np.int)
            evalBinMat[evalBinMat >= adaBin] = adaBin - 1
            evalDiffL = np.empty((evalSample, thetaN))
            for s in tqdm(range(evalSample)):
                oneLine = np.zeros(thetaN)
                oneLine[base + evalBinMat[s]] = 1
                evalDiffL[s] = oneLine * evalWeight[s] * (- evalLabel[s])
            return hessian, learnDiffL, evalDiffL, evalLabel, evalScore
        
if "__main__" == __name__:
    param = CInfluenceParam()
    influence = CInfluence(inParam = param)
    influence.RefineLearningSample()
    print("Done.")