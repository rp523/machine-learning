#coding: utf-8
from common.origLib import *
from adaBoost import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from common.mathtool import SolveLU

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

        self.__hessian, self.__learnMat, self.__evalMat, self.__evalLabel, self.__evalScore = self.__Prepare()
        
    def RefineLearningSample(self):
        
        evalTarget = None
        if self.__param["evalTarget"]["largeNeg"]:
            negMat     = self.__evalMat[  self.__evalLabel == -1]
            negScore   = self.__evalScore[self.__evalLabel == -1]
            maxId = np.argmax(negScore)
            if negScore[maxId] > 0.5:
                evalTarget = negMat[maxId]
        elif self.__param["evalTarget"]["smallPos"]:
            posMat     = self.__evalMat[  self.__evalLabel == 1]
            posScore   = self.__evalScore[self.__evalLabel == 1]
            minId = np.argmin(posScore)
            if negScore[minId] < 0.5:
                evalTarget = posMat[minId]
        assert(not np.isnan(self.__hessian).any())
        assert(not np.isnan(evalTarget).any())
        tgtEvalSample = np.where((self.__evalMat == evalTarget).all(axis = 1))
        tgtEvalSample = int(tgtEvalSample[0][0])

        ref = - SolveLU(self.__hessian, evalTarget)
        assert(not np.isnan(ref).any())
        
        learnSample = self.__learnMat.shape[0]
        print("searching harmful learning-sample...")
        upWeightLoss = np.empty(learnSample)
        for s in tqdm(range(learnSample)):
            learnVec = self.__learnMat[s]
            upWeightLoss[s] = np.sum(learnVec * ref)
        
        upWeightLoss_argsort = np.argsort(upWeightLoss)
        remainIdx = np.sort(upWeightLoss_argsort[:-self.__param["removeMax"]])
        removeIdx = np.sort(upWeightLoss_argsort[-self.__param["removeMax"]:])
        assert(remainIdx.size + removeIdx.size == upWeightLoss_argsort.size)
        
        return remainIdx, removeIdx, tgtEvalSample
    
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
                hessian = hessian + oneMat * learnWeight[s]
            
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
            
            # 平均スケールを1に近づける処理のつもりだったが、これやると結果が変わってしまう
            # hessian = hessian / np.average(hessian[hessian != 0.0])

            # 対角成分に小さい値を足すことで、無理やり正則化する。あまり良くないかも
            hessian = hessian + (np.min(hessian[hessian != 0.0]) * 1E-7) * np.eye(hessian.shape[0])
            
            return hessian, learnDiffL, evalDiffL, evalLabel, evalScore
        
if "__main__" == __name__:
    param = CInfluenceParam()
    influence = CInfluence(inParam = param)
    influence.RefineLearningSample()
    print("Done. ")