#coding: utf-8
from common.origLib import *
from adaBoost import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from common.mathtool import *

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

        self.__hessian, \
        self.__learnMat, \
        self.__evalMat, \
        self.__evalLabel, \
        self.__evalScore = self.__Prepare(self.__param)

    # スコア低減対称の評価サンプルを選び、そのIDを返す    
    def __SelectTarget(self, param, mat, score, label):
        targetID = None
        if param["evalTarget"]["largeNeg"]:
            targetScore = np.max(score[label == -1])
            if targetScore > 0.5:
                targetID = np.where(score == targetScore)
        elif self.__param["evalTarget"]["smallPos"]:
            targetScore = np.min(score[label == 1])
            if targetScore < 0.5:
                targetID = np.where(score == targetScore)
        return targetID
    
    # 各学習サンプルの重みが増えた場合の評価サンプル(指定済)損失の変動を得る。
    def __CalcUpWeighLoss(self, param, mat, score, label, targetID):
        
        evalTarget = self.__evalMat[targetID].flatten()
	
        ref = - SolveLU(self.__hessian, evalTarget)
        assert(not np.isnan(ref).any())
        
        learnSample = self.__learnMat.shape[0]
        print("searching harmful learning-sample...")
        upWeightLoss = np.empty(learnSample)
        for s in tqdm(range(learnSample)):
            learnVec = self.__learnMat[s]
            upWeightLoss[s] = np.sum(learnVec * ref)
        
        return upWeightLoss
        
    def RefineLearningSample(self):
        
        targetID = self.__SelectTarget(param = self.__param,
                                       mat   = self.__evalMat,
                                       score = self.__evalScore,
                                       label = self.__evalLabel)
        
        upWeightLoss = \
        self.__CalcUpWeighLoss(param = self.__param,
                               mat   = self.__evalMat,
                               score = self.__evalScore,
                               label = self.__evalLabel,
                               targetID = targetID)
        
        upWeightLoss_argsort = np.argsort(upWeightLoss)
        remainIdx = np.sort(upWeightLoss_argsort[:-self.__param["removeMax"]])
        removeIdx = np.sort(upWeightLoss_argsort[-self.__param["removeMax"]:])
        assert(remainIdx.size + removeIdx.size == upWeightLoss_argsort.size)
        
        return remainIdx, removeIdx, targetID
    
    def __Prepare(self, param):
        if param["learner"]["RealAdaBoost"]:
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

            print("making learnVec...")
            learnDiffL = np.empty((learnSample, thetaN))
            for s in tqdm(range(learnSample)):
                oneLine = np.zeros(thetaN)
                oneLine[base + learnBinMat[s]] = learnWeight[s] * (- learnLabel[s])
                learnDiffL[s] = oneLine

            print("making hessian...")
            hessian = np.zeros((thetaN * thetaN))
            dim = learnBinMat.shape[1]
            for s in tqdm(range(learnSample)):
                idxMat = np.empty((dim, dim)).astype(np.int)
                idxMat[np.arange(dim)] = base + learnBinMat[s]
                idxVec = (idxMat + idxMat.T * thetaN).flatten()
                hessian[idxVec] = hessian[idxVec] + learnWeight[s]
            hessian = hessian.reshape(thetaN, thetaN)

            # 対角成分に小さい値を足すことにより、損失関数の大域解になっていない場合も
            # 強制的に正定値化する
            hessian = hessian + 0.01 * np.eye(hessian.shape[0])
            assert(not np.isnan(hessian).any())
            assert((hessian == hessian.T).all())    # 転置対称性を確認
            
            if 0:   # 正定値性(全固有値が正)を確認。かなり重いので一旦OFF
                eigen, _ = GetEigen(hessian)
                assert((eigen > 0.0).all())
        
            # for debug
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
                oneLine[base + evalBinMat[s]] = evalWeight[s] * (- evalLabel[s])
                evalDiffL[s] = oneLine
            
            return hessian, learnDiffL, evalDiffL, evalLabel, evalScore
        
if "__main__" == __name__:
    param = CInfluenceParam()
    influence = CInfluence(inParam = param)
    influence.RefineLearningSample()
    print("Done. ")
