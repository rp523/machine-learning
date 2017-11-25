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
        self.__dLdtheta_learn, \
        self.__dLdtheta_eval, \
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
    
    def CalcUpWeightParam(self):
        upWeightParam = np.dot(self.__dLdtheta_learn, np.linalg.inv(self.__hessian))
        assert(upWeightParam.shape == self.__dLdtheta_learn.shape)
        return upWeightParam

    # 各学習サンプルの重みが増えた場合の評価サンプル(指定済)損失の変動を得る。
    def CalcUpWeighLoss(self, targetID):
        
        upWeightParam = self.CalcUpWeightParam()
        evalTarget = self.__dLdtheta_eval[targetID].flatten()
        upWeightLoss = np.dot(upWeightParam, evalTarget)
        
        return upWeightLoss
        
    def RefineLearningSample(self):
        
        targetID = self.__SelectTarget(param = self.__param,
                                       mat   = self.__dLdtheta_eval,
                                       score = self.__evalScore,
                                       label = self.__evalLabel)
        
        upWeightLoss = self.CalcUpWeighLoss(targetID = targetID)
        
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
            learnWeight[learnLabel ==  1] = learnWeight[learnLabel ==  1] / np.sum(learnLabel ==  1)
            learnWeight[learnLabel == -1] = learnWeight[learnLabel == -1] / np.sum(learnLabel == -1)
            
            assert(learnScore.ndim == 1)
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
            
            if 1:   # 正定値性(全固有値が正)を確認。かなり重い
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
            evalWeight[evalLabel ==  1] = evalWeight[evalLabel ==  1] / np.sum(evalLabel ==  1)
            evalWeight[evalLabel == -1] = evalWeight[evalLabel == -1] / np.sum(evalLabel == -1)

            evalSample = evalFtrMat.shape[0]

            # スコアをAdaBoostのBinで量子化
            evalBinMat = (evalFtrMat * adaBin).astype(np.int)
            evalBinMat[evalBinMat >= adaBin] = adaBin - 1
            evalDiffL = np.empty((evalSample, thetaN))
            for s in tqdm(range(evalSample)):
                oneLine = np.zeros(thetaN)
                oneLine[base + evalBinMat[s]] = evalWeight[s] * (- evalLabel[s])
                evalDiffL[s] = oneLine
            
            self.__learnLabel = learnLabel
            self.__evalLabel  = evalLabel
            return hessian, learnDiffL, evalDiffL, evalLabel, evalScore

from adaBoost import *
from feature.hog import *
def smallSampleTry(remLearnIdx, 
                   save, 
                   learn, 
                   learnLabel, 
                   eval,
                   evalLabel,
                   tgtIdx = None):
    if remLearnIdx:
        remainIdx = np.ones(len(learn)).astype(np.bool)
        remainIdx[remLearnIdx] = False
        learn = learn[remainIdx]
        learnLabel = learnLabel[remainIdx]
    hogParam = CHogParam()
    hogParam["Bin"] = 8
    hogParam["Cell"]["X"] = 2
    hogParam["Cell"]["Y"] = 4
    hogParam["Block"]["X"] = 1
    hogParam["Block"]["Y"] = 1
    detectorList = [CHog(hogParam)]

    adaBoostParam = AdaBoostParam()
    adaBoostParam["Regularizer"] = 0.0#1e-5
    adaBoostParam["Bin"] = 32
    adaBoostParam["Type"].setTrue("Real")
    adaBoostParam["verbose"] = False
    adaBoostParam["saveDetail"] = save
    adaBoostParam["Saturate"] = True
    adaBoostParam["Regularizer"] = 0.0
    adaBoostParam["BoostLoop"] = 1
        
    adaBoost = CAdaBoost()
    adaBoost.SetParam(  inAdaBoostParam = adaBoostParam,
                        inImgList = learn,
                        inLabelList = learnLabel,
                        inDetectorList = detectorList)

    trainScoreMat = np.empty((len(learn), 0))
    for detector in detectorList:
        trainScoreMat = np.append(trainScoreMat,
                                  detector.calc(learn),
                                  axis = 1)

    adaBoost.Boost(trainScoreMat = trainScoreMat,
                   labelList = learnLabel)
    
    # 評価用の特徴量行列を準備    
    testScoreMat = np.empty((len(eval), 0))
    for detector in detectorList:
        testScoreMat = np.append(testScoreMat,
                                 detector.calc(eval),
                                 axis = 1)
    
    evalScore = adaBoost.Evaluate(testScoreMat = testScoreMat,
                                  label = evalLabel)
    lossVec = np.exp(-evalLabel * evalScore)
    lossVec[evalLabel ==  1] = lossVec[evalLabel ==  1] / np.sum(evalLabel ==  1)
    lossVec[evalLabel == -1] = lossVec[evalLabel == -1] / np.sum(evalLabel == -1)

    if tgtIdx:
        loss = lossVec[tgtIdx]
        print(evalLabel[tgtIdx], loss)
        return loss
    else:
        return lossVec

def calcError():
    for xlsxFile in  GetFileList(".", includingText = ".xlsx"):
        os.remove(xlsxFile)
    for matFile in  GetFileList(".", includingText = ".mat"):
        os.remove(matFile)
    for csvFile in  GetFileList(".", includingText = ".csv"):
        os.remove(csvFile)
    skip = 100
    lp = dirPath2NumpyArray("dataset/INRIAPerson/LearnPos")
    ln = dirPath2NumpyArray("dataset/INRIAPerson/LearnNeg")
    learn = RGB2Gray(np.append(lp, ln, axis = 0), "green")
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    ep = dirPath2NumpyArray("dataset/INRIAPerson/EvalPos" )[:99]
    en = dirPath2NumpyArray("dataset/INRIAPerson/EvalNeg" )[:98]
    eval  = RGB2Gray(np.append(ep, en, axis = 0), "green")
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))

    lossList = smallSampleTry(remLearnIdx = None, 
                             tgtIdx = None, 
                             save = True,
                             learn = learn,
                             learnLabel = learnLabel,
                             eval = eval,
                             evalLabel = evalLabel)
    
    tgt = np.argsort(lossList)[-1]
    refLoss = lossList[tgt]
    
    param = CInfluenceParam()
    influence = CInfluence(inParam = param)
    upLossVec = influence.CalcUpWeighLoss(targetID = tgt)
    
    skippedIdx = np.arange(upLossVec.size)[::skip]
    real = np.arange(skippedIdx.size).astype(np.float)

    n = 0
    for i in skippedIdx:
        print("realvalue:", i)
        real[n] = smallSampleTry(remLearnIdx = i,
                                  tgtIdx = tgt, 
                                  save = False,
                                  learn = learn,
                                  learnLabel = learnLabel,
                                  eval = eval,
                                  evalLabel = evalLabel)
        n += 1

    print(real)
    plt.plot(upLossVec[skippedIdx], real - refLoss, ".")
    plt.grid(True)
    plt.show()
    
if "__main__" == __name__:
    calcError()
    print("Done. ")
    exit()
    
    param = CInfluenceParam()
    influence = CInfluence(inParam = param)
    influence.RefineLearningSample()
