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
        setDicts["damping"] = 1E-4
        super().__init__(setDicts)

class CInfluence:
    def __init__(self, inflParam, hyperParam, learnedParam, optLearnedParam, 
                 learnFtrMat, learnScore, learnLabel,
                 evalFtrMat, evalScore, evalLabel):
        assert(isinstance(inflParam, CInfluenceParam))
        
        self.__hessian, \
        self.__dLdtheta_learn, \
        self.__dLdtheta_eval = self.__Prepare(param = inflParam,
                                              hyperParam = hyperParam,
                                              learnedParam = learnedParam,
                                              optLearnedParam = optLearnedParam,
                                              learnFtrMat = learnFtrMat,
                                              learnScore = learnScore,
                                              learnLabel = learnLabel,
                                              evalFtrMat = evalFtrMat,
                                              evalScore = evalScore,
                                              evalLabel = evalLabel)

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
    
    def __Prepare(self, param, hyperParam, learnedParam, optLearnedParam, learnFtrMat, learnScore, learnLabel, evalFtrMat, evalScore, evalLabel):
        
        if param["learner"]["RealAdaBoost"]:
            adaBoost = CAdaBoost()

            learnWeight = np.exp(- learnLabel * learnScore)
            learnWeight[learnLabel ==  1] = learnWeight[learnLabel ==  1] / np.sum(learnLabel ==  1)
            learnWeight[learnLabel == -1] = learnWeight[learnLabel == -1] / np.sum(learnLabel == -1)
            
            assert(learnScore.ndim == 1)
            assert(learnFtrMat.shape[0] == learnScore.size)
            assert((learnFtrMat >= 0.0).all())
            assert((learnFtrMat <= 1.0).all())
            assert(learnWeight.ndim == 1)
            
            learnSample = learnFtrMat.shape[0]
            featureNum = learnFtrMat.shape[1]
            adaBin = learnedParam.shape[1]
            thetaN = featureNum * adaBin
            
            # スコアをAdaBoostのBinで量子化
            learnBinMat = (learnFtrMat * adaBin).astype(np.int)
            learnBinMat[learnBinMat >= adaBin] = adaBin - 1
            
            base = np.arange(featureNum) * adaBin
            
            print("making hessian...")
            hessian = np.zeros((thetaN * thetaN))
            dim = learnBinMat.shape[1]
            for s in tqdm(range(learnSample)):
                idxMat = np.empty((dim, dim)).astype(np.int)
                idxMat[np.arange(dim)] = base + learnBinMat[s]
                idxVec = (idxMat + idxMat.T * thetaN).flatten()
                hessian[idxVec] = hessian[idxVec] + learnWeight[s]
            hessian = hessian.reshape(thetaN, thetaN)
            hessian = hessian + np.diag(hyperParam["Regularizer"] * 2.0 * np.cosh(learnedParam.flatten()))

            # damping
            dampHessian = hessian + param["damping"] * np.eye(hessian.shape[0])
            assert(not np.isnan(dampHessian).any())
            assert((dampHessian == dampHessian.T).all())    # 転置対称性を確認
            
            if 1:   # 正定値性(全固有値が正)を確認。かなり重い
                eigen, _ = GetEigen(dampHessian)
                assert((eigen > 0.0).all())
            print("making learnVec...")
 
            if None != optLearnedParam:
                assert(optLearnedParam.shape == learnedParam.shape)
                opt = optLearnedParam
            else:
                opt = learnedParam
            
            learnDiffL = np.empty((learnSample, thetaN))
            for s in tqdm(range(learnSample)):
                oneLine = np.zeros(thetaN)
                oneLine[base + learnBinMat[s]] = learnWeight[s] * (- learnLabel[s])
                learnDiffL[s] = oneLine
            learnDiffL = learnDiffL + hyperParam["Regularizer"] * 2.0 * np.sinh(learnedParam.flatten())
            # damping
            dampDiffL = np.dot(dampHessian,(opt - learnedParam).flatten())
            assert(dampDiffL.shape == (learnDiffL.shape[1],))
            dampDiffLMat = np.empty(learnDiffL.shape)
            dampDiffLMat[np.arange(dampDiffLMat.shape[0])] = dampDiffL
            assert(dampDiffLMat.shape == learnDiffL.shape)
            dampLearnDiffL = learnDiffL + dampDiffLMat

            print("making evalVec...")
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
            evalDiffL = evalDiffL + hyperParam["Regularizer"] * 2.0 * np.sinh(learnedParam.flatten())
            # damping
            assert(dampDiffL.shape == (evalDiffL.shape[1],))
            dampDiffLMat = np.empty(evalDiffL.shape)
            dampDiffLMat[np.arange(dampDiffLMat.shape[0])] = dampDiffL
            assert(dampDiffLMat.shape == evalDiffL.shape)
            dampEvalDiffL = evalDiffL + dampDiffLMat
            
            return dampHessian, dampLearnDiffL, dampEvalDiffL
        
from adaBoost import *
from feature.hog import *
def smallSampleTry(hyperParam,
                   learnFtrMat, 
                   learnLabel, 
                   evalFtrMat,
                   evalLabel):

    adaBoostParam = hyperParam
        
    adaBoost = CAdaBoost()
    adaBoost.SetParam(inAdaBoostParam = adaBoostParam)

    learnScore = adaBoost.Boost(trainScoreMat = learnFtrMat,
                   labelList = learnLabel)
    
    evalScore = adaBoost.Evaluate(testScoreMat = evalFtrMat,
                                  label = evalLabel)
    evalLossVec = np.exp(-evalLabel * evalScore)
    evalLossVec[evalLabel ==  1] = evalLossVec[evalLabel ==  1] / np.sum(evalLabel ==  1)
    evalLossVec[evalLabel == -1] = evalLossVec[evalLabel == -1] / np.sum(evalLabel == -1)
    
    boostOrder, adaTable = adaBoost.GetLearnedParam()

    return adaTable[np.argsort(boostOrder)], learnScore, evalScore, evalLossVec

def calcError():
    for xlsxFile in  GetFileList(".", includingText = ".xlsx"):
        os.remove(xlsxFile)
    for matFile in  GetFileList(".", includingText = ".mat"):
        os.remove(matFile)
    for csvFile in  GetFileList(".", includingText = ".csv"):
        os.remove(csvFile)

    lp = dirPath2NumpyArray("dataset/INRIAPerson/LearnPos")
    ln = dirPath2NumpyArray("dataset/INRIAPerson/LearnNeg")
    learnImg = RGB2Gray(np.append(lp, ln, axis = 0), "green")
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    ep = dirPath2NumpyArray("dataset/INRIAPerson/EvalPos" )[:99]
    en = dirPath2NumpyArray("dataset/INRIAPerson/EvalNeg" )[:98]
    evalImg = RGB2Gray(np.append(ep, en, axis = 0), "green")
    evalLabel = np.array([1] * len(ep) + [-1] * len(en))

    hogParam = CHogParam()
    hogParam["Bin"] = 8
    hogParam["Cell"]["X"] = 2
    hogParam["Cell"]["Y"] = 4
    hogParam["Block"]["X"] = 1
    hogParam["Block"]["Y"] = 1
    detectorList = [CHog(hogParam)]

    learnFtrMat = np.empty((learnImg.shape[0], 0))
    for detector in detectorList:
        learnFtrMat = np.append(learnFtrMat,
                                  detector.calc(learnImg),
                                  axis = 1)
    evalFtrMat = np.empty((evalImg.shape[0], 0))
    for detector in detectorList:
        evalFtrMat = np.append(evalFtrMat,
                                 detector.calc(evalImg),
                                 axis = 1)

    adaBoostParam = AdaBoostParam()
    adaBoostParam["Regularizer"] = 0.0#1e-5
    adaBoostParam["Bin"] = 32
    adaBoostParam["Type"].setTrue("Real")
    adaBoostParam["verbose"] = False
    adaBoostParam["saveDetail"] = False
    adaBoostParam["Saturate"] = False
    adaBoostParam["Regularizer"] = 0.8
    adaBoostParam["BoostLoop"] = 1
    
    adaBoostParam_opt = adaBoostParam.copy()
    adaBoostParam_opt["BoostLoop"] = 4
    optAdaTable, _1, _2, _3 = smallSampleTry(hyperParam = adaBoostParam_opt,
                                             learnFtrMat = learnFtrMat,
                                             learnLabel = learnLabel,
                                             evalFtrMat = evalFtrMat,
                                             evalLabel = evalLabel)
    refAdaTable, learnScore, evalScore, evalLossVec = smallSampleTry(hyperParam = adaBoostParam,
                                                                     learnFtrMat = learnFtrMat,
                                                                     learnLabel = learnLabel,
                                                                     evalFtrMat = evalFtrMat,
                                                                     evalLabel = evalLabel)

    evalTgtIdx = np.argsort(evalLossVec)[-1]
    refEvalLoss = evalLossVec[evalTgtIdx]
    
    influence = CInfluence(inflParam = CInfluenceParam(),
                           hyperParam = adaBoostParam,
                           learnedParam = refAdaTable,
                           optLearnedParam = optAdaTable,
                           learnFtrMat = learnFtrMat,
                           learnScore = learnScore,
                           learnLabel = learnLabel,
                           evalFtrMat = evalFtrMat,
                           evalScore = evalScore,
                           evalLabel = evalLabel)
    upLossVec = influence.CalcUpWeighLoss(targetID = evalTgtIdx)
    
    skip = 100
    skippedIdx = np.arange(upLossVec.size)[::skip]
    
    # ポジネガそれぞれで最も悪影響を与えてる学習サンプルを必ず評価に入れる
    upLossArgSort = np.argsort(upLossVec)
    skippedIdx = np.append(skippedIdx, upLossArgSort[learnLabel ==  1][-1])
    skippedIdx = np.append(skippedIdx, upLossArgSort[learnLabel == -1][-1])
    skippedIdx = np.unique(skippedIdx)
    
    relearnLoss = np.empty(skippedIdx.size).astype(np.float)
    n = 0
    print("re-training")
    for i in tqdm(skippedIdx):
        _1, _2, _3, evalLossVec = smallSampleTry(hyperParam = adaBoostParam,
                                                 learnFtrMat = np.delete(learnFtrMat, i, axis = 0),
                                                 learnLabel = np.delete(learnLabel, i),
                                                 evalFtrMat = evalFtrMat,
                                                 evalLabel = evalLabel)
        relearnLoss[n] = evalLossVec[evalTgtIdx]
        n += 1

    print(relearnLoss)
    plt.plot(upLossVec[skippedIdx], relearnLoss - refEvalLoss, ".")
    plt.grid(True)
    plt.show()
    
if "__main__" == __name__:
    calcError()
    print("Done. ")
    exit()
    
    param = CInfluenceParam()
    influence = CInfluence(inParam = param)
    influence.RefineLearningSample()
