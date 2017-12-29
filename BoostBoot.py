#coding: utf-8
from common.origLib import *
from adaBoost import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from common.mathtool import *


def BoostBoot(inLearnFtrMat, inLearnLabel, evalVec, evalLabel, inAdaBoostParam, inBootNum, inBootRatio, inUseFeatNum):
    sampleNum = inLearnFtrMat.shape[0]
    featureNum = inLearnFtrMat.shape[1]
    
    distMat = np.zeros(sampleNum).astype(np.float)
    for l in range(inBootNum):
        learnIdx = np.random.choice(np.arange(sampleNum), int(sampleNum * inBootRatio), replace = False)
        
        learnFtrMat = inLearnFtrMat[learnIdx]
        learnLabel = inLearnLabel[learnIdx]
    
        adaBoost = CAdaBoost()
        adaBoost.SetParam(inAdaBoostParam = inAdaBoostParam)
    
        adaBoost.Boost(trainScoreMat = learnFtrMat,
                                    labelList = learnLabel)
        weakScoreMat = adaBoost.CalcWeakScore()
        refScoreVec = adaBoost.CalcWeakScore(label = evalLabel,
                                             scoreMat = evalVec.reshape(1, -1))
        distMat += np.sum(np.abs(weakScoreMat - refScoreVec), axis = 1)
    return distMat / inBootNum
        
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
    
    _, table = adaBoost.GetLearnedParam()
    evalLossVec = np.exp(-evalLabel * evalScore) + adaBoostParam["Regularizer"] * np.sum(np.cosh(table))
    evalLossVec[evalLabel ==  1] = evalLossVec[evalLabel ==  1] / np.sum(evalLabel ==  1)
    evalLossVec[evalLabel == -1] = evalLossVec[evalLabel == -1] / np.sum(evalLabel == -1)
    
    boostOrder, adaTable = adaBoost.GetLearnedParam()

    return adaTable[np.argsort(boostOrder)], learnScore, evalScore, evalLossVec

if "__main__" == __name__:
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
    hogParam["Cell"]["X"] = 1
    hogParam["Cell"]["Y"] = 1
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
    adaBoostParam["Bin"] = 32
    adaBoostParam["Type"].setTrue("Real")
    adaBoostParam["verbose"] = False
    adaBoostParam["saveDetail"] = False
    adaBoostParam["Saturate"] = True
    adaBoostParam["Regularizer"] = 1e-2
    adaBoostParam["BoostLoop"] = 8

    refAdaTable, reflearnScoreVec, refEvalScoreVec, refEvalLossVec = smallSampleTry(hyperParam = adaBoostParam,
                                                                     learnFtrMat = learnFtrMat,
                                                                     learnLabel = learnLabel,
                                                                     evalFtrMat = evalFtrMat,
                                                                     evalLabel = evalLabel)
    evalTgtIdx = np.argsort(refEvalLossVec)[-1]
    refTgtEvalScore = refEvalScoreVec[evalTgtIdx]
    
    learnDistVec = BoostBoot(inLearnFtrMat = learnFtrMat,
                             inLearnLabel = learnLabel,
                             evalVec = evalFtrMat[evalTgtIdx],
                             evalLabel = evalLabel[evalTgtIdx], 
                             inAdaBoostParam = adaBoostParam,
                             inBootNum = 1,
                             inBootRatio = 1.0,
                             inUseFeatNum = learnFtrMat.shape[1])
    
    print("OK")
    plotNum = 30
    
    # 最も悪影響を与えてる学習サンプルを必ず評価に入れる
    distArgSort = np.argsort(learnDistVec)

    skippedIdx = np.linspace(0, distArgSort.size - 1, plotNum // 5 * 3).astype(np.int)
    skippedIdx = np.append(skippedIdx, distArgSort[- plotNum // 5:])
    skippedIdx = np.append(skippedIdx, distArgSort[ :plotNum // 5 ])
    skippedIdx = np.unique(skippedIdx)

    plotPosIdx = learnLabel[skippedIdx] ==  1
    plotNegIdx = learnLabel[skippedIdx] == -1
    
    plotModEvalScore = np.empty(0).astype(np.float)
    print("checking re-training")
    for i in tqdm(skippedIdx):
        _1, _2, modEvalScoreVec, _3 = smallSampleTry(hyperParam = adaBoostParam,
                                                 learnFtrMat = np.delete(learnFtrMat, i, axis = 0),
                                                 learnLabel = np.delete(learnLabel, i),
                                                 evalFtrMat = evalFtrMat,
                                                 evalLabel = evalLabel)
        plotModEvalScore = np.append(plotModEvalScore, modEvalScoreVec[evalTgtIdx])

    print("tgt cls:", evalLabel[evalTgtIdx])
    x = plotModEvalScore - refTgtEvalScore
    y1 = distArgSort[skippedIdx]
    y2 = np.sqrt(np.sum((learnFtrMat[skippedIdx] - evalFtrMat[evalTgtIdx] ** 2), axis = 1))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(x[plotPosIdx], y1[plotPosIdx], ".", color="red")
    ax1.plot(x[plotNegIdx], y1[plotNegIdx], ".", color="blue")
    ax1.grid(True)
    ax1.set_title("BoostBoot Dist.")
    ax2 = fig.add_subplot(122)
    ax2.plot(x[plotPosIdx], y2[plotPosIdx], ".", color="red")
    ax2.plot(x[plotNegIdx], y2[plotNegIdx], ".", color="blue")
    ax2.grid(True)
    ax2.set_title("Euclid Dist.")
    plt.show()

    print("Done.")



