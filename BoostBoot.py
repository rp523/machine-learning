#coding: utf-8
from common.origLib import *
from adaBoost import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
from common.mathtool import *


def BoostBoot(inLearnFtrMat, inLearnLabel, evalVec, evalLabel, inAdaBoostParam, inBootNum, inBootRatio, inUseFeatNum):
    sampleNum = inLearnFtrMat.shape[0]
    featureNum = inLearnFtrMat.shape[1]
    
    distMat = np.zeros((sampleNum, inBootNum)).astype(np.float)
    distCnt = np.zeros(sampleNum).astype(np.int)
    
    for l in range(inBootNum):
        learnIdx = np.random.choice(np.arange(sampleNum), int(sampleNum * inBootRatio), replace = False)
        
        learnFtrMat = inLearnFtrMat[learnIdx]
        learnLabel = inLearnLabel[learnIdx]
    
        adaBoost = CAdaBoost()
        adaBoost.SetParam(inAdaBoostParam = inAdaBoostParam)
    
        adaBoost.Boost( trainScoreMat = learnFtrMat,
                        labelList = learnLabel)
        weakScoreMat = adaBoost.CalcWeakScore()
        refScoreVec = adaBoost.CalcWeakScore(label = evalLabel,
                                             scoreMat = evalVec.reshape(1, -1))
        
        distMat[learnIdx, distCnt[learnIdx]] += np.sum(np.abs(weakScoreMat - refScoreVec), axis = 1)
        distCnt[learnIdx] += 1
        
    distVec = np.zeros(sampleNum).astype(np.float)
    for i in range(sampleNum):
        if distCnt[i] > 0:
            distVec[i] = np.median(distMat[i][:distCnt[i]])
        else:
            distVec[i] = 0.0
    return distVec
        
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

def main():
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
                             inBootNum = 10,
                             inBootRatio = 0.5,
                             inUseFeatNum = learnFtrMat.shape[1])
    

    plotNum = 300
    
    # 最も悪影響を与えてる学習サンプルを必ず評価に入れる
    argsortedDist = np.argsort(learnDistVec)

    skippedIdx = np.linspace(0, argsortedDist.size - 1, plotNum // 5 * 3).astype(np.int)
    skippedIdx = np.append(skippedIdx, argsortedDist[- plotNum // 5:])
    skippedIdx = np.append(skippedIdx, argsortedDist[ :plotNum // 5 ])
    skippedIdx = np.unique(skippedIdx)
    
    plotPosIdx = (learnLabel[skippedIdx] ==  1)
    plotNegIdx = (learnLabel[skippedIdx] == -1)
    
    resFile = "result.xlsx"
    if not os.path.exists(resFile):
        print("write")
        plotModEvalScore = np.empty(0).astype(np.float)
        print("checking re-training")
        for i in tqdm(skippedIdx):
            _1, _2, modEvalScoreVec, _3 = smallSampleTry(hyperParam = adaBoostParam,
                                                     learnFtrMat = np.delete(learnFtrMat, i, axis = 0),
                                                     learnLabel = np.delete(learnLabel, i),
                                                     evalFtrMat = evalFtrMat,
                                                     evalLabel = evalLabel)
            plotModEvalScore = np.append(plotModEvalScore, modEvalScoreVec[evalTgtIdx])
 
        writer = pd.ExcelWriter(resFile, engine = 'xlsxwriter')
        df = pd.DataFrame()
        df["plotModEvalScore"] = plotModEvalScore
        df.to_excel(writer, sheet_name = "result")
        pd.DataFrame.from_dict(adaBoostParam).to_excel(writer, sheet_name = "AdaParam")
        pd.DataFrame.from_dict(hogParam).to_excel(writer, sheet_name = "HogParam")
        writer.save()
        writer.close()
        print(plotModEvalScore)

    else:
        print("read")
        plotModEvalScore = np.array(pd.read_excel(resFile, sheetname = "result")["plotModEvalScore"])

    print("tgt cls:", evalLabel[evalTgtIdx])
    x = plotModEvalScore - refTgtEvalScore
    
    # 近似
    y1 = learnDistVec[skippedIdx]
    # 実際
    y2 = np.sqrt(np.sum((learnFtrMat[skippedIdx] - evalFtrMat[evalTgtIdx]) ** 2, axis = 1))

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

if "__main__" == __name__:
    main()
    print("Done.")



