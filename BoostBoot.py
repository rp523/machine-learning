#coding: utf-8
from common.origLib import *
from adaBoost import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
from common.mathtool import *


def BoostBoot(inLearnFtrMat, inLearnLabel, evalVec, evalLabel, inAdaBoostParam, inBootNum, inBootRatio, inUseFeatNum, adaLoop, fastScanRate):
    sampleNum = inLearnFtrMat.shape[0]
    featureNum = inLearnFtrMat.shape[1]
    
    distMat = np.zeros((sampleNum, inBootNum)).astype(np.float)
    distCnt = np.zeros(sampleNum).astype(np.int)
    
    ran = 0
    learnFtrBin = (inLearnFtrMat * inAdaBoostParam["Bin"]).astype(np.int)
    learnFtrBin[learnFtrBin == inAdaBoostParam["Bin"]] = inAdaBoostParam["Bin"] - 1
    learnFtrBinFlg = np.zeros((learnFtrBin.shape[0], learnFtrBin.shape[1], inAdaBoostParam["Bin"])).astype(np.bool)
    for s in (range(learnFtrBin.shape[0])):
        for f in range(learnFtrBin.shape[1]):
            learnFtrBinFlg[s,f,max(0, learnFtrBin[s,f] - ran):min(inAdaBoostParam["Bin"], learnFtrBin[s,f] + ran + 1)] = True

    evalVecBin = (evalVec * inAdaBoostParam["Bin"]).astype(np.int)
    evalVecBin[evalVecBin == inAdaBoostParam["Bin"]] = inAdaBoostParam["Bin"] - 1
    evalVecBinFlg = np.zeros((1, evalVecBin.size, inAdaBoostParam["Bin"])).astype(np.bool)

    for l in range(inBootNum):
        learnIdx = np.random.choice(np.arange(sampleNum), int(sampleNum * inBootRatio), replace = False)
        learnFtrMat = inLearnFtrMat[learnIdx]
        learnLabel = inLearnLabel[learnIdx]

        adaBoost = CAdaBoost()
        param = inAdaBoostParam.copy()
        if None != adaLoop:
            param["Loop"] = adaLoop
        if None != fastScanRate:
            param["FastScan"] = int(fastScanRate * inLearnFtrMat.shape[1])
        adaBoost.SetParam(inAdaBoostParam = param)
        adaBoost.Boost( trainScoreMat = learnFtrMat,
                        labelList = learnLabel)

        alpha = inAdaBoostParam["SaturateLevel"]
        posVoteOrg, negVoteOrg = adaBoost.GetVoteNum()
        expPosOrg = (posVoteOrg ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10
        expNegOrg = (negVoteOrg ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10
        kiyoOrg = (expPosOrg - expNegOrg) / (expPosOrg + expNegOrg)
        voteWeight = adaBoost.GetVoteWeight()
        n = 0
        for idx in learnIdx:
            posVote = posVoteOrg.copy()
            negVote = negVoteOrg.copy()
            if inLearnLabel[idx] == 1:
                posVote[np.arange(posVote.shape[0]), learnFtrBin[idx]] -= voteWeight[n]
                posVote *= np.sum(learnLabel == 1) / (np.sum(learnLabel == 1) - 1)
                #posVote /= np.sum(posVote,axis=1).reshape(-1, 1)
            elif inLearnLabel[idx] == -1:
                negVote[np.arange(negVote.shape[0]), learnFtrBin[idx]] -= voteWeight[n]
                negVote *= np.sum(learnLabel == -1) / (np.sum(learnLabel == -1) - 1)
                #negVote /= np.sum(negVote,axis=1).reshape(-1, 1)
            expPos = (posVote ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10
            expNeg = (negVote ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10
            kiyo = (expPos - expNeg) / (expPos + expNeg)
            distMat[idx, distCnt[idx]] = np.sum((kiyo - kiyoOrg)[np.arange(kiyo.shape[0]), evalVecBin])
            n += 1
        distCnt[learnIdx] += 1
        print(l, np.sum(distCnt == 0), np.min(distCnt))
    '''    
    for f in range(evalVecBin.size):
        evalVecBinFlg[0,f,max(0, evalVecBin[0,f] - ran):min(inAdaBoostParam["Bin"], evalVecBin[0,f] + ran + 1)] = True

    for l in range(inBootNum):
        learnIdx = np.random.choice(np.arange(sampleNum), int(sampleNum * inBootRatio), replace = False)
        
        learnFtrMat = inLearnFtrMat[learnIdx]
        learnLabel = inLearnLabel[learnIdx]
        
        fastBoostParam = inAdaBoostParam.copy()
        fastBoostParam["Loop"] = adaLoop
        fastBoostParam["FastScan"] = int(fastScanRate * inLearnFtrMat.shape[1])
        adaBoost = CAdaBoost()
        adaBoost.SetParam(inAdaBoostParam = fastBoostParam)
    
        adaBoost.Boost( trainScoreMat = learnFtrMat,
                        labelList = learnLabel)
        weakScoreMat = adaBoost.CalcWeakScore()
        refScoreVec = adaBoost.CalcWeakScore(label = evalLabel,
                                             scoreMat = evalVec.reshape(1, -1))
        boostOrder, relia = adaBoost.GetLearnedParam()
        table = adaBoost.GetLearnedTable()
        
        base = np.arange(table.shape[0]).astype(np.int) * table.shape[1]
        for idx in learnIdx:
            
            learnedKiyo = table.flatten()[base + learnFtrBin[idx][np.sort(boostOrder)]]
            assert(learnedKiyo.shape == (table.shape[0],))
            evalKiyo = table.flatten()[base + evalVecBin[0][np.sort(boostOrder)]]
            assert(evalKiyo.shape == (table.shape[0],))
            distMat[idx, distCnt[idx]] += np.sum(np.abs(learnedKiyo - evalKiyo))
        distCnt[learnIdx] += 1
        print(l, np.sum(distCnt == 0), np.min(distCnt))
    ''' 
    medianVec = np.zeros(sampleNum).astype(np.float)
    meanVec = np.zeros(sampleNum).astype(np.float)
    for i in range(sampleNum):
        if distCnt[i] > 0:
            medianVec[i] = np.median(distMat[i][:distCnt[i]])
            meanVec[i] = np.average(distMat[i][:distCnt[i]])
        else:
            medianVec[i] = 0.0
            meanVec[i] = 0.0
    return medianVec, meanVec
        
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
    hogParam["Cell"]["X"] = 3
    hogParam["Cell"]["Y"] = 6
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

    refAdaTable, reflearnScoreVec, refEvalScoreVec, refEvalLossVec = smallSampleTry(hyperParam = adaBoostParam,
                                                                     learnFtrMat = learnFtrMat,
                                                                     learnLabel = learnLabel,
                                                                     evalFtrMat = evalFtrMat,
                                                                     evalLabel = evalLabel)
    evalTgtIdx = np.argsort(refEvalLossVec)[-1]
    refTgtEvalScore = refEvalScoreVec[evalTgtIdx]
    
    learnDistMedianVec, learnDistMeanVec = BoostBoot(inLearnFtrMat = learnFtrMat,
                             inLearnLabel = learnLabel,
                             evalVec = evalFtrMat[evalTgtIdx],
                             evalLabel = evalLabel[evalTgtIdx], 
                             inAdaBoostParam = adaBoostParam,
                             inBootNum = 1,
                             inBootRatio = 1.0,
                             inUseFeatNum = learnFtrMat.shape[1],
                             adaLoop = None,
                             fastScanRate = None)

    plotNum = 1000
    
    skippedIdx = np.linspace(0, learnImg.shape[0] - 1, plotNum).astype(np.int)
    skippedIdx = np.unique(skippedIdx)
    
    plotPosIdx = (learnLabel[skippedIdx] ==  1)
    plotNegIdx = (learnLabel[skippedIdx] == -1)
    
    resFile = "result.csv"
    if not os.path.exists(resFile):
        print("write")
        plotModEvalScore = []
        print("checking re-training")
        for i in tqdm(skippedIdx):
            _1, _2, modEvalScoreVec, _3 = smallSampleTry(hyperParam = adaBoostParam,
                                                     learnFtrMat = np.delete(learnFtrMat, i, axis = 0),
                                                     learnLabel = np.delete(learnLabel, i),
                                                     evalFtrMat = evalFtrMat,
                                                     evalLabel = evalLabel)
            plotModEvalScore.append(modEvalScoreVec[evalTgtIdx])
        
        np.savetxt(resFile, np.array(plotModEvalScore))
        '''
        writer = pd.ExcelWriter(resFile, engine = 'xlsxwriter')
        df = pd.DataFrame()
        df["plotModEvalScore"] = plotModEvalScore
        df.to_excel(writer, sheet_name = "result")
        pd.DataFrame.from_dict(adaBoostParam).to_excel(writer, sheet_name = "AdaParam")
        pd.DataFrame.from_dict(hogParam).to_excel(writer, sheet_name = "HogParam")
        writer.save()
        writer.close()
        '''

    else:
        print("read")
        plotModEvalScore = np.loadtxt(resFile).flatten()
        
    print("tgt cls:", evalLabel[evalTgtIdx])
    x = plotModEvalScore - refTgtEvalScore
    
    fig = plt.figure()

    y = np.sqrt(np.sum((learnFtrMat[skippedIdx] - evalFtrMat[evalTgtIdx]) ** 2, axis = 1))
    ax = fig.add_subplot(131)
    ax.plot(x[plotPosIdx], y[plotPosIdx], ".", color="red")
    ax.plot(x[plotNegIdx], y[plotNegIdx], ".", color="blue")
    ax.grid(True)
    ax.set_title("Euclid Dist.")

    y = learnDistMedianVec[skippedIdx]
    ax = fig.add_subplot(132)
    ax.plot(x[plotPosIdx], y[plotPosIdx], ".", color="red")
    ax.grid(True)
    ax = fig.add_subplot(132)
    ax.plot(x[plotNegIdx], y[plotNegIdx], ".", color="blue")
    ax.grid(True)
    ax.set_title("BoostBoot Median.")

    y = learnDistMeanVec[skippedIdx]
    ax = fig.add_subplot(133)
    ax.plot(x[plotPosIdx], y[plotPosIdx], ".", color="red")
    ax.grid(True)
    ax = fig.add_subplot(133)
    ax.plot(x[plotNegIdx], y[plotNegIdx], ".", color="blue")
    ax.grid(True)
    ax.set_title("BoostBoot Mean.")
    
    plt.savefig("BoostBootTrial.png")
    plt.show()

if "__main__" == __name__:
    main()
    print("Done.")



