#coding: utf-8
from common.origLib import *
from adaBoost import *
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
from common.mathtool import *

def OrgBoot(inLearnFtrMat, inLearnLabel, evalVec, evalLabel, inAdaBoostParam, inBootNum, inBootRatio, inUseFeatNum, boostOrder, adaLoop, fastScanRate, skippedIdx):
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
    
    out = np.empty(skippedIdx.size).astype(np.float)
    
    for i in tqdm(range(skippedIdx.size)):
        renIdx  = skippedIdx[i]
        adaBoost = CAdaBoost()
        param = inAdaBoostParam.copy()
        if boostOrder.size != 0:
            param["GivenOrder"] = boostOrder
        adaBoost.SetParam(inAdaBoostParam = param)
        useIdx = np.ones(sampleNum).astype(np.bool)
        useIdx[renIdx] = False
        adaBoost.Boost( trainScoreMat = inLearnFtrMat[useIdx],
                        labelList = inLearnLabel[useIdx])
        evalScore = adaBoost.Evaluate(testScoreMat = evalVec.reshape(1, -1),
                                  label = evalLabel)
        out[i] = float(evalScore)
    return out


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
    
    l = 0
    while 1:
        if inBootRatio != 1.0:
            learnIdx = np.random.choice(np.arange(sampleNum), int(sampleNum * inBootRatio), replace = False)
        else:
            learnIdx = np.arange(sampleNum)
        
        learnFtrMat = inLearnFtrMat[learnIdx]
        learnLabel = inLearnLabel[learnIdx]

        adaBoost = CAdaBoost()
        param = inAdaBoostParam.copy()
        if None != adaLoop:
            param["Loop"] = adaLoop
        if None != fastScanRate:
            param["FastScan"] = int(fastScanRate * inLearnFtrMat.shape[1])
        elif 1.0 == fastScanRate:
            param["FastScan"] = inLearnFtrMat.shape[1]
            
        adaBoost.SetParam(inAdaBoostParam = param)
        adaBoost.Boost( trainScoreMat = learnFtrMat,
                        labelList = learnLabel)

        alpha = inAdaBoostParam["SaturateLevel"]
        posVoteOrg, negVoteOrg = adaBoost.GetVoteNum()
        expPosOrg = (posVoteOrg ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10
        expNegOrg = (negVoteOrg ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10

        kiyoOrg = np.zeros(expPosOrg.shape).astype(np.float)
        validIdxOrg = np.zeros(kiyoOrg.shape).astype(np.bool)
        validIdxOrg[expPosOrg + expNegOrg > 0.0] = True
        kiyoOrg[validIdxOrg] = (expPosOrg - expNegOrg)[validIdxOrg] / (expPosOrg + expNegOrg)[validIdxOrg]
        voteWeight = adaBoost.GetVoteWeight()
        n = 0
        for idx in learnIdx:
            posVote = posVoteOrg.copy()
            negVote = negVoteOrg.copy()
            assert(not np.isnan(posVote).any())
            assert(not np.isnan(negVote).any())
            
            hikuValid = np.zeros(posVote.shape).astype(np.bool)
            hikuValid[np.arange(hikuValid.shape[0]), learnFtrBin[idx]] = True
            
            if inLearnLabel[idx] == 1:
                posVote[hikuValid] = posVote[hikuValid] - voteWeight[n]
                valid = (np.sum(posVote,axis=1)>0.0).astype(np.bool)
                negVote[valid] /= np.sum(posVote,axis=1)[valid].reshape(-1, 1)
            elif inLearnLabel[idx] == -1:
                negVote[hikuValid] = negVote[hikuValid] - voteWeight[n]
                assert(not np.isnan(negVote).any())
                valid = (np.sum(negVote,axis=1)>0.0).astype(np.bool)
                negVote[valid] /= np.sum(negVote,axis=1)[valid].reshape(-1, 1)
            assert(not np.isnan(posVote).any())
            assert(not np.isnan(negVote).any())
            expPos = np.zeros(posVote.shape).astype(np.float)
            expPos[posVote > 0.0] = (posVote[posVote > 0.0] ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10
            assert(not np.isnan(expPos).any())
            expNeg = np.zeros(negVote.shape).astype(np.float)
            expNeg[negVote > 0.0] = (negVote[negVote > 0.0] ** alpha) + inAdaBoostParam["Regularizer"] + 1e-10
            assert(not np.isnan(expNeg).any())
            validIdx = np.zeros(expPos.shape).astype(np.bool)
            validIdx[expPos + expNeg > 0.0] = True
            kiyo = np.zeros(validIdx.shape).astype(np.float)
            kiyo[validIdx] = (expPos - expNeg)[validIdx] / (expPos + expNeg)[validIdx]

            #smoothing
            if 1:
                ada_bin = posVote.shape[1]
                smoother = np.zeros((ada_bin, ada_bin)).astype(np.float)
                ran = 2
                for i in range(ada_bin):
                    smoother[i][max(i-ran,0):min(i+ran+1,ada_bin)] = 1.0
                smoother /= np.sum(smoother, axis = 1).reshape(-1, 1)
                for i in range(kiyo.shape[0]):
                    kiyo[i] = np.dot(kiyo[i], smoother.T)
            
            assert(not np.isnan(kiyo).any())
            distMat[idx, distCnt[idx]] = np.sum((kiyo - kiyoOrg)[np.arange(kiyo.shape[0]), evalVecBin])
            assert(not np.isnan(distMat).any())
            n += 1
        distCnt[learnIdx] += 1
        l += 1
        print(l, np.sum(distCnt == 0), np.min(distCnt))
        if (l >= inBootNum) and (np.min(distCnt) > 0):
            break
        
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

    return adaTable[np.argsort(boostOrder)], learnScore, evalScore, evalLossVec, boostOrder

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

    refAdaTable, reflearnScoreVec, refEvalScoreVec, refEvalLossVec, boostOrder = smallSampleTry(hyperParam = adaBoostParam,
                                                                     learnFtrMat = learnFtrMat,
                                                                     learnLabel = learnLabel,
                                                                     evalFtrMat = evalFtrMat,
                                                                     evalLabel = evalLabel)
    evalTgtIdx = np.argsort(refEvalLossVec)[-1]
    
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

    skippedIdx = np.linspace(0, learnImg.shape[0] - 1, 1000).astype(np.int)
    skippedIdx = np.unique(skippedIdx)
    plotNum = skippedIdx.size
    
    '''
    modScore = OrgBoot(inLearnFtrMat = learnFtrMat,
                             inLearnLabel = learnLabel,
                             evalVec = evalFtrMat[evalTgtIdx],
                             evalLabel = evalLabel[evalTgtIdx], 
                             inAdaBoostParam = adaBoostParam,
                             inBootNum = 1,
                             inBootRatio = 1.0,
                             inUseFeatNum = learnFtrMat.shape[1],
                             boostOrder = boostOrder,
                             adaLoop = None,
                             fastScanRate = None,
                             skippedIdx = skippedIdx)
    '''
    
    
    
    plotPosIdx = (learnLabel[skippedIdx] ==  1)
    plotNegIdx = (learnLabel[skippedIdx] == -1)
    
    resFile = "result.csv"
    if not os.path.exists(resFile):
        print("write")
        plotModEvalScore = np.empty(0)
        print("checking re-training")
        for i in tqdm(skippedIdx):
            _1, _2, modEvalScoreVec, _3, _4 = smallSampleTry(hyperParam = adaBoostParam,
                                                     learnFtrMat = np.delete(learnFtrMat, i, axis = 0),
                                                     learnLabel = np.delete(learnLabel, i),
                                                     evalFtrMat = evalFtrMat,
                                                     evalLabel = evalLabel)
            if plotModEvalScore.size == 0:
                plotModEvalScore = modEvalScoreVec.reshape(1, -1)
            else:
                plotModEvalScore = np.append(plotModEvalScore, modEvalScoreVec.reshape(1, -1), axis = 0)
        
        np.savetxt(resFile, plotModEvalScore)
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
        plotModEvalScore = np.loadtxt(resFile)
        
    print("tgt cls:", evalLabel[evalTgtIdx])
    assert(refEvalScoreVec[evalTgtIdx].size == 1)
    assert(plotModEvalScore.T[evalTgtIdx].shape == (plotNum,))
    x = plotModEvalScore.T[evalTgtIdx] - refEvalScoreVec[evalTgtIdx]
    
    fig = plt.figure(figsize=(18,9))

    y = np.sqrt(np.sum((learnFtrMat[skippedIdx] - evalFtrMat[evalTgtIdx]) ** 2, axis = 1))
    ax = fig.add_subplot(231)
    ax.plot(x[plotPosIdx], y[plotPosIdx], ".", color="red")
    ax.grid(True)
    ax.set_title("Pos Euclid Dist.")
    ax = fig.add_subplot(234)
    ax.plot(x[plotNegIdx], y[plotNegIdx], ".", color="blue")
    ax.grid(True)
    ax.set_title("Neg Euclid Dist.")

    y = learnDistMedianVec[skippedIdx]
    #y = modScore
    ax = fig.add_subplot(232)
    ax.plot(x[plotPosIdx], y[plotPosIdx], ".", color="red")
    ax.grid(True)
    ax.set_title("Pos BoostBoot Median.")
    ax = fig.add_subplot(235)
    ax.plot(x[plotNegIdx], y[plotNegIdx], ".", color="blue")
    ax.grid(True)
    ax.set_title("Neg BoostBoot Median.")

    y = learnDistMeanVec[skippedIdx]
    #y = modScore
    ax = fig.add_subplot(233)
    ax.plot(x[plotPosIdx], y[plotPosIdx], ".", color="red")
    ax.grid(True)
    ax.set_title("Pos BoostBoot Mean.")
    ax = fig.add_subplot(236)
    ax.plot(x[plotNegIdx], y[plotNegIdx], ".", color="blue")
    ax.grid(True)
    ax.set_title("Neg BoostBoot Mean.")
    
    plt.savefig("BoostBootTrial.png")
    plt.show()

if "__main__" == __name__:
    main()
    print("Done.")



