#coding: utf-8
import os, shutil, sys
import numpy as np
from feature.hog import CHogParam,CHog
from feature.chnFtrs import *
import common.imgtool as imt
import common.mathtool as mt
import pandas as pd
import common.fileIO as fio
from matplotlib import pyplot as plt
from PIL import Image
from common.origLib import *
from input import *
from preproc import *
from decisionTree import *
from tqdm import tqdm
import gui

class AdaBoostParam(CParam):
    def __init__(self):
        setDicts = dicts()
        setDicts["Type"] = selparam("Discrete", "Real", "RealTree")
        setDicts["Saturate"] = True
        setDicts["SaturateLevel"] = 0.4
        setDicts["Bin"] = 32
        setDicts["Loop"] = 10000
        setDicts["Regularizer"] = 1e-5
        setDicts["TreeDepth"] = 1
        setDicts["regDataDist"] = 0.0
        setDicts["verbose"] = True
        setDicts["saveDetail"] = False
        super().__init__(setDicts) 

class CAdaBoost:
    
    def __init__(self):
        pass
    
    def SetParam(self, inImgList,inLabelList,inDetectorList, inAdaBoostParam = None,):
        
        if None != inAdaBoostParam:
            adaBoostParam = inAdaBoostParam
        else:
            adaBoostParam = AdaBoostParam()
            
        self.__adaType = adaBoostParam["Type"].nowSelected()
        self.__bin = adaBoostParam["Bin"]
        self.__loopNum = adaBoostParam["Loop"]
        self.__regularize = adaBoostParam["Regularizer"]
        self.__saturate = adaBoostParam["Saturate"]
        self.__saturateLevel = adaBoostParam["SaturateLevel"]
        self.__treeDepth = adaBoostParam["TreeDepth"]
        self.__regDataDist = adaBoostParam["regDataDist"]
        self.__verbose = adaBoostParam["verbose"]
        self.__saveDetail = adaBoostParam["saveDetail"]
        self.__detectorList = inDetectorList
        self.__imgList = inImgList
        self.__labelList = np.array(inLabelList)
        self.__featureLen = None
        
        self.__thresh = np.zeros(len(inDetectorList),float)
        
    def __GetFeatureLength(self):
        if None == self.__featureLen:
            self.__featureLen = 0
            for detector in self.__detectorList:
                assert(detector.GetFeatureLength)
                self.__featureLen = self.__featureLen + detector.GetFeatureLength()
        return self.__featureLen

    # PosとNegが混在しているスコア行列からPosのみ、Negのみのスコア行列に分離する
    def __PosNegDevide(self,inScoreMat,inLabelList):
        assert(inScoreMat.shape[0] == np.array(inLabelList).size)
        det = inScoreMat.shape[1]
        scoreMat = inScoreMat.copy()     # 入力サンプル x 識別器
        posScoreMat = np.empty((0,det),float)   # 入力サンプル x 識別器
        negScoreMat = np.empty((0,det),float)   # 入力サンプル x 識別器
        for i in range(len(inLabelList)):
            if 1 == inLabelList[i]:
                posScoreMat = np.append(posScoreMat, [scoreMat[i]],axis=0)
            elif -1 == inLabelList[i]:
                negScoreMat = np.append(negScoreMat, [scoreMat[i]],axis=0)
            else:
                assert(0 and "bug!")

        return posScoreMat, negScoreMat
        
    class Reliability:
        def __init__(self,reliability):
            self.__reliability = reliability
        def calc(self,bin):
            return self.__reliability[bin]

    def Boost(self, trainScoreMat, labelList):

        assert(isinstance(trainScoreMat, np.ndarray))
        assert(trainScoreMat.ndim == 2)
        assert(not np.any(trainScoreMat < 0))
        assert(isinstance(labelList, np.ndarray))
        assert(labelList.ndim == 1)
        assert(trainScoreMat.shape[0] == labelList.size)
        

        sampleNum   = trainScoreMat.shape[0]
        detectorNum = trainScoreMat.shape[1]
        adaLoop = min(self.__loopNum,detectorNum)
 
        # スコア行列をPos/Negで分ける
        trainPosScore, trainNegScore = self.__PosNegDevide(trainScoreMat, labelList)
        assert(not np.any(trainPosScore < 0))
        assert(not np.any(trainNegScore < 0))
        posSample = trainPosScore.shape[0]
        negSample = trainNegScore.shape[0]

        # サンプルデータの重みを初期化
        posSampleWeight = np.array([1.0/(posSample)]*posSample)
        negSampleWeight = np.array([1.0/(negSample)]*negSample)
        assert(posSampleWeight.size == posSample)
        assert(negSampleWeight.size == negSample)
    
        # 強識別器情報の記録メモリを確保
        strongDetBin = np.zeros((adaLoop,self.__bin)).astype(np.float)
        strongDetID = np.zeros(adaLoop).astype(np.int)

        if self.__adaType == "RealTree":
            
            def assign(sortScoreVec, sortIndexVec, threshVec):
                assert(np.array(sortScoreVec).ndim == 1)
                assert(np.array(sortIndexVec).ndim == 1)
                assert(np.array(threshVec).ndim == 1)
                out = []
                sortIndex = sortIndexVec
                threshOld = -1e10
                for thresh in threshVec:
                    valid = ((threshOld < sortScoreVec) * (sortScoreVec <= thresh)).astype(np.bool)
                    out.append(list(sortIndex[valid]))

                    invalid = np.bitwise_not(valid)
                    sortIndex = sortIndex[invalid]    # delete selected-ones.
                    sortScoreVec = sortScoreVec[invalid]
                    threshOld = thresh
                out.append(list(sortIndex)) # ones over the max thresh
                return out
                        
            labelList = np.array(self.__labelList).astype(np.int)
            sampleWeights = np.ones(sampleNum)
            sampleWeights[labelList ==  1] = sampleWeights[labelList ==  1] / np.sum(sampleWeights[labelList ==  1])
            sampleWeights[labelList == -1] = sampleWeights[labelList == -1] / np.sum(sampleWeights[labelList == -1])
            
            nodes = []
            sampleIndexes = np.arange(sampleNum)
            
            assert(self.__trainScoreMat.shape[0] == detectorNum)
            for d in range(detectorNum):
                assert(np.min(self.__trainScoreMat[d]) != np.max(self.__trainScoreMat[d]))
                assert(np.min(labelList) != np.max(labelList))
                sortIndex = np.argsort(self.__trainScoreMat[d])
                node = DecisionTree.Node(   scoreVec = self.__trainScoreMat[d][sortIndex],
                                            labelVec = labelList[sortIndex],
                                            maxDepth = self.__treeDepth,
                                            regDataDist = self.__regDataDist)
                if (len(node.getThresh()) <= 0):
                    print(node.getThresh())
                    print(self.__trainScoreMat[d][sortIndex])
                    print(labelList[sortIndex])
                assert(len(node.getThresh()) > 0)
                if(np.max(node.getThresh()) == np.min(node.getThresh())):
                    print(node.getThresh())
                nodes.append(node)
            
            detIdxSaved = np.arange(detectorNum)
            detIdx = detIdxSaved
            self.__treeThresh = []
            self.__treeScore = []

            sortIndexMat = np.zeros((detectorNum, sampleNum)).astype(np.int)
            sortIndexMat[np.arange(detectorNum)] = np.arange(sampleNum)
            sortScoreMat = np.empty((detectorNum, sampleNum))
            argsortScoreMat = np.zeros((detectorNum, sampleNum)).astype(np.int)
            for d in range(detectorNum):
                argsortScoreMat[d] = np.argsort(self.__trainScoreMat[d])
                sortScoreMat[d] = self.__trainScoreMat[d][argsortScoreMat[d]]
                sortIndexMat[d] =sortIndexMat[d][argsortScoreMat[d]]

            assignedListMat = []    
            for d in range(detectorNum):
                assignedList = assign(sortScoreVec = sortScoreMat[d],
                                      sortIndexVec = sortIndexMat[d],
                                      threshVec = nodes[d].getThresh())
                assert(isinstance(assignedList, list))
                assignedListMat.append(assignedList)

            # 指定数だけ弱識別器をブースト選択するループ
            for w in range(adaLoop):
                
                # ブーストの過程で長さを変えない
                assert(sampleWeights.size == sampleNum)
                assert(labelList.size == sampleNum)
                
                # まだ選択されず残っている弱識別器の中からベストを決める
                Z  = np.zeros(detIdx.size)
                Wp = np.zeros((detIdx.size, 2 ** self.__treeDepth))
                Wm = np.zeros((detIdx.size, 2 ** self.__treeDepth))

                for d in range(detIdx.size):
                    for a in range(len(assignedListMat[d])):
                        assigned = assignedListMat[d][a]
                        assert(isinstance(assigned,list))
                        assigned = np.array(assigned)
                        if assigned.size > 0:
                            assert(assigned.ndim == 1)
                            isPositive = (labelList[assigned] ==  1)
                            isNegative = np.bitwise_not(isPositive)
                            Wp[d][a] = np.sum( sampleWeights[assigned] * isPositive)
                            Wm[d][a] = np.sum( sampleWeights[assigned] * isNegative)

                Z = np.sqrt( np.sum(Wp * Wm, axis = 1))
                assert(Wp.shape == (detIdx.size, 2 ** self.__treeDepth))
                assert(Wm.shape == (detIdx.size, 2 ** self.__treeDepth))
                assert(Z.size == detIdx.size)
                bestIdx  = np.argmin(Z)
                finalIdx = detIdx[bestIdx]
                bestAssignLen = len(assignedListMat[bestIdx])

                epsilon = 1e-10
                if not self.__saturate:
                    reliabilityBest  = 0.5 * np.log((Wp[bestIdx] + epsilon + self.__regularize) \
                                                  / (Wm[bestIdx] + epsilon + self.__regularize))
                else:
                    WpBest = (Wp[bestIdx] + self.__regularize) ** self.__saturateLevel
                    WmBest = (Wm[bestIdx] + self.__regularize) ** self.__saturateLevel
                    WpBest = WpBest[:bestAssignLen]
                    WmBest = WmBest[:bestAssignLen]
                    assert(WpBest.ndim == 1)
                    assert(WmBest.ndim == 1)
                    assert(WpBest.size == bestAssignLen)
                    assert(WmBest.size == bestAssignLen)
                    if (WpBest + WmBest != 0.0).all():
                        reliabilityBest = (WpBest - WmBest) / (WpBest + WmBest)
                    else:
                        assert(0)
                
                assert((reliabilityBest != 0.0).any())
                
                # record selected feature
                self.__treeThresh.append(nodes[finalIdx].getThresh())
                self.__treeScore.append(list(reliabilityBest))

                if not (reliabilityBest.size == bestAssignLen):
                    print(reliabilityBest.size, bestAssignLen)
                assert(reliabilityBest.size == bestAssignLen)

                # サンプル重みを更新し、ポジネガそれぞれ正規化
                scores = np.empty(sampleNum)

                for a in range(len(assignedListMat[bestIdx])):
                    assigned = assignedListMat[bestIdx][a]
                    scores[assigned] = reliabilityBest[a]
                assert(scores.size == sampleNum)

                sampleWeights = sampleWeights * (np.exp( - labelList * scores)) # - np.max( - labelList * scores)
                sampleWeights[labelList ==  1] = sampleWeights[labelList ==  1] / np.sum(sampleWeights[labelList ==  1])
                sampleWeights[labelList == -1] = sampleWeights[labelList == -1] / np.sum(sampleWeights[labelList == -1])
                assert(not np.any(np.isnan(sampleWeights)))
                assert(np.all(sampleWeights > 0))
                
                strongDetID[w] = finalIdx
                detIdx = np.delete(detIdx, bestIdx)
                sortScoreMat = np.delete(sortScoreMat, bestIdx, axis = 0)
                del assignedListMat[bestIdx]
                
                if self.__verbose:
                    print("boosting weak detector:", w + 1)

        elif self.__adaType == "Discrete":
            
            self.__decisionTree = DecisionTree(scoreMat = self.__trainScoreMat,
                                        labelVec = self.__labelList,
                                        maxDepth = 2)
            scoreMat = self.__decisionTree.predict(np.arange(detectorNum), self.__trainScoreMat)

            detIdxSaved = np.arange(detectorNum)
            detIdx = detIdxSaved    
            self.__detWeights = np.empty(adaLoop)

            sampleWeights = np.ones(sampleNum) / sampleNum
            assert(sampleWeights.size == sampleNum)
            
            yfMatSaved = (scoreMat * self.__labelList).astype(np.int)
            yfMat = yfMatSaved
            
            for w in range(adaLoop):
                
                assert(detIdx.size == yfMat.shape[0])
                errorSum = np.sum(sampleWeights * (yfMat < 0), axis = 1)
                assert(detIdx.size == errorSum.size)
                bestIdx  = np.argmin(errorSum)

                epsilon = 1e-10
                reliabilityBest  = 0.5 * np.log((1.0 - errorSum[bestIdx] + epsilon + self.__regularize) \
                                                    / (errorSum[bestIdx] + epsilon + self.__regularize))
                
                self.__detWeights[w] = reliabilityBest

                # サンプル重みを更新し、ポジネガそれぞれ正規化
                sampleWeights = sampleWeights * np.exp( - self.__detWeights[w] * yfMat[bestIdx]
                                                    - np.max( - self.__detWeights[w] * yfMat[bestIdx]))
                sampleWeights = sampleWeights / np.sum(sampleWeights)
                assert(not np.any(np.isnan(sampleWeights)))
                assert(np.all(sampleWeights > 0))
                
                strongDetID[w] = detIdx[bestIdx]
                detIdx = np.delete(detIdx, bestIdx)
                yfMat  = np.delete(yfMat,  bestIdx, axis = 0)

                if self.__verbose:
                    print("boosting weak detector:", w + 1)

        elif self.__adaType == "Real":
            
            remainDetIDList = np.arange(detectorNum)
            
            # スコアをBIN値に換算
            trainPosBin = (trainPosScore * self.__bin).astype(np.int)
            trainNegBin = (trainNegScore * self.__bin).astype(np.int)
            # 万が一値がBIN値と同じ場合はBIN-1としてカウントする
            trainPosBin = trainPosBin * (trainPosBin < self.__bin) + (self.__bin - 1) * (trainPosBin >= self.__bin)   
            trainNegBin = trainNegBin * (trainNegBin < self.__bin) + (self.__bin - 1) * (trainNegBin >= self.__bin)
            # 負の値はとらないはずだが、一応確認
            assert(not (trainPosBin < 0).any())
            assert(not (trainNegBin < 0).any())

            print("real-adaboosting...")
            for w in tqdm(range(adaLoop)):

                # まだAdaBoostに選択されず残っている識別器の数
                detRemain = trainPosBin.shape[1]
                assert(detRemain == trainNegBin.shape[1])
                
                # 各識別器の性能を計算するための重み付きヒストグラム（識別器 x AdabootBin）を計算
                histoPos = np.zeros((detRemain, self.__bin))
                histoNeg = np.zeros((detRemain, self.__bin))
                for b in range(self.__bin):
                    histoPos[np.arange(detRemain),b] = np.dot((trainPosBin == b).T, posSampleWeight)
                    histoNeg[np.arange(detRemain),b] = np.dot((trainNegBin == b).T, negSampleWeight)
                # 残っている弱識別器から最優秀のものを選択            
                bestDet = np.argmin(np.sum(np.sqrt(histoPos * histoNeg), axis=1))
                
                #ゼロ割り回避
                epsilon = 1e-10
                histoPos += epsilon
                histoNeg += epsilon
                
                # 最優秀識別器の信頼性を算出
                if self.__saturate == False:
                    h = 0.5 * np.log((histoPos[bestDet] + self.__regularize)
                                    /(histoNeg[bestDet] + self.__regularize))
                else:
                    alpha = 0.4
                    expPos = np.power(histoPos[bestDet], alpha) + self.__regularize
                    expNeg = np.power(histoNeg[bestDet], alpha) + self.__regularize
                    h = ( expPos - expNeg) / (expPos + expNeg)
                
                strongDetBin[w] = h
                strongDetID[w] = remainDetIDList[bestDet]
    
                # 選択した最優秀識別器のスコアをもとに、サンプル重みを更新
                #for i in range(sampleNum):
                #    d = bestDetectorID
                #    b = mt.IntMinMax(self.__trainScoreMat[d][i] * self.__bin, 0, self.__bin - 1)
                #    if (0 < histoPos[d][b]) and (0 < histoNeg[d][b]):
                #        sampleWeight[i] = sampleWeight[i] * np.exp(-1.0 * self.__labelList[i] * h[b])
                reliability = self.Reliability(h)
                
                # サンプル重みを更新する
                posSampleWeight *= np.exp(-1.0 * reliability.calc(bin = trainPosBin.T[bestDet]))
                negSampleWeight *= np.exp( 1.0 * reliability.calc(bin = trainNegBin.T[bestDet]))
                
                # 選択除去された弱識別器の情報を次ループでは考えない
                trainPosBin = np.delete(trainPosBin, bestDet, axis = 1)
                trainNegBin = np.delete(trainNegBin, bestDet, axis = 1)
                remainDetIDList = np.delete(remainDetIDList, bestDet)

                assert(not np.any(np.isnan(strongDetBin)))
                assert(not np.any(np.isnan(strongDetID)))

        self.__relia = strongDetBin
        self.__reliaID = strongDetID

        if self.__saveDetail:
            self.__SaveLearning(strongDetBin,
                                strongDetID,
                                trainScoreMat,
                                labelList)

    def Evaluate(self, testScoreMat, label):
        
        # 評価用サンプルに対する各弱識別器のスコアを算出

        assert((testScoreMat >= 0.0).all())
        assert((testScoreMat <= 1.0).all())
        binMat = (testScoreMat * self.__bin).astype(np.int)
        binMat[binMat >= self.__bin] = self.__bin - 1
        
        finalScore = np.zeros(testScoreMat.shape[0])
        reliaVec = self.__relia.flatten()
        base = np.arange(self.__reliaID.size) * self.__bin       
        
        if self.__adaType == "Real":
            print("evaluating sample...")
            for i in tqdm(range(finalScore.size)):
                scoreVec = reliaVec[base + binMat[i][self.__reliaID]]
                finalScore[i] = np.sum(scoreVec)

        elif self.__adaType == "Discrete":
            finalScore = np.dot(self.__detWeights, self.__decisionTree.predict(self.__reliaID, self.__testScoreMat)).T
        elif self.__adaType == "RealTree":
            didCount = 0
            for did in self.__reliaID:
                for sid in range(self.__testScoreMat.shape[1]):
                    score = self.__testScoreMat[did][sid]
                    isLarger = score > np.array(self.__treeThresh[didCount])
                    assignIdx = np.sum(isLarger)
                    finalScore[sid] += self.__treeScore[didCount][assignIdx]
                didCount += 1
        else:
            assert(0)
        
        if self.__saveDetail:
            self.__SaveEvaluation(scoreMat = testScoreMat,
                                  label = label,
                                  relia = self.__relia,
                                  reliaID = self.__reliaID)
            
        out = np.array(finalScore)
        return out
    # AdaBoostの計算過程を記録する(optional)
    def __SaveLearning(self, strongBin, strongID, scoreMat, trainLabel):

        # xlsxファイルに出力
        writer = pd.ExcelWriter("adaBoostDetail.xlsx", engine = 'xlsxwriter')
        
        # 学習サンプルごとの詳細記録シート
        learnDF = pd.DataFrame()
        
        # ラベルの記録
        learnDF["label"] = trainLabel
        
        # スコアをBIN値に換算
        trainBin = (scoreMat * self.__bin).astype(np.int)
        trainBin = trainBin * (trainBin < self.__bin) + (self.__bin - 1) * (trainBin >= self.__bin)
        scores = np.empty(trainBin.shape[0])
        print("calculating learning-sample for save...")
        base = np.arange(strongID.size) * self.__bin
        strongBinVec = strongBin.flatten()       
        for s in tqdm(range(scores.size)):
            scoreVec = strongBinVec[base + trainBin[s][strongID]]
            scores[s] = np.sum(np.sum(scoreVec))
        learnDF["score"] = scores

        # 全特徴スコアの記録
        for i in range(scoreMat.shape[1]):
            learnDF["feature{0:04d}".format(i)] = scoreMat.T[i]
        
        learnDF.to_excel(writer, sheet_name = "learn")

        # AdaBoostの詳細記録シート
        adaBoostDF = pd.DataFrame()
        
        # 特徴選択順
        adaBoostDF["boostOrder"] = strongID
        
        # 全寄与度の記録(選択順)
        for i in range(strongBin.shape[1]):
            adaBoostDF["bin{0}".format(i)] = strongBin.T[i]
        adaBoostDF.to_excel(writer, sheet_name = "adaBoost")
        
        writer.save()
        writer.close()
    
    # AdaBoostの計算過程を記録する(optional)
    def __SaveEvaluation(self, 
                         scoreMat,
                         label,
                         relia,
                         reliaID,
                         detailPath = None):
        assert(isinstance(scoreMat, np.ndarray))
        assert(isinstance(label, np.ndarray))
        assert(isinstance(relia, np.ndarray))
        assert(isinstance(reliaID, np.ndarray))
        assert(scoreMat.ndim == 2)
        assert(label.ndim == 1)
        assert(relia.ndim == 2)
        assert(reliaID.ndim == 1)
        assert(scoreMat.shape[0] == label.size)
        
        if None == detailPath:
            detailPath = "adaBoostDetail.xlsx"

        # 評価サンプルごとの詳細記録シート
        evalDF = pd.DataFrame()
        
        # 評価サンプルのラベルを記録。いらないかも
        evalDF["label"] = label

        # 評価スコアを記録
        binScoreMat = (scoreMat * self.__bin).astype(np.int)
        binScoreMat[binScoreMat >= self.__bin] = self.__bin - 1
        scores = np.empty(binScoreMat.shape[0])
        print("evaluating evaluation-sample for save...")
        base = np.arange(reliaID.size) * self.__bin
        strongBinVec = relia.flatten()       
        for s in tqdm(range(scores.size)):
            scoreVec = strongBinVec[base + binScoreMat[s][reliaID]]
            scores[s] = np.sum(np.sum(scoreVec))
        evalDF["score"] = scores

        # 全特徴スコアを記録
        featureColumn = []
        for i in range(scoreMat.shape[1]):
            evalDF["feature{0:04d}".format(i)] = scoreMat.T[i]

        # xlsxファイルに追記
        AddXlsxSheet(detailPath, "eval", evalDF)
        

    # 記録された詳細情報xlsxから情報抽出
    def Load(self, type, inDetailPath = None):
        if None == inDetailPath:
            detailPath = "adaBoostDetail.xlsx"
        
        if type == "learnLabel":
            return np.array(pd.read_excel(detailPath, sheetname = "learn")["label"])
        elif type == "learnFeature":
            out = []
            df = pd.read_excel(detailPath, sheetname = "learn")
            for col in df.columns:
                if col.find("feature") >= 0:
                    out.append(df[col])
            out = np.array(out).T
            return out
        elif type == "learnScore":
            out = np.array(pd.read_excel(detailPath, sheetname = "learn")["score"])
            return out
        elif type == "reliability":
            out = []
            df = pd.read_excel(detailPath, sheetname = "adaBoost")
            for col in df.columns:
                if col.find("bin") >= 0:
                    out.append(df[col])
            return np.array(out).T
        elif type == "boostOrder":
            return np.array(pd.read_excel(detailPath, sheetname = "adaBoost")["boostOrder"])
        elif type == "evalScore":
            return np.array(pd.read_excel(detailPath, sheetname = "eval")["score"])
        elif type == "evalFeature":
            out = []
            df = pd.read_excel(detailPath, sheetname = "eval")
            for col in df.columns:
                if col.find("feature") >= 0:
                    out.append(df[col])
            return np.array(out).T
        elif type == "evalLabel":
            return np.array(pd.read_excel(detailPath, sheetname = "eval")["label"])
        

if "__main__" == __name__:

    for xlsxFile in  GetFileList(".", includingText = ".xlsx"):
        os.remove(xlsxFile)
    for matFile in  GetFileList(".", includingText = ".mat"):
        os.remove(matFile)

    lp = dirPath2NumpyArray("dataset/INRIAPerson/LearnPos")
    ln = dirPath2NumpyArray("dataset/INRIAPerson/LearnNeg")
    ep = dirPath2NumpyArray("dataset/INRIAPerson/EvalPos" )
    en = dirPath2NumpyArray("dataset/INRIAPerson/EvalNeg" )
    learn = RGB2Gray(np.append(lp, ln, axis = 0), "green")
    eval  = RGB2Gray(np.append(ep, en, axis = 0), "green")
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))
    hogParam = CHogParam()
    hogParam["Bin"] = 8
    hogParam["Cell"]["X"] = 4
    hogParam["Cell"]["Y"] = 8
    hogParam["Block"]["X"] = 1
    hogParam["Block"]["Y"] = 1
    detectorList = [CHog(hogParam)]

    adaBoostParam = AdaBoostParam()
    adaBoostParam["Regularizer"] = 0.0#1e-4
    adaBoostParam["Bin"] = 32
    adaBoostParam["Type"].setTrue("Real")
    adaBoostParam["Saturate"] = False
    adaBoostParam["verbose"] = False
    adaBoostParam["saveDetail"] = True
    
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
    
    accuracy, auc = gui.evaluateROC(evalScore, evalLabel)
    print(auc)
    print("Done.")
