import numpy as np
#from feature.hog import CHogParam,CHog
from feature.hog import *
from feature.chnFtrs import *
import common.imgtool as imt
import common.mathtool as mt
import pandas as pd
import common.fileIO as fio
import os
import scipy.io as sio
from matplotlib import pyplot as plt
from PIL import Image
import gui
from enum import Enum
from common.origLib import *
from gaussianProcess import *
from input import *
from preproc import *
from decisionTree import *


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
        super().__init__(setDicts) 

'''
# for文を使わずに投票されたBINの数を数えるためのクラス
class VoteCount:
    def __init__(self,binNum,detectorNum,sampleNum):
        self.__filter = np.zeros((binNum,detectorNum,sampleNum))
        for b in range(binNum):
            self.__filter[b] = np.array([[b]*sampleNum]*detectorNum)
    # 入力ベクトルにおける各BIN値への投票数を重み付きでカウントする
    def calc(self,bins,weight):
        return np.transpose(np.sum((self.__filter == bins) * weight, axis=2))
    # 弱識別器の次元を１つ減らす
    def reduceDetector(self):
        self.__filter = np.delete(self.__filter, 0, axis=1) # 減らす列の位置はどこでもいい
'''
        
class CAdaBoost:
    
    def __init__(self,inAdaBoostParam,inImgList,inLabelList,inDetectorList):
        self.__adaType = inAdaBoostParam["Type"].nowSelected()
        self.__bin = inAdaBoostParam["Bin"]
        self.__loopNum = inAdaBoostParam["Loop"]
        self.__regularize = inAdaBoostParam["Regularizer"]
        self.__saturate = inAdaBoostParam["Saturate"]
        self.__saturateLevel = inAdaBoostParam["SaturateLevel"]
        self.__treeDepth = inAdaBoostParam["TreeDepth"]
        self.__regDataDist = inAdaBoostParam["regDataDist"]
        self.__verbose = inAdaBoostParam["verbose"]
        self.__detectorList = inDetectorList
        self.__imgList = inImgList
        self.__labelList = np.array(inLabelList)
        self.__featureLen = None
        
        self.__thresh = np.zeros(len(inDetectorList),float)
        
        self.__Prepare()
        self.__Boost()
        

    # (弱識別器インデックス) x (入力イメージインデックス) の行列を計算する
    # AdaBoostのループにおいて、この行列は最初に一度だけ計算する
    def __Prepare(self):
        if not os.path.exists("Train.mat"):
            self.__trainScoreMat = np.empty((0,self.__GetFeatureLength()),float)
    
            logCnt = 0;
            
            for detector in self.__detectorList:
                a = np.array(self.__imgList)
                self.__trainScoreMat = np.append(self.__trainScoreMat, detector.calc(self.__imgList), axis = 0)
            
            logCnt = logCnt + 1
            if self.__verbose:
                if (0 == logCnt % 100) or (len(self.__imgList) == logCnt):
                    print(logCnt,"/",len(self.__imgList),"file was prepared for training.")
    
            dict = {}
            dict["trainScore"] = self.__trainScoreMat
            dict["trainLabel"] = self.__labelList
            sio.savemat("Train.mat", dict)

        else:
            self.__trainScoreMat = np.asarray(sio.loadmat("Train.mat")["trainScore"])

    def __GetFeatureLength(self):
        if None == self.__featureLen:
            self.__featureLen = 0
            for detector in self.__detectorList:
                assert(detector.GetFeatureLength)
                self.__featureLen = self.__featureLen + detector.GetFeatureLength()
        return self.__featureLen

    # PosとNegが混在しているスコア行列からPosのみ、Negのみのスコア行列に分離する
    def __PosNegDevide(self,inScoreMat,inLabelList):
        assert(inScoreMat.shape[1] == np.array(inLabelList).size)
        det = inScoreMat.shape[0]
        scoreMat = np.transpose(inScoreMat)     # 入力サンプル x 識別器
        posScoreMat = np.empty((0,det),float)   # 入力サンプル x 識別器
        negScoreMat = np.empty((0,det),float)   # 入力サンプル x 識別器
        for i in range(len(inLabelList)):
            if 1 == inLabelList[i]:
                posScoreMat = np.append(posScoreMat, [scoreMat[i]],axis=0)
            elif -1 == inLabelList[i]:
                negScoreMat = np.append(negScoreMat, [scoreMat[i]],axis=0)
            else:
                print("bug!")
        posScoreMat = np.transpose(posScoreMat)   # 識別器 x 入力サンプル
        negScoreMat = np.transpose(negScoreMat)   # 識別器 x 入力サンプル
        return posScoreMat, negScoreMat
        
    class Reliability:
        def __init__(self,reliability):
            self.__reliability = reliability
        def calc(self,bin):
            return self.__reliability[bin]

    def __Boost(self):

        if os.path.exists("strong.mat"):
            return
        
        assert(isinstance(self.__trainScoreMat, np.ndarray))
        assert(self.__trainScoreMat.ndim == 2)
        assert(not np.any(self.__trainScoreMat < 0))
        # ブースティングでは識別器ごとの性能を比較するので、
        # 識別器IDが浅い階層に来るよう転置をとる
        self.__trainScoreMat = np.transpose(self.__trainScoreMat)

        detectorNum = self.__trainScoreMat.shape[0]
        sampleNum   = self.__trainScoreMat.shape[1]
        detLen = min(self.__loopNum,detectorNum)
 
        # スコア行列をPos/Negで分ける
        trainPosScore, trainNegScore = self.__PosNegDevide(self.__trainScoreMat, self.__labelList)
        assert(not np.any(trainPosScore < 0))
        assert(not np.any(trainNegScore < 0))
        posSample = trainPosScore.shape[1]
        negSample = trainNegScore.shape[1]

        # サンプルデータの重みを初期化
        posSampleWeight = np.array([1.0/(posSample)]*posSample)
        negSampleWeight = np.array([1.0/(negSample)]*negSample)
        assert(posSampleWeight.size == posSample)
        assert(negSampleWeight.size == negSample)
    
        # 強識別器情報の記録メモリを確保
        strongDetBin = np.zeros((detLen,self.__bin),float)
        strongDetID = np.zeros(detLen,int)

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
            
            for d in range(detectorNum):
                assert(np.min(self.__trainScoreMat[d]) != np.max(self.__trainScoreMat[d]))
                assert(np.min(labelList) != np.max(labelList))
                sortIndex = np.argsort(self.__trainScoreMat[d])
                node = DecisionTree.Node(   scoreVec = self.__trainScoreMat[d][sortIndex],
                                            labelVec = labelList[sortIndex],
                                            sampleIndexes = sampleIndexes[sortIndex],
                                            maxDepth = self.__treeDepth,
                                            regDataDist = self.__regDataDist)
                
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
            for w in range(detLen):
                
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
            self.__detWeights = np.empty(detLen)

            sampleWeights = np.ones(sampleNum) / sampleNum
            assert(sampleWeights.size == sampleNum)
            
            yfMatSaved = (scoreMat * self.__labelList).astype(np.int)
            yfMat = yfMatSaved
            
            for w in range(detLen):
                
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
            
            '''
            #ヒストグラム算出用のフィルタテンソルを作成
            posBinCounter = VoteCount(binNum=self.__bin,
                                      detectorNum=detectorNum,
                                      sampleNum=posSample)
            negBinCounter = VoteCount(binNum=self.__bin,
                                      detectorNum=detectorNum,
                                      sampleNum=negSample)
            '''
            remainDetIDList = np.arange(detectorNum)
            
            # スコアをBIN値に換算
            trainPosBin = (trainPosScore * self.__bin).astype(np.int64)
            trainNegBin = (trainNegScore * self.__bin).astype(np.int64)
            # 万が一値がBIN値と同じ場合はBIN-1としてカウントする
            trainPosBin = trainPosBin * (trainPosBin < self.__bin) + (self.__bin - 1) * (trainPosBin >= self.__bin)   
            trainNegBin = trainNegBin * (trainNegBin < self.__bin) + (self.__bin - 1) * (trainNegBin >= self.__bin)
            # 負の値はとらないはずだが、一応確認
            assert(not (trainPosBin < 0).any())
            assert(not (trainNegBin < 0).any())

            for w in range(detLen):

                # まだAdaBoostに選択されず残っている識別器の数
                detRemain = trainPosBin.shape[0]
                
                # 各識別器の性能を計算するための重み付きヒストグラム（識別器 x AdabootBin）を計算
                histoPos = np.zeros((detRemain, self.__bin))
                histoNeg = np.zeros((detRemain, self.__bin))
                for b in range(self.__bin):
                    histoPos[np.arange(detRemain),b] = np.dot(trainPosBin == b, posSampleWeight)
                    histoNeg[np.arange(detRemain),b] = np.dot(trainNegBin == b, negSampleWeight)
                # 残っている弱識別器から最優秀のものを選択            
                bestDet = np.argmin(np.sum(np.sqrt(histoPos * histoNeg), axis=1))
                
                #ゼロ割り回避
                epsilon = 1e-10
                histoPos += epsilon
                histoNeg += epsilon
                
                # 最優秀識別器の信頼性を算出
                if self.__saturate == False:
                    h = np.log(histoPos[bestDet] + epsilon )/(histoNeg[bestDet] + epsilon)
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
                # 指数関数の発散を防止しつつ、サンプル重みを更新&正規化する
                posMax = np.max(-1.0 * reliability.calc(bin=trainPosBin[bestDet]))
                negMax = np.max( 1.0 * reliability.calc(bin=trainNegBin[bestDet]))
    
                posSampleWeight *= np.exp(-1.0 * reliability.calc(bin=trainPosBin[bestDet]) - posMax)
                posSampleWeight /= np.sum(posSampleWeight)
    
                negSampleWeight *= np.exp( 1.0 * reliability.calc(bin=trainNegBin[bestDet]) - negMax)
                negSampleWeight /= np.sum(negSampleWeight)
                
                # 選択除去された弱識別器の情報を次ループでは考えない
                trainPosBin = np.delete(trainPosBin, bestDet, axis=0)
                trainNegBin = np.delete(trainNegBin, bestDet, axis=0)
                remainDetIDList = np.delete(remainDetIDList, bestDet)

                if self.__verbose:
                    if (0 == (w + 1) % 1) or (w + 1 == detLen):
                        print("boosting weak detector:", w + 1)
            
                assert(not np.any(np.isnan(strongDetBin)))
                assert(not np.any(np.isnan(strongDetID)))

        dict = {}
        dict["strong"] = strongDetBin
        dict["strongID"] = strongDetID
        sio.savemat("strong.mat", dict )
        
        return

    def Evaluate(self,inImgList,inLabelList):
        
        strongDetector   = np.array(sio.loadmat("strong.mat")["strong"])
        strongDetectorID = np.array(sio.loadmat("strong.mat")["strongID"]).astype(np.int)[0]
        
        assert(not np.any(np.isnan(strongDetector)))
        assert(not np.any(np.isnan(strongDetectorID)))

        # 評価用サンプルに対する各弱識別器のスコアを算出
        
        if not os.path.exists("Test.mat"):
            logCnt = 0;
            if not self.__GetFeatureLength:
                assert(0)
            self.__testScoreMat = np.empty((0,self.__GetFeatureLength()),float)

            featVec = np.empty(0,float)
            for detector in self.__detectorList:
                self.__testScoreMat = np.append(self.__testScoreMat, detector.calc(inImgList),axis=0)
            
            logCnt = logCnt + 1
            if self.__verbose:
                if (0 == logCnt % 100) or (len(inImgList) == logCnt):
                    print(logCnt,"/",len(inImgList),"file was prepared for test.")
    
            dict = {}
            dict["testScore"] = self.__testScoreMat
            dict["testLabel"] = np.array(inLabelList)
            sio.savemat("Test.mat", dict)
        else:
            self.__testScoreMat = np.asarray(sio.loadmat("Test.mat")["testScore"])

        # AdaBoostではまず転置をとる
        self.__testScoreMat = np.transpose(self.__testScoreMat)

        finalScore = np.zeros(len(inImgList))
        if self.__adaType == "Real":
            for d in range(strongDetectorID.size):
                selectedDetectorID = strongDetectorID[d]
                for i in range(len(inImgList)):
                    score = self.__testScoreMat[selectedDetectorID][i]
                    b = mt.IntMax(score * self.__bin, self.__bin - 1)
                    finalScore[i] += strongDetector[d][b]
        elif self.__adaType == "Discrete":
            finalScore = np.dot(self.__detWeights, self.__decisionTree.predict(strongDetectorID, self.__testScoreMat)).T
        elif self.__adaType == "RealTree":
            didCount = 0
            for did in strongDetectorID:
                for sid in range(self.__testScoreMat.shape[1]):
                    score = self.__testScoreMat[did][sid]
                    isLarger = score > np.array(self.__treeThresh[didCount])
                    assignIdx = np.sum(isLarger)
                    finalScore[sid] += self.__treeScore[didCount][assignIdx]
                didCount += 1
        else:
            assert(0)
        outDict = {}
        outDict["finalScore"] = np.asarray(finalScore)
        outDict["label"] = np.asarray(inLabelList)
        sio.savemat("FinalScore.mat", outDict)
        

    def DrawStrong(self):
        strong = np.array(sio.loadmat("strong.mat")["strong"])
        for d in self.__detectorList:   #今はリスト使ってないので１こだけ
            img = self.__imgList[5]
            imt.ndarray2PILimg(img).resize((400,800)).show()
            d.ShowBinImg(shape=(800,400), strong=strong, img=img)
            d.ShowBinImg(shape=(800,400), img=img)
    
if "__main__" == __name__:

    os.system("rm *mat")
    lp = dirPath2NumpyArray("INRIAPerson/LearnPos")
    ln = dirPath2NumpyArray("INRIAPerson/LearnNeg")
    ep = dirPath2NumpyArray("INRIAPerson/EvalPos")
    en = dirPath2NumpyArray("INRIAPerson/EvalNeg")
    learn = RGB2Gray(lp + ln, "green")
    eval  = RGB2Gray(ep + en, "green")
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))
    hogParam = CHogParam()
    hogParam["Bin"] = 8
    hogParam["Cell"]["X"] = 2
    hogParam["Cell"]["Y"] = 4
    hogParam["Block"]["X"] = 1
    hogParam["Block"]["Y"] = 1
    detectorList = [CHog(hogParam)]

    adaBoostParam = AdaBoostParam
    AdaBoostParam["Bin"] = 10000
    AdaBoostParam["Regularizer"] = 1e-5
    AdaBoostParam["Bin"] = 32

    AdaBoost = CAdaBoost(inAdaBoostParam=adaBoostParam,
                         inImgList=learn,
                         inLabelList=learnLabel,
                         inDetectorList=detectorList)
    
    AdaBoost.Evaluate(inImgList=eval,inLabelList=evalLabel)
    
    accuracy, auc = gui.evaluateROC(np.asarray(sio.loadmat("FinalScore.mat")["finalScore"])[0],
        np.asarray(sio.loadmat("FinalScore.mat")["label"])[0])
    print(auc)