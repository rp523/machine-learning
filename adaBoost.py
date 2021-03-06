#coding: utf-8
import os, shutil, sys, subprocess
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
import gui

class AdaBoostParam(CParam):
    def __init__(self):
        setDicts = dicts()
        setDicts["Type"] = selparam("Discrete", "Real", "RealTree")
        setDicts["Saturate"] = True
        setDicts["SaturateLevel"] = 0.4
        setDicts["SaturateLoss"] = False
        setDicts["Bin"] = 32
        setDicts["Loop"] = 10000
        setDicts["Regularizer"] = 1e-5
        setDicts["TreeDepth"] = 1
        setDicts["regDataDist"] = 0.0
        setDicts["verbose"] = True
        setDicts["saveDetail"] = False
        setDicts["BoostLoop"] = 1
        setDicts["FastScan"] = 1
        setDicts["GivenOrder"] = np.empty(0).astype(np.int)
        super().__init__(setDicts) 

class CAdaBoost:
    
    def __init__(self):
        self.__learnScoreMat = None
        self.__learnLabel = None
        pass
    
    def SetParam(self, inAdaBoostParam = None):
        
        if None != inAdaBoostParam:
            adaBoostParam = inAdaBoostParam
        else:
            adaBoostParam = AdaBoostParam()
            
        self.__adaType = adaBoostParam["Type"].nowSelected()
        self.__bin = adaBoostParam["Bin"]
        self.__loopNum = adaBoostParam["Loop"]
        self.__boostLoop = adaBoostParam["BoostLoop"]
        self.__regularize = adaBoostParam["Regularizer"]
        self.__saturate = adaBoostParam["Saturate"]
        self.__saturateLevel = adaBoostParam["SaturateLevel"]
        self.__saturateLoss = adaBoostParam["SaturateLoss"]
        self.__treeDepth = adaBoostParam["TreeDepth"]
        self.__regDataDist = adaBoostParam["regDataDist"]
        self.__verbose = adaBoostParam["verbose"]
        self.__saveDetail = adaBoostParam["saveDetail"]
        self.__fastScan = adaBoostParam["FastScan"]
        self.__givenOrder = adaBoostParam["GivenOrder"]
        self.__featureLen = None
        self.__weightRec = None
        
    def __GetFeatureLength(self):
        if None == self.__featureLen:
            self.__featureLen = 0
            for detector in self.__detectorList:
                assert(detector.GetFeatureLength)
                self.__featureLen = self.__featureLen + detector.GetFeatureLength()
        return self.__featureLen

    def Boost(self, trainScoreMat, labelList, evalScoreMat = None, evalLabel = None):

        assert(isinstance(trainScoreMat, np.ndarray))
        assert(trainScoreMat.ndim == 2)
        assert(not np.any(trainScoreMat < 0))
        assert(isinstance(labelList, np.ndarray))
        assert(labelList.ndim == 1)
        assert(trainScoreMat.shape[0] == labelList.size)
        self.__learnScoreMat = trainScoreMat
        self.__learnLabel = labelList

        posIdx = (labelList ==  1)
        negIdx = (labelList == -1)
        sampleNum   = trainScoreMat.shape[0]
        posSampleNum = np.sum(posIdx)
        negSampleNum = np.sum(negIdx)
        assert(sampleNum == posSampleNum + negSampleNum)

        featureNum = trainScoreMat.shape[1]
        adaLoop = min(self.__loopNum * self.__fastScan, featureNum)
        
        # サンプルデータの重みを初期化
        sampleWeight = np.empty(sampleNum)
        sampleWeight[posIdx] = 1.0 / posSampleNum
        sampleWeight[negIdx] = 1.0 / negSampleNum
        self.__weightRec = np.zeros((sampleNum, featureNum))
    
        self.__posVote = np.zeros((trainScoreMat.shape[1], self.__bin)).astype(np.float)
        self.__negVote = np.zeros((trainScoreMat.shape[1], self.__bin)).astype(np.float)
        
        # 強識別器情報の記録メモリを確保
        boostRelia = np.zeros((adaLoop,self.__bin)).astype(np.float)
        boostOrder = np.zeros(adaLoop).astype(np.int)

        if self.__adaType == "RealTree":
            assert(0)
        elif self.__adaType == "Discrete":
            assert(0)
        elif self.__adaType == "Real":
            
            # スコアをBIN値に換算
            trainBinMat = (trainScoreMat * self.__bin).astype(np.int)
            # 万が一値がBIN値と同じ場合はBIN-1としてカウントする
            trainBinMat = trainBinMat * (trainBinMat < self.__bin) + (self.__bin - 1) * (trainBinMat >= self.__bin)   
            # 負の値はとらないはずだが、一応確認
            assert(not (trainBinMat < 0).any())

            # まだ選択されず残っている特徴をマーキング
            remains = np.ones(featureNum).astype(np.bool)

            base = np.arange(featureNum) * self.__bin

            if self.__verbose:
                print("real-adaboosting...")
            for w in IterLog(range(min(self.__loopNum, featureNum * self.__boostLoop)), self.__verbose):

                # まだAdaBoostに選択されず残っている識別器の数
                remainNum = np.sum(remains)
                
                if remainNum > 0:
                    
                    # 1週目の2特徴目以降はサンプル重みを更新
                    if featureNum > remainNum:
                        if self.__saturateLoss:
                            for s in range(sampleNum):
                                c = self.__saturateLevel
                                sampleWeight[s] = np.prod(1.0 - labelList[s] * boostRelia.flatten()[base + trainBinMat[s][boostOrder]][:featureNum - remainNum]) ** (1.0 / c + 1.0)
                                assert((sampleWeight > 0.0).all())
                        else:
                            for s in range(sampleNum):
                                sampleWeight[s] = np.exp(-1 * labelList[s] * np.sum(boostRelia.flatten()[base + trainBinMat[s][boostOrder]][:featureNum - remainNum]))

                    sampleWeight[posIdx] /= np.sum(sampleWeight[posIdx])
                    sampleWeight[negIdx] /= np.sum(sampleWeight[negIdx])

                    if self.__givenOrder.size == 0:
                        # 各識別器の性能を計算するための重み付きヒストグラム（識別器 x AdabootBin）を計算
                        histoPos = np.zeros((remainNum, self.__bin))
                        histoNeg = np.zeros((remainNum, self.__bin))
                        for b in range(self.__bin):
                            histoPos[np.arange(remainNum),b] = np.dot((trainBinMat[posIdx].T[remains] == b), sampleWeight[posIdx])
                            histoNeg[np.arange(remainNum),b] = np.dot((trainBinMat[negIdx].T[remains] == b), sampleWeight[negIdx])
                        # 残っている弱識別器から最優秀のものを選択
                        remainGoodID = np.argsort(np.sum(np.sqrt(histoPos * histoNeg), axis=1))[:self.__fastScan]
                        selectHistPos = histoPos[remainGoodID]
                        selectHistNeg = histoNeg[remainGoodID]
                    else:
                        histoPos = np.zeros((1, self.__bin)).astype(np.float)
                        histoNeg = np.zeros((1, self.__bin)).astype(np.float)
                        remainGoodID = self.__givenOrder[w]
                        for b in range(self.__bin):
                            histoPos[0,b] = np.sum(sampleWeight[posIdx][trainBinMat[posIdx].T[remainGoodID] == b])
                            histoNeg[0,b] = np.sum(sampleWeight[negIdx][trainBinMat[negIdx].T[remainGoodID] == b])
                        selectHistPos = histoPos
                        selectHistNeg = histoNeg
                    
                else:
                    # 2週目以降の補正
                    if self.__saturateLoss:
                        for s in range(sampleNum):
                            c = self.__saturateLevel
                            sampleWeight[s] = np.prod(1.0 - labelList[s] * boostRelia.flatten()[np.delete(base + trainBinMat[s][boostOrder], w % featureNum)]) ** (1.0 / c + 1.0)
                            assert((sampleWeight > 0.0).all())
                    else:
                        for s in range(sampleNum):
                            sampleWeight[s] = np.exp(-1 * labelList[s] * np.sum(boostRelia.flatten()[np.delete(base + trainBinMat[s][boostOrder], w % featureNum)]))
                    sampleWeight[posIdx] = sampleWeight[posIdx] / posSampleNum
                    sampleWeight[negIdx] = sampleWeight[negIdx] / negSampleNum
                    # 各識別器の性能を計算するための重み付きヒストグラム（識別器 x AdabootBin）を計算
                    histoPos = np.zeros((featureNum, self.__bin))
                    histoNeg = np.zeros((featureNum, self.__bin))
                    for b in range(self.__bin):
                        histoPos[np.arange(featureNum),b] = np.dot((trainBinMat[posIdx].T == b), sampleWeight[posIdx])
                        histoNeg[np.arange(featureNum),b] = np.dot((trainBinMat[negIdx].T == b), sampleWeight[negIdx])
                    # 残っている弱識別器から最優秀のものを選択
                    selectFeature_abs = boostOrder[w % featureNum]
                    selectHistPos = histoPos[selectFeature_abs]
                    selectHistNeg = histoNeg[selectFeature_abs]
                
                # 最優秀識別器の信頼性を算出
                if self.__saturate == False:
                    h = 0.5 * np.log((selectHistPos + self.__regularize + 1e-10)
                                    /(selectHistNeg + self.__regularize + 1e-10))
                else:
                    alpha = self.__saturateLevel
                    expPos = (selectHistPos ** alpha) + self.__regularize + 1e-10
                    expNeg = (selectHistNeg ** alpha) + self.__regularize + 1e-10
                    h = ( expPos - expNeg) / (expPos + expNeg)
                
                # スムージング
                if 1:
                    smoother = np.zeros((h.shape[1], h.shape[1])).astype(np.float)
                    ran = 2
                    for i in range(h.shape[1]):
                        smoother[i][max(i-ran,0):min(i+ran+1,h.shape[1])] = 1.0
                    smoother /= np.sum(smoother, axis = 1).reshape(-1, 1)
                    h = np.dot(h, smoother.T)
                
                if remainNum > 0:
                    boostRelia[w:w + self.__fastScan] = h
                    if self.__givenOrder.size == 0:
                        selectID_abs = np.arange(featureNum)[remains][remainGoodID]
                    else:
                        selectID_abs = remainGoodID
                    
                    boostOrder[w:w + self.__fastScan] = selectID_abs
                    remains[selectID_abs] = False

                    # 別モジュールで再利用するため、各特徴計算時のサンプル重みを記録しておく
                    self.__posVote[selectID_abs] = selectHistPos
                    self.__negVote[selectID_abs] = selectHistNeg
                    self.__weightRec.T[selectID_abs] = sampleWeight
                    
                else:
                    boostRelia[np.where(boostOrder == selectFeature_abs)] = h
                
                if 0:#(w > 0) and (w % featureNum == 0 ):
                    learnLoss = self.__CalcLoss(boostRelia, boostOrder, trainScoreMat, labelList, self.__bin, (featureNum - remainNum))
                    if isinstance(evalScoreMat, np.ndarray) and isinstance(evalLabel, np.ndarray):
                        evalLoss = self.__CalcLoss(boostRelia, boostOrder, evalScoreMat, evalLabel, self.__bin, (featureNum - remainNum))
                        print(w, learnLoss, evalLoss)
                    else:
                        print(w, learnLoss)
                continue

        self.__relia = boostRelia
        self.__reliaID = boostOrder

        if self.__saveDetail:
            self.__SaveLearning(boostRelia,
                                boostOrder,
                                trainScoreMat,
                                labelList)
        '''        
        
        
        return self.__CalcScore(boostRelia = boostRelia,
                                            boostOrder = boostOrder,
                                            scoreMat = trainScoreMat,
                                            label = labelList,
                                            bin = self.__bin,
                                            selectedNum = None)
        '''
    def GetVoteNum(self):
        assert(None != self.__posVote)
        assert(None != self.__negVote)
        return self.__posVote, self.__negVote
   
    def GetVoteWeight(self):
        assert((self.__weightRec.T[self.__reliaID[:self.__fastScan]] > 0.0).all())
        return self.__weightRec
                    
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
            if self.__verbose:
                print("evaluating sample...")
            for i in IterLog(range(finalScore.size), self.__verbose):
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
                                  boostRelia = self.__relia,
                                  boostOrder = self.__reliaID)
            
        out = np.array(finalScore)
        return out
    
    def __CalcLoss(self, boostRelia, boostOrder, scoreMat, label, bin, selectedNum):
        
        if self.__saturateLoss:
            base = np.arange(boostRelia.shape[0]) * bin
            learnedParamVec = boostRelia[np.argsort(boostOrder)].flatten()       
            c = self.__saturateLevel
            ftrBinMat = (scoreMat * bin).astype(np.int)
            ftrBinMat = ftrBinMat * (ftrBinMat < bin) + (bin) * (ftrBinMat >= bin)
            lossVec = np.empty(scoreMat.shape[0]).astype(np.float)
            for s in range(scoreMat.shape[0]):
                lossVec[s] = np.prod(1.0 - label[s] * learnedParamVec[base + ftrBinMat[s]]) ** (1.0 / c + 1.0)
        else:
            scoreVec = self.__CalcScore(boostRelia, boostOrder, scoreMat, label, bin, selectedNum)
            lossVec = np.exp(-1 * label * scoreVec)
        lossVec[label ==  1] = lossVec[label ==  1] / np.sum(label ==  1)
        lossVec[label == -1] = lossVec[label == -1] / np.sum(label == -1)
        return np.sum(lossVec)
        
    def CalcWeakScore(self, 
                      label = None, 
                      bin = None,
                      scoreMat = None, 
                      boostRelia = None,
                      boostOrder = None,
                      selectedNum = None):
        
        
        # スコアをBIN値に換算
        binMat = (scoreMat * bin).astype(np.int)
        binMat = binMat * (binMat < bin) + (bin - 1) * (binMat >= bin)
        dstScoreVec = np.empty(binMat.shape[0])
        base = np.arange(boostOrder.size) * bin
        boostReliaVec = boostRelia.flatten()       
        
        if self.__verbose:
            print("calculating scores for every sample...")
            
        weakScore = np.empty((dstScoreVec.size, selectedNum)).astype(np.float)
        for s in IterLog(range(dstScoreVec.size), self.__verbose):
            weakScore[s] = boostReliaVec[base + binMat[s][boostOrder]][:selectedNum]
        return weakScore
    
    def __CalcScore(self, boostRelia, boostOrder, scoreMat, label, bin, selectedNum):
        weakScore = self.CalcWeakScore(label = label,
                                       bin = bin,
                                       scoreMat = scoreMat,
                                       boostRelia = boostRelia,
                                       boostOrder = boostOrder,
                                       selectedNum = selectedNum)
        return np.sum(weakScore, axis = 1)

    # AdaBoostの計算過程を記録する(optional)
    def __SaveLearning(self, boostRelia, boostOrder, scoreMat, trainLabel):

        # xlsxファイルに出力
        writer = pd.ExcelWriter("adaBoostDetail.xlsx", engine = 'xlsxwriter')
        
        # 学習サンプルごとの詳細記録シート
        learnDF = pd.DataFrame()
        
        # ラベルの記録
        learnDF["label"] = trainLabel
        
        learnDF["score"] = self.__CalcScore(boostRelia = boostRelia,
                                            boostOrder = boostOrder,
                                            scoreMat = scoreMat,
                                            label = trainLabel,
                                            bin = self.__bin,
                                            selectedNum = boostOrder.size)

        # 全特徴スコアの記録
        for i in range(scoreMat.shape[1]):
            learnDF["feature{0:04d}".format(i)] = scoreMat.T[i]
        
        learnDF.to_excel(writer, sheet_name = "learn")

        # AdaBoostの詳細記録シート
        adaBoostDF = pd.DataFrame()
        
        # 特徴選択順
        adaBoostDF["boostOrder"] = boostOrder
        
        # 全寄与度の記録(選択順)
        for i in range(boostRelia.shape[1]):
            adaBoostDF["bin{0}".format(i)] = boostRelia.T[i]
        adaBoostDF.to_excel(writer, sheet_name = "adaBoost")
        
        writer.save()
        writer.close()
    
    # AdaBoostの計算過程を記録する(optional)
    def __SaveEvaluation(self, 
                         scoreMat,
                         label,
                         boostRelia,
                         boostOrder,
                         detailPath = None):
        assert(isinstance(scoreMat, np.ndarray))
        assert(isinstance(label, np.ndarray))
        assert(isinstance(boostRelia, np.ndarray))
        assert(isinstance(boostOrder, np.ndarray))
        assert(scoreMat.ndim == 2)
        assert(label.ndim == 1)
        assert(boostRelia.ndim == 2)
        assert(boostOrder.ndim == 1)
        assert(scoreMat.shape[0] == label.size)
        
        if None == detailPath:
            detailPath = "adaBoostDetail.xlsx"

        # 評価サンプルごとの詳細記録シート
        evalDF = pd.DataFrame()
        
        # 評価サンプルのラベルを記録。いらないかも
        evalDF["label"] = label

        # 評価スコアを記録
        evalDF["score"] = self.__CalcScore(boostRelia = boostRelia,
                                           boostOrder = boostOrder,
                                           scoreMat = scoreMat,
                                           label = label,
                                           bin = self.__bin,
                                           selectedNum = boostOrder.size)

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
    
    def GetLearnedParam(self):
        return self.__reliaID, self.__relia
    
    def GetLearnedTable(self):
        return self.__relia[np.argsort(self.__reliaID)]
    
def main(boostLoop):

    for xlsxFile in  GetFileList(".", includingText = ".xlsx"):
        os.remove(xlsxFile)
    for matFile in  GetFileList(".", includingText = ".mat"):
        os.remove(matFile)
    
    currDir = subprocess.check_output("pwd")
    currDir = str(currDir)[2:-3]
    inriaPath = os.path.join(currDir, "dataset", "INRIAPerson")
    learnPosPath = os.path.join(inriaPath, "LearnPos")
    learnNegPath = os.path.join(inriaPath, "LearnNeg")
    evalPosPath = os.path.join(inriaPath, "EvalPos")
    evalNegPath = os.path.join(inriaPath, "EvalNeg")
    lp = dirPath2NumpyArray(learnPosPath)
    ln = dirPath2NumpyArray(learnNegPath)
    ep = dirPath2NumpyArray(evalPosPath)
    en = dirPath2NumpyArray(evalNegPath)

    learn = RGB2Gray(np.append(lp, ln, axis = 0), "green")
    eval  = RGB2Gray(np.append(ep, en, axis = 0), "green")
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))
    hogParam = CHogParam()
    hogParam["Bin"] = 8
    hogParam["Cell"]["X"] = 2
    hogParam["Cell"]["Y"] = 4
    hogParam["Block"]["X"] = 1
    hogParam["Block"]["Y"] = 1
    detectorList = [CHog(hogParam)]

    adaBoostParam = AdaBoostParam()
    adaBoostParam["Regularizer"] = 1e-4
    adaBoostParam["Bin"] = 32
    adaBoostParam["Type"].setTrue("Real")
    adaBoostParam["Saturate"] = True
    adaBoostParam["SaturateLoss"] = False
    adaBoostParam["verbose"] = False
    adaBoostParam["saveDetail"] = True
    adaBoostParam["Loop"] = 999999
    adaBoostParam["BoostLoop"] = boostLoop
    adaBoostParam["FastScan"] = 1   # Boostingのイテレーション回で同時に選択する特徴量数
    
    adaBoost = CAdaBoost()
    adaBoost.SetParam(  inAdaBoostParam = adaBoostParam)

    # 学習用の特徴量行列を準備    
    trainScoreMat = np.empty((len(learn), 0))
    for detector in detectorList:
        trainScoreMat = np.append(trainScoreMat,
                                  detector.calc(learn),
                                  axis = 1)
    # 評価用の特徴量行列を準備    
    testScoreMat = np.empty((len(eval), 0))
    for detector in detectorList:
        testScoreMat = np.append(testScoreMat,
                                 detector.calc(eval),
                                 axis = 1)

    adaBoost.Boost(trainScoreMat = trainScoreMat,
                   labelList = learnLabel,
                   evalScoreMat=testScoreMat,
                   evalLabel=evalLabel)
    
    
    evalScore = adaBoost.Evaluate(testScoreMat = testScoreMat,
                                  label = evalLabel)
    
    accuracy, auc = gui.evaluateROC(evalScore, evalLabel)
    return auc

if "__main__" == __name__:
    print(main(int(1)))
    exit()
    
    plt.figure()
    e2len = 8
    x = 2 ** np.arange(e2len).astype(np.int)
    y = []
    for x_i in x:
        y.append(100.0 * main(int(x_i)))
    plt.plot(x, y, ".")
    plt.show()
    
    print("Done.")
