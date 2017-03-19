import numpy as np
from hog import CHogParam,CHog
import imgtool as imt
import mathtool as mt
import pandas as pd
import fileIO as fio
import os
import scipy.io as sio
from matplotlib import pyplot as plt
from PIL import Image
import gui
from enum import Enum
from conda._vendor.toolz.itertoolz import accumulate

class adaType(Enum):
    Discrete = 1
    Real = 2
    SaturatedReal = 3

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

class CAdaBoost:
    
    def __init__(self,inAdaType,inAdaBin,inImgList,inLabelList,inDetectorList,loopNum):
        self.__adaType = inAdaType
        if self.__adaType == adaType.Discrete:
            self.__bin = 2
        else:
            self.__bin = inAdaBin
        self.__loopNum = loopNum
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
            for img in self.__imgList:
            
                featVec = np.empty(0,float)
                for detector in self.__detectorList:
                    featVec = np.append(featVec,np.array(detector.calc(img)))
                self.__trainScoreMat = np.append(self.__trainScoreMat,np.array([featVec]),axis=0)
                
                logCnt = logCnt + 1
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
        
        # ブースティングでは識別器ごとの性能を比較するので、
        # 識別器IDが浅い階層に来るよう転置をとる
        self.__trainScoreMat = np.transpose(self.__trainScoreMat)

        detectorNum = self.__trainScoreMat.shape[0]
        detLen = min(self.__loopNum,detectorNum)
 
        # スコア行列をPos/Negで分ける
        trainPosScore, trainNegScore = self.__PosNegDevide(self.__trainScoreMat, self.__labelList)
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
        parity = np.zeros(detLen,int)

        if self.__adaType == adaType.Discrete:

            for w in range(detLen):

                errorRateVec = np.zeros(detLen,float)
                threshVec = np.zeros(detLen,float)
                parityVec = np.zeros(detLen,float)

                for weakCandidate in range(detLen):
                    scoreColumn = np.transpose(np.array([list(self.__trainScoreMat[weakCandidate])]))
                    oneRow = np.ones((1,posSample+negSample),float)
                    scoreCopyMat = np.dot(scoreColumn, oneRow)
    
                    print(w,weakCandidate,self.__trainScoreMat[weakCandidate].shape,scoreCopyMat.shape)
                    errorRateP = np.sum(   \
                        (self.__trainPosScore[weakCandidate] < scoreCopyMat) * (self.__labelList ==  1) +    \
                        (self.__trainPosScore[weakCandidate] > scoreCopyMat) * (self.__labelList == -1)      \
                        ,axis=1)
                    errorRatePmin = np.min(errorRateP)
                    errorRateN = np.sum(   \
                        (self.__trainNegScore[weakCandidate] > scoreCopyMat) * (self.__labelList ==  1) +    \
                        (self.__trainNegScore[weakCandidate] < scoreCopyMat) * (self.__labelList == -1)      \
                        ,axis=1)
                    errorRateNmin = np.min(errorRateN)
                    
                    # for文を抜けても残る情報
                    if (errorRatePmin < errorRateNmin):
                        errorRateVec[weakCandidate] = errorRatePmin
                        threshVec[weakCandidate] = self.__trainPosScore[weakCandidate][np.argmin(errorRatePmin)]
                        parityVec[weakCandidate] = 1
                    else:
                        errorRateVec[weakCandidate] = errorRateNmin
                        threshVec[weakCandidate] = self.__trainNegScore[weakCandidate][np.argmin(errorRateNmin)]
                        parityVec[weakCandidate] = -1
                
                # 選択した最優秀識別器のスコアをもとに、サンプル重みを更新
                bestDet = np.argmin(errorRateVec)
                self.__thresh[w] = threshVec[bestDet]
                errorRate = errorRateVec[bestDet]

                epsilon = 0.00001
                reliability = 0.5 * np.log((1.0 - errorRate + epsilon) / (errorRate + epsilon))
                strongDetBin[w] = np.array([- parityVec[bestDet] * reliability, parityVec[bestDet] * reliability])
                strongDetID[w] = bestDet
    
                # サンプル重みを更新&正規化する
                posExponent = -1.0 * (strongDetBin[w][0] * (trainPosScore[bestDet] <= self.__thresh[w]) + \
                                      strongDetBin[w][1] * (trainPosScore[bestDet] >  self.__thresh[w]))
                posMax = np.max(posExponent)
                posSampleWeight *= np.exp(posExponent - posMax)
                posSampleWeight /= np.sum(posSampleWeight)

                negExponent =  1.0 * (strongDetBin[w][0] * (trainNegScore[bestDet] <= self.__thresh[w]) + \
                                      strongDetBin[w][1] * (trainNegScore[bestDet] >  self.__thresh[w]))
                negMax = np.max(negExponent)
                negSampleWeight *= np.exp(negExponent - negMax)
                negSampleWeight /= np.sum(negSampleWeight)

                # 選択除去された弱識別器の情報を次ループでは考えない
                trainPosScore = np.delete(trainPosScore, bestDet, axis=0)
                trainNegScore = np.delete(trainNegScore, bestDet, axis=0)
                posSample -= 1
                negSample -= 1

        elif (self.__adaType == adaType.Real) or (self.__adaType == adaType.SaturatedReal):
            #ヒストグラム算出用のフィルタテンソルを作成
            posBinCounter = VoteCount(binNum=self.__bin,
                                      detectorNum=detectorNum,
                                      sampleNum=posSample)
            negBinCounter = VoteCount(binNum=self.__bin,
                                      detectorNum=detectorNum,
                                      sampleNum=negSample)
            remainDetIDList = np.arange(detectorNum)
            
            # スコアをBIN値に換算
            trainPosBin = (trainPosScore * self.__bin).astype(np.int64)
            trainNegBin = (trainNegScore * self.__bin).astype(np.int64)   
            # 万が一値がBIN値と同じ場合はBIN-1としてカウントする
            trainPosBin -= 1*(trainPosBin >= self.__bin)    
            trainNegBin -= 1*(trainNegBin >= self.__bin)
            # 負の値はとらないはずだが、一応確認
            assert(not (trainPosBin < 0).any())
            assert(not (trainNegBin < 0).any())

            '''
            histP = np.zeros(self.__bin,int)
            histN = np.zeros(self.__bin,int)
            for d in range(len(trainPosBin)):
                for i in range(len(trainPosBin[d])):
                    for b in range(self.__bin):
                        if trainPosBin[d][i] == b:
                            histP[b] += 1
            for d in range(len(trainNegBin)):
                for i in range(len(trainNegBin[d])):
                    for b in range(self.__bin):
                        if trainNegBin[d][i] == b:
                            histN[b] += 1
            plt.plot(np.arange(self.__bin)/self.__bin,histP/np.sum(histP),'.')
            plt.plot(np.arange(self.__bin)/self.__bin,histN/np.sum(histN),'.')
            plt.plot(np.arange(self.__bin)/self.__bin,(histP+histN)/np.sum(histP+histN),'.')
            plt.show()
            exit()
            '''
                            
            for w in range(detLen):

                # 各識別器の性能を計算するための重み付きヒストグラム（識別器 x AdabootBin）を計算
                histoPos = posBinCounter.calc(trainPosBin, posSampleWeight)
                histoNeg = negBinCounter.calc(trainNegBin, negSampleWeight)

                # 残っている弱識別器から最優秀のものを選択            
                bestDet = np.argmin(np.sum(np.sqrt(histoPos * histoNeg), axis=1))
                
                #ゼロ割り回避
                epsilon = 0.00001
                histoPos += epsilon
                histoNeg += epsilon
                
                # 最優秀識別器の信頼性を算出
                if self.__adaType == adaType.Real:
                    h = np.log(histoPos[bestDet] + epsilon )/(histoNeg[bestDet] + epsilon)
                elif self.__adaType == adaType.SaturatedReal:
                    alpha = 0.4
                    expPos = np.power(histoPos[bestDet], alpha)
                    expNeg = np.power(histoNeg[bestDet], alpha)
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
                posSample -= 1
                negSample -= 1
                posBinCounter.reduceDetector()
                negBinCounter.reduceDetector()

                if (0 == (w + 1) % 100) or (w + 1 == detLen):
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
            self.__testScoreMat = np.empty((0,self.__GetFeatureLength()),float)
            for img in inImgList:
            
                featVec = np.empty(0,float)
                for detector in self.__detectorList:
                    featVec = np.append(featVec,np.array(detector.calc(img)))
                self.__testScoreMat = np.append(self.__testScoreMat,np.array([featVec]),axis=0)
                
                logCnt = logCnt + 1
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
        for d in range(strongDetectorID.size):
            selectedDetectorID = strongDetectorID[d]
            for i in range(len(inImgList)):
                score = self.__testScoreMat[selectedDetectorID][i]
                b = mt.IntMax(score * self.__bin, self.__bin - 1)
                finalScore[i] += strongDetector[d][b]
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

    TrainPosDir = "TrainPos"
    TrainNegDir = "TrainNeg"
    TestPosDir  = "TestPos"
    TestNegDir  = "TestNeg"
    detectorList = [CHog(CHogParam(bin=8, cellX=6,cellY=12,blockX=1,blockY=1))]
    
    trainImgList = []
    trainLabelList = []

    for imgPath in fio.GetFileList(TrainPosDir):
        trainImgList.append(imt.imgPath2ndarray(imgPath))
        trainLabelList.append(1)
    for imgPath in fio.GetFileList(TrainNegDir):
        trainImgList.append(imt.imgPath2ndarray(imgPath))
        trainLabelList.append(-1)
    AdaBoost = CAdaBoost(inAdaType=adaType.SaturatedReal,
                         inAdaBin=16,
                         inImgList=trainImgList,
                         inLabelList=trainLabelList,
                         inDetectorList=detectorList,
                         loopNum=4000)
    
    testImgList = []
    testLabelList = []

    for imgPath in fio.GetFileList(TestPosDir):
        testImgList.append(imt.imgPath2ndarray(imgPath))
        testLabelList.append(1)
    for imgPath in fio.GetFileList(TestNegDir):
        testImgList.append(imt.imgPath2ndarray(imgPath))
        testLabelList.append(-1)
    AdaBoost.Evaluate(inImgList=testImgList,inLabelList=testLabelList)
    
    gui.DrawROC(np.asarray(sio.loadmat("FinalScore.mat")["finalScore"])[0],
        np.asarray(sio.loadmat("FinalScore.mat")["label"])[0])
