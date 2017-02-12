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
    def __init__(self,inImgList,inLabelList,inDetectorList,loopNum):
        self.__bin = 8
        self.__loopNum = loopNum
        self.__detectorList = inDetectorList
        self.__imgList = inImgList
        self.__labelList = np.array(inLabelList)
        self.__featureLen = None
        if( self.__labelList.size != len(self.__imgList) ):
            print("size of image files and label data are not the same.")
            exit()
        
        
        self.__Prepare()
        self.__Boost()
        

    # (弱識別器インデックス) x (入力イメージインデックス) の行列を計算する
    # AdaBoostのループにおいて、この行列は最初に一度だけ計算する
    def __Prepare(self):
        
        if not os.path.exists("test.mat"):
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
    
            self.__trainScoreMat = np.transpose(self.__trainScoreMat)
            sio.savemat("test.mat", {"ability":self.__trainScoreMat} )

        else:
            self.__trainScoreMat = np.asarray(sio.loadmat("test.mat")["ability"])

    def __GetFeatureLength(self):
        if None == self.__featureLen:
            self.__featureLen = 0
            for detector in self.__detectorList:
                self.__featureLen = self.__featureLen + detector.GetFeatureLength()
        return self.__featureLen

    def __PosNegDevide(self,inScoreMat,inLabelList):
        det = inScoreMat.shape[0]
        scoreMat = np.transpose(inScoreMat)
        posScoreMat = np.empty((0,det),float)
        negScoreMat = np.empty((0,det),float)
        for i in range(len(inLabelList)):
            if 1 == inLabelList[i]:
                posScoreMat = np.append(posScoreMat, [scoreMat[i]],axis=0)
            elif -1 == inLabelList[i]:
                negScoreMat = np.append(negScoreMat, [scoreMat[i]],axis=0)
            else:
                print("bug!")
        posScoreMat = np.transpose(posScoreMat)
        negScoreMat = np.transpose(negScoreMat)
        return posScoreMat, negScoreMat
        
    class Reliability:
        def __init__(self,reliability):
            self.__reliability = reliability
        def calc(self,bin):
            return self.__reliability[bin]

    def __Boost(self):
        
        if os.path.exists("strong.mat"):
            return
        
        detectorNum = self.__trainScoreMat.shape[0]
        if 0 == self.__trainScoreMat.shape[1]:
            print("Abort! NO sample found.")
            exit()

        # スコア行列をPos/Negで分ける
        trainPosScore, trainNegScore = self.__PosNegDevide(self.__trainScoreMat, self.__labelList)
        posSample = trainPosScore.shape[1]
        negSample = trainNegScore.shape[1]

        # サンプルデータの重みを初期化
        posSampleWeight = np.array([1.0/(posSample)]*posSample)
        negSampleWeight = np.array([1.0/(negSample)]*negSample)
        
        #ヒストグラム算出用のフィルタテンソルを作成
        posBinCounter = VoteCount(binNum=self.__bin,
                                  detectorNum=detectorNum,
                                  sampleNum=posSample)
        negBinCounter = VoteCount(binNum=self.__bin,
                                  detectorNum=detectorNum,
                                  sampleNum=negSample)
        
        # スコアをBIN値に換算
        trainPosBin = (trainPosScore * self.__bin).astype(np.int64)
        trainNegBin = (trainNegScore * self.__bin).astype(np.int64)   
        # 万が一値がBIN値と同じ場合はBIN-1としてカウントする
        trainPosBin -= 1*(trainPosBin == self.__bin)    
        trainNegBin -= 1*(trainNegBin == self.__bin)    

        # 強識別器情報の記録メモリを確保
        detLen = min(self.__loopNum,detectorNum)
        strongDet = np.zeros((detLen,self.__bin))
        strongDetID = np.zeros(detLen)
        for w in range(detLen):

            # 各識別器の性能を計算
            #for d in range(detectorNum):
            #    for i in range(sampleNum):
            #        bin = mt.IntMinMax(self.__trainScoreMat[d][i] * self.__bin, 0, self.__bin - 1)
            #        if (1 == self.__labelList[i]):
            #            histoPos[d][bin] = histoPos[d][bin] + sampleWeight[i]
            #        elif (-1 == self.__labelList[i]):
            #            histoNeg[d][bin] = histoNeg[d][bin] + sampleWeight[i]
            
            histoPos = posBinCounter.calc(trainPosBin, posSampleWeight)
            if np.any(np.isnan(histoPos)):
                print("histoPos has NAN when created.")
                exit()
            histoNeg = negBinCounter.calc(trainNegBin, negSampleWeight)
            if np.any(np.isnan(histoNeg)):
                print("histoNeg has NAN when created.")
                exit()

            # 残っている弱識別器から最優秀のものを選択            
            #detectorAbility = np.zeros(detectorNum)
            #for d in range(detectorNum):
            #    for b in range(self.__bin):
            #        detectorAbility[d] = detectorAbility[d] + np.sqrt(histoPos[d][b]*histoNeg[d][b])
            bestDet = np.argmin(np.sum(histoPos * histoNeg, axis=1))
            if np.any(np.isnan(bestDet)):
                print("bestDet is NAN.")
                exit()

            # 最優秀識別器の信頼性を算出
            #for b in range(self.__bin):
            #    if (0.0 != histoPos[bestDetectorID][b]) and (0.0 != histoNeg[bestDetectorID][b]): 
            #        h = np.append(h, np.log(histoPos[bestDetectorID][b]/histoNeg[bestDetectorID][b]))
            #    else:
            #        h = np.append(h, 0.0)
            # ゼロ割回避
            if np.any(np.isnan(histoPos)):
                print("histoPos has NAN before escape from zero-div.")
                exit()
            histoPos[bestDet] += 1*(0.0 == histoPos[bestDet])
            if np.any(np.isnan(histoPos)):
                print("histoPos has NAN after escape from zero-div.")
                exit()

            histoNeg[bestDet] += 1*(0.0 == histoNeg[bestDet])
            if np.any(np.isnan(histoNeg)):
                print("histoNeg has NAN after escape from zero-div.")
                exit()

            '''
            # RealAdaBoost
            epsilon = 0.00001
            h = (histoPos[bestDet] + epsilon )/(histoNeg[bestDet] + epsilon)
            '''
            # 飽和型RealAdaBoost
            alpha = 0.1
            expPos = np.power(histoPos[bestDet], alpha)
            expNeg = np.power(histoNeg[bestDet], alpha)
            h = ( expPos - expNeg) / (expPos + expNeg)
            if np.any(np.isnan(h)):
                print("h has NAN.")
                exit()

            strongDet[w] = h
            strongDetID[w] = bestDet

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
            if 0.0 == np.sum(posSampleWeight):
                print("positive sample regularize failed.")
                exit()
            posSampleWeight *= np.exp(-1.0 * reliability.calc(bin=trainPosBin[bestDet]) - posMax)
            posSampleWeight /= np.sum(posSampleWeight)
            if np.any(np.isnan(posSampleWeight)):
                print("posSampleWeight has NAN.")
                exit()
            negSampleWeight *= np.exp( 1.0 * reliability.calc(bin=trainNegBin[bestDet]) - negMax)
            if 0.0 == np.sum(negSampleWeight):
                print("negative sample regularize failed.")
                exit()
            negSampleWeight /= np.sum(negSampleWeight)
            if np.any(np.isnan(negSampleWeight)):
                print("negSampleWeight has NAN.")
                exit()
            
            # 選択除去された弱識別器の情報を次ループでは考えない
            trainPosBin = np.delete(trainPosBin, bestDet, axis=0)
            trainNegBin = np.delete(trainNegBin, bestDet, axis=0)
            posSample -= 1
            negSample -= 1
            posBinCounter.reduceDetector()
            negBinCounter.reduceDetector()

            if (0 == (w + 1) % 100) or (w + 1 == detLen):
                print("boosting weak detector:", w + 1)
        
            if np.any(np.isnan(strongDet)):
                print("strongDetector has NAN.@Boost")
            if np.any(np.isnan(strongDetID)):
                print("strongDetector has NAN.@Boost")

        dict = {}
        dict["strong"] = strongDet
        dict["strongID"] = strongDetID
        sio.savemat("strong.mat", dict )
        
        return

    def Evaluate(self,inImgList,inLabelList):
        
        if not os.path.exists("strong.mat"):
            print("No strong detector was found.")
            exit()
        strongDetector   = np.array(sio.loadmat("strong.mat")["strong"])
        strongDetectorID = np.array(sio.loadmat("strong.mat")["strongID"])[0]
        
        if np.any(np.isnan(strongDetector)):
            print("strongDetector has NAN.@Evaluate")
        if np.any(np.isnan(strongDetectorID)):
            print("strongDetectorID has NAN.@Evaluate")

        # 評価用サンプルに対する各弱識別器のスコアを算出
        
        if not os.path.exists("TestScore.mat"):
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
    
            self.__testScoreMat = np.transpose(self.__testScoreMat)
            sio.savemat("TestScore.mat", {"testScore":self.__testScoreMat} )
        else:
            self.__testScoreMat = np.asarray(sio.loadmat("TestScore.mat")["testScore"])
        
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
        
    def DrawROC(self):
        finalScore = np.asarray(sio.loadmat("FinalScore.mat")["finalScore"])[0]
        label = np.asarray(sio.loadmat("FinalScore.mat")["label"])[0]

        x = np.empty(0,float)
        y = np.empty(0,float)
        for i in range(finalScore.size):
            
            # バイアスがfinalScore[i]だったときのROCカーブ上の点を算出
            bias = finalScore[i]
            
            truePos  = np.sum(np.logical_and(( 1 == label), (finalScore[i] < finalScore)))
            falseNeg = np.sum(np.logical_and(( 1 == label), (finalScore[i] > finalScore)))
            falsePos = np.sum(np.logical_and((-1 == label), (finalScore[i] < finalScore)))
            trueNeg  = np.sum(np.logical_and((-1 == label), (finalScore[i] > finalScore)))
            if 0.0 < truePos + falseNeg:
                falseNeg = falseNeg / (truePos + falseNeg)
                truePos  = truePos  / (truePos + falseNeg)
            if 0.0 < trueNeg + falsePos:
                falsePos = falsePos / (trueNeg + falsePos)
                trueNeg  =  trueNeg / (trueNeg + falsePos)
            
            x = np.append(x, falsePos)        
            y = np.append(y,  truePos)
        
        plt.plot(x,y,'.' )
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
    
        plt.plot(np.log(x),np.log(1-y),'.' )
#        plt.xlim(0.0, 1.0)
#        plt.ylim(0.0, 1.0)
        plt.title("DET Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("false Negative Rate")
        plt.show()

    def DrawStrong(self):
        strong = np.array(sio.loadmat("strong.mat")["strong"])
        for d in self.__detectorList:   #今はリスト使ってないので１こだけ
            img = self.__imgList[5]
            imt.ndarray2PILimg(img).resize((400,800)).show()
            d.ShowBinImg(shape=(800,400), strong=strong, img=img)
            d.ShowBinImg(shape=(800,400), img=img)
    
if "__main__" == __name__:

    Hog8_8_16 = CHog(CHogParam(bin=8, cellX=8,cellY=16,blockX=1,blockY=1))
    Hog8_4_8 = CHog(CHogParam(bin=8, cellX=4,cellY=8,blockX=1,blockY=1))
    Hog888 = CHog(CHogParam(bin=8, cellX=8,cellY=8,blockX=1,blockY=1))
    Hog881 = CHog(CHogParam(bin=8, cellX=8,cellY=1,blockX=1,blockY=1))
    Hog818 = CHog(CHogParam(bin=8, cellX=1,cellY=8,blockX=1,blockY=1))
    detectorList = [Hog8_8_16]
    
    trainImgList = []
    trainLabelList = []

    for imgPath in fio.GetFileList("TrainPos"):
        trainImgList.append(imt.imgPath2ndarray(imgPath))
        trainLabelList.append(1)
    for imgPath in fio.GetFileList("TrainNeg"):
        trainImgList.append(imt.imgPath2ndarray(imgPath))
        trainLabelList.append(-1)
    AdaBoost = CAdaBoost(trainImgList,trainLabelList,inDetectorList=detectorList,loopNum=2000)
    
    testImgList = []
    testLabelList = []

    for imgPath in fio.GetFileList("TestPos"):
        testImgList.append(imt.imgPath2ndarray(imgPath))
        testLabelList.append(1)
    for imgPath in fio.GetFileList("TestNeg"):
        testImgList.append(imt.imgPath2ndarray(imgPath))
        testLabelList.append(-1)
    #AdaBoost.Evaluate(inImgList=testImgList,inLabelList=testLabelList)
    
    #AdaBoost.DrawStrong()
    AdaBoost.DrawROC()
    
    print("Done.")
    exit()
