import numpy as np
from hog import CHogParam,CHog
import imgtool as imt
import mathtool as mt
import pandas as pd
import fileIO as fio
import os
import scipy.io as sio
from matplotlib import pyplot as plt

class CAdaBoost:
    def __init__(self,inImgList,inLabelList,inDetectorList,loopNum):
        self.__bin = 8
        self.__loopNum = loopNum
        self.__detectorList = inDetectorList
        self.__imgList = inImgList
        self.__labelList = np.array(inLabelList)
        
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
            print("Ability matrix was loaded.");

    def __GetFeatureLength(self):
        ret = 0
        for detector in self.__detectorList:
            ret = ret + detector.GetFeatureLength()
        return ret
    
    def __Boost(self):
        
        if os.path.exists("strong.mat"):
            return
        
        # サンプルデータの重みを初期化
        detectorNumFirst = self.__trainScoreMat.shape[0]
        sampleNumFirst = self.__trainScoreMat.shape[1]
        if 0 == sampleNumFirst:
            print("Abort! NO sample found.")
            exit()
        
        sampleWeight = np.array([1.0/sampleNumFirst]*sampleNumFirst)
        
        strongDetectorID = np.empty(0,int)
        strongDetector = np.empty((0,self.__bin),float)
        for l in range(self.__loopNum):

            # 使える弱識別器がなくなったらAdaBoostループを終了
            detectorNum = self.__trainScoreMat.shape[0]
            if 0 == detectorNum:
                break
            
            sampleNum = self.__trainScoreMat.shape[1]
            histoPos = np.zeros((detectorNum,self.__bin))
            histoNeg = np.zeros((detectorNum,self.__bin))
            
            # 各識別器の性能を計算
            for d in range(detectorNum):
                for i in range(sampleNum):
                    bin = mt.IntMinMax(self.__trainScoreMat[d][i] * self.__bin, 0, self.__bin - 1)
                    if (1 == self.__labelList[i]):
                        histoPos[d][bin] = histoPos[d][bin] + sampleWeight[i]
                    elif (-1 == self.__labelList[i]):
                        histoNeg[d][bin] = histoNeg[d][bin] + sampleWeight[i]

            # 残っている弱識別器から最優秀のものを選択            
            detectorAbility = np.zeros(detectorNum)
            for d in range(detectorNum):
                for b in range(self.__bin):
                    detectorAbility[d] = detectorAbility[d] + np.sqrt(histoPos[d][b]*histoNeg[d][b])
            bestDetectorID = np.argmin(detectorAbility)
            h = np.empty((0),float)
            for b in range(self.__bin):
                if (0.0 != histoPos[bestDetectorID][b]) and (0.0 != histoNeg[bestDetectorID][b]): 
                    h = np.append(h, np.log(histoPos[bestDetectorID][b]/histoNeg[bestDetectorID][b]))
                else:
                    h = np.append(h, 0.0)
            strongDetector = np.append(strongDetector, np.array([h]), axis = 0)    
            strongDetectorID = np.append(strongDetectorID, bestDetectorID)    
            
            # 選択した最優秀識別器のスコアをもとに、サンプル重みを更新
            for i in range(sampleNum):
                d = bestDetectorID
                b = mt.IntMinMax(self.__trainScoreMat[d][i] * self.__bin, 0, self.__bin - 1)
                if (0 < histoPos[d][b]) and (0 < histoNeg[d][b]):
                    sampleWeight[i] = sampleWeight[i] * np.exp(-1.0 * self.__labelList[i] * h[b])
            
            self.__trainScoreMat = np.delete(self.__trainScoreMat, bestDetectorID, axis=0)
            if (0 == (l + 1) % 100) or (l + 1 == self.__loopNum):
                print("boosting weak detector:", l + 1)
        
            if np.any(np.isnan(strongDetector)):
                print("strongDetector has NAN.@Boost")
            if np.any(np.isnan(strongDetector)):
                print("strongDetector has NAN.@Boost")

        dict = {}
        dict["strong"] = strongDetector
        dict["strongID"] = strongDetectorID
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
                b = mt.IntMinMax(score * self.__bin, 0, self.__bin - 1)
                finalScore[i] = finalScore[i] + strongDetector[d][b]
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
            falseNeg = 0.0
            truePos = 0.0
            falsePos = 0.0
            trueNeg = 0.0
            bias = finalScore[i]
            
            for j in range(finalScore.size):
                if i != j:
                    if 1 == label[j]:
                        # 本当はポジティブ
                        if finalScore[j] > bias:
                            truePos = truePos + 1.0   # ポジティブ判定→正解
                        elif finalScore[j] < bias:
                            falseNeg = falseNeg + 1.0 # ネガティブ判定→不正解
                    elif -1 == label[j]:
                        # 本当はネガティブ
                        if finalScore[j] > bias:
                            falsePos = falsePos + 1.0 # ポジティブ判定→不正解
                        elif finalScore[j] < bias:
                            trueNeg = trueNeg + 1.0   # ネガティブ判定→正解
                    else:
                        print("Abort! label value =",label[j])
                        exit()
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
    
        
    
if "__main__" == __name__:

    Hog8_8_16 = CHog(CHogParam(bin=8, cellX=8,cellY=16,blockX=1,blockY=1))
    Hog8_4_8 = CHog(CHogParam(bin=8, cellX=4,cellY=8,blockX=1,blockY=1))
    Hog888 = CHog(CHogParam(bin=8, cellX=8,cellY=8,blockX=1,blockY=1))
    Hog881 = CHog(CHogParam(bin=8, cellX=8,cellY=1,blockX=1,blockY=1))
    Hog818 = CHog(CHogParam(bin=8, cellX=1,cellY=8,blockX=1,blockY=1))
    detectorList = [Hog8_4_8]
    
    trainImgList = []
    trainLabelList = []

    for imgPath in fio.GetFileList("TrainPosSub"):
        trainImgList.append(imt.imgPath2ndarray(imgPath))
        trainLabelList.append(1)
    for imgPath in fio.GetFileList("TrainNegSub"):
        trainImgList.append(imt.imgPath2ndarray(imgPath))
        trainLabelList.append(-1)
    AdaBoost = CAdaBoost(trainImgList,trainLabelList,inDetectorList=detectorList,loopNum=2000)
    
    testImgList = []
    testLabelList = []

    for imgPath in fio.GetFileList("TestPosSub"):
        testImgList.append(imt.imgPath2ndarray(imgPath))
        testLabelList.append(1)
    for imgPath in fio.GetFileList("TestNegSub"):
        testImgList.append(imt.imgPath2ndarray(imgPath))
        testLabelList.append(-1)
    AdaBoost.Evaluate(inImgList=testImgList,inLabelList=testLabelList)
    AdaBoost.DrawROC()
    
    print("Done.")
    exit()