#coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
from common.imgtool import *
from common.mathtool import *
from common.origLib import *
from feature.hog import *
from input import *


        

class CChnFtrs:
    
    class CChnFtrsParam(CParam):
        def __init__(self):
            impl = dicts()
            
            ch = dicts()
            ch["red"] = False
            ch["green"] = True
            ch["blue"] = False
            ch["histGrad"] = True
            #ch["normGrad"] = False
            impl["ch"] = ch
            
            impl["edgeBin"] = 8
            
            max = dicts()
            max["w"] = 0.5
            max["h"] = 0.5
            impl["max"] = max
            
            min = dicts()
            min["w"] = 0.1
            min["h"] = 0.05
            impl["min"] = min
            
            srcSize = dicts()
            srcSize["w"] = 64
            srcSize["h"] = 128
            impl["srcSize"] = srcSize
            
            impl["dim"] = 100
    
            super().__init__(impl)
    class RandomSelectParam(CParam):
        def __init__(self, divW = 1,
                           divH = 1,
                           minW = 1,
                           minH = 1,
                           maxW = 1,
                           maxH = 1):
            impl = dicts()
            div = dicts()
            div["w"] = divW
            div["h"] = divH
            impl["div"] = div
            
            min = dicts()
            min["w"] = minW
            min["h"] = minH
            impl["min"] = min

            max = dicts()
            max["w"] = maxW
            max["h"] = maxH
            impl["max"] = max
            
            super().__init__(impl)
            
    def __init__(self, setParam):
        assert(isinstance(setParam, self.CChnFtrsParam))
        assert(setParam["min"]["h"] < setParam["max"]["h"])
        assert(setParam["min"]["w"] < setParam["max"]["w"])
        self.__param = setParam

        
        # 特徴ベクトルの次元数に合わせてランダムにサブウィンドウを生成する
        self.__winMin = np.empty((0,1,1,1,1))
        imgChList = dicts()
        self.__chIdx = 0
        randomSelectParamList = []
        for ch, uses in self.__param["ch"].items():
            if True == uses:
                if (ch == "red") or (ch == "green") or (ch == "blue"):
                    self.__chIdx += 1
                    divW = self.__param["srcSize"]["w"]
                    divH = self.__param["srcSize"]["h"]
                    minW = int(self.__param["min"]["w"] * self.__param["srcSize"]["w"] + 0.5)# - int(self.__param["min"]["w"] == 1.0)
                    minH = int(self.__param["min"]["h"] * self.__param["srcSize"]["h"] + 0.5)# - int(self.__param["min"]["h"] == 1.0)
                    maxW = int(self.__param["max"]["w"] * self.__param["srcSize"]["w"] + 0.5)# - int(self.__param["max"]["w"] == 1.0)
                    maxH = int(self.__param["max"]["h"] * self.__param["srcSize"]["h"] + 0.5)# - int(self.__param["max"]["h"] == 1.0)
                    SelectedForThisCh = self.RandomSelectParam(divW = divW,
                                                               divH = divH,
                                                               minW = minW,
                                                               minH = minH,
                                                               maxW = maxW,
                                                               maxH = maxH)
                    randomSelectParamList.append(SelectedForThisCh)
                elif ch == "histGrad":
                    for b in range(self.__param["edgeBin"]):
                        self.__chIdx += 1
                        divW = self.__param["srcSize"]["w"] - 2
                        divH = self.__param["srcSize"]["h"] - 2
                        minW = int(self.__param["min"]["w"] * (self.__param["srcSize"]["w"] - 2) + 0.5)# - int(self.__param["min"]["w"] == 1.0)
                        minH = int(self.__param["min"]["h"] * (self.__param["srcSize"]["h"] - 2) + 0.5)# - int(self.__param["min"]["h"] == 1.0)
                        maxW = int(self.__param["max"]["w"] * (self.__param["srcSize"]["w"] - 2) + 0.5)# - int(self.__param["max"]["w"] == 1.0)
                        maxH = int(self.__param["max"]["h"] * (self.__param["srcSize"]["h"] - 2) + 0.5)# - int(self.__param["max"]["h"] == 1.0)
                        SelectedForThisCh = self.RandomSelectParam(divW = divW,
                                                                   divH = divH,
                                                                   minW = minW,
                                                                   minH = minH,
                                                                   maxW = maxW,
                                                                   maxH = maxH)
                        randomSelectParamList.append(SelectedForThisCh)
        
        self.selectedSubWins = self.__randomSelectSubWin(self.__param["dim"], randomSelectParamList)
        
        self.__imgSetFnc = dicts()
        self.__imgSetFnc["red"] = self.__getRed
        self.__imgSetFnc["green"] = self.__getGreen
        self.__imgSetFnc["blue"] = self.__getBlue
        self.__imgSetFnc["histGrad"] = self.__getGradHist
    
    def GetFeatureLength(self):
        return self.__param["dim"]

    # 特徴計算するサブWindowを指定数だけランダムに決める。
    def __randomSelectSubWin(self, total, randomSelectParamList):
        assert(isinstance(total, int))
        
        out = np.empty((0, 5), int)  # ch, xMin, xMax, yMin, yMax
        pch = total // len(randomSelectParamList)   # average dim per channel

        while out.shape[0] < total:
            chIdx = 0
            for param in randomSelectParamList:
                assert(isinstance(param, self.RandomSelectParam))
                x = np.random.randint(0, param["div"]["w"], 2 * pch).reshape(pch, 2)
                y = np.random.randint(0, param["div"]["h"], 2 * pch).reshape(pch, 2)
                x = np.sort(x, axis = 1)
                y = np.sort(y, axis = 1)
                w = x[np.arange(pch), 1] - x[np.arange(pch), 0] + 1
                h = y[np.arange(pch), 1] - y[np.arange(pch), 0] + 1
                
                # 最大、最小の制限をクリアするサブWindowsのみ抽出
                limitIdx = (w >= param["min"]["w"])\
                         * (w <= param["max"]["w"])\
                         * (h >= param["min"]["h"])\
                         * (h <= param["max"]["h"])
                limitIdx = limitIdx.astype(np.bool)
                
                # shape is (winNum x 4).
                # Here 4 is composed of (xMin, xMax, yMin, yMax)
                cand = np.append(x[limitIdx], y[limitIdx], axis = 1)

                assert(cand.ndim == 2)
                assert(cand.shape[1] == 4)
                cand = np.insert(cand, 0, chIdx, axis=1)
                assert(cand.shape[1] == 5)
                out = np.append(out, cand, axis = 0)
                chIdx += 1
            out = extractUniqueRows(out)    # 重複した分は除く
        out = out[:total] #超過分を削る
        return out
        
    # re-order to put color the shallowest index.
    def __getRed(self, srcImgs):
        return srcImgs.transpose(3, 0, 1, 2)[0]#.reshape(srcImgs.shape[0],srcImgs.shape[1],srcImgs.shape[2])
    def __getGreen(self, srcImgs):
        return srcImgs.transpose(3, 0, 1, 2)[1]#.reshape(srcImgs.shape[0],srcImgs.shape[1],srcImgs.shape[2])
    def __getBlue(self, srcImgs):
        return srcImgs.transpose(3, 0, 1, 2)[2]#.reshape(srcImgs.shape[0],srcImgs.shape[1],srcImgs.shape[2])
    def __getGradHist(self, srcImgs):
        imgNum = srcImgs.shape[0]
        edgeBin = self.__param["edgeBin"]
        magnitude, theta = CHog.calcEdge(CHog, self.__getGreen(srcImgs), edgeBin)   #エッジ画像は緑から作る
        assert(magnitude.ndim == 3)
        assert(theta.ndim == 3)
        assert(magnitude.shape == theta.shape)
        gradHistCh = np.empty((edgeBin, imgNum, theta.shape[1], theta.shape[2]))
        for b in range(edgeBin):
            gradHistCh[b] = magnitude * (theta == b)

        # bin, imgNum, reducedH, reducedW
        assert(gradHistCh.ndim == 4)
        
        # Convolutionによってサイズが縮んでいるので、numpyに格納すべくダミーでサイズを膨らす
        gradHistCh = np.insert(gradHistCh, (gradHistCh.shape[2],gradHistCh.shape[2]), 0, axis = 2)
        gradHistCh = np.insert(gradHistCh, (gradHistCh.shape[3],gradHistCh.shape[3]), 0, axis = 3)
        assert(srcImgs.shape[0] == gradHistCh.shape[1])
        assert(srcImgs.shape[1] == gradHistCh.shape[2] == self.__param["srcSize"]["h"])
        assert(srcImgs.shape[2] == gradHistCh.shape[3] == self.__param["srcSize"]["w"])
        return gradHistCh
    '''
    入力：画像（輝度の(1+3)次元マップ）の配列
    出力：1次元の特徴量ベクトル
    '''
    def calc(self, srcImgs):
        
        srcImgH = self.__param["srcSize"]["h"]
        srcImgW = self.__param["srcSize"]["w"]
        edgeBin = self.__param["edgeBin"]
        dim = self.__param["dim"]
        imgs = np.array(srcImgs)
        assert(imgs.ndim == 4)    # fileNum, h ,w, color
        sampleNum = imgs.shape[0]
        
        # まずは使用するチャンネル画像を生成する
        imgCh = np.empty((0, sampleNum, srcImgH, srcImgW))
        for ch, uses in self.__param["ch"].items():
            if True == uses:
                imgCh = np.append(imgCh, self.__imgSetFnc[ch](imgs).reshape(-1, sampleNum, srcImgH, srcImgW), axis = 0)
        assert(np.all(imgCh>=0))
        assert(imgCh.shape[0] == int(self.__param["ch"]["red"] == True)\
                               + int(self.__param["ch"]["green"] == True)\
                               + int(self.__param["ch"]["blue"] == True)\
                               + int(self.__param["ch"]["histGrad"] == True) * self.__param["edgeBin"])
        # Now shape of imgCh is (ch, N, H, W)
        
        # 積分画像化する
        intgImgCh = np.zeros((imgCh.shape[0], 
                              imgCh.shape[1], 
                              imgCh.shape[2] + 1,     # zero padding 
                              imgCh.shape[3] + 1))    # zero padding
        for ch in range(imgCh.shape[0]):
            intgImgCh[ch] = MakeIntegral(imgCh[ch])
        assert(np.all(intgImgCh>=0))

        outScore = np.zeros((dim, sampleNum))
        for d in range(dim):
            ch, x0, x1, y0, y1 = self.selectedSubWins[d]
            outScore[d] = SumFromIntegral(intg = intgImgCh[ch],
                                          y0 = y0,
                                          y1 = y1,
                                          x0 = x0,
                                          x1 = x1)
        outScore = outScore.astype(np.int)
        assert(np.all(outScore>=0))
            
        return outScore.T   # shape : (sampleNum x dim)

    '''
    #入力１：画像（輝度とかbinごとのエッジとか、事前に用意した一式）
    #入力２：計算パラメタ
    #出力：floatスコア値
    def calcScore(self, ch, para):
        assert(0)
        if "edge" == para["channel"]:
            return np.average(\
            ch[para["edge"]]\
            [para["edgeID"]]\
            [para["y"] : para["y"] + para["h"]]\
            [para["x"] : para["x"] + para["w"]])
        else:
            return np.average(\
            ch[para["channel"]]\
            [para["y"] : para["y"] + para["h"]]\
            [para["x"] : para["x"] + para["w"]])
    '''
   
    def GetFeatureLength(self):
        return self.__param["dim"]
    
        
if __name__ == "__main__":
    param = CChnFtrs.CChnFtrsParam()
    chnFtrs = CChnFtrs(param)
    print("Done.")
    
