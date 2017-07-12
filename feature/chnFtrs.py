import sys
import numpy as np
import matplotlib.pyplot as plt
from common.imgtool import *
from common.mathtool import *
from common.origLib import *
from feature.hog import *
from input import *
from skimage.transform import integral_image,integrate


        

class CChnFtrs:
    
    class CChnFtrsParam(CParam):
        def __init__(self):
            impl = dicts()
            
            ch = dicts()
            ch["red"] = False
            ch["green"] = True
            ch["blue"] = False
            ch["histGrad"] = True
            ch["normGrad"] = False
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
                    minW = int(self.__param["min"]["w"] * self.__param["srcSize"]["w"] + 0.5) - int(self.__param["min"]["w"] == 1.0)
                    minH = int(self.__param["min"]["h"] * self.__param["srcSize"]["h"] + 0.5) - int(self.__param["min"]["h"] == 1.0)
                    maxW = int(self.__param["max"]["w"] * self.__param["srcSize"]["w"] + 0.5) - int(self.__param["max"]["w"] == 1.0)
                    maxH = int(self.__param["max"]["h"] * self.__param["srcSize"]["h"] + 0.5) - int(self.__param["max"]["h"] == 1.0)
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
                        minW = int(self.__param["min"]["w"] * (self.__param["srcSize"]["w"] - 2) + 0.5) - int(self.__param["min"]["w"] == 1.0)
                        minH = int(self.__param["min"]["h"] * (self.__param["srcSize"]["h"] - 2) + 0.5) - int(self.__param["min"]["h"] == 1.0)
                        maxW = int(self.__param["max"]["w"] * (self.__param["srcSize"]["w"] - 2) + 0.5) - int(self.__param["max"]["w"] == 1.0)
                        maxH = int(self.__param["max"]["h"] * (self.__param["srcSize"]["h"] - 2) + 0.5) - int(self.__param["max"]["h"] == 1.0)
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
        pch = total // len(randomSelectParamList)

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
                limitIdx = (w >= param["min"]["w"]) * (w <= param["max"]["w"]) * (h >= param["min"]["h"]) * (w <= param["max"]["h"])
                limitIdx = limitIdx.astype(np.bool)
                cand = np.append(x[limitIdx], y[limitIdx], axis = 1)

                assert(cand.ndim == 2)
                assert(cand.shape[1] == 4)
                cand = np.insert(cand, 0, chIdx, axis=1)
                out = np.append(out, cand, axis = 0)
                chIdx += 1
            out = extractUniqueRows(out)    # 重複した分は除く
        out = out[np.random.randint(0, out.shape[0], total)] #超過分を削る
        return out
        
    def __getRed(self, srcImg):
        return srcImg.transpose(2, 0, 1)[0].reshape(1,srcImg.shape[0],srcImg.shape[1])
    def __getGreen(self, srcImg):
        return srcImg.transpose(2, 0, 1)[1].reshape(1,srcImg.shape[0],srcImg.shape[1])
    def __getBlue(self, srcImg):
        return srcImg.transpose(2, 0, 1)[2].reshape(1,srcImg.shape[0],srcImg.shape[1])
    def __getGradHist(self, srcImg):
        #エッジ画像は緑から作る
        edgeBin = self.__param["edgeBin"]
        magnitude, theta = CHog.calcEdge(CHog, srcImg.transpose(2, 0, 1)[1], edgeBin)
        gradHistCh = np.empty((edgeBin, theta.shape[0], theta.shape[1]))
        for b in range(edgeBin):
            gradHistCh[b] = magnitude * (theta == b)
        assert(gradHistCh.ndim == 3)
        
        # Convolutionによってサイズが縮んでいるので、numpyに格納すべくダミーでサイズを膨らす
        gradHistCh = np.insert(gradHistCh, (gradHistCh.shape[1],gradHistCh.shape[1]), 0, axis = 1)
        gradHistCh = np.insert(gradHistCh, (gradHistCh.shape[2],gradHistCh.shape[2]), 0, axis = 2)
        return gradHistCh
    '''
    入力：画像（輝度の3次元マップ）
    出力：1次元の特徴量ベクトル
    '''
    def calc(self, srcImg):
        
        srcImgH = self.__param["srcSize"]["h"]
        srcImgW = self.__param["srcSize"]["w"]
        edgeBin = self.__param["edgeBin"]
        dim = self.__param["dim"]
        assert(isinstance(srcImg, np.ndarray))
        assert(srcImg.ndim == 3)    # h ,w, color
        assert(srcImg.shape == (srcImgH, srcImgW, 3))
        
        # まずは使用するチャンネル画像を生成する
        imgCh = np.empty((0, srcImgH, srcImgW))
        for ch, uses in self.__param["ch"].items():
            if True == uses:
                imgCh = np.append(imgCh, self.__imgSetFnc[ch](srcImg), axis = 0)
        assert(np.all(imgCh>=0))

        # 積分画像化する
        for ch in range(imgCh.shape[0]):
            imgCh[ch] = integral_image(imgCh[ch])
        assert(np.all(imgCh>=0))

        outScore = np.zeros(dim)
        for d in range(dim):
            ch, x1, x2, y1, y2 = self.selectedSubWins[d]
            outScore[d] = integrate(imgCh[ch], (y1, x1), (y2, x2)).astype(np.int)
        assert(np.all(outScore>=0))
            
        return outScore

    '''
    入力１：画像（輝度とかbinごとのエッジとか、事前に用意した一式）
    入力２：計算パラメタ
    出力：floatスコア値
    '''
    def calcScore(self, ch, para):
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
            
   
    def GetFeatureLength(self):
        return self.__param["dim"]
    
        
if __name__ == "__main__":
    param = CChnFtrs.CChnFtrsParam()
    chnFtrs = CChnFtrs(param)
    print("Done.")
    
