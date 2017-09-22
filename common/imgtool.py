#coding: utf-8
from PIL import Image,ImageOps
import numpy as np
from common import mathtool as mt
from PIL import Image
 
    

def DrawDot(inMap,p,colorVal):

    y = p[0]
    x = p[1]
    h = inMap.shape[0]
    w = inMap.shape[1]
    
    if (0 <= x) and (w > x) and (0 <= y) and (h > y):
        inMap[p[0],p[1]] = colorVal

def DrawBoldLine(inMap,s,e,colorVal,bold):

    outMap = inMap
    
    lenY = abs(e[0] - s[0])
    lenX = abs(e[1] - s[1])
    if (0==lenX) and (0==lenY):
        # 点描画
        DrawDot(outMap,s,colorVal)
        return

    tmpS, tmpE = list(s), list(e)
    step = 0;
    sign = 1;
    if lenX > lenY:
        for i in range(bold):
            tmpS[0] += step * sign
            tmpE[0] += step * sign
            DrawLine(inMap=outMap,s=tmpS,e=tmpE,colorVal=colorVal)
            step += 1
            sign *= (-1)
    else:
        for i in range(bold):
            tmpS[1] += step * sign
            tmpE[1] += step * sign
            DrawLine(inMap=outMap,s=tmpS,e=tmpE,colorVal=colorVal)
            step += 1
            sign *= (-1)
    return

def DrawLine(inMap,s,e,colorVal):

    outMap = inMap
    lenY = abs(e[0] - s[0])
    lenX = abs(e[1] - s[1])

    if (0==lenX) and (0==lenY):
        # 点描画
        DrawDot(outMap,s,colorVal)
        return outMap
    
    if lenX > lenY:
        minY = min(s[0],e[0])
        maxY = max(s[0],e[0])
        ratio = (e[0] - s[0]) / (e[1] - s[1])
        step = np.sign(e[1]-s[1])
        for x in np.arange(s[1],e[1]+step,step):
            y = mt.IntMinMax( (ratio*(x-s[1])) + s[0], minY, maxY )
            DrawDot(outMap,(y,x),colorVal)
    else:
        minX = min(s[1],e[1])
        maxX = max(s[1],e[1])
        ratio = (e[1] - s[1]) / (e[0] - s[0])
        step = np.sign(e[0]-s[0])
        for y in np.arange(s[0],e[0]+step,step):
            x = mt.IntMinMax( (ratio*(y-s[0])) + s[1], minX, maxX)
            DrawDot(outMap,(y,x),colorVal)
    
# TODO divNumが正方形でない場合のバグあり
def DrawGrid(inMap, inDivNum, inColorVal):
    outMap = inMap
    unitX = inMap.shape[0] / inDivNum[0]
    unitY = inMap.shape[1] / inDivNum[1]
    # 横線---
    for i in range(0,inDivNum[0] + 1):
        y = mt.IntMinMax(i*unitY, 0, inMap.shape[1]-1 )
        DrawLine(outMap,(0,y),(inMap.shape[0]-1,y), inColorVal)
    # 縦線|||
    for i in range(0,inDivNum[1] + 1):
        x = mt.IntMinMax(i*unitX, 0, inMap.shape[0]-1 )
        DrawLine(outMap,(x,0),(x,inMap.shape[1]-1), inColorVal)
    return outMap

if "__main__" == __name__:
    exit()

def ndarray2PILimg(inMap):
    # グレースケール画像の場合
    return Image.fromarray(np.uint8(inMap))  

def PIL2ndarray(inarray):
    return np.asarray(inarray)

def OpenAsGrayImg(imgFileName):
    return ImageOps.grayscale(Image.open(imgFileName))

def imgPath2ndarray(inPath):
    return PIL2ndarray(OpenAsGrayImg(inPath))