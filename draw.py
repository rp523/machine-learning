#coding: utf-8
import numpy as np
import common
from common import IntMinMax
from PIL import Image
 
    

def DrawDot(inMap,p,colorVal):
    outMap = inMap
    outMap[p[0],p[1]] = colorVal
    return outMap

def DrawLine(inMap,s,e,colorVal):

    outMap = inMap

    lenY = abs(e[0] - s[0])
    lenX = abs(e[1] - s[1])

    if (0==lenX) and (0==lenY):
        # 点描画
        return DrawDot(outMap,s,colorVal)
    
    if lenX > lenY:
        minY = min(s[0],e[0])
        maxY = max(s[0],e[0])
        ratio = (e[0] - s[0]) / (e[1] - s[1])
        step = np.sign(e[1]-s[1])
        for x in np.arange(s[1],e[1]+step,step):
            y = IntMinMax( (ratio*(x-s[1])) + s[0], minY, maxY )
            outMap = DrawDot(outMap,(y,x),colorVal)
    else:
        minX = min(s[1],e[1])
        maxX = max(s[1],e[1])
        ratio = (e[1] - s[1]) / (e[0] - s[0])
        step = np.sign(e[0]-s[0])
        for y in np.arange(s[0],e[0]+step,step):
            x = IntMinMax( (ratio*(y-s[0])) + s[1], minX, maxX)
            outMap = DrawDot(outMap,(y,x),colorVal)
    
# TODO divNumが正方形でない場合のバグあり
def DrawGrid(inMap, inDivNum, inColorVal):
    outMap = inMap
    unitX = inMap.shape[0] / inDivNum[0]
    unitY = inMap.shape[1] / inDivNum[1]
    # 横線---
    for i in range(0,inDivNum[0] + 1):
        y = common.IntMinMax(i*unitY, 0, inMap.shape[1]-1 )
        DrawLine(outMap,(0,y),(inMap.shape[0]-1,y), inColorVal)
    # 縦線|||
    for i in range(0,inDivNum[1] + 1):
        x = common.IntMinMax(i*unitX, 0, inMap.shape[0]-1 )
        DrawLine(outMap,(x,0),(x,inMap.shape[1]-1), inColorVal)
    return outMap

if "__main__" == __name__:
    exit()