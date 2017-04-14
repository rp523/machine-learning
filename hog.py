import numpy as np
from PIL import Image,ImageOps
import sys
import matplotlib.pyplot as plt
import imgtool as imt
import mathtool as mt

    
class CHogParam():
    def __init__(self, 
                 bin=None,
                 tgtX = None,
                 tgtY = None,
                 tgtW = None,
                 tgtH = None,
                 cellX = None,
                 cellY = None,
                 blockX = None,
                 blockY = None,
                 jointAND = None,
                 jointOR = None,
                 jointXOR = None):
        if None != bin:
            self.bin = bin
        else:
            self.bin = 8

        if None != cellX:
            self.cellX = cellX
        else:
            self.cellX = 8

        if None != cellY:
            self.cellY = cellY
        else:
            self.cellY = 8
        
        if None != blockX:
            self.blockX = blockX
        else:
            self.blockX = 1
        
        if None != blockY:
            self.blockY = blockY
        else:
            self.blockY = 1
        
        if None != jointAND:
            self.jointAND = jointAND
        else:
            self.jointAND = False
        
        if None != jointOR:
            self.jointOR = jointOR
        else:
            self.jointOR = False
        
        if None != jointXOR:
            self.jointXOR = jointXOR
        else:
            self.jointXOR = False

        if not tgtX:
            self.tgtX = tgtX
        else:
            self.tgtX = 0
        if not tgtY:
            self.tgtY = tgtY
        else:
            self.tgtY = 0
        if not tgtW:
            self.tgtW = tgtW
        else:
            self.tgtW = self.cellX
        if not tgtH:
            self.tgtH = tgtH
        else:
            self.tgtY = self.cellY

class CHog:
    def __init__(self, HogParam):
        
        assert(isinstance(HogParam, CHogParam))
        self.__hogParam = HogParam
       
    def calc(self,srcImg):
        
        hogParam = self.__hogParam
        
        assert(isinstance(srcImg, np.ndarray))

        imgHeight = srcImg.shape[0]
        imgWidth  = srcImg.shape[1]

        dxFilter = np.array(((-1,-1,-1),
                             ( 0, 0, 0),
                             ( 1, 1, 1)))
        dyFilter = np.array(((-1, 0, 1),
                             (-1, 0, 1),
                             (-1, 0, 1)))
        dy = mt.Convolution(srcImg, dxFilter)
        dx = mt.Convolution(srcImg, dyFilter)

        magnitude = np.sqrt((dx * dx) + (dy * dy))
        theta = np.arctan2( dy, dx ) / np.pi * self.__hogParam.bin
        #真横エッジ、真縦エッジを効率良く拾うためにBin/2だけずらすを調整
        theta = theta + 0.5
        #値の範囲調整(0〜bin-1)
        theta = theta + ((1 * (theta < 0)) - (1 * (theta >= self.__hogParam.bin))) * self.__hogParam.bin 
        #int化
        theta = theta.astype(np.int) 

        # reshape,mean機能だけで和が取れるようにゼロを挿入する
        padY = hogParam.cellY - theta.shape[0] % hogParam.cellY
        padX = hogParam.cellX - theta.shape[1] % hogParam.cellX
        cellHeight = theta.shape[0] // hogParam.cellY
        cellWidth  = theta.shape[1] // hogParam.cellX
        insertY = tuple(np.arange(padY) * cellHeight)
        insertX = tuple(np.arange(padX) * cellWidth )
        theta     = np.insert(theta,     insertY, 0,   axis = 0)
        theta     = np.insert(theta,     insertX, 0,   axis = 1)
        magnitude = np.insert(magnitude, insertY, 0.0, axis = 0)
        magnitude = np.insert(magnitude, insertX, 0.0, axis = 1)
        assert(0 == (theta.shape[0] % hogParam.cellY))
        assert(0 == (theta.shape[1] % hogParam.cellX))
        
        # 画像の各点においてtheta-binの箇所だけ1になっているbin長さのVector行列を作る
        oneHot = np.zeros((theta.size, hogParam.bin),float)
        oneHot[np.arange(theta.size),theta.flatten()] = 1
        
        #ゼロ挿入によって形が変わっているので、再度セルの大きさを計算
        cellHeight = theta.shape[0] // hogParam.cellY
        cellWidth  = theta.shape[1] // hogParam.cellX
        
        hogCube = oneHot.T.reshape(hogParam.bin, theta.shape[0], theta.shape[1]) * magnitude
        hogCube = hogCube.reshape(hogParam.bin, hogParam.cellY, cellHeight, hogParam.cellX, cellWidth)
        rawHogMap = hogCube.transpose(0,1,3,2,4).sum(axis=4).sum(axis=3)

        # L2ノルムの計算
        normMap = np.sqrt((rawHogMap * rawHogMap).transpose(1,2,0).sum(axis = 2))
        
        normalizedHogMap = (rawHogMap / normMap).transpose(1,2,0)
        
        return normalizedHogMap.flatten()

        noJointLen = normalizedHogMap.size
        if hogParam.jointAND:
            pass
        if self.__hogParam.jointOR:
            pass
        if self.__hogParam.jointXOR:
            pass


    def GetFeatureLength(self):
        noJoint =  ( self.__hogParam.cellY - self.__hogParam.blockY + 1 ) * \
               ( self.__hogParam.cellX - self.__hogParam.blockX + 1 ) * \
               self.__hogParam.blockY *                                 \
               self.__hogParam.blockX *                                 \
               self.__hogParam.bin
        ret = noJoint
        if False != self.__hogParam.jointAND:
            ret += int(noJoint * (noJoint - 1) / 2)
        if False != self.__hogParam.jointOR:
            ret += int(noJoint * (noJoint - 1) / 2)
        if False != self.__hogParam.jointXOR:
            ret += int(noJoint * (noJoint - 1) / 2)
        return ret
    
    def ShowBinImg(self, shape, strong=None, img=None):
        
        if None != img:
            self.calc(img)
        
        backGroundColor = 0
        if None != strong:
            backGroundColor = 255

        map = np.array( [
                            [
                                [backGroundColor]*3
                            ]*shape[1]
                        ]*shape[0]
                       )
 
        whiteColor = np.array((255,255,255))
        
        # グリッド線を引く
        # dr.DrawGrid(grayMap, (self.__hogParam.cellX,self.__hogParam.cellY), 50)
        
        height = shape[0]
        width  = shape[1]
        unitCellY = height / self.__hogParam.cellY
        unitCellX = width / self.__hogParam.cellX
        radius = min(unitCellX,unitCellY)/2 - 1
        boldMax = int(radius/8)

        if None != strong:
            colMax = -100.0
            for y in range(0,self.__hogParam.cellY):
                for x in range(0,self.__hogParam.cellX):
                    for b in range(0,self.__hogParam.bin):
                        d = self.__hogParam.bin * ( self.__hogParam.cellX * y + x ) + b
                        colMax = max(colMax, np.max(np.max(strong[d]), -1.0*np.min(strong[d])))
            if 0.0 == colMax:
                colMax = 1.0

        for y in range(0,self.__hogParam.cellY):
            centerY = (y+0.5)*unitCellY
            sCellY = int(y/unitCellY)

            for x in range(0,self.__hogParam.cellX):
                centerX = (x+0.5)*unitCellX
                sCellX = int(x/unitCellX)

                for b in range(0,self.__hogParam.bin):
                    theta = (self.__binUnit * b) + (0.5 * np.pi)
                    sy = mt.IntMinMax( centerY + (radius * np.sin(theta)), 0, height-1 )
                    sx = mt.IntMinMax( centerX + (radius * np.cos(theta)), 0, width-1 )
                    ey = mt.IntMinMax( centerY - (radius * np.sin(theta)), 0, height-1 )
                    ex = mt.IntMinMax( centerX - (radius * np.cos(theta)), 0, width-1 )
                    
                    if None == strong:
                        colorVal = (self.__binMap[y,x,b] / self.__normMap[sCellY,sCellX]) * whiteColor
                        bold = 1
                    else:
                        d = self.__hogParam.bin * ( self.__hogParam.cellX * y + x ) + b
                        # pos/negへの貢献度によって色を変える
                        red = green = blue = backGroundColor
                        if 0 < strong[d][b]:
                            red = 255
                            blue = green = mt.IntMinMax(255.0*strong[d][b]/colMax,0,255)
                        else:
                            blue = 255
                            green = red = mt.IntMinMax(-255.0*strong[d][b]/colMax,0,255)
                        colorVal = (red,green,blue) 
                        
                        bold = int(self.__binMap[y,x,b] / self.__normMap[sCellY,sCellX] * boldMax)
                    imt.DrawBoldLine(map, (sy,sx), (ey,ex), colorVal,bold)
                    #imt.DrawLine(grayMap, (sy,sx), (ey,ex), colorVal)

        img = imt.ndarray2PILimg(map)
        img.show()

        
        
        
if __name__ == "__main__":
    
    img = imt.OpenAsGrayImg("TrainPos/person_and_bike_013e.bmp")
    img.show()
    imgArray = np.asarray(img)
    Hog = CHog(CHogParam(bin=8,cellX=4,cellY=8,blockX=1,blockY=1))
    Hog.calc(imgArray)
    Hog.ShowBinImg((800,400))
    
