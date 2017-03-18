import numpy as np
from PIL import Image,ImageOps
import sys
import matplotlib.pyplot as plt
import imgtool as imt
import mathtool as mt

    
def ZeroPadding(srcMap,pad):
    if not isinstance(srcMap, np.ndarray):
        print("ZeroPadding Input Type Error")
        return 
    
    srcMapHeight = srcMap.shape[0]
    srcMapWidth  = srcMap.shape[1]
    
    retList = []
    newCol = list(np.zeros(pad))
    for p in range(0,pad):
        retList.append(list(np.zeros(srcMapWidth+2*pad)))
    for i in range(0,srcMapHeight):
        newRow = newCol + list(srcMap[i]) + newCol
        retList.append( newRow )
    for p in range(0,pad):
        retList.append(list(np.zeros(srcMapWidth+2*pad)))

    return np.array(retList)

class CHogParam():
    def __init__(self, bin=None,
                       cellX = None,
                       cellY = None,
                       blockX = None,
                       blockY = None,
                       jointAND = None,
                       jointOR = None,
                       jointXOR = None,
                       ):
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


class CHog:
    def __init__(self, HogParam):
        
        if isinstance(HogParam, CHogParam):
            self.__hogParam = HogParam
        else:
            print("HogParam is wrong type.")
       
    def calc(self,srcImg):
        
        # 画像に関するプライベート変数の初期化
        self.__srcImg = None
        if isinstance( srcImg, np.ndarray ):
            self.__srcImg = srcImg
        else:
            print("Error! Input image should be instanced as np.ndarray but is ", type(srcImg))

        self.__srcImgHeight = self.__srcImg.shape[0]
        self.__srcImgWidth  = self.__srcImg.shape[1]
        self.__cellYUnit = self.__srcImgHeight / self.__hogParam.cellY
        self.__cellXUnit = self.__srcImgWidth  / self.__hogParam.cellX
        self.__binUnit    =np.pi           / self.__hogParam.bin

        dxFilter = np.array(((-1,-1,-1),
                             ( 0, 0, 0),
                             ( 1, 1, 1)))
        dyFilter = np.array(((-1, 0, 1),
                             (-1, 0, 1),
                             (-1, 0, 1)))
        dy = mt.Convolution(self.__srcImg, dxFilter)
        dx = mt.Convolution(self.__srcImg, dyFilter)
        magnitude = np.sqrt((dx * dx) + (dy * dy))
        theta = np.arctan2( dy, dx )
        theta = theta + (self.__binUnit * 0.5)
        theta = theta + ((1*(theta<0)) - (1*(theta>np.pi)))*np.pi

        # store magnitude to histogram
        self.__binMap = np.zeros((self.__hogParam.cellY,self.__hogParam.cellX,self.__hogParam.bin))
        self.__normMap= np.zeros((self.__hogParam.cellY,self.__hogParam.cellX))
        
        thetaHeight = theta.shape[0]
        thetaWidth  = theta.shape[1]
        unitCellHeight = thetaHeight / self.__hogParam.cellY
        unitCellWidth  = thetaWidth  / self.__hogParam.cellX
        unitBin = np.pi / self.__hogParam.bin

        for y in range(0,thetaHeight):
            sCellY = int(y/unitCellHeight)
        
            for x in range(0,thetaWidth):
                sCellX = int(x/unitCellWidth)
                sBin = int(theta[y,x]/unitBin)

                self.__binMap[sCellY,sCellX,sBin] = self.__binMap[sCellY,sCellX,sBin] + magnitude[y,x]
                self.__normMap[sCellY,sCellX]     = self.__normMap[sCellY,sCellX] + magnitude[y,x]
                
        self.__feature = self.__Normalize()
        
        noJointLen = self.__feature.size
        if False != self.__hogParam.jointAND:
            featureNew = np.empty(int(noJointLen*(noJointLen-1)/2))
            n = 0
            for i in range(noJointLen):
                for j in range(i + 1, noJointLen):
                    featureNew[n] = min(self.__feature[i], self.__feature[j])
                    n += 1
            self.__feature = np.array(list(self.__feature) + list(featureNew))
        if False != self.__hogParam.jointOR:
            featureNew = np.empty(int(noJointLen*(noJointLen-1)/2))
            n = 0
            for i in range(noJointLen):
                for j in range(i + 1, noJointLen):
                    featureNew[n] = max(self.__feature[i], self.__feature[j])
                    n += 1
            self.__feature = np.array(list(self.__feature) + list(featureNew))
        if False != self.__hogParam.jointXOR:
            featureNew = np.empty(int(noJointLen*(noJointLen-1)/2))
            n = 0
            for i in range(noJointLen):
                for j in range(i + 1, noJointLen):
                    featureNew[n] = max(self.__feature[i], self.__feature[j])   \
                                  - min(self.__feature[i], self.__feature[j])
                    n += 1
            self.__feature = np.array(list(self.__feature) + list(featureNew))

        return self.__feature

    def __WideProb(self,x):
        return x
        
        #return 1 - (x-1)**2
        '''bin = 16
        prob = np.zeros(bin,float)
        cum  = np.zeros(bin,float)
        for b in range(bin):
            prob[b] = np.power(b/bin,0.5*(8-1))
        cum[0] = prob[0]
        for b in range(bin-1):
            cum[b+1] = cum[b] + prob[b]
        cum = np.sum(cum)
        return cum(int(x/bin))
        '''
        
    def __Normalize(self):
 
        ret = np.empty((0))
               
        for by in range(0,self.__hogParam.cellY - self.__hogParam.blockY + 1):
            for bx in range(0,self.__hogParam.cellX - self.__hogParam.blockX + 1):
                
                # ブロックごとに正規化するためのノルム計算
                blockNorm = 0.0
                for cy in range(by, by + self.__hogParam.blockY):
                    for cx in range(bx, bx + self.__hogParam.blockX):
                        blockNorm = blockNorm + self.__normMap[cy,cx] * self.__normMap[cy,cx]
                blockNorm = np.sqrt(blockNorm)
                
                for cy in range(by, by + self.__hogParam.blockY):
                    for cx in range(bx, bx + self.__hogParam.blockX):
                        for b in range(0, self.__hogParam.bin):
                            
                            # ゼロ割回避
                            normalizedFeature = 0.0
                            if 0.0 != blockNorm:
                                normalizedFeature = self.__binMap[cy,cx,b] / blockNorm
                            
                            normalizedFeature = self.__WideProb(normalizedFeature)
                            '''                                
                            if normalizedFeature <= 0.5:
                                coef = [4.71075112e-02, 1.90293064e+00, -3.65323112e+01, 3.13252019e+02, -1.38217473e+03, 3.54164057e+03, -5.28482539e+03, 4.26577859e+03, -1.43923189e+03]
                            else:
                                coef = [-62.66652944,   248.20333923,   -249.09550727,   -133.37201271,  194.14734653,    410.70809106,   -755.83180409,   436.26908686,   -87.36176884]
                            normalizedFeature = mt.CalcPoly(np.array(coef), normalizedFeature * normalizedFeature)
                            if normalizedFeature < 0.0:
                                normalizedFeature = 0.0
                            elif normalizedFeature > 1.0:
                                normalizedFeature = 1.0
                            '''
                            ret = np.append(ret, normalizedFeature )
        return ret

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
    
