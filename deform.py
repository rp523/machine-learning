#coding: utf-8
import numpy as np
from scipy.fftpack import fft2
import cmath
from PIL import Image,ImageOps
import matplotlib.pyplot as plt

class calcDeform():
    
    def __init__(self,img):
        self.__imgHeight = np.shape(img)[0]
        self.__imgWidth  = np.shape(img)[1]

        self.__padImgHeight = self.__imgHeight *2
        self.__padImgWidth = self.__imgWidth *2
        
        self.__img = np.zeros((self.__padImgHeight,self.__padImgWidth))
        for y in range(0,self.__imgHeight):
            for x in range(0,self.__imgWidth):
                self.__img[y][x] = img[y][x]
        
        self.FourierTransform()
    
    def GetWidth(self):
        return self.__imgWidth
    def GetHeight(self):
        return self.__imgHeight
    def GetImg(self):
        return self.__img
    def GetFTabs(self):
        return self.__ft_a
    
    def GetMax(self):
        maxVal = np.max(self.__ft_a)
        maxArg = np.argmax(self.__ft_a)
        maxArgX = ( maxArg / self.__padImgWidth )
        maxArgY = int( maxArg % self.__padImgWidth )
        
        if (0==maxArgX) and (0==maxArgX):
            self.__ft_a[0,0] = 0
            maxVal = np.max(self.__ft_a)
            maxArg = np.argmax(self.__ft_a)
            maxArgX = ( maxArg / self.__padImgWidth )
            maxArgY = int( maxArg % self.__padImgWidth )

        return maxVal,maxArgX,maxArgY
	
    def GetCenter(self):
        cx = 0.0
        cx_norm = 0.0
        cy = 0.0
        cy_norm = 0.0
        for y in range(0, int(self.__padImgHeight/2)):
            for x in range(0, int(self.__padImgWidth/2)):
                cx = cx + x * self.__ft_a[y][x] * self.__ft_a[y][x]
                cx_norm = cx_norm + self.__ft_a[y][x] * self.__ft_a[y][x]
                cy = cy + y * self.__ft_a[y][x] * self.__ft_a[y][x]
                cy_norm = cy_norm + self.__ft_a[y][x] * self.__ft_a[y][x]
				
        icx = int( cx / cx_norm )
        icy = int( cy / cy_norm )
        
        return self.__ft_a[icy][icx], cx/cx_norm, cy/cy_norm
		
    def FourierTransform(self):
        self.__ft_c = fft2(self.__img)
        
        self.__ft_a = np.zeros((self.__padImgHeight,self.__padImgWidth))
        self.__ft_p = np.zeros((self.__padImgHeight,self.__padImgWidth))

        for y in range(0,self.__padImgHeight):
            for x in range(0,self.__padImgWidth):
                self.__ft_a[y][x] = abs(self.__ft_c[y][x])
                self.__ft_p[y][x] = np.arctan2(self.__ft_c[y][x].imag,self.__ft_c[y][x].real)


def calc(rx,ry):
    rawImg = Image.open("lena.jpg")
    rawImg = ImageOps.grayscale(rawImg)
    

    rawImgWidth,rawImgHeight = rawImg.size
    newWidth = int( rawImgWidth * 0.16 )
    newHeight = int( rawImgHeight * 0.16 )
    rawImg = rawImg.resize((newWidth,newHeight))
    rawImgWidth,rawImgHeight = rawImg.size

    craw = calcDeform( np.asarray( rawImg ) )
    rMax,rMaxX,rMaxY = craw.GetCenter() 
    

    modImg = rawImg
    
    newWidth = int( rawImgWidth * rx )
    newHeight = int( rawImgHeight * ry )
    modImg = modImg.resize((newWidth,newHeight)).crop((0,0,rawImgWidth,rawImgHeight))

    cmod = calcDeform( np.asarray( modImg ) )
    mMax,mMaxX,mMaxY = cmod.GetCenter()
    
    ratio = mMax / rMax
    if rMaxX != 0:
        ratioX = mMaxX / rMaxX
    else:
        ratioX = 0
    if rMaxY != 0:
        ratioY = mMaxY / rMaxY
    else:
        ratioY = 0
        
    #Image.fromarray(np.uint8(craw.GetImg())).show()
    #Image.fromarray(np.uint8(cmod.GetImg())).show()
    #Image.fromarray(np.uint8(craw.GetFTabs())).show()
    #Image.fromarray(np.uint8(cmod.GetFTabs())).show()
    #print(craw.GetCenter())
    #print(cmod.GetCenter())
    #print( ratio, ratioX,ratioY )
    #print( 1/ratio, 1/ratioX,1/ratioY )
    return ( ratioX/ratioY )

if __name__ == "__main__":
    minVal = 0.1
    maxVal = 1.9
    step = 0.001

    x =[]
    y = []
    yb = []
    
    ry = minVal
    while True:

        x.append(ry)
        yb.append(ry)
        y.append(calc(1.0,ry))
        
        if int(ry/step)%10==0:
            print("ry:",ry)
        
        ry = ry + step
        if ry > maxVal:
            break
        
    plt.plot(x,y)
    plt.plot(x,yb)
    plt.show()
    