#coding: utf-8
from PIL import Image
import numpy as np
from common.origLib import *
from common.fileIO import *
from common.imgtool import *

def dirPath2NumpyArray(dirPath):

    out = []
    
    for imgPath in GetFileList(dirPath):
        pilImg = Image.open(imgPath)
        npImg = np.asarray(pilImg)
        
        if npImg.ndim >= 3:
            if npImg.shape[2] >= 4:
                npImg = npImg[:,:,:3]
        out.append(npImg)
        
    return np.array(out)
