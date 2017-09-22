#coding: utf-8
import os
import numpy as np
from common import fileIO as fio
import scipy.io as sio
from common import mathtool as mt
from common import imgtool as imt
from PIL import Image

def GetMatFromDir(dir):
    fileList = fio.GetFileList(path=dir, includingText=".bmp")
    tgtShape = imt.imgPath2ndarray(fileList[0]).shape
    score = np.empty((0,tgtShape[0],tgtShape[1]),float)
    for file in fileList:
        print(file)
        score = np.append(score, [imt.imgPath2ndarray(file)], axis=0)
    return score

if "__main__" == __name__:

    dstName = fio.ExtractName(os.path.dirname(__file__)) + ".npz"
    trap=GetMatFromDir("TrainPos")
    tran=GetMatFromDir("TrainNeg")
    tesp=GetMatFromDir("TestPos")
    tesn=GetMatFromDir("TestNeg")
    np.savez(dstName,\
             TrainPos=trap,
             TrainNeg=tran,
             TestPos=tesp,
             TestNeg=tesn)
    print("Done.")
    