#coding: utf-8
import common.fileIO as fio
import os
import random as rd
from PIL import Image
import common.imgtool as imt
import numpy as np
from scipy import io as sio
import random
import shutil

def MakePositive(datasetDirPath,srcDirName,dstDirName,w,h,x,y):
    
    srcDirPath = os.path.join(datasetDirPath,srcDirName)
    dstDirPath = dstDirName
    if not os.path.exists(dstDirPath):
        os.makedirs(dstDirPath)
    
    imgList = fio.GetFileList(path=srcDirPath,recursive=True,onlyName=False)
    for img in imgList:
        extension = fio.ExtractExtension(img)
        dstImgName = fio.ExtractName(img).replace(extension,".bmp")
        dstImgPath = os.path.join(dstDirPath,dstImgName)
        os.system("convert -crop " + str(w) + "x" + str(h) + "+" + str(x) + "+" + str(y) + " " + img + " " + dstImgPath)
        
def MakeNegative(datasetDirPath,srcDirName,dstDirName,subW,subH):
    srcDirPath = os.path.join(datasetDirPath,srcDirName)
    dstDirPath = dstDirName
    if not os.path.exists(dstDirPath):
        os.makedirs(dstDirPath)
    
    imgPathList = fio.GetFileList(path=srcDirPath,recursive=True,onlyName=False)
    for imgPath in imgPathList:
        img = Image.open(imgPath)
        imgW,imgH = img.size
        for i in range(0,1):
            subX = rd.randint(0,imgW-subW)
            subY = rd.randint(0,imgH-subH)
            dstImageName = fio.ExtractName(imgPath).replace(fio.ExtractExtension(imgPath),"") + "_{0:02d}".format(i) + ".bmp"
            dstImagePath = os.path.join(dstDirPath,dstImageName)
            os.system("convert -crop " + str(subW) + "x" + str(subH) + "+" + str(subX) + "+" + str(subY) + " " + imgPath + " " + dstImagePath)

def NormalizeINRIA():
    MakeNegative(datasetDirPath="dataset/INRIAPerson",
                 srcDirName=os.path.join("Train","neg"),
                 dstDirName="TrainNeg",
                 subW=64,
                 subH=128)
    MakeNegative(datasetDirPath="dataset/INRIAPerson",
                 srcDirName=os.path.join("Test","neg"),
                 dstDirName="TestNeg",
                 subW=64,
                 subH=128)
    MakePositive(datasetDirPath="dataset/INRIAPerson",
                 srcDirName="96X160H96",
                 dstDirName="TrainPos",
                 w=64,
                 h=128,
                 x=16,
                 y=16)
    MakePositive(datasetDirPath="dataset/INRIAPerson",
                 srcDirName="70X134H96",
                 dstDirName="TestPos",
                 w=64,
                 h=128,
                 x=3,
                 y=3)
    print("All Done.")
    
def ExtractSub(ratio):
    for dir in ["INRIAPerson/LearnPos", "INRIAPerson/LearnNeg", "INRIAPerson/EvalPos", "INRIAPerson/EvalNeg"]:
        dstDir = dir + "Sub"
        if not os.path.exists(dstDir):
            os.mkdir(dstDir)
        itemPathList = fio.GetFileList(dir)
        itemPathLen = len(itemPathList)
        for itemPath in random.sample(itemPathList, int(itemPathLen * ratio)):
            dstPath = os.path.join(dstDir,fio.ExtractName(itemPath))
            shutil.copyfile(itemPath, dstPath)
    print("ExtractSub fin.")
            
    
def MakeMat(inTrainPos,inTrainNeg,inTestPos,inTestNeg):
    
    dict = {}
    trainPosName = []
    trainNegName = []
    testPosName = []
    testNegName = []
    trainPosData = []
    trainNegData = []
    testPosData = []
    testNegData = []

    for img in fio.GetFileList(inTrainPos):
        trainPosName.append(fio.ExtractName(img))
        imgArray = np.asarray(Image.open(img))
        if 4 == imgArray.shape[2]:
            imgArray = np.delete(imgArray, 3, axis=2)
        trainPosData.append(imgArray)
    
    for img in fio.GetFileList(inTrainNeg):
        trainNegName.append(fio.ExtractName(img))
        imgArray = np.asarray(Image.open(img))
        if 4 == imgArray.shape[2]:
            imgArray = np.delete(imgArray, 3, axis=2)
        trainNegData.append(imgArray)
    
    for img in fio.GetFileList(inTestPos):
        trainPosName.append(fio.ExtractName(img))
        imgArray = np.asarray(Image.open(img))
        if 4 == imgArray.shape[2]:
            imgArray = np.delete(imgArray, 3, axis=2)
        testPosData.append(imgArray)
    
    for img in fio.GetFileList(inTestNeg):
        testNegName.append(fio.ExtractName(img))
        imgArray = np.asarray(Image.open(img))
        if 4 == imgArray.shape[2]:
            imgArray = np.delete(imgArray, 3, axis=2)
        testNegData.append(imgArray)
    
    dict["TrainPosName"] = trainPosName
    dict["TrainNegName"] = trainNegName
    dict["TestPosName"] = testPosName
    dict["TestNegName"] = testNegName
    
    dict["TrainPosData"] = trainPosData
    dict["TrainNegData"] = trainNegData
    dict["TestPosData"] = testPosData
    dict["TestNegData"] = testNegData
    
    sio.savemat("INRIAPerson.mat",dict)

if '__main__' == __name__:
    ExtractSub(0.05)
    exit()
    NormalizeINRIA()
    exit()
    
    array = sio.loadmat("INRIAPerson.mat")["TrainPosData"][0]
    pilImg = Image.fromarray(np.uint8(array))
    pilImg.show()
    exit()
    MakeMat("TrainPos", "TrainNeg", "TestPos", "TestNeg")