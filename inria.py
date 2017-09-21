#coding: utf-8
import os,sys,shutil
from gaussianProcess import *
from input import *
from preproc import *
from feature.hog import *
from feature.chnFtrs import *
from adaBoost import *
from inflFunc import *

def inriaInflFuncTest():
    for xlsxFile in  GetFileList(".", includingText = ".xlsx"):
        os.remove(xlsxFile)
    for matFile in  GetFileList(".", includingText = ".mat"):
        os.remove(matFile)
    for csvFile in  GetFileList(".", includingText = ".csv"):
        os.remove(csvFile)
    
    lp = dirPath2NumpyArray("dataset/INRIAPerson/LearnPos")[:102]
    ln = dirPath2NumpyArray("dataset/INRIAPerson/LearnNeg")[:101]
    ep = dirPath2NumpyArray("dataset/INRIAPerson/EvalPos" )[:99]
    en = dirPath2NumpyArray("dataset/INRIAPerson/EvalNeg" )[:98]
    eval  = RGB2Gray(np.append(ep, en, axis = 0), "green")
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))

    def smallSampleTry(idx):
        learn = RGB2Gray(np.append(lp, ln, axis = 0), "green")[idx]
        learnLabel = np.array([1] * len(lp) + [-1] * len(ln))[idx]
        hogParam = CHogParam()
        hogParam["Bin"] = 8
        hogParam["Cell"]["X"] = 2
        hogParam["Cell"]["Y"] = 4
        hogParam["Block"]["X"] = 1
        hogParam["Block"]["Y"] = 1
        detectorList = [CHog(hogParam)]
    
        adaBoostParam = AdaBoostParam()
        adaBoostParam["Regularizer"] = 1e-5
        adaBoostParam["Bin"] = 32
        adaBoostParam["Type"].setTrue("Real")
        adaBoostParam["verbose"] = False
        adaBoostParam["saveDetail"] = True
        
        adaBoost = CAdaBoost()
        adaBoost.SetParam(  inAdaBoostParam = adaBoostParam,
                            inImgList = learn,
                            inLabelList = learnLabel,
                            inDetectorList = detectorList)
    
        trainScoreMat = np.empty((len(learn), 0))
        for detector in detectorList:
            trainScoreMat = np.append(trainScoreMat,
                                      detector.calc(learn),
                                      axis = 1)
    
        adaBoost.Boost(trainScoreMat = trainScoreMat,
                       labelList = learnLabel)
        
        # 評価用の特徴量行列を準備    
        testScoreMat = np.empty((len(eval), 0))
        for detector in detectorList:
            testScoreMat = np.append(testScoreMat,
                                     detector.calc(eval),
                                     axis = 1)
        
        evalScore = adaBoost.Evaluate(testScoreMat = testScoreMat,
                                      label = evalLabel)

        accuracy, auc = gui.evaluateROC(evalScore, evalLabel)
        return auc, evalScore
    
    sampleIdx = np.arange(len(lp) + len(ln))
    param = CInfluenceParam()
    param["removeMax"] = 1
    param["evalTarget"].setTrue("largeNeg")
    
    tgtOld = None
    while 1:
        auc, evalScore = smallSampleTry(sampleIdx)
        sampleIdx, remove, evalTgt = CInfluence(inParam = param).RefineLearningSample()
        
        with open("result.csv", "a") as f:
            f.write(str(auc) + "," + str(evalLabel[evalTgt]) + "," + str(evalTgt) + ",")
            if tgtOld:
                f.write(str(evalScore[tgtOld]) + ",")
            else:
                f.write(",")
            f.write(str(evalScore[evalTgt]) + "\n")
        tgtOld = evalTgt
        

def inriaAdaboostICF_gp():
    paramMin = dicts()
    paramMax = dicts()
    paramMin["regDataDist"] = 0.0
    paramMax["regDataDist"] = 2.0
    gp = CGaussianProcess(iterTotal = 100,
                          sampleNum = 1e5,
                          paramMin = paramMin,
                          paramMax = paramMax,
                          maximizedFunc=inriaAdaboostICF)
    gp.Execute()
    print(inriaAdaboostICF(x))
    print("Done.")
    
def inriaAdaboostICF():
    for delFile in GetFileList(path=".", includingText=".mat"):
        os.remove(delFile)
    if 0:
        lp = dirPath2NumpyArray("INRIAPerson/LearnPos")
        ln = dirPath2NumpyArray("INRIAPerson/LearnNeg")
        ep = dirPath2NumpyArray("INRIAPerson/EvalPos" )
        en = dirPath2NumpyArray("INRIAPerson/EvalNeg" )
    else:
        lp = dirPath2NumpyArray("dataset/INRIAPerson/LearnPos")[:100]
        ln = dirPath2NumpyArray("dataset/INRIAPerson/LearnNeg")[:100]
        ep = dirPath2NumpyArray("dataset/INRIAPerson/EvalPos" )[:100]
        en = dirPath2NumpyArray("dataset/INRIAPerson/EvalNeg" )[:100]
    learn = lp + ln
    eval = ep + en
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))

    '''
    param = CChnFtrs.CChnFtrsParam()
    param["dim"] = 5000
    param["min"]["w"] = 0.1
    param["max"]["w"] = 0.5
    param["min"]["h"] = 0.1
    param["max"]["h"] = 0.5
    detectorList = [CChnFtrs(param)]
    adaBoostParam = AdaBoostParam()
    adaBoostParam["Type"].setTrue("RealTree")
    adaBoostParam["Saturate"] = True
    adaBoostParam["Regularizer"] = 1e-5
    adaBoostParam["Bin"] = 32
    adaBoostParam["Loop"] = 1000
    adaBoostParam["TreeDepth"] = 5
    adaBoostParam["regDataDist"] = 0.0
    adaBoostParam["verbose"] = False
    '''
    hogParam = CHogParam()
    hogParam["Bin"] = 8
    hogParam["Cell"]["X"] = 5
    hogParam["Cell"]["Y"] = 10
    hogParam["Block"]["X"] = 2
    hogParam["Block"]["Y"] = 2
    detectorList = [CHog(hogParam)]
    learn = RGB2Gray(learn, "green")
    eval  = RGB2Gray(eval , "green")
    adaBoostParam = AdaBoostParam()
    adaBoostParam["Type"].setTrue("RealTree")
    adaBoostParam["Saturate"] = True
    adaBoostParam["Regularizer"] = 1e-5
    adaBoostParam["Bin"] = 32
    adaBoostParam["Loop"] = 1000
    adaBoostParam["TreeDepth"] = 5
    adaBoostParam["regDataDist"] = 0.0
    adaBoostParam["verbose"] = False

    AdaBoost = CAdaBoost(inAdaBoostParam=adaBoostParam,
                         inImgList=learn,
                         inLabelList=learnLabel,
                         inDetectorList=detectorList)
    AdaBoost.Evaluate(inImgList=eval,inLabelList=evalLabel)
    #print(np.asarray(sio.loadmat("FinalScore.mat")["finalScore"])[0])
    accuracy, auc = gui.evaluateROC(np.asarray(sio.loadmat("FinalScore.mat")["finalScore"])[0],
        np.asarray(sio.loadmat("FinalScore.mat")["label"])[0])
    gui.DrawROC(np.asarray(sio.loadmat("FinalScore.mat")["finalScore"])[0],
        np.asarray(sio.loadmat("FinalScore.mat")["label"])[0])
    print(auc)
    return auc


def auc(x):
    for delFile in GetFileList(path=".", includingText=".mat"):
        os.remove(delFile)
    if False:
        lp = dirPath2NumpyArray("INRIAPerson/LearnPos")
        ln = dirPath2NumpyArray("INRIAPerson/LearnNeg")
        ep = dirPath2NumpyArray("INRIAPerson/EvalPos")
        en = dirPath2NumpyArray("INRIAPerson/EvalNeg")
    else:
        lp = dirPath2NumpyArray("INRIAPerson/LearnPosSub")[:20]
        ln = dirPath2NumpyArray("INRIAPerson/LearnNegSub")[:20]
        ep = dirPath2NumpyArray("INRIAPerson/EvalPosSub")[:20]
        en = dirPath2NumpyArray("INRIAPerson/EvalNegSub")[:20]
    learn = RGB2Gray(lp + ln, "green")
    eval  = RGB2Gray(ep + en, "green")
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))
    hogParam = CHogParam()
    hogParam["Bin"] = x["hogBin"]
    hogParam["Cell"]["X"] = x["hogCellX"]
    hogParam["Cell"]["Y"] = x["hogCellY"]
    hogParam["Block"]["X"] = x["hogBlockX"];
    hogParam["Block"]["Y"] = x["hogBlockY"]
    detectorList = [CHog(hogParam)]

    adaBoostParam = AdaBoostParam()
    adaBoostParam["Regularizer"] = x["adaRegu"]
    adaBoostParam["Bin"] = x["adaBin"]
    adaBoostParam["Loop"] = 2000

    AdaBoost = CAdaBoost(inAdaBoostParam=adaBoostParam,
                         inImgList=learn,
                         inLabelList=learnLabel,
                         inDetectorList=detectorList)
    AdaBoost.Evaluate(inImgList=eval,inLabelList=evalLabel)
    accuracy, auc = gui.evaluateROC(np.asarray(sio.loadmat("FinalScore.mat")["finalScore"])[0],
        np.asarray(sio.loadmat("FinalScore.mat")["label"])[0])
    return auc

def inriaHogAdaboost():
    
    paramMin = dicts()
    paramMax = dicts()
    
    paramMin["hogBin"] = 1
    paramMax["hogBin"] = 16

    paramMin["hogCellY"] = 1
    paramMax["hogCellY"] = 20
    
    paramMin["hogCellX"] = 1
    paramMax["hogCellX"] = 10
    
    paramMin["hogBlockY"] = 1
    paramMax["hogBlockY"] = 8
    
    paramMin["hogBlockX"] = 1
    paramMax["hogBlockX"] = 4
    
    paramMin["adaRegu"] = 0.0
    paramMax["adaRegu"] = 1.0
    
    paramMin["adaBin"] = 1
    paramMax["adaBin"] = 128

    gp = CGaussianProcess(iterTotal = 100,
                          sampleNum = 1e5,
                          paramMin = paramMin,
                          paramMax = paramMax,
                          maximizedFunc=auc)
    gp.Execute()
    print(auc(x))
    print("Done.")
    
if "__main__" == __name__:
    inriaInflFuncTest()
    print("Done.")