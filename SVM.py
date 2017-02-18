import os
import scipy.io as sio
import numpy as np
import gui

from svm import *
from svmutil import *
from gui import DrawROC

if "__main__" == __name__:
    
    if not os.path.exists("Train.mat"):
        print("Train Score Matrix NOT FOUND.");exit()
    if not os.path.exists("Test.mat"):
        print("Train Score Matrix NOT FOUND.");exit()

    trainScore = sio.loadmat("Train.mat")["trainScore"]
    trainLabel = sio.loadmat("Train.mat")["trainLabel"][0]
    problem = svm_problem(trainLabel,trainScore.tolist())
    param = svm_parameter('')

    trained = svm_train(problem,param)
    
    testScore = sio.loadmat("Test.mat")["testScore"]
    testLabel = sio.loadmat("Test.mat")["testLabel"][0]
    label,acc,score = svm_predict(testLabel,testScore.tolist(),trained)
    
    score = np.sum(score,axis=1)
    DrawROC(score,testLabel)
    
    print("Done.")