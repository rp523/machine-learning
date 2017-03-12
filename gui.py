import numpy as np
import matplotlib.pyplot as plt

def DrawROC(finalScore,label):

    x = np.empty(0,float)
    y = np.empty(0,float)
    
    accuracy = 0.0
    for i in range(finalScore.size):
        
        # バイアスがfinalScore[i]だったときのROCカーブ上の点を算出
        bias = finalScore[i]
        
        truePos  = np.sum(np.logical_and(( 1 == label), (finalScore[i] < finalScore)))
        falseNeg = np.sum(np.logical_and(( 1 == label), (finalScore[i] > finalScore)))
        falsePos = np.sum(np.logical_and((-1 == label), (finalScore[i] < finalScore)))
        trueNeg  = np.sum(np.logical_and((-1 == label), (finalScore[i] > finalScore)))

        accuracy = max(accuracy,
                       (truePos + trueNeg) / \
                       (truePos + trueNeg + falsePos + falseNeg))
        
        if 0.0 < truePos + falseNeg:
            falseNeg = falseNeg / (truePos + falseNeg)
            truePos  = truePos  / (truePos + falseNeg)
        if 0.0 < trueNeg + falsePos:
            falsePos = falsePos / (trueNeg + falsePos)
            trueNeg  =  trueNeg / (trueNeg + falsePos)
        
        x = np.append(x, falsePos)        
        y = np.append(y,  truePos)
    
    plt.plot(x,y,'.' )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title("ROC Curve: accuracy=" + str(accuracy))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    return

def DrawDET(finalScore,label):

    x = np.empty(0,float)
    y = np.empty(0,float)
    
    accuracy = 0.0
    for i in range(finalScore.size):
        
        # バイアスがfinalScore[i]だったときのROCカーブ上の点を算出
        bias = finalScore[i]
        
        truePos  = np.sum(np.logical_and(( 1 == label), (finalScore[i] < finalScore)))
        falseNeg = np.sum(np.logical_and(( 1 == label), (finalScore[i] > finalScore)))
        falsePos = np.sum(np.logical_and((-1 == label), (finalScore[i] < finalScore)))
        trueNeg  = np.sum(np.logical_and((-1 == label), (finalScore[i] > finalScore)))

        accuracy = max(accuracy,
                       (truePos + trueNeg) / \
                       (truePos + trueNeg + falsePos + falseNeg))
        
        if 0.0 < truePos + falseNeg:
            falseNeg = falseNeg / (truePos + falseNeg)
            truePos  = truePos  / (truePos + falseNeg)
        if 0.0 < trueNeg + falsePos:
            falsePos = falsePos / (trueNeg + falsePos)
            trueNeg  =  trueNeg / (trueNeg + falsePos)
        
        x = np.append(x, falsePos)        
        y = np.append(y, falseNeg)
    
    plt.plot(np.log(x),np.log(y),'.' )
    plt.xlim(-7.0, 0.0)
    plt.ylim(-7.0, 0.0)
    plt.title("DET Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("false Negative Rate")
    plt.show()
    return

