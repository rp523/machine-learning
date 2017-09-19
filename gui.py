import numpy as np
import matplotlib.pyplot as plt

def evaluateROC(finalScore,label):

    assert(np.array(finalScore).shape == np.array(label).shape)

    posSample = np.sum(  1 == np.array(label))
    negSample = np.sum( -1 == np.array(label))
    assert(posSample > 0)
    assert(negSample > 0)
    assert(posSample + negSample == label.size)

    truePos  = np.empty(0,float)
    falseNeg = np.empty(0,float)
    falsePos = np.empty(0,float)
    trueNeg  = np.empty(0,float)
    
    for i in range(finalScore.size):
        # バイアスがfinalScore[i]だったときのROCカーブ上の点を算出
        bias = finalScore[i]
        truePos  = np.append(truePos ,np.sum((( 1 == label) * (bias < finalScore))))
        falseNeg = np.append(falseNeg,np.sum((( 1 == label) * (bias > finalScore))))
        falsePos = np.append(falsePos,np.sum(((-1 == label) * (bias < finalScore))))
        trueNeg  = np.append(trueNeg ,np.sum(((-1 == label) * (bias > finalScore))))

    # accuracy = MAX( (TP+TN)/(TP+TN+FP+FN) )を計算
    accuracy = 0.0
    for i in range(finalScore.size):
        accuracy = max(accuracy,
                       (truePos[i] + trueNeg[i]) / \
                       (truePos[i] + trueNeg[i] + falsePos[i] + falseNeg[i]))
        
    # ROC下部面積を求める
    curveX = np.sort(falsePos / negSample)
    curveY = np.sort(truePos  / posSample)
    area = (curveY[0]) * (curveX[0]) * 0.5
    for i in range(curveX.size - 1):
        area += (curveY[i + 1] + curveY[i]) * (curveX[i + 1] - curveX[i]) * 0.5
        if i == curveX.size - 2:
            area += (1 + curveY[i + 1]) * (1 - curveX[i + 1]) * 0.5

    return accuracy, area

def DrawROC(finalScore,label):
    assert(np.array(finalScore).shape == np.array(label).shape)

    posSample = np.sum(  1 == np.array(label))
    negSample = np.sum( -1 == np.array(label))
    assert(posSample > 0)
    assert(negSample > 0)
    assert(posSample + negSample == label.size)

    truePos  = np.empty(0,float)
    falseNeg = np.empty(0,float)
    falsePos = np.empty(0,float)
    trueNeg  = np.empty(0,float)
    
    for i in range(finalScore.size):
        # バイアスがfinalScore[i]だったときのROCカーブ上の点を算出
        bias = finalScore[i]
        truePos  = np.append(truePos ,np.sum((( 1 == label) * (bias < finalScore))))
        falseNeg = np.append(falseNeg,np.sum((( 1 == label) * (bias > finalScore))))
        falsePos = np.append(falsePos,np.sum(((-1 == label) * (bias < finalScore))))
        trueNeg  = np.append(trueNeg ,np.sum(((-1 == label) * (bias > finalScore))))

    # accuracy = MAX( (TP+TN)/(TP+TN+FP+FN) )を計算
    accuracy = 0.0
    for i in range(finalScore.size):
        accuracy = max(accuracy,
                       (truePos[i] + trueNeg[i]) / \
                       (truePos[i] + trueNeg[i] + falsePos[i] + falseNeg[i]))
        
    # ROC下部面積を求める
    curveX = np.sort(falsePos / negSample)
    curveY = np.sort(truePos  / posSample)
    area = (curveY[0]) * (curveX[0]) * 0.5
    for i in range(curveX.size - 1):
        area += (curveY[i + 1] + curveY[i]) * (curveX[i + 1] - curveX[i]) * 0.5
        if i == curveX.size - 2:
            area += (1 + curveY[i + 1]) * (1 - curveX[i + 1]) * 0.5
    
    plt.plot(falsePos/negSample,truePos/posSample,'.' )
    plt.plot([0,1],[0,1])   # 基準線 y=x
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("ROC Curve: accuracy=" + str(format(accuracy,'.4f')) + ",area=" + str(format(area,'.4f')))
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

