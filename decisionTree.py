#coding: utf-8
import numpy as np
from common.origLib import *


def distriEntropy(distri):
    assert(isinstance(distri, np.ndarray))
    prob = distri / np.sum(distri)  # Normalize
    nonZeroPart = prob[prob != 0.0]
    return - (np.sum(nonZeroPart * np.log(nonZeroPart))) / np.log(distri.size)

class DecisionTree:
    
    def __init__(self, maxDepth, scoreMat, labelVec, regDataDist = None):
        assert(isinstance(maxDepth, int))
        assert(isinstance(scoreMat, np.ndarray))
        assert(scoreMat.ndim == 2)
        assert(isinstance(labelVec, np.ndarray))
        assert(labelVec.ndim == 1)
        assert(scoreMat.shape[1] == labelVec.size)
        
        # scoreMatは(特徴次元 x サンプル数)の形であることが前提
        
        sampleNum = scoreMat.shape[0]
        self.nodes = []
        for f in range(sampleNum):
            scoreVec = scoreMat[f]
            score_sort_index = np.argsort(scoreVec)
            self.nodes.append(\
                   self.Node(scoreVec = scoreVec[score_sort_index],   # スコア昇順
                             labelVec = labelVec[score_sort_index],  # スコアの順番変更とひも付け
                             maxDepth = maxDepth,
                             regDataDist = regDataDist))
    
    def predict(self, featureIdxs, scores):
        assert(isinstance(featureIdxs, np.ndarray))
        assert(isinstance(scores, np.ndarray))
        assert(featureIdxs.ndim == 1)
        assert(scores.ndim == 2)
        assert(featureIdxs.size <= scores.shape[0])
        sampleNum = scores.shape[1]
        useFeatureNum = featureIdxs.size
        
        out = np.empty((useFeatureNum, sampleNum))
        for f in range(useFeatureNum):
            useFeatureIdx = featureIdxs[f]
            for sample in range(sampleNum):
                score = scores[useFeatureIdx][sample]
                out[f][sample] = self.nodes[useFeatureIdx].predict(score)
        return out

    def getThresh(self, featureIndex):
        assert(np.array(featureIndex).size == 1)
        return self.nodes[featureIndex].getThresh()

    def getAssigned(self, featureIndex):
        assert(np.array(featureIndex).size == 1)
        return self.nodes[featureIndex].getAssigned()

    class Node:
        def __init__(self,
                     scoreVec,
                     labelVec,
                     maxDepth,
                     parentDepth = None,
                     threshMin = None,
                     threshMax = None,
                     regDataDist = None):
            
            assert(isinstance(scoreVec, np.ndarray))
            assert(scoreVec.ndim == 1)
            assert(isinstance(labelVec, np.ndarray))
            assert(labelVec.ndim == 1)
            assert(scoreVec.size == labelVec.size)
            assert((scoreVec == np.sort(scoreVec)).all())
            assert(labelVec.size > 0)

            if parentDepth != None:
                self.__myDepth = parentDepth + 1  # 親ノードより１層深い
            else:
                self.__myDepth = 0  # 特に引数での指定がなければ、自身が一番上
                
            self.__maxDepth = maxDepth  # 最大深さは親から子へそのまま伝承する
            self.__childL = None
            self.__childR = None
            self.__majority = None
            self.__thresh = None
            self.__isTerminal = False
            self.__average = None
            
            uniqueLabels = np.unique(labelVec);
            uniqueLabelNum = uniqueLabels.size
            
            # unique label number must be larger than ZERO
            assert(uniqueLabelNum > 0)

            if uniqueLabelNum == 1:
                # 割り振られたサンプルが全て同一のクラスなので、
                # 自分がターミナルノードになる(完全に分割が完了)
                self.__isTerminal = True

            elif uniqueLabelNum > 1:

                if (np.max(scoreVec) == np.min(scoreVec)):
                    self.__isTerminal = True
                elif self.__myDepth == maxDepth:
                    self.__isTerminal = True
                else:
                    thresh = self.__calcBestThresh(scoreVec, labelVec, regDataDist)
                    if threshMin != None:
                        if thresh <= threshMin:
                            self.__isTerminal = True
                    elif threshMax != None:
                        if thresh >= threshMax:
                            self.__isTerminal = True
                    elif not (scoreVec <= thresh).any():
                        self.__isTerminal = True
                    elif not (scoreVec >  thresh).any():
                        self.__isTerminal = True
                    
            if self.__isTerminal:

                # 自分がターミナルノードになる(まだ分割が済んでいない)
                # 多数決で代表ラベルを決める
                labelCounter = np.zeros(uniqueLabelNum)
                for l in range(uniqueLabelNum):
                    labelCounter[l] = np.sum(labelVec == uniqueLabels[l])
                self.__majority = uniqueLabels[np.argmax(labelCounter)]
                self.__average = np.average(labelVec)

            else:
            
                # さらに深い子ノードを作れる
                self.__thresh = thresh
                assert(isinstance(self.__thresh, float))
                
                self.__childL = DecisionTree.Node(scoreVec = scoreVec[scoreVec <= self.__thresh],
                                        labelVec = labelVec[scoreVec <= self.__thresh],
                                        parentDepth = self.__myDepth,
                                        maxDepth = self.__maxDepth,
                                        threshMin = threshMin,
                                        threshMax = thresh,
                                        regDataDist = regDataDist)
                self.__childR = DecisionTree.Node(scoreVec = scoreVec[scoreVec > self.__thresh],
                                        labelVec = labelVec[scoreVec > self.__thresh],
                                        parentDepth = self.__myDepth,
                                        maxDepth = self.__maxDepth,
                                        threshMin = thresh,
                                        threshMax = threshMax,
                                        regDataDist = regDataDist)
                    
        def predict(self, score):
            assert(np.array(score).size == 1)
            if self.__isTerminal:
                # ターミナルノード
                return self.__majority
            elif self.__thresh >= score:
                return self.__childL.predict(score)
            else:
                return self.__childR.predict(score)
        
        def getThresh(self):

            out = []
            if self.__childL != None:
                threshL = self.__childL.getThresh()
                if len(threshL) > 0:
                    out = out + threshL

            if self.__thresh != None:
                out = out + [self.__thresh]

            if self.__childR != None:
                threshR = self.__childR.getThresh()
                if len(threshR) > 0:
                    out = out + threshR
            
            return out

        def __calcBestThresh(self, scoreVec, labelVec, regDataDist = None):
            assert(isinstance(scoreVec, np.ndarray))
            assert(isinstance(labelVec, np.ndarray))
            assert(scoreVec.ndim == 1)
            assert(scoreVec.ndim == 1)
            assert(scoreVec.size == labelVec.size)
            assert(np.min(scoreVec) != np.max(scoreVec))
            assert((scoreVec == np.sort(scoreVec)).all())   # check if sorted
            
            self.__regDataDist = 0.0
            if regDataDist:
                self.__regDataDist = regDataDist
                
            # 既にスコアはソート済であること、ラベルの順番はスコアと紐付いていることが前提
            datNum = scoreVec.size
            uniqueLabels = np.unique(labelVec)    # 付与されたラベル。(-1,1)かもしれないし(0,1,2)かもしれない
            uniqueLabelNum = uniqueLabels.size
            assert(uniqueLabelNum > 1)
    
            filterL = np.tri(datNum)[:-1].astype(np.bool)   # 左グループへの分類[全パターン]
            filterR = np.bitwise_not(filterL)               # 右グループへの分類[全パターン]
            
            labelMat = np.empty((datNum - 1, datNum))       # 全パターンのトライ用にラベルVectorをコピー
            labelMat[np.arange(datNum - 1)] = labelVec
            
            assignMatL = (labelMat * filterL).astype(np.int)    # 実際にラベルを左グループに分類[全パターン]
            assignMatR = (labelMat * filterR).astype(np.int)    # 実際にラベルをグループに分類[全パターン]
            
            assignLNum = np.arange(1, datNum)   # 左グループへの分類数[全パターン]
            assignRNum = datNum - assignLNum    # 右グループへの分類数[全パターン]
            
            # 左右グループへのラベル分配率
            assignRateL = np.empty((uniqueLabelNum, datNum - 1))
            assignRateR = np.empty((uniqueLabelNum, datNum - 1))
            for l in range(uniqueLabelNum):
                assignRateL[l] = np.sum(assignMatL == uniqueLabels[l], axis = 1) / assignLNum
                assignRateR[l] = np.sum(assignMatR == uniqueLabels[l], axis = 1) / assignRNum
            
            # ジニ不純度を計算
            giniInpurity = \
            (1 - np.sum(assignRateL ** 2, axis = 0)) * assignLNum +\
            (1 - np.sum(assignRateR ** 2, axis = 0)) * assignRNum
            giniInpurity /= datNum
            
            # ジニ純度を計算
            giniPurity = 1.0 - giniInpurity
            
            # scoreが同じサンプルの間で分けてはいけない
            # the same as right
            sameIdxR = scoreVec[:datNum - 1] == scoreVec[1:]
            giniPurity[sameIdxR] = -1
            
            giniDivIdx = (giniPurity.argmax())
            out = 0.5 * (scoreVec[giniDivIdx] + scoreVec[giniDivIdx + 1])
            assert(np.array(out).size == 1)
            assert(np.min(scoreVec) < out)
            assert(np.max(scoreVec) > out)
            return out

if "__main__" == __name__:

    N = 1000
    data = np.arange(N) / N
    data =  np.sort(data)[::-1]
    label = np.random.choice(np.array([-1] * (N // 2) + [1] * (N // 2)), N, replace = False)
    a = DecisionTree(maxDepth = 2,
                     scoreMat = np.sort(data).reshape(1, -1),
                     labelVec = label[np.argsort(data)],
                     sampleIndexes = np.arange(N))
#    test = np.random.uniform(0,1,N).reshape(1, -1)
#    ans = a.predict(score = test).astype(np.int)
#    print(data)
#    print(label)
#    print(test)
#    print(ans)
    print(len(a.getThresh(0)))
    print(len(a.getAssigned(0)))
#    print((a.getThresh()))
#    print((a.getAssigned()))
    print("Done.")
