import numpy as np
from common.origLib import *


def distriEntropy(distri):
    assert(isinstance(distri, np.ndarray))
    prob = distri / np.sum(distri)
    nonZeroPart = prob[prob != 0.0]
    return - (np.sum(nonZeroPart * np.log(nonZeroPart))) / np.log(distri.size)

class DecisionTree:
    
    def __init__(self, maxDepth, scoreMat, labelVec, sampleIndexes = None):
        assert(isinstance(maxDepth, int))
        assert(isinstance(scoreMat, np.ndarray))
        assert(scoreMat.ndim == 2)
        assert(isinstance(labelVec, np.ndarray))
        assert(labelVec.ndim == 1)
        assert(scoreMat.shape[1] == labelVec.shape[0])
        
        if sampleIndexes != None:
            indexes = sampleIndexes
        else:
            indexes = np.arange(len(labelVec))
        # scoreMatは(特徴次元 x サンプル数)の形であることが前提
        
        self.nodes = []
        for f in range(scoreMat.shape[0]):
            score_sort_index = np.argsort(scoreMat[f])
            self.nodes.append(\
                   self.Node(scoreVec = scoreMat[f][score_sort_index],   # スコア昇順
                             labelVec = labelVec[score_sort_index],  # スコアの順番変更とひも付け
                             maxDepth = maxDepth,
                             sampleIndexes = indexes))
    
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
                     sampleIndexes = None,
                     parentDepth = None,
                     threshMin = None,
                     threshMax = None):
            assert(isinstance(scoreVec, np.ndarray))
            assert(scoreVec.ndim == 1)
            assert(isinstance(labelVec, np.ndarray))
            assert(labelVec.ndim == 1)
            assert(scoreVec.size == labelVec.size == len(list(sampleIndexes)))
            assert((scoreVec == np.sort(scoreVec)).all())
            Assert(labelVec.size > 0)

            if parentDepth != None:
                self.__myDepth = parentDepth + 1  # 親ノードより１層深い
            else:
                self.__myDepth = 0  # 自身が一番上
            self.__maxDepth = maxDepth  # 最大深さは親から子へそのまま伝承する
            self.__childL = None
            self.__childR = None
            self.__majority = None
            self.__thresh = None
            self.__assignedIndexes = sampleIndexes
            self.__isTerminal = None
            
            labels = np.unique(labelVec);
            labelNum = labels.size

            if labelNum == 0:
                Assert(False)
                pass
                # サンプルが割り振られていない
            elif labelNum == 1:
                # 自分がターミナルノードになる(完全に分割が完了)
                self.__majority = labels[0]
            elif labelNum > 1:

                thresh = self.__calcBestThresh(scoreVec, labelVec)
                valid = True
                if threshMin != None:
                    if thresh <= threshMin:
                        valid = False
                if threshMax != None:
                    if thresh >= threshMax:
                        valid = False
                if not (scoreVec <= thresh).any():
                    valid = False
                if not (scoreVec >  thresh).any():
                    valid = False
                        
                if (np.max(scoreVec) == np.min(scoreVec))\
                  or not valid\
                  or (self.__myDepth == maxDepth):

                    # 自分がターミナルノードになる(まだ分割が済んでいない)
                    # 多数決で代表ラベルを決める
                    labelCount = np.zeros(labelNum)
                    for l in range(labelNum):
                        labelCount[l] = np.sum(labelVec == labels[l])
                    self.__majority = labels[np.argmax(labelCount)]

                elif labelNum >= 2:
                
                    # さらに深い子ノードを作れる
                    self.__thresh = thresh
                    assert(isinstance(self.__thresh, float))
                    
                    assignL = None
                    assignR = None
                    if sampleIndexes != None:
                        assignL = sampleIndexes[scoreVec <= self.__thresh]
                        assignR = sampleIndexes[scoreVec >  self.__thresh]
                        Assert(len(assignL) + len(assignR) == len(sampleIndexes))
                    self.__childL = DecisionTree.Node(scoreVec = scoreVec[scoreVec <= self.__thresh],
                                            labelVec = labelVec[scoreVec <= self.__thresh],
                                            sampleIndexes = assignL,
                                            parentDepth = self.__myDepth,
                                            maxDepth = self.__maxDepth,
                                            threshMin = threshMin,
                                            threshMax = thresh)
                    self.__childR = DecisionTree.Node(scoreVec = scoreVec[scoreVec > self.__thresh],
                                            labelVec = labelVec[scoreVec > self.__thresh],
                                            sampleIndexes = assignR,
                                            parentDepth = self.__myDepth,
                                            maxDepth = self.__maxDepth,
                                            threshMin = thresh,
                                            threshMax = threshMax)
                    
        def predict(self, score):
            assert(np.array(score).size == 1)
            if not self.__thresh:
                # ターミナルノードか、またはラベルが完全純粋
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

        def getAssigned(self):

            out = []
            if self.__thresh:
                left_assigned  = self.__childL.getAssigned()
                right_assigned = self.__childR.getAssigned()
                for la in left_assigned:
                    assert(isinstance(la, list))
                for ra in right_assigned:
                    assert(isinstance(ra, list))
                
                out = out + left_assigned
                out = out + right_assigned
            else:
                self_assigned = list(self.__assignedIndexes)
                if len(self_assigned) > 0:
                    out.append(self_assigned)
            assert(isinstance(out, list))
            return out

        def __calcBestThresh(self, scoreVec, labelVec):
            assert(isinstance(scoreVec, np.ndarray))
            assert(isinstance(labelVec, np.ndarray))
            assert(scoreVec.ndim == 1)
            assert(scoreVec.ndim == 1)
            assert(scoreVec.size == labelVec.size)
            assert((scoreVec == np.sort(scoreVec)).all())
                
            # 既にスコアはソート済であること、ラベルの順番はスコアと紐付いていることが前提
            datNum = scoreVec.size
            labels = np.unique(labelVec)    # 付与されたラベル。(-1,1)かもしれないし(0,1,2)かもしれない
            labelNum = labels.size
            assert(labelNum > 1)
    
            divUp = 0
            inpurityMin = 1e5  # 最初はありえない値を代入しておく(inpurityは１より小さい）
    
            for d in range(1, datNum):
                rateL = np.empty(labelNum)
                rateR = np.empty(labelNum)
                for l in range(labelNum):
                    assert(labelVec[:d].size == d)
                    assert(labelVec[d:].size == datNum - d)
                    assert(labelVec[:d].size + labelVec[d:].size == datNum)
                    rateL[l] = np.sum(labelVec[:d] == labels[l]) / d
                    rateR[l] = np.sum(labelVec[d:] == labels[l]) / (datNum - d)
    
                # 分類クラスごとのジニ係数をサンプル数を重みとして平均
                # ジニ不純度を使う
                inpurity = ((1.0 - np.sum(rateL * rateL)) * d + (1.0 - np.sum(rateR * rateR)) * (datNum - d)) / datNum
                inpurity -= distriEntropy(np.array([d, datNum - d]))    # データ分布も考慮
                #assert(1.0 >= inpurity >= 0.0)
                
                # update the mimimum of inpurity.
                if inpurityMin > inpurity:
                    inpurityMin = inpurity
                    divUp = d
            out = 0.5 * (scoreVec[divUp - 1] + scoreVec[divUp])
            assert(np.array(out).size == 1)
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
