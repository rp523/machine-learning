import numpy as np


class DecisionTree:
    
    def __init__(self, maxDepth, scoreMat, labelVec):
        assert(isinstance(maxDepth, int))
        assert(isinstance(scoreMat, np.ndarray))
        assert(scoreMat.ndim == 2)
        assert(isinstance(labelVec, np.ndarray))
        assert(labelVec.ndim == 1)
        assert(scoreMat.shape[1] == labelVec.shape[0])
        
        # scoreMatは(特徴次元 x サンプル数)の形であることが前提
        
        self.nodes = []
        for f in range(scoreMat.shape[0]):
            self.nodes.append(\
                   self.Node(scoreVec = np.sort(scoreMat[f]),   # スコア昇順
                             labelVec = labelVec[np.argsort(scoreMat[f])],  # スコアの順番変更とひも付け
                             parentDepth = 0,
                             maxDepth = maxDepth))
    
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

    class Node:
        def __init__(self, scoreVec, labelVec, parentDepth, maxDepth):
            assert(isinstance(scoreVec, np.ndarray))
            assert(scoreVec.ndim == 1)
            assert(isinstance(labelVec, np.ndarray))
            assert(labelVec.ndim == 1)
            assert(scoreVec.size == labelVec.size)
            assert((scoreVec == np.sort(scoreVec)).all())

            self.__myDepth = parentDepth + 1  # 親ノードより１層深い
            self.__maxDepth = maxDepth  # 最大深さは親から子へそのまま伝承する
            self.__childL = None
            self.__childR = None
            self.__majority = None
            
            labels = np.unique(labelVec);
            labelNum = labels.size
            if labelNum == 1:
                # 全サンプルのラベルが同じなら、子ノードをもはや作らない
                self.__majority = labelVec[0]
            elif labelNum >= 2:
                if self.__myDepth != maxDepth:
                    self.thresh = self.CalcBestThresh(scoreVec, labelVec)
                    assert(isinstance(self.thresh, float))
                    self.__childL = DecisionTree.Node(scoreVec = scoreVec[scoreVec <= self.thresh],
                                            labelVec = labelVec[scoreVec <= self.thresh],
                                            parentDepth = self.__myDepth,
                                            maxDepth = self.__maxDepth)
                    self.__childR = DecisionTree.Node(scoreVec = scoreVec[scoreVec > self.thresh],
                                            labelVec = labelVec[scoreVec > self.thresh],
                                            parentDepth = self.__myDepth,
                                            maxDepth = self.__maxDepth)
                else:
                    # 自分がターミナルノード。
                    # 多数決で代表ラベルを決める
                    labelCount = np.zeros(labelNum)
                    for l in range(labelNum):
                        labelCount[l] = np.sum(labelVec == labels[l])
                    self.__majority = labels[np.argmax(labelCount)]
            else:
                assert(0)
                    
        def predict(self, score):
            assert(np.array(score).size == 1)
            if self.__majority:
                # ターミナルノードか、またはラベルが完全純粋
                return self.__majority
            elif self.thresh >= score:
                return self.__childL.predict(score)
            else:
                return self.__childR.predict(score)

        def CalcBestThresh(self, scoreVec, labelVec):
            assert(isinstance(scoreVec, np.ndarray))
            assert(isinstance(labelVec, np.ndarray))
            assert(scoreVec.ndim == 1)
            assert(scoreVec.ndim == 1)
            assert(scoreVec.size == labelVec.size)
                
            # 既にスコアはソート済であること、ラベルの順番はスコアと紐付いていることが前提
            datNum = scoreVec.size
            labels = np.unique(labelVec)    # 付与されたラベル。(-1,1)かもしれないし(0,1,2)かもしれない
            labelNum = labels.size
    
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
                assert(1.0 >= inpurity >= 0.0)
                
                # update the mimimum of inpurity.
                if inpurityMin > inpurity:
                    inpurityMin = inpurity
                    divUp = d
            out = 0.5 * (scoreVec[divUp - 1] + scoreVec[divUp])
            assert(np.array(out).size == 1)
            return out
                
if "__main__" == __name__:

    N = 100
    data = np.arange(N) / N
    data =  np.sort(data)[::-1]
    data = data.reshape(1, -1)
    label = np.array([-1] * (N // 2) + [1] * (N // 2))    
    a = DecisionTree(maxDepth = 2,
                     scoreMat = data,
                     labelVec = label)
    test = np.random.uniform(0,1,N).reshape(1, -1)
    ans = a.predict(featureIdxs = np.array([0]),
                    scores = test).astype(np.int)
    print(data)
    print(label)
    print(test)
    print(ans)
    print("Done.")
