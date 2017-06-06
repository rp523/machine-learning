import numpy as np


class DecisionTree:
    
    def __init__(self, maxDepth, scoreMat, labelVec):
        assert(isinstance(maxDepth, int))
        assert(isinstance(scoreMat, np.ndarray))
        assert(scoreMat.ndim == 2)
        assert(isinstance(labelVec, np.ndarray))
        assert(labelVec.ndim == 1)
        
        # scoreMatは(特徴次元 x サンプル数)の形であることが前提
        
        self.nodes = []
        for f in range(scoreMat.shape[0]):
            node = self.Node(scoreVec = np.sort(scoreMat[0]),   # スコア昇順
                             labelVec = labelVec[np.argsort(scoreMat[0])],  # スコアの順番変更とひも付け
                             parentDepth = 0,
                             maxDepth = maxDepth)
            self.nodes.append(node)
    
    def predict(self, featureIdxs, scores):
        assert(isinstance(featureIdxs, np.ndarray))
        assert(isinstance(scores, np.ndarray))
        assert(featureIdxs.ndim == 1)
        assert(scores.ndim == 2)
        
        out = np.empty((featureIdxs.size, scores.shape[1]))
        for f in range(featureIdxs.size):
            idx = featureIdxs[f]
            for sample in range(scores.shape[1]):
                score = scores[f][sample]
                out[f][sample] = self.nodes[idx].predict(score)
        return out

    class Node:
        def __init__(self, scoreVec, labelVec, parentDepth, maxDepth):
            assert(isinstance(scoreVec, np.ndarray))
            assert(scoreVec.ndim == 1)
            assert(isinstance(labelVec, np.ndarray))
            assert(labelVec.ndim == 1)

            self.__scoreVec = scoreVec
            self.__labelVec = labelVec
            self.__depth = parentDepth + 1
            self.__maxDepth = maxDepth
            self.__childL = None
            self.__childR = None
            self.__majority = None
            
            if self.__depth != maxDepth:
                self.thresh = self.CalcBestThresh(scoreVec, labelVec)
                self.__childL = DecisionTree.Node(scoreVec = scoreVec[scoreVec <= self.thresh],
                                        labelVec = labelVec[scoreVec <= self.thresh],
                                        parentDepth = self.__depth,
                                        maxDepth = self.__maxDepth)
                self.__childR = DecisionTree.Node(scoreVec = scoreVec[scoreVec > self.thresh],
                                        labelVec = labelVec[scoreVec > self.thresh],
                                        parentDepth = self.__depth,
                                        maxDepth = self.__maxDepth)
            else:
                # 自分がターミナルノード
                labels = np.unique(labelVec)
                labelNum = labels.size
                dataNum = labelVec.size
                labelCount = np.empty(labelNum)
                for l in range(labels.size):
                    labelCount[l] = np.sum(labelVec == labels[l])
                self.__majority = labels[np.argmax(labelCount)]

        def predict(self, score):
            assert(np.array(score).size == 1)
            if self.__depth == self.__maxDepth:
                return self.__majority
            elif self.thresh >= score:
                return self.__childL.predict(score)
            else:
                return self.__childR.predict(score)

        def CalcBestThresh(self, scoreVec, labelVec):
                
            # 既にスコアはソート済であること、ラベルの順番はスコアと紐付いていることが前提
            datNum = scoreVec.size
            labels = np.unique(labelVec)    # 付与されたラベル。(-1,1)かもしれないし(0,1,2)かもしれない
            labelNum = labels.size
    
            divUp = 0
            inpurityMin = -1.0  # 最初は負値(ありえない値)を代入しておく
    
            for d in range(1, datNum):
                rateL = np.empty(labelNum)
                rateR = np.empty(labelNum)
                for l in range(labelNum):
                    rateL[l] = np.sum(self.__labelVec[:d] == labels[l]) / d
                    rateR[l] = np.sum(self.__labelVec[d:] == labels[l]) / (datNum - d)
    
                # 分類クラスごとのジニ係数をサンプル数を重みとして平均
                # ジニ係数を使う
                inpurity = ((1.0 - np.sum(rateL * rateL)) * d + (1.0 - np.sum(rateR * rateR)) * (datNum - d)) / datNum
                if (0.0 > inpurityMin) or (inpurityMin > inpurity):
                    inpurityMin = inpurity
                    divUp = d
            return np.average(scoreVec[d-1:d+1])
                
if "__main__" == __name__:
    
    ftrNum = 100
    datNum = 2
    s = np.random.randint(0, 100, (ftrNum, datNum))
    l = np.random.randint(0, 100, datNum)
    a = DecisionTree(maxDepth = 2,
                     scoreMat = s,
                     labelVec = l)
    ans = a.predict(featureIdxs = np.random.choice(np.arange(ftrNum),5,replace=False),
                    scores = np.random.randint(0, 100, 5))
    print(ans)
    print("Done.")
