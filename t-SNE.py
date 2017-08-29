#coding: utf-8
import numpy as np
from input import *
from matplotlib import pyplot as plt
#from preproc import *
from feature.hog import *
from PIL import Image

class CtSNE:

    def __init__(self, mapDim, perplexity, loop, saveInterval = 0, labels = None):
        self.__mapDim = mapDim
        self.__loop = loop
        self.__eta = 100
        self.__alpha = self.__selfAlpha
        self.__perp = perplexity
        self.__saveInterval = saveInterval  # 0より大のとき、そのループ間隔で経過画像を保存する
        self.__labels = labels
    
    def __selfAlpha(self, l):
        if l < 250:
            return 0.5
        else:
            return 0.8

    def __call__(self, x):
        #assert(isinstance(x, np.ndarray))
        #assert(x.ndim == 2)
        sampleNum, featureLen = x.shape
        
        # 正規分布で初期化
        y = np.random.normal(0, 1e-4, (sampleNum, self.__mapDim))
        
        sqDistX = self.__CalcHighDimSquareDistance(x)
        sqDistY = self.__CalcLowDimSquareDistance(y)
        sqSigma = self.__Perp2SqSigma(sqDistX, self.__perp)
        
        p = self.__CalcHighDimProb(sqDistX, sqSigma)
        p = (p + p.T) / (2 * sampleNum) # impose symmetry
        # 以降、元の高次元確率分布pは一定
        
        before1 = np.zeros(y.shape)
        
        y1 = y.copy()
        for l in range(self.__loop):
            q = self.__CalcLowDimProb(self.__CalcLowDimSquareDistance(y))

            # (i,j)th component = |y_i - y_j|^2
            crossDiffY = np.empty((sampleNum, sampleNum, self.__mapDim))
            crossDiffY[np.arange(sampleNum),:,:] = y    # (N x N x dim)
            crossDiffY = crossDiffY.transpose(1, 0, 2) - crossDiffY
            assert(crossDiffY.shape == (sampleNum, sampleNum, self.__mapDim))
            crossDiffY = crossDiffY.transpose(2, 0, 1)  # (dim x N x N)
            assert(crossDiffY.shape == (self.__mapDim, sampleNum, sampleNum))
            
            gradYMat = (p - q) * crossDiffY / (1.0 + np.sum(crossDiffY ** 2, axis = 0)) # (dim x N x N)
            gradY = 4 * np.sum(gradYMat, axis = 1).T    # (N x dim)
            assert(gradY.shape == (sampleNum, self.__mapDim))
            
            yOld = y.copy()
            
            y = y + self.__eta * gradY + self.__alpha(l) * (y - before1)
            before1 = yOld
            
            offDiagIdx = np.bitwise_not(np.eye(q.shape[0]).astype(np.bool))
            kl_div = np.sum(p[offDiagIdx] * np.log(p[offDiagIdx] / q[offDiagIdx]))
            if (l % 10 == 9) or (l == self.__loop - 1):
                assert(p.shape == q.shape)
                print("loop=", l + 1, "/", self.__loop, "kl-div=", kl_div)
        
            if self.__saveInterval > 0:
                if (l % self.__saveInterval == 0) or (l == self.__loop - 1):
                    self.__saveStatus(y = y,
                                      index = l,
                                      kl_div = kl_div,
                                      labels = self.__labels)
            
        return y
    
    def __Perp2SqSigma(self, sqDist, perpTgt):
        
        assert(isinstance(sqDist, np.ndarray))
        assert(sqDist.ndim == 2)
        assert(sqDist.shape[0] == sqDist.shape[1])
        sampleNum = sqDist.shape[0]
        
        def perpTry(sqDist, sqSigma):
            sampleNum = sqDist.shape[0]
            p = self.__CalcHighDimProb(sqDist, sqSigma)
            assert(not np.isnan(p).any())
            assert(p.shape == (sampleNum, sampleNum))
            
            #debug
            p[np.eye(sampleNum).astype(np.bool)] = 1
            H = np.sum( - p * np.log2(p), axis = 1)
            assert(not np.isnan(H).any())
            out = 2 ** H
            return out

        # initialize smaller end of search window
        sqSigmaMin = np.ones(sampleNum)
        cnt = 0
        while((perpTry(sqDist, sqSigmaMin) >= perpTgt).any()):
            sqSigmaMin *= 0.5
            print("binary search min-setting:",cnt + 1);cnt += 1
        assert((perpTry(sqDist, sqSigmaMin) < perpTgt).all())

        # initialize larger end of search window
        sqSigmaMax = np.ones(sampleNum)
        cnt = 0
        while((perpTry(sqDist, sqSigmaMax) <= perpTgt).any()):
            sqSigmaMax *= 2.0
            print("binary search max-setting:",cnt + 1);cnt += 1
        assert((perpTry(sqDist, sqSigmaMax) > perpTgt).all())

        # initialize center of search window
        sqSigmaCenter = 0.5 * (sqSigmaMin + sqSigmaMax)

        # allocate memory for output        
        outSqSigma = np.ones(sampleNum)
        
        # search-finish flags
        OK = np.zeros(sampleNum).astype(np.bool)
        # search-finish indexes (to be reduced)
        idx = np.arange(sampleNum)

        breakWhile = False
        for l in range(100000000):
            assert((perpTry(sqDist, sqSigmaMax) > perpTgt).all())
            assert((perpTry(sqDist, sqSigmaMax)[np.bitwise_not(OK)] >= perpTgt).all())
            
            perpTryVal = perpTry(sqDist, sqSigmaCenter) # full length
            
            isTooLarge = (perpTryVal > perpTgt)
            if isTooLarge.any():
                assert(sqSigmaMax[isTooLarge].size == np.sum(isTooLarge))
                sqSigmaMax[isTooLarge] = sqSigmaCenter[isTooLarge]
            
            isTooSmall = (perpTryVal < perpTgt)
            if isTooSmall.any():
                assert(sqSigmaMin[isTooSmall].size == np.sum(isTooSmall))
                sqSigmaMin[isTooSmall] = sqSigmaCenter[isTooSmall]

            sqSigmaCenter = 0.5 * (sqSigmaMin + sqSigmaMax)
            assert((sqSigmaMin < sqSigmaCenter).all())
            assert((sqSigmaMax > sqSigmaCenter).all())
            assert((0.0 < sqSigmaMin).all())
            assert((0.0 < sqSigmaMax).all())
            assert((0.0 < sqSigmaCenter).all())
            assert((0.0 < outSqSigma).all())
            
            if not OK.all():
                # まだ終わってない
                
                # 残りの中から今回初めてOK判定が出たものを探す
                newOK = np.abs(perpTryVal[np.bitwise_not(OK)] - perpTgt) < 0.01
                if newOK.any():
                    OK[idx[newOK]] = True
                    outSqSigma[idx[newOK]] = sqSigmaCenter[idx[newOK]]
                    
                    if not newOK.all():
                        idx = idx[np.bitwise_not(newOK)]
                    else:
                        idx = None
            if OK.all():
                breakWhile = True
            
            print("sigma binary search loop: ",l," finished: ",np.sum(OK), "/", OK.size)
            if True == breakWhile:
                break
            
        return outSqSigma
    
    # calculate squared feature distance in high-dimensional (original) space
    def __CalcHighDimSquareDistance(self, x):
        assert(isinstance(x, np.ndarray))
        assert(x.ndim == 2)
        sampleNum, featureLen = x.shape
        
        # 特徴量ペアのユークリッド距離の２乗行列を作る。
        # for文なしだとO(n^3)のメモリ確保が必要となりメモリが足りなくなってしまうため、
        # ここはfor文でやる
        diffX2 = np.empty((sampleNum, sampleNum)).astype(np.float)
        for i in range(sampleNum):
            #print("Making feature-distance matrix: ", i + 1, "/", sampleNum)
            diffX2[i] = np.sum((x - x[i]) ** 2, axis = 1)
            logInt = 10
            if (i % logInt == logInt - 1) or (i == sampleNum - 1):
                print("calculate feature pair distance^2: ({}/{})".format(i + 1, sampleNum))
        return diffX2

    # calculate squared feature distance in low-dimensional (mapped) space
    def __CalcLowDimSquareDistance(self, y):
        assert(isinstance(y, np.ndarray))
        assert(y.ndim == 2)
        sampleNum, featureLen = y.shape
        
        diff2 = np.empty((sampleNum, sampleNum, featureLen)).astype(np.float)
        diff2[np.arange(sampleNum),:,:] = y
        out = np.sum((diff2.transpose(1, 0, 2) - diff2) ** 2, axis = 2)
        assert(out.shape == (sampleNum, sampleNum))
        return out
    
    # calculate conditional probability distribution in high-dimensional (original) space
    def __CalcHighDimProb(self, sqDist, sqSigma):
        assert(isinstance(sqDist, np.ndarray))
        assert(sqDist.ndim == 2)
        assert(sqDist.shape[0] == sqDist.shape[1])
        assert(isinstance(sqSigma, np.ndarray))
        assert(sqSigma.ndim == 1)
        assert((sqSigma > 0.0).all())
        assert((sqDist == sqDist.T).all())
        
        sampleNum = sqDist.shape[0]
        diagIdx = np.eye(sampleNum).astype(np.bool)
        offDiagIdx = np.bitwise_not(diagIdx)

        power = (-1.0) * (sqDist / (2 * sqSigma.reshape(sampleNum, 1)))

        # exp演算のアンダーフロー対策してないけど、やったほうがいいかも
        neighbor = np.exp(power)
        # 対角成分をゼロとする
        neighbor[diagIdx] = 0.0
        assert((neighbor[offDiagIdx] > 0.0).all())
        assert(neighbor.shape == (sampleNum, sampleNum))

        bunbo = np.sum(neighbor, axis = 1).reshape(sampleNum, 1)
        assert(bunbo.shape == (sampleNum, 1))
        assert((bunbo > 0.0).all())

        prob = neighbor / bunbo
        assert(prob.shape == (sampleNum, sampleNum))
        assert((0.0 < prob[offDiagIdx]).all())        

        return prob

    # calculate conditional probability distribution in low-dimensional (mapped) space
    def __CalcLowDimProb(self, sqDist):
        assert(isinstance(sqDist, np.ndarray))
        assert(sqDist.ndim == 2)
        assert(sqDist.shape[0] == sqDist.shape[1])
        assert((sqDist[np.bitwise_not(np.eye(sqDist.shape[0]).astype(np.bool))] > 0.0).all())
        assert(not np.isnan(sqDist).any())
        
        sampleNum = sqDist.shape[0]
        diagIdx = np.eye(sampleNum).astype(np.bool)
        offDiagIdx = np.bitwise_not(diagIdx)

        # t-分布
        neighbor = 1.0 / (1.0 + sqDist)

        # 対角成分をゼロとする
        neighbor[diagIdx] = 0.0
        assert(neighbor.shape == (sampleNum, sampleNum))
        assert((neighbor[offDiagIdx] > 0.0).all())
        
        bunbo = np.sum(neighbor)  # i!=jを全て足しあげる
        assert(np.array(bunbo).size == 1)
        assert(bunbo > 0.0)
        
        prob = neighbor / bunbo
        assert((prob[offDiagIdx] > 0.0).all())
        assert((prob <= 1.0).all())        
        
        return prob
    
    def __saveStatus(self, y, index, kl_div = None, dst = "imgs", labels = None):
        assert(isinstance(y, np.ndarray))
        assert(y.ndim == 2)
        assert(y.shape[1] == 2)
        assert(isinstance(i, int))
        assert(isinstance(dst, str))
        
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        title = "loop = " + str(index)
        if not kl_div is None:
            title = title + ", KL-div = " + str(kl_div)
        
        plt.clf()
        if labels is None:
            plt.title(title)
            plt.plot(y.T[0], y.T[1], ".")
        else:
            for label in np.unique(labels):
                pY = y[labels == label]
                plt.plot(pY.T[0], pY.T[1], ".", label = label)
        plt.legend()
        plt.savefig(os.path.join(dst, "image{0:05d}.png".format(index)))
        

if "__main__" == __name__:
    
    '''
    # prepare toy datasets
    dat = np.empty(0)
    label = np.empty(0)
    print("Reading Start.")
    imgs = dirPath2NumpyArray("dataset/INRIAPerson/LearnPos") / 255
    imgs  = imgs[:]
    print("Finished reading.", imgs.shape)
    hogParam = CHogParam()
    hogParam["Bin"] = 8
    hogParam["Cell"]["X"] = 5
    hogParam["Cell"]["Y"] = 10
    hogParam["Block"]["X"] = 2
    hogParam["Block"]["Y"] = 2
    hog = CHog(hogParam)
    grayImgs = RGB2Gray(imgs, "green")
    x = hog.calc(grayImgs)
    print("finished calc hog.", x.shape)
    y = tSNE(x)

    # draw results
    plt.figure()
    plt.plot(y.T[0], y.T[1], ".")
    plt.legend()
    plt.show()

    '''
    # prepare toy datasets
    dat = np.empty(0)
    labels = np.empty(0)
    print("Reading Start.")
    for i in range(10):
        print("Reading: ", str(i))
        eval = dirPath2NumpyArray("dataset/mnist/eval/" + str(i)) / 255
        #index = np.arange(eval.shape[0])
        index = np.random.choice(np.arange(eval.shape[0]), 200, replace=False)
        eval = eval[index]
        if 0 == dat.size:
            dat = np.empty((0, eval.size // eval.shape[0]))
        dat = np.append(dat, eval.reshape(eval.shape[0], -1), axis = 0)
        labels = np.append(labels, np.array([i] * eval.shape[0])).astype(np.int)
    print("Finished reading.")
    #x = dat.reshape(dat.shape[0], -1)
    # prepare class for t-SNE
    tSNE = CtSNE(mapDim = 2,
                 perplexity = 50,
                 loop = 1000,
                 saveInterval = 10,
                 labels=labels)
    y = tSNE(dat)

    # draw results
    plt.figure()
    for i in range(10):
        plotY = y[label == i]
        plt.plot(plotY.T[0], plotY.T[1], ".", label = str(i))
    plt.legend()
    plt.show()
    
    print("Done.")
