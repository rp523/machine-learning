import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from common.origLib import *


class CGaussianProcess(object):
    
    def __init__(self, iterTotal, sampleNum, paramMin, paramMax, maximizedFunc):
        self.N = 0
        self.gram = np.empty(0)
        self.x = None
        self.y = None
        self.sampleNum = sampleNum
        self.maximizedFunc = maximizedFunc
        self.paramMin = paramMin
        self.paramMax = paramMax
        self.paraUnit = 128
        self.iterTotal = iterTotal
        
        assert(isinstance(paramMin,dicts))
        assert(isinstance(paramMax,dicts))

        # とりあえずminとmaxの中間を初期値とする        
        startX = dicts()
        for k, minVal in self.paramMin.items():
            assert(k in paramMax)
            maxVal = self.paramMax[k]
            if isinstance(minVal, float) and isinstance(maxVal, float):
                startX[k] = (minVal + maxVal) / 2
            elif isinstance(minVal, int) and isinstance(maxVal, int):
                startX[k] = (minVal + maxVal) // 2
            else:
                assert(0) # parameter type error.
        self.observed(startX, self.maximizedFunc(startX))
    
    def observed(self, xNewParam, yNew):

        assert(isinstance(xNewParam,dicts))
        xNew = np.empty(0)
        for v in xNewParam.values():
            xNew = np.append(xNew, v)
        
        assert(xNew.ndim == 1)
        dim = xNew.shape[0]

        if None == self.x:        
            self.x = np.empty((0, xNew.size))
        if None == self.y:        
            self.y = np.empty(0)
            
        gram = np.zeros((self.N + 1, self.N + 1))
        if 0 != self.N:
            gram[0:self.N, 0:self.N] = self.gram
            gram[self.N, np.arange(self.N)] = self.__kVector(xNew.reshape(1,dim)).reshape(self.N)
            gram[np.arange(self.N), self.N] = self.__kVector(xNew.reshape(1,dim)).reshape(self.N)
        gram[self.N, self.N] = self.__kernel(xNew.reshape(1,1,dim), xNew.reshape(1,1,dim))

        try:
            self.invGram = np.linalg.inv(gram)
        except np.linalg.linalg.LinAlgError as e:
            print("check: error")
        else:
            print("check: ok")
            self.gram = gram
            self.x = np.append(self.x, [xNew], axis=0)
            self.y = np.append(self.y, yNew)
            self.N += 1
        
        assert(self.x.shape[0] == self.N)
        assert(self.y.shape == (self.N,))
        
    def __kVector(self, xTrial):
        assert(xTrial.ndim == 2)    # トライ数 x 次元数
        base = self.x
        newSample = xTrial
        assert(base.ndim == 2)
        assert(newSample.ndim == 2)
        dim = base.shape[1]
        baseLen = base.shape[0]
        sampleLen = newSample.shape[0]
        baseMat = np.zeros((sampleLen, baseLen, dim))
        baseMat[:] = base
        
        sampleMat = np.zeros((baseLen, sampleLen, dim))
        sampleMat[:] = newSample
        sampleMat = sampleMat.transpose(1,0,2)
        
        out = self.__kernel(sampleMat, baseMat)
        assert(out.ndim == 2)
        return out

    def __kernel(self, x1, x2):
        assert(x1.ndim == 3)
        assert(x2.ndim == 3)
              
        return np.exp( - 0.5 * np.sum((x1 - x2) ** 2, axis = 2))
    
    def __mean(self, xNew):
        assert(xNew.ndim == 2)
        k = self.__kVector(xNew)
        assert(k.shape[0] == xNew.shape[0])
        f = self.y.reshape(-1 ,1)
        out = np.dot(np.dot(k, self.invGram), f).flatten()
        assert(out.ndim == 1)
        assert(out.shape[0] == xNew.shape[0])
        return out
    def __cov(self, xNew):
        
        dim = xNew.shape[xNew.ndim - 1]
        
        kVec = self.__kVector(xNew)
        assert(kVec.ndim == 2)
        sampleLen = xNew.shape[0]
        assert(kVec.shape[0] == sampleLen)
        kVecT = kVec.T
        outMat = np.dot(np.dot(kVec, self.invGram), kVecT)
        assert(outMat.ndim == 2)
        assert(outMat.shape[0] == sampleLen)
        diagIdx = np.arange(sampleLen)
        diagIdx = diagIdx * sampleLen + diagIdx
        outVec = self.__kernel(xNew.reshape(sampleLen,1,dim), xNew.reshape(sampleLen,1,dim)).flatten() - outMat.flatten()[diagIdx]
        assert(outVec.ndim == 1)
        assert(outVec.shape[0] == sampleLen)
        return outVec
    
    def EI(self, xNew):
        if 0 < self.N:
            mean = self.__mean(xNew)
            cov = self.__cov(xNew)
            yMax = np.max(self.y)
            z = (mean - yMax) / cov
            ei = (mean - yMax) * norm.cdf(z) + cov * norm.pdf(z)
            return ei
        else:
            return np.array([0.0])
    def __debug_info(self):
        return self.N, self.x.shape, self.gram.shape

    def nextParam(self):
        trialX = None
        acqTemp = None
        
        # 行列サイズは128くらいが速度的にちょうどいい
        for j in range(int(self.sampleNum) // self.paraUnit):

            scanX = np.empty((0, self.paraUnit))
            
            for k, minVal in self.paramMin.items():
                maxVal = self.paramMax[k]
                if isinstance(minVal, float):
                    assert(isinstance(maxVal, float))
                    scanX = np.append(scanX, [np.random.sample(self.paraUnit) * (maxVal - minVal) + minVal], axis = 0)
                elif isinstance(minVal, int):
                    assert(isinstance(maxVal, int))
                    add = np.random.randint(maxVal - minVal + 1, size = self.paraUnit) + minVal
                    scanX = np.append(scanX, [add], axis = 0)
                    assert(np.all(add >= minVal))
                else:
                    assert(0)
            scanX = scanX.transpose()
            assert(scanX.shape[0] == self.paraUnit)
            acq = self.EI(scanX)
            if None == trialX:
                trialX = scanX[np.argmax(acq)]
                acqTemp = np.max(acq)
            elif acqTemp < np.max(acq):
                trialX = scanX[np.argmax(acq)]
                acqTemp = np.max(acq)

        out = dicts()
        i = 0
        for k, v in self.paramMin.items():
            if isinstance(v, int):
                out[k] = int(trialX[i])
            else:
                out[k] = float(trialX[i])
            i += 1
        return out

    def Execute(self):

        globalMax = None
        globalArgMax = None
        recordX = []
        recordY = []
    
        for i in range(int(self.iterTotal)):
            
            trialX = self.nextParam()
            trialY = self.maximizedFunc(trialX)
            self.observed(trialX, trialY)
            recordX.append(trialX)
            recordY.append(trialY)
            
            if globalMax:
                if globalMax < trialY:
                    globalMax = trialY
                    globalArgMax = trialX
            else:
                globalMax = trialY
                globalArgMax = trialX
    
            print(globalArgMax, globalMax)
            print(trialX, trialY)
            print(self._CGaussianProcess__debug_info())
        
        return globalArgMax, globalMax, recordX, recordY
    

if "__main__" == __name__:
    def funcs(arg):
        return np.sin(2 * np.pi * arg["x"]) * np.sin(2 * np.pi * arg["y"])
    
    dim = 2
    paramMin = dicts()
    paramMax = dicts()
    paramMin["x"] = 0.0
    paramMax["x"] = 1.0
    paramMin["y"] = 0.0
    paramMax["y"] = 2.0
    gp = CGaussianProcess(sampleNum=int(1e5),
                          iterTotal = 30,
                          paramMin = paramMin,
                          paramMax = paramMax,
                          maximizedFunc = funcs)
    argOpt, opt, triRecordX, triRecordY = gp.Execute()
    
    dim = 2
    min = np.array([paramMin["x"],paramMin["y"]])
    max = np.array([paramMax["x"],paramMax["y"]])
    smoothBin = np.array([100, 100])
    delta = (max - min) / smoothBin
    x = dicts()
    x["x"],x["y"] = np.meshgrid(np.arange(min[0],max[0],delta[0]),np.arange(min[1],max[1],delta[1]))
    Z = funcs(x)
    Z = Z.reshape(smoothBin[0], smoothBin[1])
    plt.pcolor(x["x"],x["y"],Z,cmap="rainbow")
    
    xx = []
    yy = []
    for param in triRecordX:
        xx.append(param["x"])
        yy.append(param["y"])
    plt.scatter(xx, yy, c="pink")
    plt.scatter(argOpt["x"], argOpt["y"], c="yellow")
    #plt.colorbar()
    plt.show()

    
    
    print("Done.")