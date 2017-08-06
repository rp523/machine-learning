import numpy as np
from input import *
from matplotlib import pyplot as plt

class CtSNE:

    def __init__(self, mapDim, perplexity, loop):
        self.__mapDim = mapDim
        self.__loop = loop
        self.__eta = 0.1
        self.__alpha = self.__selfAlpha
    
    def __selfAlpha(self, l):
        if l < 250:
            return 0.5
        else:
            return 0.8

    def __call__(self, x):
        assert(isinstance(x, np.ndarray))
        assert(x.ndim == 2)
        sampleNum, featureLen = x.shape
        
        y = np.random.normal(0, 1E-4, (sampleNum, self.__mapDim))
        p = self.__MakeCondProb(x)
        before1 = np.zeros(y.shape)
        
        for l in range(self.__loop):
            q = self.__MakeCondProb(y)
            cubeY = np.empty((sampleNum, sampleNum, self.__mapDim))
            cubeY[:, np.arange(sampleNum)] = y 
            crossDiffY = cubeY - cubeY.transpose(1, 0, 2)
            gradYbeforeSigma = (p - q) * crossDiffY.transpose(2, 0, 1)
            gradYbeforeSigma = gradYbeforeSigma.transpose(1, 2, 0)
            gradY = 4 * np.sum(gradYbeforeSigma, axis = 1)
            assert(gradY.shape == (sampleNum, self.__mapDim))
            
            yOld = y.copy()
            y = y + self.__eta * gradY + self.__alpha(l) * (y - before1)
            before1 = yOld
            #print("loop:", l)
        return y
    
    def __MakeCondProb(self, x):
        assert(isinstance(x, np.ndarray))
        assert(x.ndim == 2)
        sampleNum, featureLen = x.shape
        
        # calclate cross-component distance
        cube = np.empty((sampleNum, sampleNum, featureLen))
        cube[:, np.arange(sampleNum)] = x
        diffX2 = np.sum((cube - cube.transpose(1, 0, 2)) ** 2, axis = 2)

        # set diagonal component 0
        diffX2[np.eye(sampleNum).astype(np.bool)] = 0
        assert(diffX2.shape == (sampleNum, sampleNum))
        
        # calclate Student's t-distribution
        tDistX = 1.0 / (diffX2 + 1.0)
        
        # normalize
        sigma = np.sum(tDistX, axis = 1)
        bunboMat = np.empty((sampleNum, sampleNum))
        bunboMat[np.arange(sampleNum)] = sigma
        sigma = sigma.T
        out = tDistX / sigma
        
        return out

if "__main__" == __name__:
    
    # prepare class for t-SNE
    tSNE = CtSNE(mapDim = 2,
                 perplexity=50,
                 loop=10000)
    
    # prepare toy datasets
    dat = np.empty((0, 28, 28))
    label = np.empty(0)
    for i in range(10):
        eval = dirPath2NumpyArray("dataset/mnist/eval/" + str(i))
        dat = np.append(dat, eval, axis = 0)
        label = np.append(label, np.array([i] * dat.shape[0])).astype(np.int)
    index = np.random.choice(np.arange(dat.shape[0]), 100, replace=False)
    x = dat[index].reshape(-1, 28 * 28)
    l = label[index]
    y = tSNE(x)

    # draw results
    plt.figure()
    for i in range(10):
        plotY = y[l == i]
        plt.plot(plotY.T[0], plotY.T[1], ".")
    plt.legend()
    plt.show()

    print("Done.")
