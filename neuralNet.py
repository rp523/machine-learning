#coding: utf-8
import os
import numpy as np
import common.fileIO as fio
import scipy.io as sio
import common.mathtool as mt
import common.imgtool as imt
from input import *


def SignBinarize(x):
    out = 1*(x >= 0.0) - 1*(x < 0.0)
    assert(out.shape == x.shape)
    return out

class CAdamParam:
    
    def __init__(self, input, size, eta = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8, zeroInit=None):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        
        self.m = np.zeros(size,float)
        self.v = np.zeros(size,float)
        self.beta1Pow = 1.0
        self.beta2Pow = 1.0

        if zeroInit:
            self.param = np.zeros(size)
        else:
            self.param = np.random.randn(size) / np.sqrt(input)
        
    def update(self, grad):
        assert(isinstance(grad,np.ndarray))
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta1) * grad * grad
        self.beta1Pow *= self.beta1
        self.beta2Pow *= self.beta2
        mhat = self.m / (1.0 - self.beta1Pow)
        vhat = self.v / (1.0 - self.beta2Pow)
        self.param = self.param - self.eta * mhat / (np.sqrt(vhat) + self.epsilon)
        
        # Noise Injection
        #self.param = self.param + np.random.randn(self.input,self.output) / self.input
    
    def Get(self):
        return self.param

class CLayer:
    def __init__(self,input,output,validRate):
        print("to be overridden.")
    def forward(self,x):
        print("to be overridden.")
    def backward(self,dzdy):
        print("to be overridden.")

class CAffineLayer(CLayer):
    def __init__(self,outShape,cost=0.0):
        self.initOK = False
        self.inShape = None
        self.inSize = None
        self.outShape = outShape
        self.outSize = mt.MultipleAll(outShape)
        self.cost = cost
    def __initInpl(self,x):
        self.inShape = x[0].shape
        self.inSize = x[0].size
        #パラメタは１次元ベクトルの形で確保
        self.w = CAdamParam(size = self.inSize*self.outSize, input=self.inSize)  
        self.b = CAdamParam(size = self.outSize, input=self.inSize)
    
    def forward(self,x):
        if False == self.initOK:
            self.__initInpl(x)
            self.initOK = True
        
        batch = x.shape[0]
        self.xMat = x.reshape(batch, -1)  #確認用
        self.yMat = np.dot(self.xMat,self.w.Get().reshape(self.inSize,self.outSize)) + self.b.Get().reshape((1,-1))

        outBatShape = tuple([batch] + list(self.outShape))
        self.y = self.yMat.reshape(outBatShape)
        return self.y

    def predict(self,x):
        batch = x.shape[0]
        xMat = x.reshape(batch,self.inSize)
        y = np.dot(xMat,self.w.Get().reshape(self.inSize,self.outSize)) + self.b.Get().reshape((1,-1))
        outBatShape = tuple([batch] + list(self.outShape))
        y = y.reshape(outBatShape)
        return y
        
    def backward(self,dzdy):
   
        batch = dzdy.shape[0]     
        inBatShape = ([batch] + list(self.inShape))
        dzdyMat = dzdy.reshape(batch, -1)

        # 自レイヤー内のパラメタを更新
        dzdwVec = np.dot(self.xMat.T, dzdyMat).reshape(self.inSize*self.outSize)
        self.w.update(dzdwVec + self.cost * self.w.Get())
        db = np.sum(dzdyMat,axis=0).flatten()
        self.b.update(db)   # バイアスにはWeightDecayを使わない
        
        # １つ前のlayerへgradientを伝搬させる
        dzdxMat = np.dot(dzdyMat, np.transpose(self.w.Get().reshape(self.inSize,self.outSize)))
        return dzdxMat.reshape(inBatShape)

class CSimpleSumLayer(CLayer):
    def __init__(self):
        self.initOK = None
        self.inShape = None
        self.inSize = None
    def __initInpl(self,x):
        self.inShape = x[0].shape
        self.inSize = x[0].size
    
    def forward(self,x):
        if not self.initOK:
            self.__initInpl(x)
            self.initOK = True
        
        batch = x.shape[0]
        xMat = x.reshape(batch, -1)  #確認用
        return np.sum(xMat,axis=1).reshape((-1,1))

    def predict(self,x):
        return self.forward(x)
        
    def backward(self,dzdy):
   
        batch = dzdy.shape[0]
        dx = np.dot(dzdy.reshape(batch,-1), np.ones((1,self.inSize),float))
        assert(dx.shape == (batch,self.inSize))
        return dx

class CHistoLayer(CLayer):
    def __init__(self,bin,cost=0.0):
        self.initOK = False
        self.inShape = None
        self.inSize = None
        self.bin = bin
        self.cost = cost
    def __initInpl(self,x):
        self.inShape = x[0].shape
        self.inSize = x[0].size
        #パラメタは１次元ベクトルの形で確保
        self.w = CAdamParam(size = self.inSize*self.bin, input=self.inSize, zeroInit=True)  
    
    def forward(self,x):
        if False == self.initOK:
            self.__initInpl(x)
            self.initOK = True
        
        batch = x.shape[0]
        xbin = np.array(x * self.bin).astype(np.int).reshape(batch, self.inSize)
        xbin = xbin * (xbin < self.bin) + (xbin >= self.bin) * (self.bin - 1)

        self.onehot = np.zeros((batch,self.inSize,self.bin),int)
        for ba in range(batch):
            self.onehot[ba,np.arange(self.inSize),xbin[ba]] = 1
        
        w = self.w.Get().reshape(self.inSize,self.bin)
        wbatch = np.zeros((batch,self.inSize,self.bin),float)
        wbatch[:] = w
        y = np.sum(wbatch * self.onehot, axis = 2)
        assert(y.shape == (batch,self.inSize))
        return y

    def predict(self,x):
        return self.forward(x)
        
    def backward(self,dzdy):
        onehotT = self.onehot.transpose(2,0,1)
        onehotT = onehotT * dzdy
        dw = np.sum(onehotT.transpose(1,2,0),axis=0).flatten()
        self.w.update(dw + self.cost * self.w.Get())
        
        return dzdy
        
class CConvolutionLayer(CLayer):
    def __init__(self,filterShape,stride,pad=0,cost=0.0):
        self.initOK = False
        self.pad = pad
        assert(3 == np.array(filterShape).size)
        self.outC = filterShape[0]  # Output Channel (Not necessarily equals to input channel.)
        self.fh = filterShape[1]    # Filter Height
        self.fw = filterShape[2]    # Filter Width
        self.stride = stride
        self.cost = cost
        
    def __initInpl(self,x):
        assert(4 == np.array(x).ndim)
        batch,self.inC,self.inH,self.inW = x.shape
        assert(0 == (self.inH + 2 * self.pad - self.fh) % self.stride)
        assert(0 == (self.inW + 2 * self.pad - self.fw) % self.stride)
        self.outH = 1 + (self.inH + 2 * self.pad - self.fh) // self.stride
        self.outW = 1 + (self.inW + 2 * self.pad - self.fw) // self.stride
        
        assert(4 == np.array(x).ndim)
        batch, self.inC, self.inH, self.inW = x.shape
        inputNeuron = self.inC * self.inH * self.inW
        
        self.w = CAdamParam(input=inputNeuron, size=self.outC*self.inC*self.fh*self.fw)
        self.b = CAdamParam(input=inputNeuron, size=self.outC)
       
    def forward(self, x):
        if False == self.initOK:
            self.__initInpl(x)
            self.initOK = True

        batch = x.shape[0]
        col = mt.im2col(input_data=x,
                        filter_h=self.fh,
                        filter_w=self.fw,
                        stride=self.stride,
                        pad=self.pad)
        col_w = self.w.Get().reshape(self.outC, -1).T

        out = np.dot(col, col_w) + self.b.Get().reshape((1,-1))
        out = out.reshape(batch, self.outH, self.outW, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_w = col_w

        return out
    
    def predict(self,x):
        #
        return self.forward(x)

    def backward(self, dout):

        dout = dout.transpose(0,2,3,1).reshape(-1, self.outC)

        dw = np.dot(self.col.T, dout)
        dw = dw.transpose(1, 0).reshape(self.outC, self.inC, self.fh, self.fw)
        self.w.update(dw.flatten() + self.cost * self.w.Get())
        db = np.sum(dout, axis=0).flatten()
        self.b.update(db)   # バイアスにはWeightDecayを採用しない
        
        dcol = np.dot(dout, self.col_w.T)
        dx = mt.col2im(dcol, self.x.shape, self.fh, self.fw, self.stride, self.pad)

        return dx

class CPoolingLayer(CLayer):
    def __init__(self, shape, stride=1, pad=0):
        assert(2 == np.array(shape).size)
        self.pool_h = shape[0]
        self.pool_w = shape[1]
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):

        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = mt.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def predict(self,x):
        return self.forward(x)

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = mt.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


class CBatchNormLayer(CLayer):
    def __init__(self, gamma = 1.0, beta = 0.0, moment=0.9):
        self.gamma = gamma
        self.beta = beta
        self.moment = moment
        self.initOK = None
    def __initInpl(self,x):
        self.inShape = x[0].shape
        self.inSize = x[0].size
        #パラメタは１次元ベクトルの形で確保
        self.m = np.zeros(self.inShape)
        self.v = np.zeros(self.inShape)
        
    def forward(self, x):
        if not self.initOK:
            self.__initInpl(x)
            self.initOK = True

        self.input_shape = x.shape
        out = self.__forward(x, True)
        return out.reshape(*self.input_shape)
    def predict(self, x):
        self.input_shape = x.shape
        out = self.__forward(x, False)
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):

        if train_flg:
            m = x.mean(axis=0).reshape(self.inShape)
            self.xc = x - m  # centered
            v = np.mean(self.xc**2, axis=0)
            self.std = np.sqrt(v + 10e-7)
            self.xn = self.xc / self.std   # normalized
            
            self.m = self.moment * self.m + (1-self.moment) * m
            self.v = self.moment * self.v + (1-self.moment) * v
        else:
            xc = x - self.m
            self.xn = xc / ((np.sqrt(self.v + 10e-7)))
            
        out = self.gamma * self.xn + self.beta 
        return out

    def backward(self, dout):
        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        batch = dout.shape[0]
        dbeta = dout.sum(axis=0).reshape(self.inShape)
        dgamma = np.sum(self.xn * dout, axis=0).reshape(self.inShape)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / batch) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / batch
        return dx



class CDropOutLayer(CLayer):
    def __init__(self,validRate):
        assert(validRate <= 1.0)
        assert(validRate >= 0.0)
        self.validRate = validRate
    def forward(self,x):
        batch = x.shape[0]
        # DropOutのためのフィルタ行列を生成
        self.validVec = np.random.rand(x[0].size) < self.validRate
        xMat = x.reshape((batch,x[0].size))
        yMat = np.zeros(xMat.shape,float)
        for i in range(batch):
            yMat[i] = xMat[i] * self.validVec
        y = yMat.reshape(x.shape)
        return y
    def predict(self,x):
        return x * self.validRate
    def backward(self,dzdy):
        batch = dzdy.shape[0]
        dzdyMat = dzdy.reshape((dzdy.shape[0],dzdy[0].size))
        dzdxMat = np.zeros(dzdyMat.shape,float)
        for i in range(batch):
            dzdxMat[i] = dzdyMat[i] * self.validVec
        dzdx = dzdxMat.reshape(dzdy.shape)
        return dzdx

class CReLU(CLayer):
    def __init__(self):
        pass
    def forward(self,x):
        self.valid = (x > 0.0)
        return x * self.valid
    def predict(self,x):
        return x * (x > 0.0)
    def backward(self,dzdy):
        assert(dzdy.shape == self.valid.shape)
        return dzdy * self.valid

class CLogistic(CLayer):
    def __init__(self):
        pass
    def forward(self,x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y
    def predict(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    def backward(self,dzdy):
        assert(dzdy.shape == self.y.shape)
        ret = self.y * (1.0 - self.y) * dzdy
        return ret
    
class CTanh(CLayer):
    def __init__(self):    
        pass
    def forward(self,x):
        self.y = self.predict(x)
        return self.y
    def predict(self,x):
        return np.tanh(x)
    def backward(self,dzdy):
        assert(dzdy.shape == self.y.shape)
        ret = (1.0 - self.y * self.y) * dzdy
        return ret
    
class CSquare(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.label = label
        self.x = x
        self.xv = x.flatten()
        assert(self.label.shape == self.xv.shape)

        return 0.5 * np.average(((self.xv - self.label)**2))

    def backward(self):

        # １つ前のlayerへgradientを伝搬させる
        dydx = (self.xv * self.label - 1.0) * self.label
        dydx /= self.xv.size
        dydx = np.transpose([dydx])

        assert(dydx.shape == self.x.shape)
        return dydx

class CHinge(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.label = label
        self.x = x
        self.xv = x.flatten()
        assert(self.label.shape == self.xv.shape)

        return 0.5 * np.average(((self.xv - self.label)**2))

    def backward(self):

        # １つ前のlayerへgradientを伝搬させる
        dydx = (1.0 > self.xv * self.label) * (- self.label)
        dydx = dydx / self.xv.size
        dydx = np.transpose([dydx])

        assert(dydx.shape == self.x.shape)
        return dydx

class CSquareHingeLoss(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.x = x
        self.label = label
        self.xv = self.x.flatten()
        assert(self.label.shape == self.xv.shape)

        return 0.5 * np.average(\
            (self.label ==  1) * (self.xv <   1) * ((self.xv - 1)**2) +\
            (self.label == -1) * (self.xv >  -1) * ((self.xv + 1)**2))
        
    def backward(self):

        dydx = (self.label > 0) * (self.xv <  1) * (self.xv - 1) +\
               (self.label < 0) * (self.xv > -1) * (self.xv + 1)
        dydx /= self.xv.size
        dydx = np.transpose([dydx])
        assert(dydx.shape == self.x.shape)
        return dydx

class CSparseHuberLoss(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.x = x  #確認用
        self.label = label
        xv = self.x.flatten()
        assert(self.label.shape == xv.shape)

        self.margin = self.label * xv
        return np.average(\
            (self.margin >= -1) * (self.margin < 1) * ((self.margin - 1)**2) +\
            (self.margin <  -1) * (- 4.0 * self.margin))
        
    def backward(self):

        dydx = (self.margin >= -1) * (self.margin < 1) * (2 * (self.margin - 1) * self.label) +\
            (self.margin <  -1) * (- 4.0 * self.label)
        dydx /= self.margin.size
        dydx = np.transpose([dydx])
        assert(dydx.shape == self.x.shape)
        return dydx

class CExponentialLoss(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.x = x
        self.label = label
        self.xv = x.flatten()
        
        self.expArray = np.exp(- self.xv * self.label) 
        return np.sum(self.expArray)
    
    def backward(self):

        # １つ前のlayerへgradientを伝搬させる
        return (- self.label * self.expArray).reshape(-1,1)

class CExponentialSparseLoss(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.x = x
        self.label = label
        self.xv = x.flatten()
        
        self.expArray = np.exp(- self.xv * self.label) 
        return np.sum(self.expArray)
    
    def backward(self):

        # １つ前のlayerへgradientを伝搬させる
        return (1 > self.label * self.xv) * (- self.label * self.expArray)

class CLayerController:
    def __init__(self):
        self.layers = []
    def append(self,layer):
        assert(isinstance(layer,CLayer))
        self.layers.append(layer)
    def setOut(self,layer):
        assert(isinstance(layer,CLayer))
        self.output = layer
    def forward(self,x,y):
        act = x
        for l in range(len(self.layers)):
            act = self.layers[l].forward(act)
        return self.output.forward(act,y)
    def predict(self,x):
        act = x
        for l in range(len(self.layers)):
            act = self.layers[l].predict(act)
        return act
    def predictBinary(self,x,y):
        act = x
        for l in range(len(self.layers)):
            act = self.layers[l].forwardBinary(act)
        return act
    def backward(self):
        grad = self.output.backward()
        for l in range(len(self.layers)-1, -1, -1):
            grad = self.layers[l].backward(grad)

if "__main__" == __name__:
    
    lp = dirPath2NumpyArray("dataset/INRIAPerson/LearnPos")
    ln = dirPath2NumpyArray("dataset/INRIAPerson/LearnNeg")
    learnScore = np.append(lp,ln,axis=0)
    
    ep = dirPath2NumpyArray("dataset/INRIAPerson/EvalPos")
    en = dirPath2NumpyArray("dataset/INRIAPerson/EvalNeg")
    evalScore = np.append(ep,en,axis=0)
    
    learnLabel = np.array([1] * len(lp) + [-1] * len(ln))
    evalLabel  = np.array([1] * len(ep) + [-1] * len(en))

    shrinkX = 4
    shrinkY = 4
    n,h,w,c = learnScore.shape
    learnScore = learnScore.reshape(n,h//shrinkY,shrinkY,w//shrinkX,shrinkX,c).transpose(0,1,3,2,4,5).mean(axis=4).mean(axis=3)
    n,h,w,c = evalScore.shape
    evalScore = evalScore.reshape(n,h//shrinkY,shrinkY,w//shrinkX,shrinkX,c).transpose(0,1,3,2,4,5).mean(axis=4).mean(axis=3)
    
#    imt.ndarray2PILimg(trainScore[1]).resize((200,400)).show();exit()
    
    n,h,w,c = learnScore.shape
    learnScore = learnScore.reshape(n,c,h,w)
    n,h,w,c = evalScore.shape
    evalScore = evalScore.reshape(n,c,h,w)
    
    learnScore /= 255
    evalScore /= 255
    '''

    trainScore = np.asarray(sio.loadmat("Train.mat")["trainScore"])
    trainLabel = np.asarray(sio.loadmat("Train.mat")["trainLabel"]).flatten()
    testScore = np.asarray(sio.loadmat("Test.mat")["testScore"])
    testLabel = np.asarray(sio.loadmat("Test.mat")["testLabel"]).flatten()
    '''
    
    
    sample = learnScore.shape[0]
    batchSize = 64

    assert(sample >= batchSize)
    layers = CLayerController()

    layers.append(CConvolutionLayer(filterShape=(6,5,5),stride=1,cost=5e-4))
    layers.append(CPoolingLayer(shape=(2,2),stride=2))
    layers.append(CBatchNormLayer())
    layers.append(CDropOutLayer(validRate=0.5))
    layers.append(CReLU())
    
    layers.append(CConvolutionLayer(filterShape=(16,5,5),stride=1,cost=5e-4))
    layers.append(CPoolingLayer(shape=(2,2),stride=2))
    layers.append(CBatchNormLayer())
    layers.append(CDropOutLayer(validRate=0.5))
    layers.append(CReLU())
    
    layers.append(CAffineLayer(outShape=(120,),cost=5e-4))
    layers.append(CBatchNormLayer())
    layers.append(CDropOutLayer(validRate=0.5))
    layers.append(CReLU())
    
    layers.append(CAffineLayer(outShape=(84,),cost=5e-4))
    layers.append(CBatchNormLayer())
    layers.append(CDropOutLayer(validRate=0.5))
    layers.append(CReLU())
    
    layers.append(CAffineLayer(outShape=(1,),cost=5e-4))
    layers.setOut(CExponentialLoss())
    
    for epoch in range(100000000000):

        batchID = np.random.choice(range(sample),batchSize,replace = True)
        loss = layers.forward(learnScore[batchID], learnLabel[batchID])
        layers.backward()
        if epoch % 100 == 0:
            print('%09d'%epoch,'%5f'%loss,end=",")
            print('%3.10f' % (100*mt.CalcAccuracy(layers.predict(evalScore),evalLabel)),end=",")
            print('%3.10f' % (100*mt.CalcROCArea(layers.predict(evalScore),evalLabel)) ,end=",")
            print()
    print("Done.")
