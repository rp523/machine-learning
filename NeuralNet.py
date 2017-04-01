import os
import numpy as np
import fileIO as fio
import scipy.io as sio
import mathtool as mt
import imgtool as imt



def SignBinarize(x):
    out = 1*(x >= 0.0) - 1*(x < 0.0)
    assert(out.shape == x.shape)
    return out

class CAdamParam:
    
    def __init__(self, input, output, eta = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.shape = (input,output)
        
        self.m = np.zeros(self.shape,float)
        self.v = np.zeros(self.shape,float)
        self.param = np.random.randn(input,output) / np.sqrt(input)
        self.beta1Pow = 1.0
        self.beta2Pow = 1.0
        
    def update(self, grad):
        assert(isinstance(grad,np.ndarray))
        assert(grad.ndim == 2)
        assert(self.m.shape == grad.shape)
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
    def __init__(self,outShape):
        self.initOK = False
        self.inShape = None
        self.inSize = None
        self.outShape = outShape
        self.outSize = mt.MultipleAll(outShape)
    def __initInpl(self,x):
        self.inShape = x[0].shape
        self.inSize = x[0].size
        self.w = CAdamParam(input=self.inSize,output=self.outSize)  #パラメタは１次元ベクトルの形で確保
    
    def forward(self,x):
        if False == self.initOK:
            self.__initInpl(x)
            self.initOK = True
        
        self.xMat = x.reshape(x.shape[0],self.inSize)  #確認用
        # DropOutのためのフィルタ行列
        self.yMat = np.dot(self.xMat,self.w.Get())

        outBatShape = tuple([x.shape[0]] + list(self.outShape))
        self.y = self.yMat.reshape(outBatShape)
        return self.y

    def predict(self,x):
        outBatShape = tuple([x.shape[0]] + list(self.outShape))
        return np.dot(x.reshape(x.shape[0],self.inSize),self.w.Get())\
                .reshape(outBatShape)
        
    def backward(self,dzdy):
        
        inBatShape = ([dzdy.shape[0]] + list(self.inShape))
        dzdyMat = dzdy.reshape(dzdy.shape[0],self.outSize)

        # 自レイヤー内のパラメタを更新
        dzdwMat = np.dot(np.transpose(self.xMat), dzdyMat)
        self.w.update(dzdwMat)
        
        # １つ前のlayerへgradientを伝搬させる
        dzdxMat = np.dot(dzdyMat, np.transpose(self.w.Get()))
        return dzdxMat.reshape(inBatShape)

    def OutputParam(self):
        return self.w.Get()

class Convolution(CLayer):
    def __init__(self,outShape,stride,pad=0):
        self.initOK = False
        self.inShape = None
        self.inSize = None
        
        assert(3 == np.array(outShape).ndim)
        self.outShape = outShape
        self.outSize = mt.MultipleAll(outShape)
        self.stride = stride
        self.pad = pad
    def __initInpl(self,x):
        self.inShape = x[0].shape
        self.inSize = x[0].size
        self.w = CAdamParam(input=self.inSize,output=self.outSize)  #パラメタは１次元ベクトルの形で確保

        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

    
class CDropOutLayer(CLayer):
    def __init__(self,validRate):
        assert(validRate <= 1.0)
        assert(validRate >= 0.0)
        self.validRate = validRate
    def forward(self,x):
        # DropOutのためのフィルタ行列を生成
        self.validMat = np.diag(np.random.rand(x[0].size) < self.validRate)
        xMat = x.reshape((x.shape[0],x[0].size))
        yMat = np.dot(xMat, self.validMat)
        y = yMat.reshape(x.shape)
        return y
    def predict(self,x):
        return x * self.validRate
    def backward(self,dzdy):
        dzdyMat = dzdy.reshape((dzdy.shape[0],dzdy[0].size))
        dzdxMat = np.dot(dzdyMat,self.validMat)
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

class CSigmoid(CLayer):
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
    
class CSquare(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.x = x
        self.label = label
        self.xv = np.transpose(self.x)[0]
        assert(self.label.shape == self.xv.shape)

        return 0.5 * np.average(((self.xv - self.label)**2))

    def backward(self):

        # １つ前のlayerへgradientを伝搬させる
        dydx = self.xv - self.label
        dydx /= self.xv.size
        dydx = np.transpose([dydx])

        assert(dydx.shape == self.x.shape)
        return dydx

class CSquareHinge(CLayer):
    def __init__(self):
        pass
    def forward(self,x,label):  
        assert(label.size == x.shape[0])
        self.x = x
        self.label = label
        self.xv = np.transpose(self.x)[0]
        assert(self.label.shape == self.xv.shape)

        return 0.5 * np.average(\
            (self.label ==  1) * (self.xv <   1) * ((self.xv - 1)**2) +\
            (self.label == -1) * (self.xv >  -1) * ((self.xv + 1)**2))
        
    def backward(self):

        dydx = (self.label ==  1) * (self.xv <   1) * (self.xv - 1) +\
               (self.label == -1) * (self.xv >  -1) * (self.xv + 1)
        dydx /= self.xv.size
        dydx = np.transpose([dydx])
        assert(dydx.shape == self.x.shape)
        return dydx

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
    
    trainScorePos = np.load("grayINRIA.npz")["TrainPos"]
    trainScoreNeg = np.load("grayINRIA.npz")["TrainNeg"]
    trainScore = np.append(trainScorePos,trainScoreNeg,axis=0)
    trainScore = trainScore.reshape((trainScore.shape[0],trainScore.shape[1]*trainScore.shape[2]))
    
    testScorePos = np.load("grayINRIA.npz")["TestPos"]
    testScoreNeg = np.load("grayINRIA.npz")["TestNeg"]
    testScore = np.append(testScorePos,testScoreNeg,axis=0)
    testScore = testScore.reshape((testScore.shape[0],testScore.shape[1]*testScore.shape[2]))
    
    trainLabel = np.array([1] * trainScorePos.shape[0] + [-1] * trainScoreNeg.shape[0])
    testLabel  = np.array([1] *  testScorePos.shape[0] + [-1] *  testScoreNeg.shape[0])
    
    sample = trainScore.shape[0]
    detector = trainScore.shape[1]
    
    batchSize = 32
    
    trainScore /= 255
    testScore /= 255
    
    layers = CLayerController()
    layers.append(CAffineLayer(outShape=(512,)))
    layers.append(CReLU())
    layers.append(CDropOutLayer(validRate=0.6))
    layers.append(CAffineLayer(outShape=(512,)))
    layers.append(CReLU())
    layers.append(CDropOutLayer(validRate=0.7))
    layers.append(CAffineLayer(outShape=(512,)))
    layers.append(CReLU())
    layers.append(CDropOutLayer(validRate=0.8))
    layers.append(CAffineLayer(outShape=(32,)))
    layers.append(CReLU())
    layers.append(CAffineLayer(outShape=(1,)))
    layers.append(CSigmoid())
    layers.setOut(CSquare())
    
    for epoch in range(100000000000):

        batchID = np.random.choice(range(sample),batchSize,replace = False)
        loss = layers.forward(trainScore[batchID], trainLabel[batchID])
        layers.backward()
        if epoch % 100 == 0:
            print('%06d'%epoch,'%5f'%loss,end=",")
            print('%3.10f' % (100*mt.CalcAccuracy(layers.predict(testScore),testLabel)),end=",")
            print('%3.10f' % (100*mt.CalcROCArea(layers.predict(testScore),testLabel)) ,end=",")
            print()
    print("Done.")
