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
        self.input = input
        self.output = output

        self.m = np.zeros((input,output),float)
        self.v = np.zeros((input,output),float)
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

class CFullConnectlayer(CLayer):
    def __init__(self,input,output,regu = 0.0):
        self.w = CAdamParam(input = input, output = output)
        self.regu = regu
        self.output = output
    
    def forward(self,x):
        self.x = x  #確認用
        # DropOutのためのフィルタ行列
        self.y = np.dot(x,self.w.Get())
        return self.y

    def predict(self,x):
        return np.dot(x,self.w.Get())

    def backward(self,dzdy):

        dzdw = np.dot(np.transpose(self.x), dzdy)
        w_buf = self.w.Get()
        
        # １つ前のlayerへgradientを伝搬させる準備
        dzdx = np.dot(dzdy, np.transpose(self.w.Get()))
        assert(dzdx.shape == self.x.shape)

        # 自レイヤー内のパラメタを更新
        reguGrad = self.regu * w_buf * (1.0 - w_buf * w_buf)
        self.w.update(dzdw + reguGrad)
        
        return dzdx

    def OutputParam(self):
        return self.w.Get()

class CMaxPoolingLayer(CLayer):
    def __init__(self,filterShape,step):
        assert(np.array(filterShape).ndim == 2)
        self.fh = np.array(filterShape).shape[0]    # filter height
        self.fw = np.array(filterShape).shape[1]    # filter width
        self.step = step
        pass
    def forward(self,x):
        assert(x.ndim >= 2)
        assert(x.ndim <= 3)
        self.x_selected = np.zeros(x.shape,float)
        if x.ndim == 2:
            inH = x.shape[0]
            inW = x.shape[1]
            ouH = int((inH - (self.fh - 1)) / self.step) + 1
            ouW = int((inW - (self.fw - 1)) / self.step) + 1
            y = np.zeros(ouH*ouW,float)
            n = 0
            for y in range(0, inH - (self.fh - 1), self.step):
                for x in range(0, inW - (self.fw - 1), self.step):
                    y[n] = np.max(x[y : y + self.fh, x : x + self.fw])
                    n += 1
            assert(n == ouH*ouW)
        if x.ndim == 3:
            inH = x.shape[0]
            inW = x.shape[1]
            inC = x.shape[2]
            ouH = int((inH - (self.fh - 1)) / self.step) + 1
            ouW = int((inW - (self.fw - 1)) / self.step) + 1
            ouC = inC
            y = np.zeros(ouH*ouW*ouC,float)
            n = 0
            for y in range(0, inH - (self.fh - 1), self.step):
                for x in range(0, inW - (self.fw - 1), self.step):
                    for c in range(inC):
                        y[n] = np.max(x[y : y + self.fh, x : x + self.fw, c])
                        n += 1
            assert(n == ouC*ouH*ouW)
        return y

class CDropOutLayer(CLayer):
    def __init__(self,validRate):
        assert(validRate <= 1.0)
        self.validRate = validRate
    def forward(self,x):
        # DropOutのためのフィルタ行列
        self.valid = np.diag(1*(np.random.rand(x.shape[1]) < self.validRate))
        return np.dot(x, self.valid)
    def predict(self,x):
        return x * self.validRate
    def backward(self,dzdy):
        return np.dot(dzdy,self.valid)

class CSTE(CLayer):
    def __init__(self):
        pass
    def forward(self,x):
        return SignBinarize(x)
    def backward(self,dzdy):
        return (dzdy < 1) * (dzdy > -1) * dzdy

class CReLU(CLayer):
    def __init__(self):
        pass
    def forward(self,x):
        self.valid = (x > 0.0)
        return x * self.valid
    def predict(self,x):
        return x * (x > 0.0)
    def backward(self,dzdy):
        return dzdy * self.valid

class CSigmoid(CLayer):
    def __init__(self):
        pass
    def forward(self,x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        assert(x.shape == self.y.shape)
        return self.y
    def predict(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    def backward(self,dzdy):
        assert(dzdy.shape == self.y.shape)
        ret = self.y * (1.0 - self.y) * dzdy
        assert(ret.shape == self.y.shape)
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
    layers.append(CFullConnectlayer(input=detector,output=128))
    layers.append(CReLU())
    layers.append(CFullConnectlayer(input=128,output=128))
    layers.append(CReLU())
    layers.append(CDropOutLayer(validRate=0.6))
    layers.append(CFullConnectlayer(input=128,output=32))
    layers.append(CReLU())
    layers.append(CDropOutLayer(validRate=0.8))
    layers.append(CFullConnectlayer(input=32,output=1))
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
