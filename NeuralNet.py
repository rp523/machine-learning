import os
import numpy as np
import fileIO as fio
import scipy.io as sio
import mathtool as mt

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
    def __init__(self,input,output):
        print("to be overridden.")
    def forward(self,x):
        print("to be overridden.")
    def backward(self,dzdy):
        print("to be overridden.")

class CFullConnectlayer(CLayer):
    def __init__(self,input,output,regu):
        self.w = CAdamParam(input = input, output = output)
        self.regu = regu
    
    def forward(self,x):
        
        self.x = x  #確認用
        self.y = np.dot(x,self.w.Get())
        
        return self.y

    def backward(self,dzdy):

        '''
        dzdw = np.zeros(self.w.shape,float)
        for s in range(dzdy.shape[0]):
            for yidx in range(self.w.shape[1]):
                for xidx in range(self.w.shape[0]):
                    dzdw[xidx,yidx] += self.x[s,xidx] * 
        '''
        dzdw = np.dot(np.transpose(self.x), dzdy)
        w_buf = self.w.Get()
        reguGrad = self.regu * w_buf * (1.0 - w_buf * w_buf)
        self.w.update(dzdw + reguGrad)
        
        # １つ前のlayerへgradientを伝搬させる
        '''
        dzdx = np.zeros(self.x.shape,float)
        for s in range(dzdy.shape[0]):
            for yidx in range(self.w.shape[1]):
                for xidx in range(self.w.shape[0]):
                    dzdx[s,xidx] += dzdy[s,yidx] * self.w[xidx,yidx]
        '''
        dzdx = np.dot(dzdy, np.transpose(self.w.Get()))
        assert(dzdx.shape == self.x.shape)
        return dzdx

    def forwardBinary(self,x):
        return SignBinarize(np.dot(x,SignBinarize(self.w.Get())))
    
    def OutputParam(self):
        return self.w.Get()

class CSTE(CLayer):
    def __init__(self):
        pass
    def forward(self,x):
        return SignBinarize(x)
    def backward(self,dzdy):
        return (dzdy < 1) * (dzdy > -1) * dzdy
    def forwardBinary(self,x):
        return self.forward(x)

class CReLU(CLayer):
    def __init__(self):
        pass
    def forward(self,x):
        self.valid = (x > 0.0)
        return x * self.valid
    def forwardBinary(self,x):
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
        pre = self.predict(x)
        return self.output.forward(pre,y)
    def predict(self,x):
        act = x
        for l in range(len(self.layers)):
            act = self.layers[l].forward(act)
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
    
    trainScore = sio.loadmat("Train.mat")["trainScore"]
    trainLabel = sio.loadmat("Train.mat")["trainLabel"][0]
    sample = trainScore.shape[0]
    detector = trainScore.shape[1]
    testScore = sio.loadmat("Test.mat")["testScore"]
    testLabel = sio.loadmat("Test.mat")["testLabel"][0]
    
    batchSize = 64
    regu = 0.0#1e-2
    layers = CLayerController()
    layers.append(CFullConnectlayer(detector,128,regu))
    layers.append(CSTE())
    layers.append(CFullConnectlayer(128,32,regu))
    layers.append(CSTE())
    layers.append(CFullConnectlayer(32,1,regu))
    layers.setOut(CSquare())
    
    for epoch in range(100000000000):

        batchID = np.random.choice(range(sample),batchSize,replace = False)
        loss = layers.forward(trainScore[batchID], trainLabel[batchID])
        layers.backward()
        if epoch % 100 == 0:
            print(epoch,loss,end=",")
            print(mt.CalcAccuracy(layers.predict(testScore),testLabel),end=",")
            print(mt.CalcROCArea(layers.predict(testScore),testLabel),end=",")
            print(mt.CalcAccuracy(layers.predictBinary(testScore,testLabel),testLabel),end=",")
            print(mt.CalcROCArea(layers.predictBinary(testScore,testLabel),testLabel),end=",")
            print()
    print("Done.")
