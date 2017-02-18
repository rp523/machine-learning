import numpy as np
import scipy.io as sio
import gui
import mathtool as mt

class layer1:
    def __init__(self,a,rho):
        self.a = a
        self.rho = rho
    def forward(self,x):
        if x.shape[1] != a.shape[0]: print("layer1 forward error!")

        self.x = x
        sample = self.x.shape[0]
        detector = self.x.shape[1]
        poly = self.a.shape[1]

        u = np.zeros(x.shape,float);
        for i in range(sample):
            for d in range(detector):
#                u[i][d] = mt.CalcPoly(a, x[i][d]])
                u[i][d] = mt.CalcPoly(a[d], x[i][d])
        return u

    def backward(self,grad_u):
        sample = self.x.shape[0]
        detector = self.x.shape[1]
        poly = self.a.shape[1]
        grad_a = np.zeros(self.a.shape,float)
        for i in range(sample):
            for d in range(detector):
                for k in range(poly):
                    grad_a[d][k] += (self.x[i][d] ** k) * grad_u[i][d]
        self.a -= self.rho * grad_a

class layer2:
    def __init__(self):
        pass
    def forward(self,u):
        self.v = np.tanh(u)
        return self.v
    def backward(self,grad_v):
        grad_u = np.zeros(grad_v.shape,float)
        grad_u = (1 - (self.v * self.v)) * grad_v
        return grad_u
            
class layer3:
    def __init__(self,b,rho):
        self.b = b
        self.rho = rho
        
    def forward(self,v):
        self.v = v
        return np.dot(v,self.b)

    def backward(self,grad_z):
        sample = grad_z.shape[0]
        detector = self.b.shape[0]
        grad_v = np.zeros((sample,detector),float)
        
        for i in range(sample):
            for d in range(detector):
                grad_v[i][d] = b[d] * grad_z[i]

        grad_b = np.zeros(self.b.shape,float)
        for d in range(detector):
            grad_b[d] = self.v[i][d] * grad_z[i]

        self.b -= self.rho * grad_b
        return grad_v

class layer4:
    def __init__(self,y):
        self.y = y
    def forward(self,z):
        self.z = z
        return np.sum(0.5 * ((z - self.y) ** 2))
    def backward(self):
        gradz =  (self.z - self.y)
        return gradz
    
class Propagate:
    def __init__(self,l1,l2,l3,l4,x):
        self.x = x
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
    def forward(self):
        u = self.l1.forward(self.x)
        v = self.l2.forward(u)
        z = self.l3.forward(v)
        w = self.l4.forward(z)
        return w
    def backward(self):
        grad_z = self.l4.backward()
        grad_v = self.l3.backward(grad_z)
        grad_u = self.l2.backward(grad_v)
        self.l1.backward(grad_u)
    def calcScore(self):
        u = self.l1.forward(self.x)
        v = self.l2.forward(u)
        z = self.l3.forward(v)
        return z
    
if "__main__" == __name__:
    x = sio.loadmat("Train.mat")["trainScore"]
    y = sio.loadmat("Train.mat")["trainLabel"][0]
    sample = x.shape[0]
    detector = x.shape[1]
    adim = 4
    rho = 0.00005

    a = np.array([[0.1]*adim]*detector)
    b = np.array([0.1]*detector)
    l1 = layer1(a,rho)
    l2 = layer2()
    l3 = layer3(b,rho)
    l4 = layer4(y)
    
    train = Propagate(l1,l2,l3,l4,x)
    
    for epoch in range(200000000):
        print("epoch=",epoch+1,"lost-func val:",train.forward())
        train.backward()
        if 0==(epoch+1)%100:
            np.savez(("nn"+"{0:05d}".format(epoch)+".npz"),a=a,b=b,w=w)
    
    x_ = sio.loadmat("Test.mat")["testScore"]
    y_ = sio.loadmat("Test.mat")["testLabel"][0]
    l4_ = layer4(y_)
    test = Propagate(l1,l2,l3,l4_,x_)
    gui.DrawROC(test.calcScore(),y_)
    
    print("Done.")