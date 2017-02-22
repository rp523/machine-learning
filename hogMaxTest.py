import numpy as np
import random
import mathtool as mt


class layer1:
    def __init__(self,w,h,rho):
        self.w = w
        self.h = h
        self.rho = rho
        self.rawfield = np.random.uniform(0.0,1.0,(h+2)*(w+2))
        self.rawfield = self.rawfield.reshape((h+2,w+2))
    def forward(self):
        out = np.zeros((self.h,self.w),float)
        for i in range(self.rawfield.shape[0]-2):
            for j in range(self.rawfield.shape[1]-2):
                out[i  ,j  ] = 2 * self.rawfield[i  ,j  ] ** 2 \
                             + 2 * self.rawfield[i+2,j  ] ** 2 \
                             + 2 * self.rawfield[i  ,j+2] ** 2 \
                             + 2 * self.rawfield[i+2,j+2] ** 2 \
                             +     self.rawfield[i+1,j  ] ** 2 \
                             +     self.rawfield[i  ,j+1] ** 2 \
                             +     self.rawfield[i+1,j+2] ** 2 \
                             +     self.rawfield[i+2,j+1] ** 2 \
                             - 2 * self.rawfield[i  ,j  ] * self.rawfield[i+2,j  ] \
                             - 2 * self.rawfield[i  ,j+1] * self.rawfield[i+2,j+1] \
                             - 2 * self.rawfield[i  ,j+2] * self.rawfield[i+2,j+2] \
                             - 2 * self.rawfield[i  ,j  ] * self.rawfield[i  ,j+2] \
                             - 2 * self.rawfield[i+1,j  ] * self.rawfield[i+1,j+2] \
                             - 2 * self.rawfield[i+2,j  ] * self.rawfield[i+2,j+2]
        return out
    def backward(self,grad):
        assert isinstance(grad,np.ndarray)
        assert grad.ndim == 2
        
        out = np.zeros((self.h+2,self.w+2),float)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                out[i-1,j-1] += 4 * self.rawfield[i-1,j-1] - 2 * (self.rawfield[i+1,j-1] + self.rawfield[i-1,j+1]) * grad[i,j]
                out[i+1,j-1] += 4 * self.rawfield[i+1,j-1] - 2 * (self.rawfield[i+1,j+1] + self.rawfield[i-1,j-1]) * grad[i,j]
                out[i-1,j+1] += 4 * self.rawfield[i-1,j+1] - 2 * (self.rawfield[i-1,j-1] + self.rawfield[i+1,j+1]) * grad[i,j]
                out[i+1,j+1] += 4 * self.rawfield[i+1,j+1] - 2 * (self.rawfield[i-1,j+1] + self.rawfield[i+1,j-1]) * grad[i,j]
                out[i+1,j  ] += 2 * self.rawfield[i+1,j  ] - 2 * (self.rawfield[i-1,j  ]                         ) * grad[i,j]
                out[i-1,j  ] += 2 * self.rawfield[i-1,j  ] - 2 * (self.rawfield[i+1,j  ]                         ) * grad[i,j]
                out[i  ,j+1] += 2 * self.rawfield[i  ,j+1] - 2 * (self.rawfield[i  ,j-1]                         ) * grad[i,j]
                out[i  ,j-1] += 2 * self.rawfield[i  ,j-1] - 2 * (self.rawfield[i  ,j+1]                         ) * grad[i,j]
        self.rawfield += out * self.rho

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if self.rawfield[i,j] > 1.0:
                    self.rawfield[i,j] = 1.0
                elif self.rawfield[i,j] < 0.0:
                    self.rawfield[i,j] = 0.0
        return

    def output(self):
        return self.rawfield

class layer2:
    def __init__(self):
        pass
    def forward(self,x):
        assert isinstance(x,np.ndarray)
        assert x.ndim == 2

        self.h = x.shape[0]
        self.w = x.shape[1]
        self.x = x
        self.y = np.sqrt(self.x)
        return np.sum(self.x)
    def backward(self):
        out = np.ones((self.h, self.w),float)
        return out

if "__main__" == __name__:
    l1 = layer1(1000,2000,0.000001)
    l2 = layer2()
    
    old = l1.output()
    for l in range(1000000):
        w = l2.forward(l1.forward())
        l1.backward(l2.backward())
        
        
        print(w)
        #print(l1.output())
        if l % 100 == 0:
            np.savez("hogMax.npz",l1.output())
    print(l1.output())
        
    