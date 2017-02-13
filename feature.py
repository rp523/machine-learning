import numpy as np



class CFeature:
    def __init__(self):
        self.jointAND = False
        self.jointOR = False
        self.jointXOR = False

    def GetFeatureLength(self):
        print("GetFeatureLength should be overridden!")
        exit()
    def calc(self,img):        
        print("calc should be overridden!")
        exit()

    
    def GetTotalFeatureLength(self):
        n = self.GetFeatureLength()
        if False != self.jointAND:
            n += int((n * (n - 1)) / 2)
        if False != self.jointOR:
            n += int((n * (n - 1)) / 2)
        if False != self.jointXOR:
            n += int((n * (n - 1)) / 2)
        return n

    def totalCalc(self,img):

        self.feature = self.calc(self,img)
        
        if False != self.jointAND:
            n = self.GetFeatureLength()
            nC2 = int((n * (n - 1)) / 2)
            featureAnd = np.zeros(nC2)
            for i in range(n):
                for j in range(n):
                    featureAnd[(i * j) + n] = min(self.feature[i], self.feature[j])
            self.feature = np.array(list(self.feature) + list(featureAnd))

        if False != self.jointOR:
            n = self.GetFeatureLength()
            nC2 = int((n * (n - 1)) / 2)
            featureOR = np.zeros(nC2)
            for i in range(n):
                for j in range(n):
                    featureOR[(i * j) + n] = max(self.feature[i], self.feature[j])
            self.feature = np.array(list(self.feature) + list(featureOR))

        if False != self.jointXOR:
            n = self.GetFeatureLength()
            nC2 = int((n * (n - 1)) / 2)
            featureXOR = np.zeros(nC2)
            for i in range(n):
                for j in range(n):
                    featureXOR[(i * j) + n] = \
                        max(self.feature[i], self.feature[j]) - \
                        min(self.feature[i], self.feature[j])   
            self.feature = np.array(list(self.feature) + list(featureXOR))
                
    