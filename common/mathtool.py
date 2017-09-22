#coding: utf-8
import sys
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import integral_image,integrate
import unittest
import scipy.linalg as linalg

def MakeIntegral(img):
    assert(isinstance(img, np.ndarray))
    assert(2 <= img.ndim <= 3)

    if img.ndim == 2:
        out = img.cumsum(0).cumsum(1)
        out = np.insert(out, 0, 0, axis = 0)
        out = np.insert(out, 0, 0, axis = 1)
        return out
    elif img.ndim == 3:
        out = img.cumsum(1).cumsum(2)
        out = np.insert(out, 0, 0, axis = 1)
        out = np.insert(out, 0, 0, axis = 2)
        return out

def SumFromIntegral(intg, y0, y1, x0, x1):
    assert(isinstance(intg, np.ndarray))
    assert(2 <= intg.ndim <= 3)

    if intg.ndim == 2:
        out = intg[y1 + 1, x1 + 1]  \
            - intg[y1 + 1, x0    ]  \
            - intg[y0    , x1 + 1]  \
            + intg[y0    , x0    ]
        return out
    elif intg.ndim == 3:
        out = intg[:, y1 + 1, x1 + 1]  \
            - intg[:, y1 + 1, x0    ]  \
            - intg[:, y0    , x1 + 1]  \
            + intg[:, y0    , x0    ]
        return out
    

# 行列内にある指定の列ベクトルを昇順ソートする。
# 他の列は指定列ベクトルと同じ順番変更を受ける。
def sortMatBasedOneCol(mat, baseColID):
    return mat[ mat[1,:].argsort() ]

# 行列内にある指定の行ベクトルを昇順ソートする。
# 他の行は指定行ベクトルと同じ順番変更を受ける。
def sortMatBasedOneRow(mat, baseRowID):
    return mat[:,mat[baseRowID,:].argsort()]

'''
行列内にある行ベクトルの重複を覗いたものを返す
ex .
入力:
[[1 2 2]
 [1 2 2]
 [3 4 5]]
出力：
[[1 2 2]
 [3 4 5]]

参考URL：https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
'''
def extractUniqueRows(mat):
    assert(isinstance(mat, np.ndarray))
    assert(mat.ndim == 2)
    b = np.ascontiguousarray(mat).view(np.dtype((np.void, mat.dtype.itemsize * mat.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return mat[idx]


def sort(inSrc,inTargetIndex):
    assert(isinstance(inSrc,np.ndarray))
    assert(inSrc.shape[1] >= inTargetIndex)
    vec = np.transpose(inSrc)
    vec = np.array(sorted(vec,key=lambda vec:vec[inTargetIndex]))
    return np.transpose(vec)


def DischargeCalculation(A,y):
    assert isinstance(A,np.ndarray)
    assert isinstance(y,np.ndarray)
    assert 2 == A.ndim
    assert 1 == y.ndim
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == y.shape[0]
    
    N = A.shape[0]
    
    for j1 in range(N):
        
        # 左辺行列の対角成分を１にする
        Aj1j1 = A[j1,j1]
        if Aj1j1 == 0.0:
            return False    # 対角成分に０がある→行列が正則でないので計算不能
        for i in range(N):
            A[j1,i] /= Aj1j1
        y[j1] /= Aj1j1
        
        # 左辺行列の非対角成分を０にする
        for j2 in range(0,N):
            if j2 != j1:
                Aj2j1 = A[j2,j1]
                for i in range(N):
                    A[j2,i] -= A[j1,i] * Aj2j1
                y[j2] -= y[j1] * Aj2j1
    return True

# 多項式Σ_i^N (C_i * x^i)を計算する
# coef: N次元多項式の係数。長さN+1のnumpy一次元配列を想定。C0~CNの次元昇順。
def CalcPoly(coef, x):
    len = coef.size
    ret = coef[len - 1]
    for i in range(len - 1, 0, -1):
        ret = (ret * x) + coef[i - 1]
    return ret

def ApproxPoly(x,y,polyDim):
    assert(isinstance(polyDim,int))
    assert(polyDim > 0)
    assert(isinstance(x,np.ndarray))
    assert(isinstance(y,np.ndarray))
    
    dim = polyDim + 1
    
    sumx = np.zeros(2 * dim + 1, float)
    xpoly = np.ones(x.size, float)
    sumx[0] = np.sum(xpoly)
    for d in range(2 * dim):
        xpoly = xpoly * x
        sumx[d + 1] = np.sum(xpoly)
    sumy = np.zeros(dim, float)
    xypoly = np.array(y)
    sumy[0] = np.sum(xypoly)
    for d in range(dim - 1):
        xypoly = xypoly * x
        sumy[d + 1] = np.sum(xypoly)
    A = np.zeros((dim,dim),float)
    for d1 in range(dim):
        for d2 in range(dim):
            A[d1,d2] = sumx[d1 + d2]
    y = sumy
    if False != DischargeCalculation(A,y):
        ret = y
    else:
        ret = False
    return ret
    
def Convolution(rawMaps, rawFilter,stride = 1, pad = 0):
    
    assert(isinstance(rawMaps, np.ndarray))
    assert(isinstance(rawFilter,np.ndarray))
    map2dim = False

    if 2 == rawMaps.ndim:
        map = rawMaps.reshape(1,1,rawMaps.shape[0],rawMaps.shape[1])
        map2dim = True
    else:
        map = rawMaps.reshape(rawMaps.shape[0],1,rawMaps.shape[1],rawMaps.shape[2])
    assert(4 == map.ndim)
    
    batch = map.shape[0]
    mapH = map.shape[2]
    mapW = map.shape[3]

    if 2 == rawFilter.ndim:
        filter = rawFilter.reshape(1,rawFilter.shape[0],rawFilter.shape[1])
    else:
        filter = rawFilter
    assert(3 == filter.ndim)
    filH = filter.shape[1]
    filW = filter.shape[2]

    assert(0 == (mapH + 2 * pad - filH) % stride)
    assert(0 == (mapW + 2 * pad - filW) % stride)
    outH = 1 +  (mapH + 2 * pad - filH) // stride
    outW = 1 +  (mapW + 2 * pad - filW) // stride

    col = im2col(input_data = map,
                 filter_h = filH,
                 filter_w = filW,
                 stride = stride,
                 pad = pad)
    col_w = filter.reshape(filter.shape[0], -1).T

    out = np.dot(col, col_w).reshape(batch, outH, outW, -1).transpose(0, 3, 1, 2)

    if map2dim:
        out = out.reshape(outH, outW)
    else:
        out = out.reshape(-1, outH, outW)

    return out


def IntMinMax(fVal,iMin,iMax):
    if not isinstance(fVal,float):
        if not isinstance(fVal,int):
            func = func = sys._getframe().f_code.co_name
            print( func + ": fVal type error: " + str(type(fVal)) )
    if not isinstance(iMin,int):
        func = func = sys._getframe().f_code.co_name
        print( func + ": iMin type error." )
    if not isinstance(iMax,int):
        func = func = sys._getframe().f_code.co_name
        print( func + ": iMax type error." )
        
        
    iVal = int( fVal )
    if( iMin > iVal ):
        return iMin
    elif ( iMax < iVal ):
        return iMax
    else:
        return iVal
    
def IntMax(fVal,iMax):
    if not isinstance(fVal,float):
        if not isinstance(fVal,int):
            func = func = sys._getframe().f_code.co_name
            print( func + ": fVal type error: " + str(type(fVal)) )
    if not isinstance(iMax,int):
        func = func = sys._getframe().f_code.co_name
        print( func + ": iMax type error." )
        
    iVal = int( fVal )
    if( iMax < iVal ):
        return iMax
    else:
        return iVal
    
def CalcAccuracy(finalScore,label):
    
    if finalScore.ndim > 0:
        finalScore = np.transpose(finalScore)[0]
    
    assert(np.array(finalScore).shape == np.array(label).shape)

    posSample = np.sum(  1 == label)
    negSample = np.sum( -1 == label)
    assert(posSample > 0)
    assert(negSample > 0)
    assert(posSample + negSample == label.size)

    truePos  = np.empty(0,float)
    falseNeg = np.empty(0,float)
    falsePos = np.empty(0,float)
    trueNeg  = np.empty(0,float)
    
    # accuracy = MAX( (TP+TN)/(TP+TN+FP+FN) )を計算
    accuracy = 0.0
    for i in range(finalScore.size):
        # バイアスがfinalScore[i]だったときのROCカーブ上の点を算出
        bias = finalScore[i]
        truePos  = np.append(truePos ,np.sum((( 1 == label) * (bias < finalScore))))
        falseNeg = np.append(falseNeg,np.sum((( 1 == label) * (bias > finalScore))))
        falsePos = np.append(falsePos,np.sum(((-1 == label) * (bias < finalScore))))
        trueNeg  = np.append(trueNeg ,np.sum(((-1 == label) * (bias > finalScore))))
        accuracy = max(accuracy,                    \
                       (truePos[i] + trueNeg[i]) / \
                       (truePos[i] + trueNeg[i] + falsePos[i] + falseNeg[i]))
        
    return accuracy

def CalcROCArea(finalScore,label):
    
    if finalScore.ndim > 0:
        finalScore = np.transpose(finalScore)[0]
    
    assert(np.array(finalScore).shape == np.array(label).shape)

    posSample = np.sum(  1 == label)
    negSample = np.sum( -1 == label)
    assert(posSample > 0)
    assert(negSample > 0)
    assert(posSample + negSample == label.size)

    truePos  = np.empty(0,float)
    falseNeg = np.empty(0,float)
    falsePos = np.empty(0,float)
    trueNeg  = np.empty(0,float)
    
    for i in range(finalScore.size):
        # バイアスがfinalScore[i]だったときのROCカーブ上の点を算出
        bias = finalScore[i]
        truePos  = np.append(truePos ,np.sum((( 1 == label) * (bias < finalScore))))
        falseNeg = np.append(falseNeg,np.sum((( 1 == label) * (bias > finalScore))))
        falsePos = np.append(falsePos,np.sum(((-1 == label) * (bias < finalScore))))
        trueNeg  = np.append(trueNeg ,np.sum(((-1 == label) * (bias > finalScore))))
        
    # ROC下部面積を求める
    curveX = np.sort(falsePos / negSample)
    curveY = np.sort(truePos  / posSample)
    area = (curveY[0]) * (curveX[0]) * 0.5
    for i in range(curveX.size - 1):
        area += (curveY[i + 1] + curveY[i]) * (curveX[i + 1] - curveX[i]) * 0.5
        if i == curveX.size - 2:
            area += (1 + curveY[i + 1]) * (1 - curveX[i + 1]) * 0.5
    
    return area

def MultipleAll(x):
    out = 1
    if 0 == np.array(x).ndim:
        return x
    else:
        for d in range(np.array(x).ndim):
            out *= x[d]
    return out
    
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Test_MakeIntegral(unittest.TestCase):
    def test_MakeIntegral(self):
        a = np.array([[0, 1, 2],
                      [0, 1, 2],
                      [0, 1, 2]])
        b = MakeIntegral(a)
        self.assertEqual((b == np.array([[0, 0, 0, 0],
                                         [0, 0, 1, 3],
                                         [0, 0, 2, 6],
                                         [0, 0, 3, 9]])).all(), True)
class Test_SumFromIntegral(unittest.TestCase):
    def test_SumFromIntegral1(self):
        a = np.array([[0, 1, 2],
                      [0, 1, 2],
                      [0, 1, 2]])
        b = MakeIntegral(a)
        self.assertEqual(6, SumFromIntegral(b, y0 = 1, y1 = 2, x0 = 1, x1 = 2))
    def test_SumFromIntegral2(self):
        a = np.array([[0, 1, 2],
                      [0, 1, 2],
                      [0, 1, 2]])
        b = MakeIntegral(a)
        self.assertEqual(6, SumFromIntegral(b, y0 = 0, y1 = 1, x0 = 0, x1 = 2))

# Ax=bの解xを、LU分解で高速に求める
def SolveLU(A, b):
    LU = linalg.lu_factor(A)
    return np.array(linalg.lu_solve(LU, b))
class Test_LU(unittest.TestCase):
    def test_LU(self):        
        A = np.array([[6, 4, 1],
                      [1, 8, -2],
                      [3, 2, 0]])
        b = np.array([7, 6, 8])
        x = SolveLU(A, b)
        xExpected = np.dot(np.linalg.inv(A), b)
        self.assertEqual(False, np.isnan(x).any())        

if "__main__" == __name__:
    unittest.main()
    