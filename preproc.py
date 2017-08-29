from input import *
import numpy as np

from common.imgtool import *


def RGB2Gray(rgbList, color):
    
    grayList = []
    for rgb in rgbList:
        assert(isinstance(rgb, np.ndarray))
        assert(rgb.ndim == 3)
        
        h = rgb.shape[0]
        w = rgb.shape[1]
        if "red" == color:
            gray = rgb[:,:,0]
        elif "green" == color:
            gray = rgb[:,:,1]
        elif "blue" == color:
            gray = rgb[:,:,2]
        else:
            assert(0) # color parameter error.
        grayList.append(gray)
    return np.array(grayList)

if "__main__" == __name__:
    learnPosList = dirPath2NumpyArray("INRIAPerson/LearnPos")
    rgb = learnPosList[0]
    
    ndarray2PILimg(rgb).show()
    ndarray2PILimg(RGB2Gray(rgb, "green")).show()
    ndarray2PILimg(RGB2Gray(rgb, "red")).show()
    ndarray2PILimg(RGB2Gray(rgb, "blue")).show()
        
