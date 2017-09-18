#coding: utf-8
import os, sys, shutil
import enum
from matplotlib.font_manager import FontProperties
from datetime import datetime
import pandas as pd
myrica = FontProperties(fname="/home/isgsktyktt/.fonts/Myrica.TTC")

# 自動でアイテムが追加順にsortされるディクショナリ
class dicts(dict):
    # 初期化子のoverride
    def __init__(self, *args, **kwargs):
        # 追加順にkey値が保存されるlist
        self.__order = []
        
        for newDict in args:
            for newKey in newDict.keys():
                self.__order.append(newKey)
        for key in kwargs.keys():
            self.__order.append(key)
        return super().__init__(*args,**kwargs)
    # 値取得operator[]のoverride
    def __getitem__(self, *args, **kwargs):
        # 特にやることなし
        return super().__getitem__(*args)
    # 値設定operator[]のoverride
    def __setitem__(self, *args, **kwargs):
        newKey = args[0]
        if not newKey in self.__order:
            self.__order.append(newKey)
        return super().__setitem__(*args, **kwargs)
    # アイテム削除operator del のoverride
    def __delitem__(self, *args, **kwargs):
        delKey = args[0]
        for i in range(len(self.__order)):
            if self.__order[i] == delKey:
                del self.__order[i]
                break
        return dict.__delitem__(self, *args, **kwargs)
    def update(self, *args, **kwargs):
        for newDict in args:
            for newKey in newDict.keys():
                if not newKey in self.__order:
                    self.__order.append(newKey)
        return super().update(*args,**kwargs)
    def keys(self, *args, **kwargs):
        # 継承元のkeysは使わない
        return self.__order
    def values(self, *args, **kwargs):
        # 追加順のキーを基準にvalueの配列を生成して返す
        out = []
        for key in self.__order:
            out.append(super().__getitem__(key))
        return out
    def items(self, *args, **kwargs):
        # 追加順のキーを基準にitemの配列を生成して返す
        out = []
        for key in self.__order:
            out.append([key,super().__getitem__(key)])
        return out
    def clear(self, *args, **kwargs):
        self.__order = []
        return super().clear(*args, **kwargs)

# __init__()以外で新しい要素を追加できないdictionary
# ただし既存要素の変数型を変えずに値のみ変更するのはＯＫ
class CParam(dicts):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args,**kwargs)
    def __setitem__(self, *args, **kwargs):
        newKey = args[0]
        newVal = args[1]
        assert(newKey in self.keys()) # 要素の追加は禁止
        oldType = type(super().__getitem__(newKey))
        newType = type(newVal)
        assert(newType == oldType) # 要素の型変更は禁止
        return super().__setitem__(*args, **kwargs)
    def update(self, *args, **kwargs):
        for newDict in args:
            for newKey in newDict.keys():
                assert(newKey in self.keys()) # 要素の追加は禁止
                newVal = newDict[newKey]
                newType = type(newVal)
                oldType = type(super().__getitem__(newKey))
                assert(newType == oldType) # 要素の型変更は禁止
        return super().update(*args,**kwargs)

class selparam(CParam):
    # valueにはbool型しか設定しちゃダメ
    def __init__(self,  *args, **kwargs):
        setDicts = dicts()
        firstFlg = True
        for keyStr in args:
            assert(isinstance(keyStr,str))
            if firstFlg == True:
                setDicts[keyStr] = True
                firstFlg = False
            else:
                setDicts[keyStr] = False
        return super().__init__(setDicts)
    def setTrue(self,setKey):
        for key in self.keys():
            if setKey == key:
                super().update({key:True})
            else:
                super().update({key:False})
    def option(self):
        for k,v in self.items():
            out = ""
            if True == v:
                out = k
        return k
    def nowSelected(self):
        for k, v in self.items():
            if True == v:
                return k
    def __setitem__(self, *args, **kwargs):
        assert(1) # select関数以外での値変更は禁止
    def update(self, *args, **kwargs):
        assert(0) # select関数以外での値変更は禁止

def GetTimeMark():
    now = datetime.now()
    nowStr = "{0:04d}".format(now.year)   + \
             "{0:02d}".format(now.month)  + \
             "{0:02d}".format(now.day)    + \
             "_"                          + \
             "{0:02d}".format(now.hour)   + \
             "{0:02d}".format(now.minute) + \
             "{0:02d}".format(now.second) + \
             "_"                          + \
             "{0:06d}".format(now.microsecond)
    return nowStr

def AddXlsxSheet(filePath,
                 addSheetName,
                 addDataFrame):
    
    if not os.path.exists(filePath):
        return False
    file = pd.ExcelFile(filePath)
    if addSheetName in file.sheet_names:
        return False

    tmpPath = "temp" + GetTimeMark() + ".xlsx"
    assert(not os.path.exists(tmpPath))
    
    tmpWriter = pd.ExcelWriter(tmpPath, engine = 'xlsxwriter')
    for sheetName in file.sheet_names:
        pd.read_excel(filePath, sheetname = sheetName).to_excel(tmpWriter, sheet_name = sheetName)
    
    addDataFrame.to_excel(tmpWriter, sheet_name = addSheetName)

    tmpWriter.save()
    tmpWriter.close()
    shutil.move(tmpPath, filePath)
