#coding: utf-8
import os

def GetDirList(path,onlyName=None):
    ret = []
    itemList = os.listdir(path)
    for item in itemList:
        if os.path.isdir(os.path.join(path,item)):
            if True == onlyName:
                addItem = item
            else:
                addItem = os.path.join(path,item)
            ret.append(addItem)
    return ret

def GetFileList(path,includingText=None,onlyName=None,recursive=None):
    ret = []
    itemNameList = os.listdir(path)
    for itemName in itemNameList:
        itemPath = os.path.join(path,itemName)
        if os.path.isfile(itemPath):
            if (None != includingText):
                if (0 > itemName.find(includingText)):
                    continue
            if True == onlyName:
                addItem = itemName
            else:
                addItem = itemPath
            ret.append(addItem)
            
        elif os.path.isdir(itemPath):
            if True == recursive:
                ret = ret + GetFileList(itemPath,includingText,onlyName,recursive)
    return ret

def ExtractName(path):
    sepa = os.path.join("aaa","aaa").strip("aaa")
    nameStartIndex = path.rfind(sepa) + len(sepa)
    return path[nameStartIndex:]

def ExtractExtension(path):
    extensionStartIndex = path.rfind(".")
    return path[extensionStartIndex:]


if "__main__" == __name__:
    print( GetFileList(".", includingText=".txt",onlyName=False, recursive=True) )
    print( ExtractName( os.path.join("dir1","dir2","filename") ) )
    exit()
    