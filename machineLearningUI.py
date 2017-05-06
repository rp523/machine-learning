import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
import pandas as pd
from common.origLib import *
from AdaBoost import AdaBoostParam
import xlsxwriter
from feature import *


class CHRadioButtonSet(QGroupBox):
    def __init__(self, label, setParam):
        super().__init__(label)
        self.paramVal = setParam
        self.rb = dicts()
        self.param2gui(label, setParam)

    def param2gui(self, label, setParam):
        assert(isinstance(setParam, selparam))
        hlay = QHBoxLayout()
        for rbLabel, setval in setParam.items():
            self.rb[rbLabel] = QRadioButton(rbLabel)
            self.rb[rbLabel].setChecked(setval)
            hlay.addWidget(self.rb[rbLabel])
        self.setLayout(hlay)

    def gui2param(self):
        out = self.paramVal
        for rbLabel, val in self.rb.items():
            if True == val.isChecked():
                out.set(rbLabel)
        return out


class CNumericUpDownSet(QWidget):
    def __init__(self, label, setParam):
        super().__init__()
        self.param2gui(label, setParam)

    def param2gui(self, label, setParam):
        assert(isinstance(setParam, int))
        hlay = QHBoxLayout()
        hlay.addWidget(QLabel(label))
        self.sb = QSpinBox()
        self.sb.setMaximum(99999)
        self.sb.setValue(setParam)
        hlay.addWidget(self.sb)
        self.setLayout(hlay)

    def gui2param(self):
        return self.sb.value()


class CLineFloatEdit(QLineEdit):
    def __init__(self):
        self.oldVal = ""
        return super().__init__()
    # このコントロールにフォーカスが移った時のイベント

    def focusInEvent(self, *args, **kwargs):
        # ユーザーがいじる前の値を記録しておく
        self.oldVal = self.text()
        return super().focusInEvent(*args, **kwargs)
    # このコントロールからフォーカスが外れた時のイベント

    def focusOutEvent(self, *args, **kwargs):
        # float変換できる文字列が入力されているのかをチェック
        try:
            floatVal = float(self.text())
            strVal = str(floatVal)
            self.setText(strVal)
        except ValueError:
            # float変換できない値が入力されていたので、値を過去の値に戻す
            self.setText(self.oldVal)
        return super().focusOutEvent(*args, **kwargs)


class CLineFloatEditSet(QWidget):
    def __init__(self, label, setParam):
        super().__init__()
        self.param2gui(label, setParam)

    def param2gui(self, label, setParam):
        assert(isinstance(setParam, float))
        hlay = QHBoxLayout()
        hlay.addWidget(QLabel(label))
        self.le = CLineFloatEdit()
        self.le.setText(str(setParam))
        hlay.addWidget(self.le)
        self.setLayout(hlay)

    def gui2param(self):
        return float(self.le.text())


class CCheckBoxSet(QCheckBox):
    def __init__(self, label):
        super().__init__(label)

    def gui2param(self):
        return self.isChecked()


class CHoriEditSet(QGroupBox):
    def __init__(self, label, setParam):
        self.paramType = setParam
        super().__init__(label)
        self.param2gui(label, setParam)

    def param2gui(self, label, setParam):
        hbox = QHBoxLayout()
        self.gui2paramFuncs = dicts()
        for k, v in setParam.items():
            if isinstance(v, selparam):
                ctrlWidget = CHRadioButtonSet(k, v)
            if isinstance(v, int):
                ctrlWidget = CNumericUpDownSet(k, v)
            if isinstance(v, float):
                ctrlWidget = CLineFloatEditSet(k, v)
            if isinstance(v, bool):
                ctrlWidget = CCheckBoxSet(k)
            hbox.addWidget(ctrlWidget)
            self.gui2paramFuncs[k] = ctrlWidget.gui2param
        self.setLayout(hbox)

    def gui2param(self):
        out = self.paramType
        for k, v in self.gui2paramFuncs.items():
            try:
                out[k] = v()
            except AssertionError:
                print("type conversion error.")
                print("parameter labelled ", k,
                      " cannot be converted from ", type(out[k]), " ", out(v()))
        return out


class CParamSetWidget(QWidget):
    def __init__(self, setParam):
        self.paramType = setParam
        super().__init__()
        self.gui2paramFuncs = dicts()
        self.param2gui(setParam)

    def param2gui(self, setParam):
        assert(isinstance(setParam, param))
        vbox = QVBoxLayout()
        for k, v in setParam.items():
            if isinstance(v, selparam):
                ctrlWidget = CHRadioButtonSet(k, v)
            if isinstance(v, int):
                ctrlWidget = CNumericUpDownSet(k, v)
            if isinstance(v, float):
                ctrlWidget = CLineFloatEditSet(k, v)
            if isinstance(v, dicts):
                ctrlWidget = CHoriEditSet(k, v)
            self.gui2paramFuncs[k] = ctrlWidget.gui2param
            vbox.addWidget(ctrlWidget)
        self.setLayout(vbox)

    def gui2param(self):
        out = self.paramType
        for k, v in self.gui2paramFuncs.items():
            try:
                out[k] = v()
            except AssertionError:
                print("type conversion error.")
                print("parameter labelled ", k, " cannot be converted from ", type(
                    out[k]), " ", type(v()))
        return out

# ディクショナリ引数で与えられた機能のボタンが横に並んだレイアウトを返す
# keyの値はボタンのラベル情報を兼ねている


class CHPushButtonSet(QWidget):
    def __init__(self, pushEvent=None):
        super().__init__()
        if pushEvent:
            self.__set(pushEvent)

    def __set(self, pushEvent):
        assert(isinstance(pushEvent, dicts))
        layout = QHBoxLayout(self)
        for btLabel in pushEvent.keys():
            bt = QPushButton(btLabel)
            bt.clicked.connect(pushEvent[btLabel])
            bt.setFixedHeight(32)
            layout.addWidget(bt)
        self.setLayout(layout)


def testMessage():
    print("Came.")


class CTableWidgetSet(QWidget):
    def __init__(self, df=None):
        super().__init__()
        self.table = QTableWidget()
        box = QVBoxLayout()
        box.addWidget(QLabel("Recipe"))
        box.addWidget(self.table)
        self.setLayout(box)

        if df:
            self.set(df)

    def set(self, df):

        assert(type(df) == type(pd.DataFrame()))
        colList = df.columns
        colLen = len(colList)
        rowLen = len(df.index)

        # 入力が空なら何もしない
        if (0 == colLen) or (0 == rowLen):
            return

        self.table.setColumnCount(colLen)
        self.table.setRowCount(rowLen)
        self.table.setHorizontalHeaderLabels(colList)

        for x in range(colLen):
            for y in range(rowLen):
                newitem = QTableWidgetItem(str(df.iloc[y, x]))
                self.table.setItem(y, x, newitem)

    def get(self):
        header = []
        for x in range(self.table.columnCount()):
            header.append(self.table.horizontalHeaderItem(x).text())
        dfImpl = []
        for y in range(self.table.rowCount()):
            oneRow = []
            for x in range(self.table.columnCount()):
                strVal = self.table.item(y, x).text()
                oneRow.append(strVal)
            dfImpl.append(oneRow)
        df = pd.DataFrame(dfImpl)
        df.columns = header
        df.index = range(1, self.table.rowCount() + 1)
#        df.to_csv("sample.csv")
        return df


# Label、パス入力用LineEdit、パス設定ダイアログ呼び出しPushButtonのセット
class CPathEditAndEditBT(QWidget):
    def __init__(self, name, default=None):
        super().__init__()
        assert(isinstance(name, str))
        if not default:
            default = ""
        else:
            assert(isinstance(default, str))
        self.__lineEdit = QLineEdit(default)

        editBt = QPushButton()
        editBt.setText("set")
        editBt.clicked.connect(self.__selectDialog)

        hbox = QHBoxLayout()
        hbox.addWidget(self.__lineEdit)
        hbox.addWidget(editBt)

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel(name))
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def __selectDialog(self):
        selectedPath = str(QFileDialog.getExistingDirectory(
            self, "入力ファイルのパスを選択して下さい"))
        self.__lineEdit.setText(selectedPath)

    def getPathStr(self):
        return self.__lineEdit.text()


class CInputSetWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.learnPos = CPathEditAndEditBT("学習Pos")
        self.learnNeg = CPathEditAndEditBT("学習Neg")
        self.evalPos = CPathEditAndEditBT("評価Pos")
        self.evalNeg = CPathEditAndEditBT("評価Neg")

        grid = QGridLayout()

        grid.addWidget(self.learnPos, 0, 0)
        grid.addWidget(self.learnNeg, 0, 1)
        grid.addWidget(self.evalPos,  1, 0)
        grid.addWidget(self.evalNeg,  1, 1)

        self.setLayout(grid)

    def get(self):
        out = dicts()
        out["学習Posパス"] = self.learnPos.getPathStr()
        out["学習Negパス"] = self.learnNeg.getPathStr()
        out["評価Posパス"] = self.evalPos.getPathStr()
        out["評価Negパス"] = self.evalNeg.getPathStr()
        return out

# タブWidget.dicts型でセットできる


class CTabWidget(QTabWidget):

    def __init__(self, initDicts=None):
        super().__init__()
        if initDicts:
            self.__set(initDicts)

    def __set(self, setDicts):
        for key in setDicts.keys():
            assert(isinstance(key, str))
            value = setDicts[key]
            assert(isinstance(value, QWidget))
            self.addTab(value, key)


def Dicts2List(inDicts, supStr=None):
    assert(isinstance(inDicts, dicts))
    key, value = [], []
    print(supStr, inDicts.keys())
    for k, v in inDicts.items():
        if not isinstance(v, dicts):
            if None != supStr:
                key.append(supStr + "\n" + k)
            else:
                key.append(k)
            value.append(v)
        else:
            if None != supStr:
                recKey, recVal = Dicts2List(v,supStr + "\n" + k)
            else:
                recKey, recVal = Dicts2List(v,k)
            key = key + recKey
            value = value + recVal
    return key, value


def isEmptyList(inList):
    assert(isinstance(inList, list))
    flat = sum(inList, [])

    out = True
    for i in range(len(flat)):
        if flat[i] != "":
            out = False
            break
    return out


def HierarchiedDicts(inDicts, parent=None):
    for key in inDicts.keys():
        if not isinstance(inDicts[key], dicts):
            if parent:
                inDicts[parent + "\n" + key] = inDicts[key]
                del inDicts[key]
        else:
            HierarchiedDicts(inDicts[key], parent=key)
    return inDicts


class CMainForm(QWidget):

    def addParam2Recipe(self):
        pass

    def __init__(self):
        super().__init__()

        # ウィジェットを生成

        self.layout = QVBoxLayout()

        mainTabs = dicts()
        self.input = CInputSetWidget()
        mainTabs["input"] = self.input
        self.feature = CParamSetWidget(hog.CHogParam)
        mainTabs["feature"] = self.feature
        self.learner = CParamSetWidget(AdaBoostParam)
        mainTabs["learner"] = self.learner
        self.layout.addWidget(CTabWidget(mainTabs))

        # プッシュボタンを追加
        dic = dicts()
        dic["↓Add"] = self.addParam2table
        dic["↑Back Copy"] = testMessage
        dic["×Delete"] = testMessage
        dic["→Exec"] = testMessage
        self.layout.addWidget(CHPushButtonSet(pushEvent=dic))

        # レシピ表示を追加
        self.table = CTableWidgetSet()
        self.layout.addWidget(self.table)
        self.setLabel()
        # Windowを表示してアプリケーションのメインループに入る
        self.setLayout(self.layout)

    def setLabel(self):
        label, value = self.extractAllParam()
        df = pd.DataFrame([[""] * len(label)])
        df.columns = label
        self.table.set(df)

    def extractAllParam(self):
        allParam = dicts()
        allParam["input"] = self.input.get()
        allParam["feature"] = self.feature.gui2param()
        allParam["learner"] = self.learner.gui2param()
        return Dicts2List(allParam)

    def extractAllParamFromTable(self, index):
        out = self.extractAllParam()  # dicts書式を取得
        df = self.table.get()
        out = self.tableDF2dicts(out, df, index)
        return out

    def tableDF2dicts(self, out, df, index):
        for key in out.keys():
            if not isinstance(out[key], dicts):
                out[key] = df.ix[index, key]
            else:
                self.tableDF2dicts(out[key], df)
        return out

    def addParam2table(self):
        dfNow = self.table.get()
        label, value = self.extractAllParam()
        if isEmptyList(dfNow.values.tolist()):
            # 最初のレシピ
            dfNext = pd.DataFrame([value])
            dfNext.columns = label
        else:
            # 既にレシピがある
            dfAdd = pd.DataFrame([value])
            dfAdd.columns = label
            dfNext = dfNow.append(dfAdd, ignore_index=True)
        self.table.set(dfNext)

    def param2gui(self):
        label, value = self.extractAllParam()
        df = pd.DataFrame([value])
        df.columns = label
        #df.index = range(1,self.table.rowCount()+1)
        print(df)
        # df.to_csv("sample.csv")
        self.table.set(df)


class CMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning")
        self.setCentralWidget(CMainForm())


if "__main__" == __name__:
    # アプリケーションのオブジェクトを生成。後の拡張用にコマンド引数を渡す
    app = QApplication(sys.argv)
    mainWindow = CMainWindow()
    mainWindow.show()
    sys.exit(app.exec())
