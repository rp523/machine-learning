import sys
from PyQt5 import QtWidgets as qt

# ディクショナリ引数で与えられた機能のボタンが横に並んだレイアウトを返す
# keyの値はボタンのラベル情報を兼ねている
def ButtonRow(pushEvent, supWindow = None):

        out = False
            
        assert(isinstance(pushEvent,dict))
        assert(isinstance(supWindow,qt.QWidget))

        layout = qt.QHBoxLayout()
        
        # key兼ラベルの順にソートする
        sorted(pushEvent.items(), key=lambda x: x[0])

        for btLabel in pushEvent.keys():
            bt = qt.QPushButton(btLabel)
            bt.clicked.connect(pushEvent[btLabel])
            layout.addWidget(bt)
            out = True
        
        supWindow.setLayout(layout)

        return out

def testMessage():
    print("Came.")
    

class CMainWindow:
    def __init__(self):
           
        # アプリケーションのオブジェクトを生成。後の拡張用にコマンド引数を渡す
        app = qt.QApplication(sys.argv)
        
        # ウィジェットを生成表示
        window = qt.QWidget()
        
        dic = {}
        print(type(dic))
        assert(isinstance(dic,dict))
        dic["exec1"] = testMessage
        dic["exec2"] = testMessage
        
        # プッシュボタンを追加
        assert(ButtonRow(pushEvent=dic,supWindow=window))
            
        # Windowを表示してアプリケーションのメインループに入る
        window.show()
        sys.exit(app.exec())
        
if "__main__" == __name__:
    CMainWindow()
