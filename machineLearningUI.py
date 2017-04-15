import sys
from PyQt5 import QtWidgets as qt


if "__main__" == __name__:
    
    # アプリケーションのオブジェクトを生成。後の拡張用にコマンド引数を渡す
    app = qt.QApplication(sys.argv)
    
    # ウィジェットを生成して表示
    window = qt.QWidget()
    window.show()
    
    # アプリケーションのメインループに入る
    sys.exit(app.exec())