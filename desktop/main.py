import sys

from PyQt5 import QtWidgets
from desktop.ui import Ui

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    with open("qdarkstyle/style.qss") as f:
        app.setStyleSheet(f.read())
    ui = Ui()
    ui.setWindowTitle("Face Recognition")
    ui.show()
    app.exec_()
