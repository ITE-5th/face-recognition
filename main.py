import sys

from PyQt5 import QtWidgets

from file_path_manager import FilePathManager
from ui import Ui_MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    with open(FilePathManager.load_path("qdarkstyle/style.qss")) as f:
        app.setStyleSheet(f.read())
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setWindowTitle("Face Recognition")
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
