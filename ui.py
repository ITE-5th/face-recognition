import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog

from file_path_manager import FilePathManager
from recognition.predictor.evm_predictor import EvmPredictor


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(530, 353)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.loadButton = QtWidgets.QPushButton(self.centralWidget)
        self.loadButton.setGeometry(QtCore.QRect(60, 120, 89, 25))
        self.loadButton.setObjectName("loadButton")
        self.resultLabel = QtWidgets.QLabel(self.centralWidget)
        self.resultLabel.setGeometry(QtCore.QRect(230, 270, 121, 31))
        self.resultLabel.setObjectName("resultLabel")
        MainWindow.setCentralWidget(self.centralWidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.predictor = EvmPredictor(FilePathManager.load_path("models/evm/evm_model.model"))
        self.setupEvents()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadButton.setText(_translate("MainWindow", "Load.."))
        self.resultLabel.setText(_translate("MainWindow", "Result"))

    def setupEvents(self):
        self.loadButton.clicked.connect(self.predict)

    def predict(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select An Image")
        image = cv2.imread(image_path)
        result = self.predictor.predict(image_path)
        result = result.replace("_", " ")
        self.resultLabel.setText(result)
        cv2.imshow("Image", image)
