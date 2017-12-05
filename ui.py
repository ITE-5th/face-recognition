from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileSystemModel
from PyQt5.QtWidgets import QWidget

from file_path_manager import FilePathManager
from recognition.predictor.evm_predictor import EvmPredictor


class Ui_MainWindow(QWidget):
    root_path = FilePathManager.load_path("images")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(926, 635)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.imageLabel = QtWidgets.QLabel(self.centralWidget)
        self.imageLabel.setGeometry(QtCore.QRect(400, 0, 521, 491))
        self.imageLabel.setObjectName("imageLabel")
        self.treeView = QtWidgets.QTreeView(self.centralWidget)
        self.treeView.setGeometry(QtCore.QRect(0, 0, 401, 491))
        self.treeView.setObjectName("treeView")
        self.layoutWidget = QtWidgets.QWidget(self.centralWidget)
        self.layoutWidget.setGeometry(QtCore.QRect(390, 530, 131, 71))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.predictButton = QtWidgets.QPushButton(self.layoutWidget)
        self.predictButton.setObjectName("predictButton")
        self.verticalLayout.addWidget(self.predictButton)
        self.resultLabel = QtWidgets.QLabel(self.layoutWidget)
        self.resultLabel.setObjectName("resultLabel")
        self.verticalLayout.addWidget(self.resultLabel)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # custom
        self.predictor = EvmPredictor(FilePathManager.load_path("models/evm/evm_model.model"))
        self.setupEvents()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.imageLabel.setText(_translate("MainWindow", "Your Image:"))
        self.predictButton.setText(_translate("MainWindow", "Predict"))
        self.resultLabel.setText(_translate("MainWindow", "Result"))

    def setupEvents(self):
        model = QFileSystemModel()
        root = model.setRootPath(Ui_MainWindow.root_path)
        self.treeView.setModel(model)
        self.treeView.setRootIndex(root)
        self.treeView.selectionModel().selectionChanged.connect(self.item_selection_changed_slot)
        self.predictButton.clicked.connect(self.predict)

    def predict(self):
        index = self.treeView.selectedIndexes()[0]
        item = self.treeView.model().itemData(index)[0]
        image_path = "{}/{}".format(Ui_MainWindow.root_path, item)
        predicted = self.predictor.predict(image_path)
        self.resultLabel.setText(predicted)

    def item_selection_changed_slot(self):
        index = self.treeView.selectedIndexes()[0]
        item = self.treeView.model().itemData(index)[0]
        image_path = "{}/{}".format(Ui_MainWindow.root_path, item)
        pixmap = QtGui.QPixmap(image_path)
        scaledPixmap = pixmap.scaled(self.imageLabel.size())
        self.imageLabel.setPixmap(scaledPixmap)
