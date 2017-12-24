import os
from random import random as rand

import cv2
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileSystemModel
from PyQt5.QtWidgets import QWidget
import matplotlib.pyplot as plt


from recognition.predictor.evm_predictor import EvmPredictor
from recognition.predictor.predictor import Predictor
from util.file_path_manager import FilePathManager


class Ui_MainWindow(QWidget):
    root_path = FilePathManager.load_path("test_images")
    # matplotlib or opencv
    drawing_method = "opencv"
    with_prop = False

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
        self.layoutWidget.setGeometry(QtCore.QRect(390, 530, 151, 71))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.predictButton = QtWidgets.QPushButton(self.layoutWidget)
        self.predictButton.setObjectName("predictButton")
        self.verticalLayout.addWidget(self.predictButton)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # custom
        Predictor.scale = 0
        self.predictor = EvmPredictor(FilePathManager.load_path("models/evm/evm.model"))
        self.setupEvents()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Recognition"))
        self.imageLabel.setText(_translate("MainWindow", ""))
        self.predictButton.setText(_translate("MainWindow", "Predict"))

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
        predicted = self.predictor.predict_from_path(image_path)
        image = cv2.imread(image_path)
        self.show_boxes(image, predicted)

    def item_selection_changed_slot(self):
        index = self.treeView.selectedIndexes()[0]
        item = self.treeView.model().itemData(index)[0]
        image_path = "{}/{}".format(Ui_MainWindow.root_path, item)
        self.set_image(image_path)

    def set_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(scaled_pixmap)

    def show_boxes(self, image, predicted):
        if Ui_MainWindow.drawing_method == "matplotlib":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.cla()
            plt.axis("off")
            plt.imshow(image)
            for (name, rect, prop) in predicted:
                name = name.replace("_", " ")
                color = (rand(), rand(), rand())
                x, y, w, h = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
                rect = plt.Rectangle((x, y),
                                     w,
                                     h,
                                     fill=False,
                                     edgecolor=color,
                                     linewidth=2.5)
                plt.gca().add_patch(rect)
                plt.gca().text(x, y - 10,
                               '{:s}\n{:.3f}'.format(name, prop) if Ui_MainWindow.with_prop else "{:s}".format(name),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
            plt.show()
        else:
            font_scale = 1
            for (name, rect, prop) in predicted:
                name = name.replace("_", " ")
                color = (rand() * 255, rand() * 255, rand() * 255)
                x, y, w, h = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                cv2.putText(image,
                            '{:s}\n{:.3f}'.format(name, prop) if Ui_MainWindow.with_prop else "{:s}".format(name),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_COMPLEX,
                            font_scale, (255, 255, 255),
                            2)
            cv2.imwrite("temp.jpg", image)
            self.set_image("temp.jpg")
            os.system("rm temp.jpg")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.predict()
