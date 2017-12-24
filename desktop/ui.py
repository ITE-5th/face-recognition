import os
import sys
import threading
from queue import Queue
from random import random as rand

import cv2
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileSystemModel, QTreeView, QPushButton

from recognition.predictor.evm_predictor import EvmPredictor
from util.file_path_manager import FilePathManager

FormClass = uic.loadUiType("ui.ui")[0]
running = False
q = Queue()


def grab(queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    while running:
        capture.grab()
        _, img = capture.retrieve(0)
        queue.put(img)


class FilesTreeView(QtWidgets.QTreeView):
    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def keyPressEvent(self, event):
        self.func(event)





class ImageWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_image = None
        self.image = None

    def setImage(self, image, raw_image):
        self.image = image
        self.raw_image = raw_image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class Ui(QtWidgets.QMainWindow, FormClass):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.root_path = FilePathManager.load_path("test_images")
        self.drawing_method = "matplotlib"
        self.with_prop = True
        self.window_width = self.videoWidget.frameSize().width()
        self.window_height = self.videoWidget.frameSize().height()
        self.filesTreeView = FilesTreeView(self.keyPressEvent, self.filesTreeView)
        self.videoWidget = ImageWidget(self.videoWidget)
        self.predictor = EvmPredictor(FilePathManager.load_path("models/evm/evm.model"))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        self.setup_events()

    def setup_events(self):
        model = QFileSystemModel()
        root = model.setRootPath(self.root_path)
        self.filesTreeView.setModel(model)
        self.filesTreeView.setRootIndex(root)
        self.filesTreeView.selectionModel().selectionChanged.connect(self.item_selection_changed_slot)

    def item_selection_changed_slot(self):
        index = self.filesTreeView.selectedIndexes()[0]
        item = self.filesTreeView.model().itemData(index)[0]
        image_path = "{}/{}".format(self.root_path, item)
        self.set_image(image_path)

    def set_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(scaled_pixmap)

    def keyPressEvent(self, event):
        global running, q
        if event.key() == QtCore.Qt.Key_Space:

            tab = self.tabWidget.currentIndex()
            if tab == 0:
                index = self.filesTreeView.selectedIndexes()[0]
                item = self.filesTreeView.model().itemData(index)[0]
                image_path = "{}/{}".format(self.root_path, item)
                predicted = self.predictor.predict_from_path(image_path)
                image = cv2.imread(image_path)
                self.show_boxes(image, predicted)
            else:
                if running:
                    running = False
                    img = self.videoWidget.raw_image
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    predicted = self.predictor.predict_from_image(img)
                    self.show_boxes(img, predicted, True)
                else:
                    running = True
                    capture_thread = threading.Thread(target=grab, args=(q, 1920, 1080, 30))
                    capture_thread.start()

    def show_boxes(self, image, predicted, video=False):
        if self.drawing_method == "matplotlib":
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
                plt.gca().text(x + 15, y - 10,
                               '{:s}\n{:.3f}%'.format(name, prop * 100) if self.with_prop else "{:s}".format(name),
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
                            '{:s}\n{:.3f}%'.format(name, prop * 100) if self.with_prop else "{:s}".format(name),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_COMPLEX,
                            font_scale, (255, 255, 255),
                            2)
            if not video:
                cv2.imwrite("temp.jpg", image)
                self.set_image("temp.jpg")
                os.system("rm temp.jpg")
            else:
                cv2.imshow("image", image)
                img = image
                img_height, img_width, img_colors = img.shape
                scale_w = float(self.window_width) / float(img_width)
                scale_h = float(self.window_height) / float(img_height)
                scale = min([scale_w, scale_h])
                if scale == 0:
                    scale = 1
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, bpc = img.shape
                bpl = bpc * width
                image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
                self.videoWidget.setImage(image, img)

    def update_frame(self):
        global running
        if not q.empty() and running:
            img = q.get()
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])
            if scale == 0:
                scale = 1
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.videoWidget.setImage(image, img)

    def closeEvent(self, event):
        global running
        running = False


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    with open("qdarkstyle/style.qss") as f:
        app.setStyleSheet(f.read())
    ui = Ui()
    ui.setWindowTitle("Face Recognition")
    ui.show()
    app.exec_()
