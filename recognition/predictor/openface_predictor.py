import os

import cv2
import numpy as np
import openface

from file_path_manager import FilePathManager
from recognition.predictor.predictor import Predictor


class OpenfacePredictor(Predictor):
    def __init__(self):
        super().__init__(siamese=True)
        self.net = openface.TorchNeuralNet(FilePathManager.resolve("data/nn4.small2.v1.t7"), 224, cuda=True)
        classes = os.listdir(FilePathManager.resolve("data/custom_images"))
        t = {}
        for clz in classes:
            path = FilePathManager.resolve(f"data/custom_images/{clz}")
            temp = os.listdir(path)[0]
            temp = f"{path}/{temp}"
            face = cv2.imread(temp)
            face = self.preprocessor.preprocess(face)[0][0]
            t[clz] = self.net.forward(face).reshape(1, -1)
        self.classes = t

    def recognize_person(self, face, threshold=0.3):
        max_sim = 0
        max_pers = None
        for (clz, feat) in self.classes.items():
            sim = np.dot(face, feat) / (np.linalg.norm(feat) * np.linalg.norm(face))
            if sim > max_sim:
                max_pers = clz
                max_sim = sim
        return max_pers if max_sim >= threshold else Predictor.UNKNOWN

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            face = face.data.cpu().numpy()
            face = face.reshape(-1, 1)
            person = self.recognize_person(face)
            result.append((person, rect, 1))
        return result
