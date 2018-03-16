import os

import cv2
import torch.nn.functional as F
from dlt.util import cv2torch
from torch.autograd import Variable

from file_path_manager import FilePathManager
from recognition.predictor.predictor import Predictor


class SiameseNetPredictor(Predictor):

    def __init__(self):
        super().__init__(siamese=True)
        classes = os.listdir(FilePathManager.resolve("data/custom_images"))
        t = {}
        for clz in classes:
            path = FilePathManager.resolve(f"data/custom_images/{clz}")
            temp = os.listdir(path)[0]
            temp = f"{path}/{temp}"
            face = cv2.imread(temp)
            face = self.preprocessor.preprocess(face)[0][0]
            face = cv2.resize(face, (224, 224))
            face = cv2torch(face).float()
            face = face.unsqueeze(0)
            x = Variable(face).cuda()
            x = self.extractor(x)
            t[clz] = x
        self.classes = t

    def recognize_person(self, face, threshold=1.5):
        min_dis = 1e9
        min_per = None
        for (clz, feat) in self.classes.items():
            distance = F.pairwise_distance(face, feat)
            distance = distance.cpu().data.numpy()[0][0]
            if distance < min_dis:
                min_per = clz
                min_dis = distance
        return min_per if min_dis <= threshold else Predictor.UNKNOWN

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            person = self.recognize_person(face)
            result.append((person, rect, 1))
        return result
