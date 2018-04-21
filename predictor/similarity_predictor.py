import os

import cv2
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from bases.pipeline import Pipeline
from classifiers.evm import EVM
from detectors.dlib_detector import DLibDetector
from extractors.base_extractor import BaseExtractor
from extractors.dlib_extractor import DLibExtractor
from file_path_manager import FilePathManager


class SimilarityPredictor:

    def __init__(self, model_path, extractor: BaseExtractor = DLibExtractor(), threshold: float = 0.9, align=True):
        super().__init__()
        self.pipeline = Pipeline([
            DLibDetector(scale=1),
            # OneMillisecondAligner(extractor.resize.size) if align else NoAligner(),
            extractor
        ])
        self.model_path = model_path
        # for vgg face, 0.7 seems to be the best
        self.threshold = threshold
        if os.path.exists(model_path):
            self.reload()
            return
        classes = os.listdir(FilePathManager.resolve("faces"))
        self.classes = {}
        for clz in classes:
            path = FilePathManager.resolve(f"faces/{clz}")
            temp = os.listdir(path)
            images = []
            for t in temp:
                fi = f"{path}/{t}"
                images.append(cv2.imread(fi))
            self.add_person(clz, images)

    def detect(self, features):
        features = features.reshape(1, -1)
        max_sim = -1
        max_per = None
        for (clz, class_features) in self.classes.items():
            sim = cosine_similarity(class_features, features).max()
            if sim > max_sim:
                max_per = clz
                max_sim = sim
        return max_per if max_sim >= self.threshold else EVM.UNKNOWN, max_sim

    def predict_from_image(self, image):
        faces, rects, _ = self.pipeline(image)
        return [(*self.detect(faces[i]), rects[i]) for i in range(faces.shape[0])]

    def predict_from_path(self, path):
        return self.predict_from_image(cv2.imread(path))

    def reload(self):
        self.classes = joblib.load(self.model_path)

    def save(self):
        joblib.dump(self.classes, self.model_path)

    def extract_from_images(self, images):
        result = []
        for image in images:
            temp = self.pipeline(image)[0].reshape(-1)
            if temp[0] == 0:
                continue
            result.append(temp)
        return np.asarray(result)

    def add_person(self, person_name, images):
        feats = self.extract_from_images(images)
        self.classes[person_name] = np.array(feats)
        self.save()

    def remove_person(self, person_name):
        self.classes.pop(person_name)
        self.save()

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


if __name__ == '__main__':
    p = SimilarityPredictor(FilePathManager.resolve("trained_models/similarity.model"))
    p.save()
