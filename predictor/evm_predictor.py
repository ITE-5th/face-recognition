import cv2
import joblib
import numpy as np

from aligners.one_millisecond_aligner import OneMillisecondAligner
from bases.pipeline import Pipeline
from classifiers.evm import EVM
from detectors.dlib_detector import DLibDetector
from extractors.openface_extractor import OpenfaceExtractor
from extractors.vgg_extractor import VggExtractor


class EvmPredictor:
    pipeline = Pipeline([
        DLibDetector(scale=1),
        OneMillisecondAligner(224),
        VggExtractor()
    ])

    def __init__(self, evm_model_path: str):
        self.model_path = evm_model_path
        self.evm: EVM = joblib.load(self.model_path)

    def reload(self):
        self.evm = joblib.load(self.model_path)

    def save(self):
        joblib.dump(self.evm, self.model_path)

    @staticmethod
    def extract_from_images(images):
        result = []
        for image in images:
            temp = EvmPredictor.pipeline(image)[0].reshape(-1)
            if temp[0] == 0:
                continue
            result.append(temp)
        return np.asarray(result)

    def add_person(self, person_name, images):
        X = EvmPredictor.extract_from_images(images)
        y = np.full((len(images), 1), person_name)
        self.evm.fit(X, y)
        self.save()

    def remove_person(self, person_name):
        self.evm.remove(person_name)
        self.save()

    def predict_from_image(self, image):
        faces = EvmPredictor.pipeline(image)[0]
        return self.evm.predict_with_prop(faces)

    def predict_from_path(self, path):
        return self.predict_from_image(cv2.imread(path))
