import os
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from util.file_path_manager import FilePathManager
from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor
from recognition.pretrained.extractors import vgg_extractor


class Predictor(metaclass=ABCMeta):
    scale = 1
    preprocessor = AlignerPreprocessor(scale)
    extractor = vgg_extractor()

    def __init__(self, use_custom: bool = True, use_cuda: bool = True):
        self.use_cuda = use_cuda
        self.names = sorted(
            os.listdir(FilePathManager.load_path("data/{}".format("custom_images2" if use_custom else "lfw2"))))

    def predict_from_path(self, image_path: str):
        return self.predict_from_image(cv2.imread(image_path))

    @abstractmethod
    def predict_from_image(self, image):
        items = Predictor.preprocessor.preprocess(image)
        result = []
        for (face, rect) in items:
            face = cv2.resize(face, (224, 224))
            face = np.swapaxes(face, 0, 2)
            face = np.swapaxes(face, 1, 2)
            face = torch.from_numpy(face).float()
            face = face.unsqueeze(0)
            x = Variable(face.cuda())
            x = Predictor.extractor(x)
            result.append((x, rect))
        return result
