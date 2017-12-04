import os
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor
from recognition.pretrained.extractors import vgg_extractor, inception_extractor
from file_path_manager import FilePathManager


class Predictor(metaclass=ABCMeta):

    def __init__(self, use_custom: bool = True, use_vgg: bool = True, use_cuda: bool = True):
        self.use_vgg = use_vgg
        self.use_cuda = use_cuda
        self.names = sorted(os.listdir(FilePathManager.load_path("data/{}".format("custom_images2" if use_custom else "lfw2"))))

    @abstractmethod
    def predict(self, image_path: str):
        image = cv2.imread(image_path)
        preprocessor = AlignerPreprocessor()
        image = preprocessor.preprocess(image)
        cv2.imwrite(FilePathManager.load_path("temp.jpg"), image)
        image = cv2.imread(FilePathManager.load_path("temp.jpg"))
        image = cv2.resize(image, (224, 224))
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        x = Variable(image.cuda())
        extractor = vgg_extractor(self.use_cuda) if self.use_vgg else inception_extractor(self.use_cuda)
        x = extractor(x)
        return x
