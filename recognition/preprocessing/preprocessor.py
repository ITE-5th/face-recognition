from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torch


def to_tensor(image_path):
    image = cv2.imread(image_path)
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return torch.from_numpy(image).float()


class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, image):
        raise NotImplementedError()
