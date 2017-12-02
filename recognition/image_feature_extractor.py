import os

import cv2
import numpy as np
import torch


class ImageFeatureExtractor:
    @staticmethod
    def extract(root_dir: str):
        names = os.listdir(root_dir + "/lfw2")
        if not os.path.exists(root_dir + "/lfw_features"):
            os.makedirs(root_dir + "/lfw_features")
        for name in names:
            path = root_dir + "/lfw2/" + name
            if not os.path.exists(root_dir + "/lfw_features/" + name):
                os.makedirs(root_dir + "/lfw_features/" + name)
            faces = os.listdir(path)
            for face in faces:
                p = path + "/" + face
                image = cv2.imread(p)
                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 1, 2)
                res = torch.from_numpy(image)
                torch.save(res, root_dir + "/lfw_features/" + name + "/" + face[:face.rfind(".")] + ".features")


if __name__ == '__main__':
    ImageFeatureExtractor.extract("../data")
