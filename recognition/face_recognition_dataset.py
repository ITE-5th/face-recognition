import os

import cv2
import dlib
import numpy as np
import openface
import torch
from torch.utils.data.dataset import Dataset


class FaceRecognitionDataset(Dataset):
    path_to_pretrained_model = "models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_to_pretrained_model)
    aligner = openface.AlignDlib(path_to_pretrained_model)

    def __init__(self, root_path):
        self.root_path = root_path
        paths = [os.path.join(root_path, path) for path in os.listdir(root_path)]
        self.names = [path[path.rfind("/") + 1:] for path in paths]
        temp = [[os.path.join(path, t) for t in os.listdir(path)] for path in paths]
        self.faces = [item for sublist in temp for item in sublist]

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        face = self.faces[index]
        name = face[face.rfind("/") + 1:]
        name = name[:name.rfind("_")]
        print(face)
        image = cv2.imread(face)
        rect = FaceRecognitionDataset.detector(image, 1)[0]
        aligned = FaceRecognitionDataset.aligner.align(299, image, rect)
        aligned = np.swapaxes(aligned, 0, 2)
        aligned = np.swapaxes(aligned, 1, 2)
        x = torch.from_numpy(aligned)
        # y = torch.zeros(len(self.names))
        # y[self.names.index(name)] = 1
        # return x.float(), y.long()
        return x.float(), self.names.index(name)
