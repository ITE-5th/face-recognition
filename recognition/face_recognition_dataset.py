import os

import cv2
import dlib
import numpy as np
import openface
import torch
import torchvision.models as models
from torch.utils.data.dataset import Dataset
import glob


class FaceRecognitionDataset(Dataset):
    path_to_pretrained_model = "../data/shape_predictor_68_face_landmarks.dat"
    path_to_cnn_model = "../data/mmod_human_face_detector.dat"
    detector = dlib.cnn_face_detection_model_v1(path_to_cnn_model)
    predictor = dlib.shape_predictor(path_to_pretrained_model)
    aligner = openface.AlignDlib(path_to_pretrained_model)

    def __init__(self, root_path: str):
        self.names = [path for path in os.listdir(root_path)]
        faces_path = glob.glob(root_path + "/**/*.jpg")
        self.faces = [(self.names.index(self.extract_name(face_path)), self.prepare_image(face_path)) for face_path in
                      faces_path]
        print(self.faces)

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        face = self.faces[index]
        return face[1], face[0]

    def extract_name(self, path: str):
        return path[path.rfind("/") + 1:path.rfind("_")]

    def prepare_image(self, image_path: str):
        image = cv2.imread(image_path)
        rect = FaceRecognitionDataset.detector(image, 1)[0]
        # aligned = FaceRecognitionDataset.aligner.align(299, image, rect,
        #                                                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        # aligned = np.swapaxes(aligned, 0, 2)
        # aligned = np.swapaxes(aligned, 1, 2)
        # return torch.from_numpy(aligned)
        return []
