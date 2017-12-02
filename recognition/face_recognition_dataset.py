import glob
import os
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from preprocessing.preprocessor import Preprocessor


class FaceRecognitionDataset(Dataset):
    def __init__(self, root_path: str, preprocessor: Preprocessor = None, vgg_face : bool=True):
        self.vgg_face = vgg_face
        self.preprocessor = preprocessor
        self.names = list(set([path for path in os.listdir(root_path)]))
        faces_path = glob.glob(root_path + "/**/*.jpg")
        # with Pool(cpu_count()) as p:
        #     self.faces = p.map(self.extract_item, faces_path)
        #     p.close()
        #     p.join()
        self.faces = []
        for face in faces_path:
            self.faces.append(self.extract_item(face))

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        return self.faces[index]

    def name_from_index(self, index: int):
        return self.names[index]

    def extract_item(self, item):
        name = self.extract_name(item)
        ind = self.names.index(name)
        image = self.prepare_image(item)
        tu = image.float(), ind
        return tu

    def extract_name(self, path: str):
        return path[path.rfind("/") + 1:path.rfind("_")]

    def prepare_image(self, image_path: str):
        image = cv2.imread(image_path)
        if self.vgg_face:
            image = cv2.resize(image, (224, 224))
        if self.preprocessor is None:
            image = np.swapaxes(image, 0, 2)
            image = np.swapaxes(image, 1, 2)
            res = torch.from_numpy(image)
            return res
        aligned = self.preprocessor.preprocess(image)
        return torch.from_numpy(aligned)
