import glob
import os
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from preprocessing.preprocessor import Preprocessor


class FaceRecognitionDataset(Dataset):
    def __init__(self, root_path: str, preprocessor: Preprocessor = None):
        self.preprocessor = preprocessor
        self.names = [path for path in os.listdir(root_path)]
        faces_path = glob.glob(root_path + "/**/*.jpg")
        with Pool(cpu_count()) as p:
            self.faces = p.map(self.extract_item, faces_path)
            p.close()
            p.join()

    # def __init__(self):
    #     self.preprocessor = None
    #     self.names = ["Aaron_Eckhart", "Aaron_Guiel", "Abdullah", "Abudl_Rahman"]
    #     faces_path = glob.glob("../data/lfw2/**/*.jpg")
    #     faces_path = [path for path in faces_path if self.extract_name(path) in self.names]
    #     # self.faces = [self.extract_item(item) for item in faces_path]
    #     with Pool(cpu_count()) as p:
    #         self.faces = p.map(self.extract_item, faces_path)
    #         p.close()
    #         p.join()

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
        if self.preprocessor is None:
            image = np.swapaxes(image, 0, 2)
            image = np.swapaxes(image, 1, 2)
            res = torch.from_numpy(image)
            return res
        aligned = self.preprocessor.preprocess(image)
        return torch.from_numpy(aligned)
