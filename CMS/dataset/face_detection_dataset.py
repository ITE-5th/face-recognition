import json
import os

import numpy as np
import torch
from cv2 import cv2
from torch.utils.data import Dataset


class FaceDetectionDataset(Dataset):
    """Face Detection dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        :param json_file (string): Path to the json file with bounding boxes.
        :param root_dir (string): Directory with all images.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.bboxes = json.load(open(json_file))
        self.names = list(self.bboxes.keys())

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.names[index])
        image = cv2.imread(img_name)

        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        image = torch.from_numpy(image)

        bboxes = self.bboxes[self.names[index]]
        sample = {'image': image, 'bboxes': bboxes}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    fdd = FaceDetectionDataset('../data/wider_face_split/wider_face_train_bbx_gt.json', '../data/WIDER_train/images/')

    image = fdd.__getitem__(0)['image'].numpy()
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)

    cv2.imshow('image', image)
    cv2.waitKey(0)
