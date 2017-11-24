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
        sample = {'image': image, 'bboxes': [[int(coord) for coord in bbox] for bbox in bboxes]}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    fdd = FaceDetectionDataset('../data/wider_face_split/wider_face_train_bbx_gt.json', '../data/WIDER_train/images/')

    row = fdd.__getitem__(3)
    image = row['image'].numpy()
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)

    im2show = np.copy(image)
    for i, det in enumerate(row['bboxes']):
        # for j, a in enumerate(det):
        # det[2] += det[0]
        # det[3] += det[1]
        # det = list(map(int, det))
        det[2] += det[0]
        det[3] += det[1]
        det = tuple(int(x) for x in det[:4])
        print(det)

        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        # cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
        #             1.0, (0, 0, 255), thickness=1)

    cv2.imshow('image', im2show)
    cv2.waitKey(0)
