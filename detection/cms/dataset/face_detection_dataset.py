import json
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from cv2 import cv2
from torch.utils.data import Dataset


class FaceDetectionDataset(Dataset):
    """Face Detection dataset."""

    def __init__(self, json_file, root_dir, transform=None, max_bboxes=900):
        """
        :param json_file (string): Path to the json file with bounding boxes.
        :param root_dir (string): Directory with all images.
        :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.max_bboxes = max_bboxes
        self.root_dir = root_dir
        self.transform = transform
        self.bboxes = json.load(open(json_file))
        self.names = list(self.bboxes.keys())
        with Pool(cpu_count()) as p:
            self.samples = p.map(self.process_bbox, self.bboxes.items())
            p.close()
            p.join()
        self.length = len(self.bboxes)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.samples[index]

    def process_bbox(self, item):
        name, val = item
        img_name = os.path.join(self.root_dir, name)
        image = cv2.imread(img_name).astype(np.float32)

        actualBBoxes = self.bboxes[name]
        bboxes_count = len(actualBBoxes)

        bboxes = np.array(actualBBoxes, dtype=np.float)[:, :5]
        try:
            padding = np.ones((self.max_bboxes - bboxes_count, 5)) * -1
        except:
            print(bboxes_count)

        bboxes = np.concatenate((bboxes, padding), axis=0)

        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        bboxes[:, 4] = 1

        sample = {'image': image, 'bboxes': bboxes, 'bboxes_count': bboxes_count}

        if self.transform:
            self.transform(sample)

        return sample

    @staticmethod
    def remove_padded_bboxes(sample):
        gt_bboxes = sample["bboxes"].numpy()
        bboxes_count = sample["bboxes_count"].numpy()

        # remove padded bboxes
        temp = []
        for i in range(len(gt_bboxes)):
            temp.append(gt_bboxes[i, :bboxes_count[i]])

        return temp


if __name__ == "__main__":
    fdd = FaceDetectionDataset('../data/wider_face_split/wider_face_train_bbx_gt.json', '../data/WIDER_train/images/')

    row = fdd.__getitem__(3)
    image = row['image'].numpy()
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 0, 1)

    im2show = np.copy(image)
    for i, det in enumerate(row['bboxes']):
        det[2:4] += det[:2]
        # det = tuple(int(x) for x in det[:4])
        det = tuple(det[:4])
        print(det)

        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        # cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
        #             1.0, (0, 0, 255), thickness=1)

    cv2.imshow('image', im2show)
    cv2.waitKey(0)
