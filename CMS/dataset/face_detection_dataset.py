import json
import os

import numpy as np
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
        self.max_bboxes = 500
        self.root_dir = root_dir
        self.transform = transform
        self.bboxes = json.load(open(json_file))
        self.names = list(self.bboxes.keys())

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.names[index])
        image = cv2.imread(img_name).astype(np.float)

        # image = np.swapaxes(image, 0, 2)
        # image = np.swapaxes(image, 1, 2)
        # image = network.np_to_variable(image, is_cuda=True)
        # image = torch.FloatTensor(image)

        actualBBoxes = self.bboxes[self.names[index]]
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
