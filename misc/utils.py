from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import cv2
import dlib
import dlt
import numpy as np
import torch


class Utils(object):

    @staticmethod
    def rect2points(rect):
        """
        transform dlib.rectangle to numpy array of points
        :param rect: dlib.rectangle
        :return: numpy array of points [ first_point, second_point ]
        """

        y1, y2 = rect.top(), rect.bottom()
        x1, x2 = rect.left(), rect.right()

        return np.array([x1, y1, x2, y2])

    @staticmethod
    def points2rect(points):
        """
        transform numpy rect to dlib.rectangle
        :param points: rect points
        :return: dlib.rectangle
        """
        x1, y1, x2, y2 = points
        return dlib.rectangle(x1, y1, x2, y2)

    @staticmethod
    def points2rects(arr):
        return np.array(Utils.map(Utils.points2rect, arr))

    @staticmethod
    def rects2points(rectangles):
        return np.array(Utils.map(Utils.rect2points, rectangles))

    @staticmethod
    def cv2torch(img):
        result = Utils.map(dlt.util.cv2torch, img)

        return torch.stack(result)

    @staticmethod
    def torch2cv(tensors):
        return Utils.map(dlt.util.torch2cv, tensors)

    @staticmethod
    def map(func, inputs):
        with Pool(cpu_count()) as pool:
            result = pool.map(func, inputs)

        return result

    @staticmethod
    def to_module(state_dict):
        new_state_dict = dict()
        for key in state_dict.keys():
            new_name = key[key.index(".") + 1:]
            new_state_dict[new_name] = state_dict[key]
        return new_state_dict

    @staticmethod
    def to_gray(images):
        temp = []
        for img in images:
            temp.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        return np.asarray(temp)
