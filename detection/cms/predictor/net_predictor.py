import numpy as np
from cv2 import cv2

from detection.cms.estimator.minibatch_cms import CMSRCNN


class NetPredictor:
    def __init__(self, model_path: str = None, use_cuda: bool = True):
        super().__init__()

        net = CMSRCNN()
        net.load_pretrained(model_path)
        if use_cuda:
            net = net.cuda()
        net.eval()
        self.net = net

    def predict(self, image_path: str):
        '''
        predict the output of image and show it
        :param image_path: path to image
        :return: bounding boxes of faces
        '''

        image = cv2.imread(image_path)

        dets, scores, classes = self.net.detect(image, 0.7)

        im2show = np.copy(image)
        for i, det in enumerate(dets):
            det = tuple(int(x) for x in det)
            cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
            cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)

        cv2.imshow('output', im2show)
        cv2.waitKey(0)

        return dets
