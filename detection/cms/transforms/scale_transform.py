import cv2
import numpy as np


class Scale(object):
    sizes = (800, 600)

    def __init__(self, sizes=None, include_scale: bool = False):
        super().__init__()
        if sizes is not None:
            self.sizes = sizes

        self.include_scale = include_scale

    def __call__(self, sample):
        if sample is None:
            return

        img = sample['image']

        h, w, _ = img.shape

        im_scale = 0
        if self.include_scale:
            im_scale = float(self.sizes[1]) / float(h)

        im = cv2.resize(img, dsize=self.sizes, interpolation=cv2.INTER_LINEAR)

        # sample['image'] = np.array((len(self.scales)))
        # sample['image'] = blob
        sample['image'] = im.astype(np.float32)
        sample['scale'] = np.array([self.sizes[0], self.sizes[1], im_scale], dtype=np.float32)

        if 'bboxes' in sample.keys() and sample['bboxes'] is not None:
            sample = self.scale_bboxes(sample, w, h)

        return sample

    def scale_bboxes(self, sample, oldW, oldH):
        bboxes = sample['bboxes']

        w, h = self.sizes

        width_scale = w / oldW
        height_scale = h / oldH

        bboxes[:, 0::2] = (bboxes[:, 0::2] * width_scale).astype(np.int)
        bboxes[:, 1::2] = (bboxes[:, 1::2] * height_scale).astype(np.int)
        bboxes[:, 4] = 1

        sample['bboxes'] = bboxes
        return sample
