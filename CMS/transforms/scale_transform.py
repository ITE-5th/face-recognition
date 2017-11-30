import cv2
import numpy as np


class Scale(object):
    sizes = (800, 600)

    def __init__(self, sizes=None):
        super().__init__()
        if sizes is not None:
            self.sizes = sizes

    def __call__(self, sample):
        if sample is None:
            return

        img = sample['image']

        w, h, _ = img.shape

        im_scale = float(self.sizes[1]) / float(h)

        im = cv2.resize(img, dsize=self.sizes, interpolation=cv2.INTER_LINEAR)

        # sample['image'] = np.array((len(self.scales)))
        # sample['image'] = blob
        sample['image'] = im.astype(np.float32)
        sample['scale'] = np.array([self.sizes[0], self.sizes[1], im_scale], dtype=np.float32)
        sample = self.scale_bboxes(sample, w, h)

        return sample

    @staticmethod
    def im_list_to_blob(ims):
        """Convert a list of images into a network input.

        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob

    def scale_bboxes(self, sample, oldW, oldH):
        bboxes = sample['bboxes']

        w, h = self.sizes

        width_scale = w / oldW
        height_scale = h / oldH

        bboxes[:, 0::2] = (bboxes[:, 0::2] * width_scale).astype(np.int)
        bboxes[:, 1::2] = (bboxes[:, 1::2] * height_scale).astype(np.int)
        bboxes[:, 4] = 1

        # bboxes[:, 0:4] = bboxes[:, 0:4] * scale[0, 2]

        sample['bboxes'] = bboxes
        return sample
