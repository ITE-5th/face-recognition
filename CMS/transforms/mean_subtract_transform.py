import numpy as np


class MeanSubtract(object):
    pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

    def __init__(self, pixel_means=None):
        super().__init__()

        if pixel_means is not None:
            self.pixel_means = pixel_means

    def __call__(self, sample):
        img = sample['image']
        img = img - self.pixel_means

        sample['image'] = img
        return sample
