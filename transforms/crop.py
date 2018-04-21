from multiprocessing import cpu_count
from multiprocessing.pool import Pool

from transforms.base_transform import BaseTransform


class Crop(BaseTransform):
    """
    Crop Image into corresponding bounding boxes
    """

    def __init__(self) -> None:
        super().__init__()
        self.image = None

    def forward(self, inputs):
        rects, self.image = inputs
        with Pool(cpu_count()) as pool:
            cropped_images = pool.map(self.crop, rects)

        return cropped_images, self.image

    def crop(self, rect):
        x1, y1, x2, y2 = rect
        return self.image[int(y1):int(y2), int(x1):int(x2)]
