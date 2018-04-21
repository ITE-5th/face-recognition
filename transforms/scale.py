import numpy as np

from misc.utils import Utils
from transforms.base_transform import BaseTransform


class Scale(BaseTransform):
    """
    Scale Bounding Boxes
    """
    def __init__(self, scale) -> None:
        """
        :param scale: percent of width and height to add to the bounding box
        """
        super().__init__()
        self.scale = scale
        assert self.scale >= 0
        self.scale /= 2
        self.image = None

    def scale_bbox(self, bbox):
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        new_x1 = x1 - self.scale * width
        new_y1 = y1 - self.scale * height
        new_x2 = x2 + self.scale * width
        new_y2 = y2 + self.scale * height

        image_width = self.image.shape[1]
        image_height = self.image.shape[0]

        new_x1 = np.clip(new_x1, a_min=0, a_max=image_width)
        new_y1 = np.clip(new_y1, a_min=0, a_max=image_height)
        new_x2 = np.clip(new_x2, a_min=0, a_max=image_width)
        new_y2 = np.clip(new_y2, a_min=0, a_max=image_height)

        return np.array([int(new_x1), int(new_y1), int(new_x2), int(new_y2)])

    def forward(self, inputs):
        if self.scale <= 0:
            return inputs

        bboxes, self.image = inputs
        return Utils.map(self.scale_bbox, bboxes), self.image
