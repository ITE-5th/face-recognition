import cv2

from misc.utils import Utils
from transforms.base_transform import BaseTransform


class Resize(BaseTransform):

    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def resize(self, img):
        return cv2.resize(img, (self.size, self.size))

    def forward(self, inputs):
        faces, image = inputs
        return Utils.map(self.resize, faces), image
