import glob

import cv2

from aligners.base_aligner import BaseAligner
from bases.pipeline import Pipeline
from detectors.dlib_detector import DLibDetector
from file_path_manager import FilePathManager
from transforms.crop import Crop
from transforms.scale import Scale


class NoAligner(BaseAligner):
    def __init__(self, scale: float = 0.0):
        """
        :param scale:  between 0 (0%) and 1 (100%)
        """
        self.scale = scale
        self.pipeline = Pipeline([Scale(scale), Crop()])

    def forward(self, inputs):
        return self.pipeline(inputs)


if __name__ == '__main__':
    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("images/*")))

    pipeline = Pipeline([DLibDetector(), NoAligner(scale=1.5)])

    for i, face in enumerate(faces):
        face = cv2.imread(face)

        cropped_output, image = pipeline(face, True)
        for j, cropped_image in enumerate(cropped_output):
            cv2.imwrite(path + "/test{}-{}.jpeg".format(i, j), cropped_image)
