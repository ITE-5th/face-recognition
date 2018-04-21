import glob

import cv2

from aligners.base_aligner import BaseAligner
from bases.pipeline import Pipeline
from detectors.dlib_detector import DLibDetector
from file_path_manager import FilePathManager


class PassAligner(BaseAligner):
    def forward(self, inputs):
        return inputs


if __name__ == '__main__':
    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("images/*")))

    pipeline = Pipeline([DLibDetector(), PassAligner()])

    for i, face in enumerate(faces):
        face = cv2.imread(face)

        cropped_output, image = pipeline(face, True)
        for j, cropped_image in enumerate(cropped_output):
            cv2.imwrite(path + "/test{}-{}.jpeg".format(i, j), cropped_image)
