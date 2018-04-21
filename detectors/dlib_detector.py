import glob

import cv2
import dlib

from bases.pipeline import Pipeline
from detectors.base_detector import BaseDetector
from file_path_manager import FilePathManager
from misc.utils import Utils
from transforms.crop import Crop
from transforms.scale import Scale


class DLibDetector(BaseDetector):

    def __init__(self, scale=1, use_cnn=False) -> None:
        super().__init__()
        self.scale = scale
        self.use_cnn = use_cnn
        self.detector = dlib.get_frontal_face_detector() if not use_cnn else dlib.cnn_face_detection_model_v1(
            FilePathManager.resolve("data/mmod_human_face_detector.dat"))

    def forward(self, image):
        temp = self.detector(image, self.scale)
        items = temp if not self.use_cnn else [item.rect for item in temp]
        return items, image

    def postprocess(self, inputs):
        items, image = inputs
        return Utils.rects2points(items), image


if __name__ == '__main__':

    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("faces/*/*")))

    pipeline = Pipeline([DLibDetector(scale=1), Scale(0.2), Crop()])

    for i, face in enumerate(faces):
        print(face)
        face = cv2.imread(face)
        cropped_output, _ = pipeline(face)
        for j, cropped_image in enumerate(cropped_output):
            cv2.imwrite(path + "/test{}-{}.jpeg".format(i, j), cropped_image)
