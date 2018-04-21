import glob

import cv2
import dlib
import face_recognition
import numpy as np
from face_recognition.api import _rect_to_css

from aligners.no_aligner import NoAligner
from bases.pipeline import Pipeline
from detectors.dlib_detector import DLibDetector
from extractors.base_extractor import BaseExtractor
from file_path_manager import FilePathManager
from misc.utils import Utils


class DLibExtractor(BaseExtractor):

    def __init__(self):
        super().__init__()

    def preprocess(self, inputs):
        return inputs

    def forward(self, inputs):
        faces, image = inputs
        rects = dlib.rectangles(Utils.points2rects(faces))
        csses = []
        for rect in rects:
            csses.append(_rect_to_css(rect))
        result = face_recognition.face_encodings(image, csses)
        return np.array(result), image


if __name__ == '__main__':
    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("images/*")))

    # pipeline = Pipeline([DLibDetector(), OneMillisecondAligner(), VggExtractor()])
    pipeline = Pipeline([DLibDetector(), NoAligner(scale=0), DLibExtractor()])
    # pipeline = Pipeline([DLibDetector(), Crop(), Resize(224), VggExtractor()])

    for i, face in enumerate(faces):
        face = cv2.imread(face)

        features, _ = pipeline(face, True)
        print("{} image: #{} Features.".format(i, features.shape))
