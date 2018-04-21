import glob

import cv2
import numpy as np

from aligners.no_aligner import NoAligner
from bases.pipeline import Pipeline
from detectors.dlib_detector import DLibDetector
from extractors.base_extractor import BaseExtractor
from extractors.models.insightface import face_preprocess
from extractors.models.insightface.face_embedding import FaceModel
from file_path_manager import FilePathManager


class ArcFaceExtractor(BaseExtractor):
    image_size = [112, 112]
    insight_extractor = FaceModel(threshold=1.24, det=2, image_size=image_size,
                                  model=FilePathManager.resolve(
                                      "data/model-r50-am-lfw/model,0"))

    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, inputs):
        faces, image = inputs
        result = []
        for face in faces:
            result.append(self.preprocess_face(face))

        # TODO: map doesn't work with mxnet
        # result = Utils.map(self.preprocess_face, faces)
        return result, image

    def preprocess_face(self, face):
        face = cv2.resize(face, (112, 112))
        face = face_preprocess.preprocess(face, image_size=self.image_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1))

        return face

    def forward(self, inputs):
        faces, image = inputs
        return self.insight_extractor.get_feature(faces), image

    def postprocess(self, inputs):
        features, image = inputs
        return np.array(features), image


if __name__ == '__main__':
    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("images/*")))

    # pipeline = Pipeline([DLibDetector(), Scale(5), Crop(), ArcFaceExtractor()])
    # pipeline = Pipeline([DLibDetector(), OneMillisecondAligner(size=112), ArcFaceExtractor()])
    pipeline = Pipeline([DLibDetector(), NoAligner(scale=3), ArcFaceExtractor()])

    for i, face in enumerate(faces):
        face = cv2.imread(face)
        features, _ = pipeline(face)
        if features is None:
            continue
        print("{} image: #{} Features.".format(i, features.shape))
