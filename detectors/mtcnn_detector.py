import glob

import cv2
import mxnet as mx

from bases.pipeline import Pipeline
from detectors.base_detector import BaseDetector
from extractors.models.insightface.mtcnn_detector import MtcnnDetector
from file_path_manager import FilePathManager
from transforms.crop import Crop


class MTCNNDetector(BaseDetector):

    def __init__(self, det=2, use_gpu=False) -> None:
        super().__init__()

        self.det = det
        ctx = mx.gpu() if use_gpu else mx.cpu()

        mtcnn_path = FilePathManager.resolve("data/mtcnn-model")
        self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=4, accurate_landmark=True,
                                      threshold=[0.0, 0.0, 0.2])

    def forward(self, image):
        # face_img is bgr image
        ret = self.detector.detect_face_limited(image, det_type=self.det)
        return ret, image

    def postprocess(self, inputs):
        ret, image = inputs
        if ret is None:
            return None, image

        bbox, points = ret
        if bbox.shape[0] == 0:
            return None, image
        bbox = bbox[:, 0:4]
        return bbox, image


if __name__ == '__main__':

    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("images/*")))

    pipeline = Pipeline([MTCNNDetector(), Crop()])

    for i, face in enumerate(faces):
        face = cv2.imread(face)

        cropped_output, _ = pipeline(face)
        if cropped_output is None:
            continue
        for j, cropped_image in enumerate(cropped_output):
            cv2.imwrite(path + "/test{}-{}.jpeg".format(i, j), cropped_image)
