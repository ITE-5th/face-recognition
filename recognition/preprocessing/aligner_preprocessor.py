import glob
import os

import cv2
import dlib
import openface

from recognition.preprocessing.preprocessor import Preprocessor
from file_path_manager import FilePathManager


class AlignerPreprocessor(Preprocessor):
    path_to_pretrained_model = FilePathManager.load_path("data/shape_predictor_68_face_landmarks.dat")
    path_to_cnn_model = FilePathManager.load_path("data/cmsrcnn_face_detector")
    detector = FilePathManager.load_detection_model()
    predictor = dlib.shape_predictor(path_to_pretrained_model)
    aligner = openface.AlignDlib(path_to_pretrained_model)

    def __init__(self, scale: int = 1, size=200):
        self.lfw = None
        self.scale = scale
        self.size = size

    def preprocess(self, image):
        items = AlignerPreprocessor.detector(image, self.scale)
        result = []
        for item in items:
            rect = item.rect
            aligned = AlignerPreprocessor.aligner.align(self.size, image, rect,
                                                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            cv2.imwrite(FilePathManager.load_path("test.jpg"), aligned)
            result.append((aligned, rect))
        return result

    def preprocess_faces(self, faces, lfw: bool = False):
        self.lfw = lfw
        for face in faces:
            self.process_face(face)

    def preprocess_face(self, image):
        image = cv2.imread(image)
        rect = AlignerPreprocessor.detector(image, self.scale)[0].rect
        temp = AlignerPreprocessor.aligner.align(self.size, image, rect,
                                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return temp

    def process_face(self, face):
        ind = face.index("lfw" if self.lfw else "custom_images") + (len("lfw") if self.lfw else len("custom_images"))
        try:
            aligned = self.preprocess_face(face)
            temp = face[:ind] + "2" + face[ind:]
            dirname = temp[:temp.rfind("/")]
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            cv2.imwrite(temp, aligned)
            del aligned
            print("correct : {}".format(face))

        except Exception as e:
            print("wrong : {}".format(face))
            print(e)


if __name__ == '__main__':
    faces = sorted(glob.glob(FilePathManager.load_path("data/custom_images/**/*")))
    p = AlignerPreprocessor(scale=1)
    p.preprocess_faces(faces)
