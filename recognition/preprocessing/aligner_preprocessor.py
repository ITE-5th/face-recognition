import glob
import os
from multiprocessing import Pool, cpu_count

import cv2
import dlib
import openface

from recognition.preprocessing.preprocessor import Preprocessor
from file_path_manager import FilePathManager


class AlignerPreprocessor(Preprocessor):
    path_to_pretrained_model = FilePathManager.load_path("data/shape_predictor_68_face_landmarks.dat")
    path_to_cnn_model = FilePathManager.load_path("data/mmod_human_face_detector.dat")
    detector = dlib.cnn_face_detection_model_v1(path_to_cnn_model)
    predictor = dlib.shape_predictor(path_to_pretrained_model)
    aligner = openface.AlignDlib(path_to_pretrained_model)

    def __init__(self):
        self.lfw = None

    def preprocess(self, image):
        rect = AlignerPreprocessor.detector(image, 2)[0].rect
        aligned = AlignerPreprocessor.aligner.align(299, image, rect,
                                                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return aligned

    def preprocess_faces(self, faces, lfw: bool = False):
        self.lfw = lfw
        with Pool(cpu_count()) as p:
            p.map(self.process_face, faces)
            p.close()
            p.join()

    def process_face(self, face):
        lfw = self.lfw
        try:
            image = cv2.imread(face)
            rect = AlignerPreprocessor.detector(image, 1)[0].rect
            aligned = AlignerPreprocessor.aligner.align(299, image, rect,
                                                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            ind = face.index("lfw" if lfw else "custom_images") + (len("lfw") if lfw else len("custom_images")) - 1
            temp = face[:ind] + "2" + face[ind:]
            dirname = temp[:temp.rfind("/")]
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            cv2.imwrite(temp, aligned)
            print("correct : {}".format(face))

        except:
            print("wrong : {}".format(face))


if __name__ == '__main__':
    faces = glob.glob(FilePathManager.load_path("data/custom_images/**/*"))
    p = AlignerPreprocessor()
    p.preprocess_faces(faces)
