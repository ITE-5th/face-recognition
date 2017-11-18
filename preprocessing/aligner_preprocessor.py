import dlib
import numpy as np
import openface

from preprocessing.preprocessor import Preprocessor


class AlignerPreprocessor(Preprocessor):
    path_to_pretrained_model = "../data/shape_predictor_68_face_landmarks.dat"
    path_to_cnn_model = "../data/mmod_human_face_detector.dat"
    detector = dlib.cnn_face_detection_model_v1(path_to_cnn_model)
    predictor = dlib.shape_predictor(path_to_pretrained_model)
    aligner = openface.AlignDlib(path_to_pretrained_model)

    def preprocess(self, image):
        rect = AlignerPreprocessor.detector(image, 1)[0].rect
        aligned = AlignerPreprocessor.aligner.align(299, image, rect,
                                                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        aligned = np.swapaxes(aligned, 0, 2)
        aligned = np.swapaxes(aligned, 1, 2)
        return aligned
