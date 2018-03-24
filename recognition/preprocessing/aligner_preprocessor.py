import glob
import os

import cv2
import dlib
import openface

from file_path_manager import FilePathManager


class AlignerPreprocessor:
    path_to_landmarks_model = FilePathManager.resolve("data/shape_predictor_68_face_landmarks.dat")
    detector = dlib.cnn_face_detection_model_v1(FilePathManager.resolve("data/mmod_human_face_detector.dat"))
    predictor = dlib.shape_predictor(path_to_landmarks_model)
    aligner = openface.AlignDlib(path_to_landmarks_model)

    def __init__(self, scale: int = 1, size=224):
        self.scale = scale
        self.size = size

    def preprocess_image(self, image):
        items = AlignerPreprocessor.detector(image, self.scale)
        result = []
        for item in items:
            rect = item.rect
            aligned = AlignerPreprocessor.aligner.align(self.size, image, rect,
                                                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            result.append((aligned, rect))
        return result

    def preprocess_face_from_path(self, path):
        image = cv2.imread(path)
        return self.preprocess_face_from_image(image)

    def preprocess_face_from_image(self, image):
        rect = AlignerPreprocessor.detector(image, self.scale)[0].rect
        aligned = AlignerPreprocessor.aligner.align(self.size, image, rect,
                                                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return aligned

    def process_faces(self, faces):
        for face in faces:
            self.process_face(face)

    def process_face(self, face):
        ind = face.index("custom_images") + (len("custom_images"))
        try:
            aligned = self.preprocess_face_from_path(face)
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
    faces = sorted(glob.glob(FilePathManager.resolve("data/custom_images/**/*")))
    p = AlignerPreprocessor(scale=1)
    p.process_faces(faces)
