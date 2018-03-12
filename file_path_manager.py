import os
import dlib


class FilePathManager:
    root_path = os.path.dirname(os.path.abspath(__file__)) + "/"

    @staticmethod
    def load_path(path: str):
        return FilePathManager.root_path + path

    @staticmethod
    def load_detection_model():
        path_to_cnn_model = FilePathManager.load_path("data/mmod_human_face_detector.dat")
        return dlib.cnn_face_detection_model_v1(path_to_cnn_model)
