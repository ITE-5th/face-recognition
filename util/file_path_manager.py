import dlib


class FilePathManager:
    root_path = "/home/obada/PycharmProjects/face-recognition/"

    @staticmethod
    def load_path(path: str):
        return FilePathManager.root_path + path

    @staticmethod
    def load_detection_model():
        path_to_cnn_model = FilePathManager.load_path("data/cmsrcnn_face_detector")
        return dlib.cnn_face_detection_model_v1(path_to_cnn_model)