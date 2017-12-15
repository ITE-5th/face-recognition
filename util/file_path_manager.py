class FilePathManager:
    root_path = "/home/obada/PycharmProjects/face-recognition/"

    @staticmethod
    def load_path(path: str):
        return FilePathManager.root_path + path
