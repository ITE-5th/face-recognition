import os


class FilePathManager:
    root_path = os.path.dirname(os.path.abspath(__file__)) + "/"

    @staticmethod
    def resolve(path: str):
        return FilePathManager.root_path + path
