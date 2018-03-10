from joblib import load
from sklearn.ensemble import RandomForestClassifier

from recognition.predictor.predictor import Predictor
from util.file_path_manager import FilePathManager


class RandomForestPredictor(Predictor):
    UNKNOWN = "Unknown"

    def __init__(self, random_forest_model_path: str, use_custom: bool = True, use_cuda: bool = True, scale=0):
        super().__init__(use_custom, use_cuda, scale)
        self.rf = load(random_forest_model_path)
        RandomForestClassifier().predict_proba()

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            x = face.data.cpu().numpy().reshape(1, -1)
            predicted = self.rf.predict(x)
            clz, prop = predicted, 1
            result.append((self.names[int(clz)], rect, prop))
        return result


if __name__ == '__main__':
    path = FilePathManager.load_path("models/evm/evm.model")
    evm = EvmPredictor(path)
    evm.predict_from_path(FilePathManager.load_path("test_images/image_1.jpg"))
