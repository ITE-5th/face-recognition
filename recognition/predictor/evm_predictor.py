import joblib
from recognition.predictor.predictor import Predictor
from file_path_manager import FilePathManager


class EvmPredictor(Predictor):

    def __init__(self, evm_model_path: str, use_cuda: bool = True, scale=1):
        super().__init__(use_cuda=use_cuda, scale=scale)
        self.evm = joblib.load(evm_model_path)

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            x = face.data.cpu().numpy().reshape(1, -1)
            predicted = self.evm.predict_with_prop(x)
            clz, prop = predicted[0]
            result.append((clz, rect, prop))
        return result


if __name__ == '__main__':
    path = FilePathManager.resolve("models/evm/evm.model")
    evm = EvmPredictor(path)
    evm.predict_from_path(FilePathManager.resolve("test_images/image_1.jpg"))
