from recognition.estimator.evm import load
from recognition.predictor.predictor import Predictor
from util.file_path_manager import FilePathManager


class EvmPredictor(Predictor):
    UNKNOWN = "Unknown"

    def __init__(self, evm_model_path: str, use_custom: bool = True, use_cuda: bool = True, scale=0):
        super().__init__(use_custom, use_cuda, scale)
        self.evm = load(evm_model_path)

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            x = face.data.cpu().numpy().reshape(1, -1)
            predicted = self.evm.predict_with_prop(x)
            clz, prop = predicted[0]
            result.append((self.names[int(clz)] if clz != -1 else EvmPredictor.UNKNOWN, rect, prop))
        return result


if __name__ == '__main__':
    path = FilePathManager.load_path("models/evm/evm.model")
    evm = EvmPredictor(path)
    evm.predict_from_path(FilePathManager.load_path("test_images/image_1.jpg"))
