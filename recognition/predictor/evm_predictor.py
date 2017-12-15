from recognition.estimator.evm import load
from recognition.predictor.predictor import Predictor


class EvmPredictor(Predictor):

    UNKNOWN = "Unknown"

    def __init__(self, evm_model_path: str, use_custom: bool = True, use_cuda: bool = True):
        super().__init__(use_custom, use_cuda)
        self.evm = load(evm_model_path)

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            x = face.data.cpu().numpy().reshape(1, -1)
            predicted = self.evm.predict(x)
            predicted = predicted[0]
            result.append((self.names[predicted] if predicted != -1 else EvmPredictor.UNKNOWN, rect))
        return result
