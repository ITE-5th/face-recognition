import os

from file_path_manager import FilePathManager
from recognition.estimator.evm import EVM
from recognition.predictor.predictor import Predictor


class EvmPredictor(Predictor):

    UNKNOWN = "unknown"

    def __init__(self, evm_model_path: str, use_custom: bool = True, use_vgg: bool = True, use_cuda: bool = True):
        super().__init__(use_custom, use_vgg, use_cuda)
        self.evm = EVM.load(evm_model_path)

    def predict(self, image_path: str):
        x = super().predict(image_path)
        x = x.data.cpu().numpy()
        x = x.reshape(1, -1)
        predicted = self.evm.predict(x)
        predicted = predicted[0]
        os.system("rm {}".format(FilePathManager.load_path("temp.jpg")))
        return self.names[predicted] if predicted != -1 else EvmPredictor.UNKNOWN
