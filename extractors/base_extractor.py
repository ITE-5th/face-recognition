from abc import abstractmethod

from bases.stage import Stage
from transforms.resize import Resize


class BaseExtractor(Stage):

    def __init__(self):
        self.resize = Resize(100)

    @abstractmethod
    def preprocess(self, inputs):
        pass

    @abstractmethod
    def forward(self, inputs):
        """
        Takes Image as input and returns features as output
        Must resize the input to the required input size.
        :param inputs: image
        :return: features as numpy array
        """
        pass

    def postprocess(self, inputs):
        return inputs
