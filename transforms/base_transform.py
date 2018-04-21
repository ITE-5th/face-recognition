from abc import abstractmethod

from bases.stage import Stage


class BaseTransform(Stage):
    def preprocess(self, inputs):
        return inputs

    @abstractmethod
    def forward(self, inputs):
        pass

    def postprocess(self, inputs):
        return inputs
