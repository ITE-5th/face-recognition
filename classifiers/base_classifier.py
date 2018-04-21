from abc import abstractmethod

from bases.stage import Stage


class BaseClassifier(Stage):
    @abstractmethod
    def preprocess(self, inputs):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def postprocess(self, inputs):
        pass
