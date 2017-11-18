from abc import ABCMeta, abstractmethod


class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, image):
        raise NotImplementedError()
