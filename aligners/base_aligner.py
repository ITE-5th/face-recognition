from abc import abstractmethod

from bases.stage import Stage


class BaseAligner(Stage):
    def preprocess(self, inputs):
        """
        preprocess some images
        :param inputs: list of images to align
        :return: preprocessed list of images
        """
        return inputs

    @abstractmethod
    def forward(self, inputs):
        """
        align images
        :param inputs: list of images to align
        :return: list of aligned images
        """
        pass

    def postprocess(self, inputs):
        """
        postprocess aligned images
        :param inputs: list of aligned images
        :return: list of post-processed aligned images
        """
        return inputs
