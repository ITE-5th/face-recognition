from abc import abstractmethod

from bases.stage import Stage


class BaseDetector(Stage):
    def preprocess(self, inputs):
        return inputs

    @abstractmethod
    def forward(self, image):
        """
        detect all faces in the input image
        :param image: the image (cv2 array)
        :return: returns 2 items
        1. the image (cv2 array)
        2. np.array of list of 4 values (x1, y1, x2, y2)
        """
        pass

    @abstractmethod
    def postprocess(self, inputs):
        pass
