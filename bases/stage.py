from abc import ABC, abstractmethod

from misc.time_logger import TimeLogger


class Stage(ABC):
    @abstractmethod
    def preprocess(self, inputs):
        pass

    def __call__(self, inputs, verbose: bool = False):
        if verbose:
            logger = TimeLogger()

        outputs = self.preprocess(inputs)
        outputs = self.forward(outputs)
        outputs = self.postprocess(outputs)

        if verbose:
            logger.log()
            print("Stage: {}, time {}".format(self, logger))

        return outputs

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def postprocess(self, inputs):
        pass

    def __str__(self):
        return type(self).__name__
