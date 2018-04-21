from dlt.util import torch2cv
from torch.autograd import Variable

from extractors.base_extractor import BaseExtractor
from extractors.models.openface.load_openface_net import prepareOpenFace
from misc.utils import Utils
from transforms.resize import Resize


class OpenfaceExtractor(BaseExtractor):

    def __init__(self, use_cuda: bool = False):
        super().__init__()
        self.use_cuda = use_cuda
        self.resize = Resize(96)
        self.net = prepareOpenFace(use_cuda)

    def preprocess(self, inputs):
        inputs = self.resize(inputs)
        faces, image = inputs
        faces = Utils.cv2torch(faces).float()
        # if len(faces.shape) == 3:
        #     faces = faces.unsqueeze(0)

        if self.use_cuda:
            faces = faces.cuda()
        return Variable(faces), image

    def forward(self, inputs):
        faces, image = inputs
        return self.net(faces)[0], image

    def postprocess(self, inputs):
        features, image = inputs
        return torch2cv(features), image
