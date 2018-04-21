import glob

import cv2
import torch
from dlt.util import torch2cv
from torch.autograd import Variable

from aligners.no_aligner import NoAligner
from aligners.one_millisecond_aligner import OneMillisecondAligner
from bases.pipeline import Pipeline
from detectors.dlib_detector import DLibDetector
from extractors.base_extractor import BaseExtractor
from extractors.models.vgg.vgg_face import vgg_face
from file_path_manager import FilePathManager
from misc.utils import Utils
from transforms.crop import Crop
from transforms.resize import Resize


class VggExtractor(BaseExtractor):

    def __init__(self, use_cuda: bool = True):
        super().__init__()
        self.use_cuda = use_cuda
        self.vgg_face = vgg_face
        self.load(FilePathManager.resolve('data/VGG_FACE.pth'))
        self.resize = Resize(224)
        if self.use_cuda:
            self.vgg_face.cuda()

    def load(self, path: str):
        state = torch.load(path)
        self.vgg_face.load_state_dict(state)
        self.vgg_face = torch.nn.Sequential(*list(self.vgg_face.children())[:-7])
        for param in self.vgg_face.parameters():
            param.requires_grad = False
        self.vgg_face.eval()

    def preprocess(self, inputs):
        inputs = self.resize(inputs)
        faces, image = inputs
        faces = Utils.cv2torch(faces).float()
        if len(faces.shape) == 3:
            faces = faces.unsqueeze(0)

        if self.use_cuda:
            faces = faces.cuda()
        return Variable(faces), image

    def forward(self, inputs):
        faces, image = inputs
        return self.vgg_face(faces), image

    def postprocess(self, inputs):
        features, image = inputs
        return torch2cv(features), image


if __name__ == '__main__':
    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("images/*")))

    # pipeline = Pipeline([DLibDetector(), OneMillisecondAligner(), VggExtractor()])
    pipeline = Pipeline([DLibDetector(), NoAligner(scale=3), VggExtractor()])
    # pipeline = Pipeline([DLibDetector(), Crop(), Resize(224), VggExtractor()])

    for i, face in enumerate(faces):
        face = cv2.imread(face)

        features, _ = pipeline(face, True)
        print("{} image: #{} Features.".format(i, features.shape))
