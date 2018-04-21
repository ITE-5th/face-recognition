import glob

import cv2
import torch
from dlt.util import torch2cv
from torch.autograd import Variable

from aligners.no_aligner import NoAligner
from bases.pipeline import Pipeline
from detectors.dlib_detector import DLibDetector
from extractors.base_extractor import BaseExtractor
from extractors.models.lightcnn.light_cnn import LightCNN_29Layers_v2
from file_path_manager import FilePathManager
from misc.utils import Utils
from transforms.resize import Resize


class LightCNNExtractor(BaseExtractor):

    def __init__(self, use_cuda: bool = True):
        super().__init__()
        self.use_cuda = use_cuda
        self.light_cnn = torch.nn.DataParallel(LightCNN_29Layers_v2(num_classes=80013))
        self.load(FilePathManager.resolve('data/LightCNN_29Layers_V2_checkpoint.pth.tar'))
        self.resize = Resize(128)
        if self.use_cuda:
            self.light_cnn.cuda()

    def load(self, path: str):
        state = torch.load(path)
        self.light_cnn.load_state_dict(state['state_dict'])
        for param in self.light_cnn.parameters():
            param.requires_grad = False
        self.light_cnn.eval()

    def preprocess(self, inputs):
        inputs = self.resize(inputs)
        faces, image = inputs
        faces = Utils.to_gray(faces)
        faces = Utils.cv2torch(faces).float()
        if len(faces.shape) == 3:
            faces = faces.unsqueeze(1)

        if self.use_cuda:
            faces = faces.cuda()
        return Variable(faces), image

    def forward(self, inputs):
        faces, image = inputs
        _, features = self.light_cnn(faces)
        return features, image

    def postprocess(self, inputs):
        features, image = inputs
        return torch2cv(features), image


if __name__ == '__main__':
    FilePathManager.clear_dir("output")

    path = FilePathManager.resolve("output")
    faces = sorted(glob.glob(FilePathManager.resolve("images/*")))

    # pipeline = Pipeline([DLibDetector(), OneMillisecondAligner(), VggExtractor()])
    pipeline = Pipeline([DLibDetector(), NoAligner(scale=3), LightCNNExtractor()])
    # pipeline = Pipeline([DLibDetector(), Crop(), Resize(224), VggExtractor()])

    for i, face in enumerate(faces):
        face = cv2.imread(face)

        features, _ = pipeline(face, True)
        print("{} image: #{} Features.".format(i, features.shape))
