import cv2
import torch
from dlt.util.misc import cv2torch
from torch import nn
from torch.autograd import Variable

from file_path_manager import FilePathManager


def inception_extractor(use_cuda=True):
    from recognition.pretrained.inceptionresnetv2 import inceptionresnetv2

    path_of_pretrained_model = FilePathManager.load_path("data/inceptionresnetv2.pth")
    extractor = inceptionresnetv2(path_of_pretrained_model)
    extractor = nn.Sequential(*list(extractor.children())[:-1])
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    if use_cuda:
        extractor = extractor.cuda()
    return extractor


def vgg_extractor(use_cuda=True):
    from recognition.pretrained.VGG_FACE import VGG_FACE
    extractor = VGG_FACE
    state = torch.load(FilePathManager.load_path('data/VGG_FACE.pth'))
    extractor.load_state_dict(state)
    extractor = nn.Sequential(*list(extractor.children())[:-7])
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    if use_cuda:
        extractor = extractor.cuda()
    return extractor


if __name__ == '__main__':
    extractor = vgg_extractor()
    image = cv2.imread(FilePathManager.load_path("test_images/image_1.jpg"))
    image = cv2.resize(image, (200, 200))
    image = cv2torch(image).float()
    image = image.unsqueeze(0)
    image = Variable(image.cuda())
    print(extractor(image))