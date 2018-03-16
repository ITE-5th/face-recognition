import cv2
import torch
import torch.nn as nn
from dlt.util.misc import cv2torch
from torch.autograd import Variable

from file_path_manager import FilePathManager
from recognition.extractor.vgg_face import vgg_face


def remove_net(state):
    new_state = {}
    for key in state.keys():
        new_key = key[key.index(".") + 1:]
        new_state[new_key] = state[key]
    return new_state


def vgg_extractor(siamese: bool = False):
    extractor = vgg_face
    if not siamese:
        state = torch.load(FilePathManager.resolve('data/VGG_FACE.pth'))
        extractor.load_state_dict(state)
        extractor = nn.Sequential(*list(extractor.children())[:-7])
    else:
        state = torch.load(FilePathManager.resolve('data/VGG_FACE_MODIFIED.pth.tar'))
        extractor = nn.Sequential(*list(extractor.children())[:-1])
        state = state["state_dict"]
        state = remove_net(state)
        extractor.load_state_dict(state)
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    extractor = extractor.cuda()
    return extractor


if __name__ == '__main__':
    extractor = vgg_extractor()
    image = cv2.imread(FilePathManager.resolve("test_images/image_1.jpg"))
    image = cv2.resize(image, (200, 200))
    image = cv2torch(image).float()
    image = image.unsqueeze(0)
    image = Variable(image.cuda())
    print(extractor(image))
