import os

import cv2
import torch
from torch.autograd import Variable

from preprocessing.aligner_preprocessor import AlignerPreprocessor
from recognition.net import Net, extractor


def to_module(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        new_name = key[key.index(".") + 1:]
        new_state_dict[new_name] = state_dict[key]
    return new_state_dict


state = torch.load("checkpoint.pth.tar")
temp = state["state_dict"]
state_dict = to_module(temp)
names = sorted(os.listdir("../data/lfw2"))
num_classes = len(names)
net = Net(state["num_classes"])
net.load_state_dict(state_dict)
net.eval()
net = net.cuda()
image = cv2.imread("../test_image.jpeg")
preprocessor = AlignerPreprocessor()
image = preprocessor.preprocess(image)
image = torch.from_numpy(image).float()
image = image.unsqueeze(0)
x = Variable(image.cuda())
x = extractor(x)
x = x.view(1, -1)
y = net(x)
k = 10
vals, inds = y.data.topk(k)
inds = inds.view(k)
for ind in inds:
    print(names[ind])
