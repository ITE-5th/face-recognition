import torch
from torch.autograd import Variable
from recognition.net import Net
from preprocessing.aligner_preprocessor import AlignerPreprocessor
import cv2
import os
import numpy as np


def to_module(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        new_name = key[key.index(".") + 1:]
        new_state_dict[new_name] = state_dict[key]
    return new_state_dict


names = os.listdir("../data/lfw/")
state = torch.load("checkpoint.pth.tar")
temp = state["state_dict"]
state_dict = to_module(temp)
num_classes = len(names)
net = Net(state["num_classes"])
net.load_state_dict(state_dict)
image = cv2.imread("../test_image2.jpeg")
preprocessor = AlignerPreprocessor()
image = preprocessor.preprocess(image)
aligned = np.swapaxes(image, 0, 2)
aligned = np.swapaxes(aligned, 1, 2)
image = torch.from_numpy(aligned).float()
image = image.unsqueeze(0)
x = Variable(image)
y = net(x)
vals, inds = y.data.topk(10)
inds = inds.view(10)
for ind in inds:
    print(names[ind])
