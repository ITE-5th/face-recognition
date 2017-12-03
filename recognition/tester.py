import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from evm.evm import EVM
from preprocessing.aligner_preprocessor import AlignerPreprocessor
from recognition.extractors import vgg_extractor
from recognition.net import Net


def to_module(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        new_name = key[key.index(".") + 1:]
        new_state_dict[new_name] = state_dict[key]
    return new_state_dict


use_evm = True
names = sorted(os.listdir("../data/custom_images2"))
image = cv2.imread("../test_image.jpeg")
preprocessor = AlignerPreprocessor()
image = preprocessor.preprocess(image)
cv2.imwrite("temp.jpg", image)
image = cv2.imread("temp.jpg")
image = cv2.resize(image, (224, 224))
image = np.swapaxes(image, 0, 2)
image = np.swapaxes(image, 1, 2)
image = torch.from_numpy(image).float()
image = image.unsqueeze(0)
x = Variable(image.cuda())
extractor = vgg_extractor()
x = extractor(x)

if not use_evm:
    state = torch.load("../models/checkpoint-200.pth.tar")
    temp = state["state_dict"]
    state_dict = to_module(temp)
    num_classes = len(names)
    net = Net(state["num_classes"]).cuda()
    net.load_state_dict(state_dict)
    net.eval()
    x = x.view(1, -1)
    y = net(x)
    k = 10
    vals, inds = y.data.topk(k)
    inds = inds.view(k)
    for ind in inds:
        print(names[ind])
else:
    evm = EVM.load("../evm_model.model")
    x = x.view(-1)
    x = x.data.cpu().numpy()
    x = x.reshape(1, -1)
    predicted = evm.predict(x)
    predicted = predicted[0]
    if predicted == -1:
        print("unknown")
    else:
        print(names[predicted])

os.system("rm temp.jpg")
