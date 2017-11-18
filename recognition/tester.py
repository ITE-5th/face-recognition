import torch
from torch.autograd import Variable
from recognition.net import Net
from preprocessing.aligner_preprocessor import AlignerPreprocessor
import cv2
import os

names = os.listdir("../data/lfw/")
state = torch.load("checkpoint.pth.tar")
temp = state["state_dict"]
state_dict = dict()
for key in temp.keys():
    new_name = key[key.index(".") + 1:]
    state_dict[new_name] = temp[key]
num_classes = len(names)
net = Net(state["num_classes"]).cuda()
net.load_state_dict(state_dict)
image = cv2.imread("../test_image.jpeg")
preprocessor = AlignerPreprocessor()
image = preprocessor.preprocess(image)
image = torch.from_numpy(image).float().cuda()
image = image.unsqueeze(0)
x = Variable(image).cuda()
y = net(x)
_, ind = torch.max(y.data, 1)
print(names[ind[0]])
