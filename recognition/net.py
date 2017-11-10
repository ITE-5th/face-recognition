import torch.nn as nn
import torch.nn.functional as F

from recognition.inceptionresnetv2 import inceptionresnetv2

path_of_pretrained_model = "../models/inceptionresnetv2.pth"
extractor = inceptionresnetv2(path_of_pretrained_model)
for param in extractor.parameters():
    param.requires_grad = False


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(1001, 100)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = extractor.forward(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        return F.softmax(self.linear3(x))
