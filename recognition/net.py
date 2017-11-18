import torch.nn as nn
import torch.nn.functional as F

from recognition.inceptionresnetv2 import inceptionresnetv2




class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        path_of_pretrained_model = "../data/inceptionresnetv2.pth"
        self.extractor = inceptionresnetv2(path_of_pretrained_model)
        for param in self.extractor.parameters():
            param.requires_grad = False
        self.num_classes = num_classes
        self.linear1 = nn.Linear(1000, 100)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.extractor(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
