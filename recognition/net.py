import torch.nn as nn
import torch.nn.functional as F

from recognition.inceptionresnetv2 import inceptionresnetv2

path_of_pretrained_model = "../data/inceptionresnetv2.pth"
extractor = inceptionresnetv2(path_of_pretrained_model)
extractor = nn.Sequential(*list(extractor.children())[:-1])
for param in extractor.parameters():
    param.requires_grad = False
extractor.eval()
extractor = extractor.cuda()


# extractor = VGG_FACE
# extractor.load_state_dict(torch.load('../data/VGG_FACE.pth'))
# extractor = nn.Sequential(*list(extractor.children())[:-1])
# extractor.eval()
# extractor = extractor.cuda()


class Net(nn.Module):
    def __init__(self, num_classes, vgg_face: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.linear1 = nn.Linear(2622 if vgg_face else 1536, 150)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = extractor.forward(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
