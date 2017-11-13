import torch
import torch.nn as nn

from CMS.vgg_face import vgg_face


class CMSRCNN(nn.Module):
    def __init__(self):
        super(CMSRCNN, self).__init__()
        # VGG-16 structure: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.vgg = vgg_face
        self.vgg.load_state_dict(torch.load('../model/vgg_face.pth'))

    def forward(self, x):
        return self.vgg(x)
