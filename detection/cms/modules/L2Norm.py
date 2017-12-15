import torch
import torch.nn.functional as F


class L2Norm(torch.nn.Module):
    def __init__(self, scaling_factor):
        super(L2Norm, self).__init__()
        # create learn-able parameters
        self.scaling_factor = torch.nn.Parameter(torch.FloatTensor(scaling_factor))

    def forward(self, x):
        bs, c, h, w = x.size()

        x_bar = F.normalize(x.view(bs, -1), p=2, dim=1).view(bs, c, h, w)

        return x_bar * self.scaling_factor.view(1, c, 1, 1)
