import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes, vgg_face: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.fcs = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096 if vgg_face else 1536, num_classes),
        )

    def forward(self, x):
        return self.fcs(x)


if __name__ == '__main__':
    pass
