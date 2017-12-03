import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes, vgg_face: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.fcs = nn.Sequential(
            nn.Linear(4096 if vgg_face else 1536, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.fcs(x)


if __name__ == '__main__':
    pass
