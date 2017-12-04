import torch.nn as nn


def to_module(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        new_name = key[key.index(".") + 1:]
        new_state_dict[new_name] = state_dict[key]
    return new_state_dict


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

