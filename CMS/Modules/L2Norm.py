import torch


# from CMS.Functions.L2NormFunc import L2NormFunc


class L2Norm(torch.nn.Module):
    def __init__(self, scaling_factor):
        super(L2Norm, self).__init__()
        # create learn-able parameters
        self.scaling_factor = torch.nn.Parameter(torch.FloatTensor(scaling_factor))

    def forward(self, x):
        # apply L2Norm function
        # return L2NormFunc()(x, self.scaling_factor)

        bs, c, h, w = x.size()
        input_data = x.view(bs, c, -1)

        denominator = input_data.view(bs, -1).norm(2, 1)

        x_bar = input_data / denominator.view(bs, 1, 1)
        assert x_bar.size() == input_data.size()

        # normalize the pixels and multiply by scaling_factor
        y = x_bar * self.scaling_factor.view(1, c, 1)  # .expand_as(x_bar)
        assert y.size() == input_data.size()

        return y.view(bs, c, h, w)
