import torch


# from CMS.Functions.L2NormFunc import L2NormFunc


class L2Norm_2(torch.nn.Module):
    def __init__(self, scaling_factor):
        super(L2Norm, self).__init__()
        # create learn-able parameters
        self.scaling_factor = torch.nn.Parameter(torch.FloatTensor(scaling_factor))

    def forward(self, x):
        if len(x.size()) != 4 and len(x.size()) != 5:
            raise Exception("Invalid inputs, got " + str(len(x.size())) + "! expected 4 or 5")
        # apply L2Norm function
        input_data = x
        if len(input_data.size()) != 5:
            input_data = torch.unsqueeze(x, 1)

        bs, rois_num, c, h, w = input_data.size()
        input_data = input_data.view(bs, rois_num, c, -1)

        # TODO: modify 1e-8 to not divide by zero
        denominator = input_data.view(bs, rois_num, c, -1).norm(2, 3) + 1e-8

        x_bar = input_data / denominator.view(bs, rois_num, c, 1)
        assert x_bar.size() == input_data.size()

        # normalize the pixels and multiply by scaling_factor
        y = x_bar * self.scaling_factor.view(1, 1, c, 1)
        assert y.size() == input_data.size()

        y = y.view(bs, rois_num, c, h, w)
        if x.size() != 5:
            y = torch.squeeze(y)

        print(y.size())
        return y
        # return y.view(bs, c, h, w)


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

        # TODO: modify 1e-8 to not divide by zero
        denominator = input_data.view(bs, -1).norm(2, 1) + 1e-8

        x_bar = input_data / denominator.view(bs, 1, 1)
        assert x_bar.size() == input_data.size()

        # normalize the pixels and multiply by scaling_factor
        y = x_bar * self.scaling_factor.view(1, c, 1)  # .expand_as(x_bar)
        assert y.size() == input_data.size()

        return y.view(bs, c, h, w)
