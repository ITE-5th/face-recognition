import torch
from torch.autograd import Function


class L2Norm(torch.nn.Module):
    def __init__(self, scaling_factor):
        super(L2Norm, self).__init__()
        # create learn-able parameters
        self.scaling_factor = torch.nn.Parameter(torch.FloatTensor(scaling_factor))

    def forward(self, x):
        # apply L2Norm function
        return L2NormFunc()(x, self.scaling_factor)


class L2NormFunc(Function):
    def forward(self, input_data, scaling_factor):
        # compute L2Norm for pixels in channels
        bs, c, _, _ = input_data.size()
        denominator = input_data.view(bs, c, -1).abs().sum(2).sqrt()

        x_bar = input_data.div(
            denominator.view(bs, c, 1, 1).expand_as(input_data))

        # normalize the pixels and multiply by scaling_factor
        y = x_bar * scaling_factor.view(1, c, 1, 1).expand_as(x_bar)

        # TODO: uncomment the next lines
        # commented 'cuz of low gpu memory
        # self.scaling_factor = scaling_factor
        # self.input_data = input_data
        # self.x_bar = x_bar
        # self.denominator = denominator
        # self.y = y

        return y

    def backward(self, grad_output):
        # restore the values
        input_data, scaling_factor, x_bar, denominator, y = self.input_data, self.scaling_factor, self.x_bar, self.denominator, self.y

        # element-wise multiplication -> sum elements in every channel
        grad_scaling_factor = torch.FloatTensor(scaling_factor.shape)

        # TODO: replace "for loop" with vectorization
        # compute scaling_factor gradients
        for i in range(len(grad_scaling_factor)):
            grad_scaling_factor[i] = (grad_output * x_bar[:, :, i]).sum()

        # compute gradients of x bar
        grad_x_bar = grad_output * scaling_factor.view(1, 1, -1).expand_as(grad_output)

        x_by_xt = input_data * input_data.transpose(0, 1)
        denominator_cubed = torch.pow(denominator, 3)

        # compute gradients of inputs
        grad_inputs = grad_x_bar * (1 / denominator - x_by_xt / denominator_cubed)

        return grad_inputs, grad_scaling_factor
