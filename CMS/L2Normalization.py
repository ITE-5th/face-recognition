import torch
from torch.autograd import Function


class L2Norm(torch.nn.Module):
    def __init__(self, scaling_factor):
        super(L2Norm, self).__init__()
        # create learn-able parameters
        self.scaling_factor = torch.nn.Parameter(scaling_factor)
        # L2Norm Function
        self.l2n_func = L2NormalizationFunc

    def forward(self, x):
        # apply L2Norm function
        x = self.l2n_func(x, self.scaling_factor)
        return x


class L2NormalizationFunc(Function):
    def forward(self, input_data, scaling_factor):
        # compute L2Norm for pixels in channels
        denominator = input_data.view(-1, 3).abs().sum(1).sqrt()

        x_bar = torch.div(input_data, denominator.view(1, 1, -1).exapnd_as(input_data))

        # normalize the pixels and multiply by scaling_factor
        y = x_bar * scaling_factor.view(1, 1, -1).expand_as(x_bar)

        self.save_for_backward(input_data, scaling_factor, x_bar, denominator, y)

        return y

    def backward(self, grad_output):
        # restore the values
        input_data, scaling_factor, x_bar, denominator, y = self.saved_variables

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
