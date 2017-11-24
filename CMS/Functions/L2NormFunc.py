import torch
from torch.autograd import Function


class L2NormFunc(Function):
    def forward(self, input_data, scaling_factor):
        # compute L2Norm for pixels in channels
        bs, c, h, w = input_data.size()
        input_data = input_data.view(bs, c, -1)

        # denominator = input_data.view(bs, c, -1).abs().sum(2).sqrt()
        # denominator = input_data.norm(2, 1).view(bs, 1, -1)
        denominator = input_data.view(bs, -1).norm(2, 1)

        x_bar = input_data / denominator.view(bs, 1, 1)
        assert x_bar.size() == input_data.size()
        # x_bar = input_data.div(
        #     denominator.view(bs, c, 1, 1).expand_as(input_data))

        # normalize the pixels and multiply by scaling_factor
        y = x_bar * scaling_factor.view(1, c, 1)#.expand_as(x_bar)
        assert y.size() == input_data.size()

        # TODO: uncomment the next lines
        # commented 'cuz of low gpu memory
        self.scaling_factor = scaling_factor
        self.input_data = input_data
        self.x_bar = x_bar
        self.denominator = denominator
        self.y = y

        return y.view(bs, c, h, w)

    def backward(self, grad_output):
        # restore the values
        input_data, scaling_factor, x_bar, denominator, y = self.input_data, self.scaling_factor, self.x_bar, self.denominator, self.y
        bs, c, h, w = grad_output.size()

        grad_output = grad_output.view(bs, c, -1)

        # compute scaling_factor gradients
        # for i in range(len(grad_scaling_factor)):
        #     grad_scaling_factor[i] = (grad_output * x_bar[:, :, i]).sum()
        grad_scaling_factor = (grad_output.data * x_bar).sum(2).mean(0)
        assert grad_scaling_factor.size() == scaling_factor.size()

        # compute gradients of x bar
        grad_x_bar = grad_output.data * scaling_factor.view(1, c, 1).expand_as(grad_output)
        assert grad_x_bar.size() == x_bar.size()

        denominator_cubed = torch.pow(denominator, 3)
        # x_by_xt = input_data.pow(2).sum(2).view(denominator_cubed.size())
        x_by_xt = torch.matmul(input_data, input_data.transpose(1, 2))

        # compute gradients of inputs
        grad_inputs = grad_x_bar * (1 / denominator - x_by_xt / denominator_cubed)
        assert grad_inputs.size() == input_data.size()

        return grad_inputs.view(bs, c, h, w), grad_scaling_factor
