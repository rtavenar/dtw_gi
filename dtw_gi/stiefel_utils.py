import torch
import numpy as np
import torch.nn.functional as F
import geoopt


def stiefel_uniform_(tensor):
    with torch.no_grad():
        tensor.normal_()
        tensor.proj_()
    return tensor


def stiefel_uniform_npy(shape):
    tensor = geoopt.ManifoldParameter(
        data=torch.Tensor(*shape),
        manifold=geoopt.Stiefel()
    )
    stiefel_uniform_(tensor)
    return tensor.detach().cpu().numpy()


def stiefel_eye_(tensor):
    torch.nn.init.eye_(tensor)
    return tensor


class StiefelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 init_eye=True, init_zero=True, init_map=None):
        super(StiefelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_eye = init_eye
        self.init_zero = init_zero
        self.weight = geoopt.ManifoldParameter(
            data=torch.Tensor(out_features, in_features),
            manifold=geoopt.Stiefel()
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(init_map=init_map)

    def reset_parameters(self, init_map=None):
        if init_map is not None:
            with torch.no_grad():
                self.weight.data = init_map.data
        elif self.init_eye:
            stiefel_eye_(self.weight)
        else:
            stiefel_uniform_(self.weight)
        if self.bias is not None:
            if self.init_zero:
                torch.nn.init.zeros_(self.bias)
            else:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                    self.weight)
                bound = 1 / np.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class StiefelLinearPerGroup(torch.nn.Module):
    def __init__(self, group_size=3, bias=True, init_eye=True, init_zero=True):
        super(StiefelLinearPerGroup, self).__init__()
        self.group_size = group_size
        self.init_eye = init_eye
        self.init_zero = init_zero
        self.weight = geoopt.ManifoldParameter(
            data=torch.Tensor(group_size, group_size),
            manifold=geoopt.Stiefel()
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(group_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_eye:
            stiefel_eye_(self.weight)
        else:
            stiefel_uniform_(self.weight)
        if self.bias is not None:
            if self.init_zero:
                torch.nn.init.zeros_(self.bias)
            else:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                    self.weight)
                bound = 1 / np.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        shape = list(input.size())
        new_shape = shape[:-2] + [-1, self.group_size]
        return F.linear(input.view(new_shape), self.weight, self.bias).view(
            shape)
