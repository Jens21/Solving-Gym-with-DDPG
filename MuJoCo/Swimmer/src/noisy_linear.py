import torch
import torch.nn as nn
import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('weight_noise', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_noise', torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = self.std_init / np.sqrt(self.in_features)
        self.weight.data.normal_(0, std)
        self.bias.data.normal_(0, std)
        self.weight_noise.zero_()
        self.bias_noise.zero_()

    def forward(self, input):
        if self.training:
            weight = self.weight + self.weight_noise
            bias = self.bias + self.bias_noise
        else:
            weight = self.weight
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

    def sample_noise(self):
        self.weight_noise.normal_(0, std=self.std_init)
        self.bias_noise.normal_(0, std=self.std_init)