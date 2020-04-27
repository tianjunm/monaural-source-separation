# https://github.com/f90/Wave-U-Net-Pytorch/blob/master/waveunet.py

import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

class Conv1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, transpose=False):
        super(Conv1Layer, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        if self.transpose:
            self.layer = nn.ConvTranspose1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding=kernel_size-1)
        else:
            self.layer = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        # Apply the convolution
        out =  self.layer(x)
        out = F.leaky_relu(out)
        out = self.bn(out)
        return out

    def get_input_size(self, output_size):
        # Strided conv/decimation
        if not self.transpose:
            curr_size = (output_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = output_size
        # Conv
        curr_size = curr_size + self.kernel_size - 1 # o = i + p - k + 1

        # Transposed
        if self.transpose:
            assert ((curr_size - 1) % self.stride == 0)# We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        assert(curr_size > 0)
        return curr_size

    def get_output_size(self, input_size):
        # Transposed
        if self.transpose:
            assert(input_size > 1)
            curr_size = (input_size - 1)*self.stride + 1 # o = (i-1)//s + 1 => i = (o - 1)*s + 1
        else:
            curr_size = input_size
        # Conv
        curr_size = curr_size - self.kernel_size + 1 # o = i + p - k + 1
        assert (curr_size > 0)

        # Strided conv/decimation
        if not self.transpose:
            assert ((curr_size - 1) % self.stride == 0)  # We need to have a value at the beginning and end
            curr_size = ((curr_size - 1) // self.stride) + 1
        return curr_size

