import torch
import torch.nn as nn
from . import ConvLayer


def crop(x, target):
    '''
    Center-crop 3-dim. input tensor along last axis so it fits the target tensor shape
    :param x: Input tensor
    :param target: Shape of this tensor will be used as target shape
    :return: Cropped input tensor
    '''
    if x is None:
        return None
    if target is None:
        return x
    target_shape = target.shape
    diff = x.shape[-1] - target_shape[-1]

    assert (diff % 2 == 0)
    crop = diff // 2
    if crop == 0:
        return x
    if crop < 0:
        raise ArithmeticError

    return x[:, :, crop:-crop].contiguous()

class DownSamplingBlock(nn.Module):
    def __init__(self, kernel_size, stride, in_channels, shortcut_channels, out_channels, pre_depth, post_depth):
        super(DownSamplingBlock, self).__init__()
        assert stride > 1
        self.in_channels = in_channels
        self.shortcut_channels = shortcut_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.pre_depth = pre_depth
        self.post_depth = post_depth
        self.__constPreShortcut__()
        self.__constPostShortcut__()
        self.__constDecimate__()
        
    # private function to create the pre_shortcut layers
    def __constPreShortcut__(self):
        self.pre_shortcut = nn.ModuleList()
        for d in range(self.pre_depth):
            in_channels = self.in_channels if d == 0 else self.shortcut_channels
            layer = ConvLayer.Conv1Layer(in_channels, self.shortcut_channels, kernel_size=self.kernel_size, stride=1)
            self.pre_shortcut.append(layer)
    
    # private function to create the pre_shortcut layers
    def __constPostShortcut__(self):
        self.post_shortcut = nn.ModuleList()
        for d in range(self.post_depth):
            in_channels = self.shortcut_channels if d==0 else self.out_channels
            layer = ConvLayer.Conv1Layer(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=1)
            self.post_shortcut.append(layer)
    
    # private function to create the Decimate layers(decreases the 'resolution')
    def __constDecimate__(self):
        self.decimate = ConvLayer.Conv1Layer(self.out_channels, self.out_channels,  kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        shortcut = x
        for layer in self.pre_shortcut:
            shortcut = layer(shortcut)
        out = shortcut 
        for layer in self.post_shortcut:
            out = layer(out)
        out = self.decimate(out)
        return out, shortcut
    
    def get_output_size(self, input_size):
        curr_size = input_size
        for layer in self.pre_shortcut:
            curr_size = layer.get_output_size(curr_size)
        for layer in self.post_shortcut:
            curr_size = layer.get_output_size(curr_size)
        curr_size = self.decimate.get_output_size(curr_size)
        return curr_size
    
    def get_input_size(self, output_size):
        curr_size = self.decimate.get_input_size(output_size)
        for layer in reversed(self.post_shortcut):
            curr_size = layer.get_input_size(curr_size)
        for layer in reversed(self.pre_shortcut):
            curr_size = layer.get_input_size(curr_size)
        return curr_size

class BottleNeckBlock(nn.Module):
    
    def __constLayers__(self):
        self.layers = nn.ModuleList()
        for d in range(depth):
            layer = ConvLayer.Conv1Layer(self.channels, self.channels, kernel_size=self.kernel_size, stride=1)
            self.layers.append(layer)

    def __init__(self, channels, kernel_size, stride, depth):
        super(BottleNeckBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.__constLayers__()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class UpSamplingBlock(nn.Module):
    # private function to create the pre_shortcut layers
    
    def __init__(self, kernel_size, stride, in_channels, shortcut_channels, out_channels, pre_depth, post_depth):
        super(UpSamplingBlock, self).__init__()
        assert stride > 1
        self.in_channels = in_channels
        self.shortcut_channels = shortcut_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.pre_depth = pre_depth
        self.post_depth = post_depth
        self.__constRevamp__()
        self.__constPreShortcut__()
        self.__constPostShortcut__()
    
    def __constPreShortcut__(self):
        self.pre_shortcut = nn.ModuleList()
        for d in range(self.pre_depth):
            in_channels =  self.in_channels if d==0 else self.out_channels
            layer = ConvLayer.Conv1Layer(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=1)
            self.pre_shortcut.append(layer)
    
    def __constPostShortcut__(self):
        self.post_shortcut = nn.ModuleList()
        for d in range(self.post_depth):
            in_channels =  (self.out_channels + self.shortcut_channels) if d==0 else self.out_channels
            layer = ConvLayer.Conv1Layer(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=1)
            self.post_shortcut.append(layer)
    
    def __constRevamp__(self):
        self.revamp = ConvLayer.Conv1Layer(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, transpose=True)
  
        
    def forward(self, up_input, shortcut):
        x = up_input 
        x = self.revamp(x)
        for layer in self.pre_shortcut:
            x = layer(x)
        shortcut = crop(shortcut, x)
        x = crop(x, shortcut )
        combined = torch.cat([shortcut, x], dim = 1)
        for layer in self.post_shortcut:
            combined = layer(combined)
        return combined

    def get_output_size(self, input_size):
        curr_size = self.revamp.get_output_size(input_size)
        for layer in self.pre_shortcut:
            curr_size = layer.get_output_size(curr_size)
        for layer in self.post_shortcut:
            curr_size = layer.get_output_size(curr_size)
        return curr_size