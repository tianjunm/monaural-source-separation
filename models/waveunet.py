import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from . import Blocks
from . import ConvLayer



class WaveUNet(nn.Module):

    def __createDownSampling__(self):
        self.downsampling_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            in_channels = self.input_channels if i == 0 else self.num_channels[i]
            block = Blocks.DownSamplingBlock(self.kernel_size, self.strides[i], in_channels, 
                        self.num_channels[i], self.num_channels[i+1], self.pre_depth, self.post_depth)
            self.downsampling_blocks.append(block)
        self.module.downsampling_blocks = self.downsampling_blocks
        
    
    def __createBottleNeck__(self):
        self.bottleneck_block =  nn.ModuleList()
        for d in range(self.pre_depth):
            layer = ConvLayer.Conv1Layer(self.num_channels[-1], self.num_channels[-1], kernel_size=self.kernel_size, stride=1)
            self.bottleneck_block.append(layer)
        self.module.bottleneck_block = self.bottleneck_block

    
    def __createUpSampling__(self):
        self.upsampling_blocks = nn.ModuleList()
        n = self.num_levels
        for i in range(self.num_levels - 1):
            block = Blocks.UpSamplingBlock(self.kernel_size, self.strides[-1-i], self.num_channels[-1-i], 
                        self.num_channels[-2-i], self.num_channels[-2-i], self.pre_depth, self.post_depth)
            self.upsampling_blocks.append(block)
        self.module.upsampling_blocks = self.upsampling_blocks
    
    def __createOutputLayer__(self):
        outputs = self.out_channels * (self.num_cats-1)
        self.outputLayer = ConvLayer.Conv1Layer(self.num_channels[0], outputs, self.kernel_size, 1)
        self.module.outputLayer = self.outputLayer
    
    def __init__(self, in_channels, num_channels, out_channels, num_cats, kernel_size, target_output_size, pre_depth, post_depth, depth, stride_start=2, stride_change = 0):
        super(WaveUNet, self).__init__()
        self.input_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_cats = num_cats
        self.kernel_size = kernel_size
        self.target_output_size = target_output_size
        self.num_levels = len(num_channels)
        self.pre_depth = pre_depth
        self.post_depth = post_depth
        self.depth = depth
        self.stride_change = stride_change
        self.strides = [stride_start - i * self.stride_change for i in range(self.num_levels)]
        self.module = nn.Module()
        self.__createDownSampling__()
        self.__createBottleNeck__()
        self.__createUpSampling__()
        self.__createOutputLayer__()   
        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")
        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1
        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        try:
            curr_size = bottleneck

            for idx, block in enumerate(self.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            curr_size = self.outputLayer.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(self.bottleneck_block):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(self.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)
            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward(self, x):
        shortcuts = []
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) 
        # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size
        for block in self.module.downsampling_blocks:
            x, shortcut =  block(x)
            shortcuts.append(shortcut)
        shortcuts.reverse()

        for block in self.module.bottleneck_block:
            x = block(x)
        i = 0
        for block in self.module.upsampling_blocks:
            x = block(x, shortcuts[i])
            i += 1
        x = self.module.outputLayer(x)
        return x





        



          