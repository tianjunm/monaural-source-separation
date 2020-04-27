# reference from https://github.com/f90/Wave-U-Net-Pytorch/blob/master/waveunet.py
import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from . import Blocks
from . import ConvLayer
import logging


from guppy import hpy
import sys
import torch
import gc

from memory_profiler import profile

class OutputToSTFT:
    #@profile
    def __init__(self, dataset_info):
        super().__init__()
        self._info = dataset_info
        self._window_size = 256
        self._hop_length = 192
    
    #@profile
    def __call__(self, item):
        output_dimensions = self._info['output_dimensions']
        sr = self._info['sample_rate']
        duration = self._info['mixture_duration']
        logging.info("ITEM"+str(item.shape))
        spects = self._convert_output(item, output_dimensions) 
        print(spects.shape)
        return spects

    #@profile
    def _convert_output(self, all_outputs, output_dimensions):
        if output_dimensions == 'BS2NM':
            spects = []
            for output in all_outputs:
                spect = [self._stft(pcm).permute(2, 1, 0) for pcm in
                      output]
                spect = torch.stack(spect, dim=0)
                spects.append(spect)
            spects = torch.stack(spects, dim=0)
            

        elif output_dimensions == 'BN(S2M)':
            # n = self._stft(ground_truths[0]).size(1)
            # spects = [self._stft(pcm).permute(1, 2, 0) for pcm in
            #           ground_truths]
            # spects = torch.stack(spects, dim=1).view(n, -1)

            # spects = []
            # for pcm in ground_truths:
            #     spect = self._stft(pcm).permute(1, 2, 0)
            #     spects.append(spect)
            spects = [self._stft(pcm).permute(1, 2, 0) for pcm in
                      output]

            n = spects[0].size(0)
            spects = torch.stack(spects, dim=1).view(n, -1)
            dummy = torch.ones(1, spects.size(1))
            spects = torch.cat((dummy, spects))
        return spects

    #@profile
    def _stft(self, pcm):
        """Utilizes the pytorch library for STFT transformation.

        Args:
            pcm: .wav file in as numpy array

        Return:
            spect: pytorch tensor with shape BMN2

        """
        pcm_tensor = pcm.float()
        spect = torch.stft(pcm_tensor, self._window_size, self._hop_length,
                           window=torch.hann_window(256))

        return spect


class WaveUNet(nn.Module):
    #@profile
    def __createDownSampling__(self):
        self.module.downsampling_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            in_channels = self.input_channels if i == 0 else self.num_channels[i]
            block = Blocks.DownSamplingBlock(self.kernel_size, self.strides[i], in_channels, 
                        self.num_channels[i], self.num_channels[i+1], self.pre_depth, self.post_depth)
            self.module.downsampling_blocks.append(block)

    #@profile
    def __createBottleNeck__(self):
        self.module.bottleneck_block =  nn.ModuleList()
        for d in range(self.pre_depth):
            layer = ConvLayer.Conv1Layer(self.num_channels[-1], self.num_channels[-1], kernel_size=self.kernel_size, stride=1)
            self.module.bottleneck_block.append(layer)

    #@profile
    def __createUpSampling__(self):
        self.module.upsampling_blocks = nn.ModuleList()
        n = self.num_levels
        for i in range(self.num_levels - 1):
            block = Blocks.UpSamplingBlock(self.kernel_size, self.strides[-1-i], self.num_channels[-1-i], 
                        self.num_channels[-2-i], self.num_channels[-2-i], self.pre_depth, self.post_depth)
            self.module.upsampling_blocks.append(block)

    #@profile
    def __createOutputLayer__(self):
        outputs = self.out_channels * (self.num_cats-1)
        self.module.outputLayer = ConvLayer.Conv1Layer(self.num_channels[0], outputs, self.kernel_size, 1)
    
    #@profile
    def __init__(self, in_channels, num_channels, out_channels, num_cats, kernel_size, target_output_size, pre_depth, post_depth, depth, dataset_info, stride_start=2, stride_change = 0):
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
        self.convert_output = OutputToSTFT(dataset_info['config'])
        self.stride_change = stride_change
        self.strides = [stride_start - i * self.stride_change for i in range(self.num_levels)]
        self.module = nn.Module()
        self.__createDownSampling__()
        self.__createBottleNeck__()
        self.__createUpSampling__()
        self.__createOutputLayer__()   
        self.set_output_size(target_output_size)
        self._gt_start = self.shapes['output_start_frame']
        self._gt_end = self.shapes['output_end_frame']
    
    #@profile
    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        self.input_size, self.output_size = self.check_padding(target_output_size)
        logging.info("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")
        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}
    #@profile
    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1
        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1
    #@profile
    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        try:
            curr_size = bottleneck

            for idx, block in enumerate(self.module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            curr_size = self.module.outputLayer.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(self.module.bottleneck_block):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(self.module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)
            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False
    #@profile
    def forward(self, x):
        shortcuts = []
        clipped_model_input = x[:, :, self._gt_start: self._gt_end][:,:, :self.target_output_size]
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
        model_output = self.module.outputLayer(x)[:, :, :self.target_output_size]
        #print(x.shape, clipped_model_input.shape, model_output.shape)
        all_ouput_sum =torch.sum(model_output, dim=1).reshape(model_output.shape[0], 1, -1)
        last_output = clipped_model_input -  all_ouput_sum
        model_output = torch.cat((model_output, last_output), dim=1).float()
        torch.cuda.empty_cache() 

        #model_output = self.convert_output(model_output)
        #stft_input = self._convert_input(clipped_model_input)
        return model_output
        #clipped_model_input
        
    #@profile
    def _convert_input(self, input_val):
        all_spects = []
        print(input_val.shape)
        input_val = input_val.reshape(input_val.shape[0], input_val.shape[2])
        for pcm in input_val: 
            pcm_tensor = pcm.float()
            spect = torch.stft(pcm_tensor, 256, 192,
                           window=torch.hann_window(256))
            spect = spect.permute(2, 1, 0)
            all_spects.append(spect)
        return torch.stack(all_spects, dim =0)

        ''' TODO

        elif input_dimensions == 'BN(2M)':
            n = spect.size(1)
            spect = spect.permute(1, 2, 0).reshape(n, -1)
            # print('o')
        '''
        return spect

class WeightedCombination(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()
    
    def forward(self, tensors, weights):
        mult = (w*(tensors.T)).T
        return torch.sum(mult, dim =0)

'''
class WeightedWaveUNet(nn.Module):
    def __createDownSampling__(self, module):
        module.downsampling_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            in_channels = self.input_channels if i == 0 else self.num_channels[i]
            block = Blocks.DownSamplingBlock(self.kernel_size, self.strides[i], in_channels, 
                        self.num_channels[i], self.num_channels[i+1], self.pre_depth, self.post_depth)
            module.downsampling_blocks.append(block)
        
    
    def __createBottleNeck__(self, module):
        module.bottleneck_block =  nn.ModuleList()
        for d in range(self.pre_depth):
            layer = ConvLayer.Conv1Layer(self.num_channels[-1], self.num_channels[-1], kernel_size=self.kernel_size, stride=1)
            module.bottleneck_block.append(layer)
    
    def __createUpSampling__(self, module):
        module.upsampling_blocks = nn.ModuleList()
        n = self.num_levels
        for i in range(self.num_levels - 1):
            block = Blocks.UpSamplingBlock(self.kernel_size, self.strides[-1-i], self.num_channels[-1-i], 
                        self.num_channels[-2-i], self.num_channels[-2-i], self.pre_depth, self.post_depth)
            module.upsampling_blocks.append(block)
    
    def __createOutputLayer__(self, module):
        outputs = self.out_channels * (self.num_cats-1)
        module.outputLayer = ConvLayer.Conv1Layer(self.num_channels[0], outputs, self.kernel_size, 1)
    
    def __init__(self, in_channels, num_channels, out_channels, num_cats, kernel_size, target_output_size, pre_depth, post_depth, depth, stride_start=2, stride_change = 0,  num_models=1, num_ont_cats=6):
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
        self.model_list = []
        self.model_classifier = nn.linear(num_ont_cats, num_models)
        self.waveunets = nn.ModuleDict()
        self.num_models = num_models
        self.output_layer = WeightedCombination()
        for model_num in range(num_models):
            module = nn.Module()
            self.__createDownSampling__(module)
            self.__createBottleNeck__(module)
            self.__createUpSampling__(module)
            self.__createOutputLayer__(module)   
            self.waveunets[model_num] = module
        
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
        module = self.waveunets[0]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            curr_size = module.outputLayer.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottleneck_block):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)
            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    
    inputs: 
        x - pcm
        label_values - the transformer outputs for the ontological categories in the model
    

    def forward(self, x, label_values):
        shortcuts = []
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) 
        model_wts = torch.sigmoid(self.model_classifier(label_values))
        # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size
        all_outs = []
        for model_num in range(self.num_models): 
            module = self.waveunets[model_num]
            for block in module.downsampling_blocks:
                x, shortcut =  block(x)
                shortcuts.append(shortcut)
            shortcuts.reverse()
            for block in module.bottleneck_block:
                x = block(x)
            i = 0
            for block in module.upsampling_blocks:
                x = block(x, shortcuts[i])
                i += 1
            x = module.outputLayer(x)
            all_outs.append(x)
        torch_out = torch.stack(all_outs, dim = 0)
        output = self.output_layer(torch_out, model_wts)
        return output
'''


class WeightedWaveUNet(nn.Module):
    def __init__(self, in_channels, num_channels, out_channels, num_cats, kernel_size, target_output_size, pre_depth, post_depth, depth, stride_start=2, stride_change = 0,  num_models=1, num_ont_cats=6):
        super(WeightedWaveUNet, self).__init__()
        self.waveunets = nn.ModuleList()
        self.model_classifier = nn.linear(num_ont_cats, num_models)
        self.num_models = num_models
        self.output_layer = WeightedCombination()

        for model_num in range(num_models):
            model = WaveUNet(in_channels, num_channels, out_channels, num_cats, kernel_size, target_output_size, pre_depth, post_depth, depth, dataset_info, stride_start, stride_change )
            self.waveunets.append(model)
        self._gt_start = self.waveunets[0]._gt_start
        self._gt_end = self.waveunets[0]._gt_end
        self._target_output_size = self.waveunets[0]._target_output_size

    def forward(self, x, labels):
        model_wts = torch.softmax(self.model_classifier(label_values))
        clipped_model_input = x[:, :, self._gt_start: self._gt_end][:,:, :self.target_output_size]
        all_outs = []
        for model in self.waveunets: 
            x = model(x)
            all_outs.append(x)
        torch_out = torch.stack(all_outs, dim = 0)
        output = self.output_layer(torch_out, model_wts)
        stft_input = self._convert_input(clipped_model_input)
        return model_output, stft_input

    def _convert_input(self, pcm_tensor):
        all_spects = []
        for x in pcm_tensor: 
            pcm_tensor = pcm.float()
            spect = torch.stft(pcm_tensor, 256, 192,
                           window=torch.hann_window(256))
            spect = spect.permute(2, 1, 0)
            all_spects.append(spect)
        return torch.stack(all_spects, dim =0)
