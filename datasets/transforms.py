"""Transformations of data before loading into models."""


import torch
import numpy as np
import torch.nn as nn
import logging

from guppy import hpy
import sys
import torch
import gc
from memory_profiler import profile


class STFT():
    """Transforms PCM files to spectrogram tensors using STFT."""

    def __init__(self, dataset_info):
        # pre-defined STFT parameters
        self._info = dataset_info
        self._window_size = 256
        self._hop_length = 192


    def __call__(self, item):
        input_dimensions = self._info['config']['input_dimensions']
        output_dimensions = self._info['config']['output_dimensions']

        model_input = self._convert_input(item['model_input'],
                                          input_dimensions)
        ground_truths = self._convert_gt(item['ground_truths'],
                                         output_dimensions)
        transformed = {
            'model_input': model_input,
            'ground_truths': ground_truths,
            'component_info': item['component_info']
        }

        return transformed

    def _convert_input(self, model_input, input_dimensions):
        spect = self._stft(model_input)

        if input_dimensions == 'B2NM':
            spect = spect.permute(2, 1, 0)

        elif input_dimensions == 'BN(2M)':
            n = spect.size(1)
            spect = spect.permute(1, 2, 0).reshape(n, -1)
            # print('o')

        return spect

    def _convert_gt(self, ground_truths, output_dimensions):
        if output_dimensions == 'BS2NM':
            spects = [self._stft(pcm).permute(2, 1, 0) for pcm in
                      ground_truths]
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
                      ground_truths]

            n = spects[0].size(0)
            spects = torch.stack(spects, dim=1).view(n, -1)
            dummy = torch.ones(1, spects.size(1))
            spects = torch.cat((dummy, spects))

        return spects

    def _stft(self, pcm):
        """Utilizes the pytorch library for STFT transformation.

        Args:
            pcm: .wav file in as numpy array

        Return:
            spect: pytorch tensor with shape BMN2

        """
        pcm_tensor = torch.from_numpy(pcm).float()
        spect = torch.stft(pcm_tensor, self._window_size, self._hop_length,
                           window=torch.hann_window(256))

        return spect

TEMP_ROOT = '/work/sbali/monaural-source-separation/TEMP'


class WaveunetWindow:
    def __init__(self, dataset_info):
        self._gt_start = dataset_info['gt_start']
        self._gt_end = dataset_info['gt_end']
        self._agg_len = dataset_info['agg_len']
        self._info = dataset_info
        self._window_size = 256
        self._hop_length = 192
        self.i = 0
   
    #@profile
    def __call__(self, item):
        output_dimensions = self._info['config']['output_dimensions']
        input_dimensions = self._info['config']['input_dimensions']
        model_input = self._convert_input(item)
        ground_truths = self._convert_gt(item)
        transformed =  {
            'model_input': model_input,
            'ground_truths': ground_truths,
            'component_info': item['component_info']
        }
        return transformed
    #@profile
    def _convert_input(self, item):
        model_input = np.zeros(self._agg_len)
        total_input = item['model_input']
        gt_len_needed = self._gt_end-self._gt_start
        end = min(gt_len_needed, len(total_input))
        model_input[self._gt_start: end+ self._gt_start] = total_input[0: end]
        return torch.from_numpy(model_input).float()

    #@profile  
    def _convert_gt(self,item ):
        return torch.tensor(np.stack(item['ground_truths'], axis=0)).float()
        '''
        if output_dimensions == 'BS2NM':
            spects = [self._stft(pcm).permute(2, 1, 0) for pcm in
                      ground_truths]
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
                      ground_truths]

            n = spects[0].size(0)
            spects = torch.stack(spects, dim=1).view(n, -1)
            dummy = torch.ones(1, spects.size(1))
            spects = torch.cat((dummy, spects))
        return spects
        '''
    
    #@profile
    def _stft(self, pcm):
        """Utilizes the pytorch library for STFT transformation.

        Args:
            pcm: .wav file in as numpy array

        Return:
            spect: pytorch tensor with shape BMN2

        """
        pcm_tensor = torch.from_numpy(pcm).float()
        spect = torch.stft(pcm_tensor, self._window_size, self._hop_length,
                           window=torch.hann_window(256))

        return spect


class SSTFT():
    """Transforms PCM files to spectrogram tensors using STFT."""

    def __init__(self, dataset_info):
        # pre-defined STFT parameters
        self._info = dataset_info
        self._window_size = 256
        self._hop_length = 192


    def __call__(self, item):
        input_dimensions = self._info['config']['input_dimensions']
        output_dimensions = self._info['config']['output_dimensions']
        model_input = self._convert_input(item,
                                          input_dimensions)

        return model_input

    def _convert_input(self, model_input, input_dimensions):
        spect = self._stft(model_input)

        if input_dimensions == 'B2NM':
            spect = spect.permute(2, 1, 0)

        elif input_dimensions == 'BN(2M)':
            n = spect.size(1)
            spect = spect.permute(1, 2, 0).reshape(n, -1)
            # print('o')
        return spect

    def _stft(self, pcm):
        """Utilizes the pytorch library for STFT transformation.

        Args:
            pcm: .wav file in as numpy array

        Return:
            spect: pytorch tensor with shape BMN2

        """
        pcm_tensor = torch.from_numpy(pcm).float()
        spect = torch.stft(pcm_tensor, self._window_size, self._hop_length,
                           window=torch.hann_window(256))

        return spect
