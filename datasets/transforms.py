"""Transformations of data before loading into models."""


import torch
import numpy as np


class STFT():
    """Transforms PCM files to spectrogram tensors using STFT."""

    def __init__(self, dataset_info):
        # pre-defined STFT parameters
        self._info = dataset_info
        self._window_size = 256
        self._self._hop_length = 192


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
        spect = torch.stft(pcm_tensor, self._window_size, self._self._hop_length,
                           window=torch.hann_window(256))

        return spect

class WaveunetWindow:
    def __init__(self, dataset_info):
        self._info = dataset_info
        self._window_size = 256
        self._hop_length = 192
    
    def __call__(self, item):
        curr_gt_start = self._info['gt_start']
        curr_gt_end = self._info['gt_end']
        curr_agg_start = 0
        agg_len = self._info['agg_len']
        lzero_end = curr_gt_start
        rzero_start = curr_gt_end
        gts = []
        aggs = []
        clipped_aggs = []


        MIXTURE_DURATION = len(item['model_input'])
        while curr_gt_end < MIXTURE_DURATION: 
            agg = np.zeros(agg_len)
            clipped_agg = item['model_input'][curr_gt_start:curr_gt_end]
            agg[lzero_end:rzero_start] = clipped_agg
            aggs.append(torch.tensor(agg))
            clipped_aggs.append(torch.DoubleTensor(clipped_agg))
            gt_new = []
            for gt in item['ground_truths']:
                gt_new.append(torch.tensor(gt[curr_gt_start:curr_gt_end]))
            gts.append(torch.stack(gt_new, dim = 0))
            curr_gt_start += self._hop_length
            curr_gt_end += self._hop_length


        model_input = torch.stack(aggs, dim = 0)
        ground_truths = torch.stack(gts, dim = 0)
        clipped_model_input = torch.stack(clipped_aggs, dim = 0)

        transformed = {
            'model_input': model_input,
            'ground_truths': ground_truths,
            'clipped_model_input': clipped_model_input,
            'component_info': item['component_info']
        }

        return transformed