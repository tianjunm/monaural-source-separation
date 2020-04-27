"""Transformations of data before loading into models."""


import torch


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

        elif output_dimensions == 'B2NM':
            spects = self._stft(ground_truths).permute(2, 1, 0)

        elif output_dimensions == 'BN(2M)':
            spects = self._stft(ground_truths).permute(1, 2, 0)
            n = spects.size(0)
            spects = spects.reshape(n, -1)

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
