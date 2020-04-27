"""Implementation of customized loss functions.
"""


import math
import torch
import torch.nn as nn
from . import helpers
import logging


class CSALoss(nn.Module):
    """Masking-based cIRM.

    Source: Monaural Source Separation in Complex Domain
            With Long Short-Term Memory Neural Network

    Computes the loss of both real and imaginary components separately.
    The loss is based on applying estimated cIRM to mixture spectrogram
    and comparing the result with the ground truth.
    """

    def __init__(self, dataset_config, waveunet=False):
        
        super().__init__()
        self.s = dataset_config['num_sources']
        self.input_dimensions = dataset_config['input_dimensions']
        self.output_dimensions = dataset_config['output_dimensions']
        self.iswaveunet = waveunet

    @helpers.perm_invariant
    def forward(self, model_input, model_output, ground_truths):
        
        """Calculates the loss using the cost function of cSA-based method.

        Assumes that features are represented as spectrograms.

        Args:
            model_input: [b * 2 * n * m]
            model_output: [b * s * 2 * n * m]
            ground_truths: [b * s * 2 * n * m]

        Returns:
            loss: [batch_size]

        """

        print(model_output.shape, model_input.shape, ground_truths.shape)

        # print(model_input.size())
        # print(model_output.size())
        # print(ground_truths.size())

        Y_r = model_input[:, 0].unsqueeze(1)
        Y_i = model_input[:, 1].unsqueeze(1)

        M_r, M_i = model_output[:, :, 0], model_output[:, :, 1]
        S_r, S_i = ground_truths[:, :, 0], ground_truths[:, :, 1]

        s = model_output.size(1)
        n = model_input.size(2)
        m = model_input.size(3)

        norm_factor = math.sqrt(n * m)
        # J_1 = ((M_r * Y_r - M_i * Y_i - S_r) ** 2).mean(axis=[1, 2, 3])
        # J_2 = ((M_r * Y_i + M_i * Y_r - S_i) ** 2).mean(axis=[1, 2, 3])
        J_1 = torch.norm((M_r * Y_r - M_i * Y_i - S_r) / norm_factor,
                         dim=[2, 3]).mean(axis=1)
        J_2 = torch.norm((M_r * Y_i - M_i * Y_r - S_i) / norm_factor,
                         dim=[2, 3]).mean(axis=1)

        loss = (J_1 + J_2) / 2

        return loss


class Difference(nn.Module):
    """Masking-based

    Computes the loss of both real and imaginary components separately.
    The loss is based on the difference between the predicted spectrogram
    and the ground truth.
    """

    def __init__(self, dataset_config):
        
        super().__init__()
        self.s = dataset_config['num_sources']
        self.input_dimensions = dataset_config['input_dimensions']
        self.output_dimensions = dataset_config['output_dimensions']
            


    @helpers.perm_invariant
    def forward(self, model_input, model_output, ground_truths):
        
        """Calculates the loss using the difference only.

        Assumes that features are represented as spectrograms.

        Args:
            model_input: [b * 2 * n * m]
            model_output: [b * s * 2 * n * m]
            ground_truths: [b * s * 2 * n * m]

        Returns:
            loss: [batch_size]

        """
        
        # if (self.input_dimensions == 'BN(2M)' and
        #    self.output_dimensions == 'BN(S2M)'):
        #     b = model_input.size(0)
        #     n = model_input.size(1)

        #     model_input = model_input.view(b, n, 2, -1).permute(0, 2, 1, 3)
        #     model_output = model_output.view(b, n, self.s, 2, -1).permute(
        #         0, 2, 3, 1, 4)
        #     ground_truths = ground_truths.view(b, n, self.s, 2, -1).permute(
        #         0, 2, 3, 1, 4)
        # c = model_output.size(1)

        # Y_r = model_input[:, 0].unsqueeze(1)
        # Y_i = model_input[:, 1].unsqueeze(1)

        Y = model_input.unsqueeze(1)

        s = model_output.size(1)
        n = model_input.size(2)
        m = model_input.size(3)

        norm_factor = math.sqrt(s * n * m)
        # J_1 = torch.norm((M_r * Y_r - S_r) / norm_factor)
        # J_2 = torch.norm((M_i * Y_i - S_i) / norm_factor)
        loss = torch.norm((Y * model_output - ground_truths) /
                          norm_factor, dim=[3, 4]).mean(axis=[1, 2])

        # M_r, M_i = model_output[:, :, 0], model_output[:, :, 1]
        # S_r, S_i = ground_truths[:, :, 0], ground_truths[:, :, 1]

        # J_1 = ((M_r * Y_r - S_r) ** 2).mean(axis=[1, 2, 3])
        # J_2 = ((M_i * Y_i - S_i) ** 2).mean(axis=[1, 2, 3])

        # loss = (J_1 + J_2) / c
        # loss = (J_1 + J_2) / 2

        return loss

        
class CatLoss(nn.Module):
    """Masking-based

    Computes the loss of both real and imaginary components separately.
    The loss is based on the difference between the predicted spectrogram
    and the ground truth.
    """

    def __init__(self):
        super().__init__()
        self.label_loss = nn.BCELoss()
        self.sum_loss = nn.MSELoss()

    def forward(self, model_output, one_hot_labels):
        loss1 = self.label_loss(model_output, one_hot_labels) 
        return loss1 
