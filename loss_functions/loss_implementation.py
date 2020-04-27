"""Implementation of customized loss functions.
"""


import math
import torch
import torch.nn as nn
from . import helpers


class CSALoss(nn.Module):
    """Masking-based cIRM.

    Source: Monaural Source Separation in Complex Domain
            With Long Short-Term Memory Neural Network

    Computes the loss of both real and imaginary components separately.
    The loss is based on applying estimated cIRM to mixture spectrogram
    and comparing the result with the ground truth.
    """

    def __init__(self, dataset_config, no_pit=False):
        super().__init__()
        self.s = dataset_config['num_sources']
        self.input_dimensions = dataset_config['input_dimensions']
        self.output_dimensions = dataset_config['output_dimensions']
        self.no_pit = no_pit

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

        # norm_factor = math.sqrt(s * n * m)
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
        Y = model_input.unsqueeze(1)

        s = model_output.size(1)
        n = model_input.size(2)
        m = model_input.size(3)

        norm_factor = math.sqrt(s * n * m)
        loss = torch.norm((Y * model_output - ground_truths) /
                          norm_factor, dim=[3, 4]).mean(axis=[1, 2])

        return loss


class NoMask(nn.Module):
    def __init__(self, dataset_config):
        super().__init__()
        self.s = dataset_config['num_sources']
        self.input_dimensions = dataset_config['input_dimensions']
        self.output_dimensions = dataset_config['output_dimensions']

    @helpers.perm_invariant_nomask
    def forward(self, prediction, ground_truths):
        """Calculates the loss using the difference only.

        Assumes that features are represented as spectrograms.

        Args:
            model_input: [b * 2 * n * m]
            model_output: [b * s * 2 * n * m]
            ground_truths: [b * s * 2 * n * m]

        Returns:
            loss: [batch_size]

        """

        n = ground_truths.size(2)
        m = ground_truths.size(3)

        norm_factor = math.sqrt(self.s * n * m)
        loss = torch.norm((prediction - ground_truths) /
                          norm_factor, dim=[3, 4]).mean(axis=[1, 2])

        return loss


class UniSrc(nn.Module):
    def __init__(self, dataset_config):
        super().__init__()
        self.input_dimensions = dataset_config['input_dimensions']
        self.output_dimensions = dataset_config['output_dimensions']

    def forward(self, model_input, model_output, ground_truths):
        """Calculates the loss using the cost function of cSA-based method.

        Assumes that features are represented as spectrograms.

        Args:
            model_input: [b * 2 * n * m]
            model_output: [b * 2 * n * m]
            ground_truths: [b * 2 * n * m]

        Returns:
            loss: [batch_size]

        """

        Y_r = model_input[:, 0].unsqueeze(1)
        Y_i = model_input[:, 1].unsqueeze(1)

        M_r, M_i = model_output[:, 0], model_output[:, 1]
        S_r, S_i = ground_truths[:, 0], ground_truths[:, 1]

        n = model_input.size(1)
        m = model_input.size(2)

        norm_factor = math.sqrt(n * m)

        J_1 = torch.norm((M_r * Y_r - M_i * Y_i - S_r) / norm_factor,
                         dim=[1, 2])
        J_2 = torch.norm((M_r * Y_i - M_i * Y_r - S_i) / norm_factor,
                         dim=[1, 2])

        loss = (J_1 + J_2) / 2

        return loss
