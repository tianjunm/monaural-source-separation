"""Implementation of customized loss functions.
"""


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

    def __init__(self):
        super().__init__()

    @helpers.perm_invariant
    def forward(self, model_input, model_output, ground_truths):
        """Calculates the loss using the cost function of cSA-based method.

        Assumes that features are represented as spectrograms.

        Args:
            model_input: [b * 2 * n * m]
            model_output: [b * c * 2 * n * m]
            ground_truths: [b * c * 2 * n * m]

        Returns:
            loss: [batch_size]

        """
        Y_r, Y_i = model_input[:, 0], model_input[:, 1]
        M_r, M_i = model_output[:, :, 0], model_output[:, :, 1]
        S_r, S_i = ground_truths[:, :, 0], ground_truths[:, :, 1]

        J_1 = M_r * Y_r - M_i * Y_i - S_r
        J_2 = M_r * Y_i + M_i * Y_r - S_i

        return J_1 + J_2
        # print(model_input)
