import torch
import torch.nn as nn
import torch.nn.functional as F


class BLSTM(nn.Module):
    """BLSTM"""

    def __init__(self, input_size, num_sources, dmodel, hidden_size=512):
        super().__init__()
        self.m = input_size
        self.c = num_sources
        self.enc_r = nn.Linear(input_size, dmodel)
        self.enc_i = nn.Linear(input_size, dmodel)
        self.blstm_r = nn.LSTM(dmodel, hidden_size, batch_first=True,
                               bidirectional=True)
        self.blstm_i = nn.LSTM(dmodel, hidden_size, batch_first=True,
                               bidirectional=True)
        self.fc_r = nn.Linear(2 * hidden_size, self.c * self.m)
        self.fc_i = nn.Linear(2 * hidden_size, self.c * self.m)

    def forward(self, x):
        """

        Assumes that features are represented as spectrograms.

        Args:
            x: input mixture [b * 2 * n * m]

        Returns:
            model_output: [b * c * 2 * n * m]

        """
        # batch_size
        b = x.size(0)
        # seq_len
        n = x.size(2)

        x_r = x[:, 0]
        x_i = x[:, 1]

        # x = torch.cat([x_real, x_imag], dim=-1)

        out_r, _ = self.blstm_r(F.relu(self.enc_r(x_r)))
        out_i, _ = self.blstm_i(F.relu(self.enc_i(x_i)))

        M_r = self.fc_r(F.relu(out_r)).view(b, self.c, n, self.m)
        M_i = self.fc_i(F.relu(out_i)).view(b, self.c, n, self.m)

        model_output = torch.stack([M_r, M_i], dim=2)

        return model_output


class SingleBLSTM(nn.Module):
    """BLSTM that works with the combination of real and imaginary using one
    model

    """
    def __init__(self, input_size, num_sources, dmodel, hidden_size=512,
                 num_layers=3):
        super().__init__()
        self.m = input_size
        self.c = num_sources
        self.enc = nn.Linear(2 * input_size, dmodel)
        self.blstm = nn.LSTM(dmodel, hidden_size, num_layers, batch_first=True,
                             bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, 2 * self.c * self.m)

    def forward(self, x):
        """

        Assumes that features are represented as spectrograms.

        Args:
            x: input mixture [b * 2 * n * m]

        Returns:
            model_output: [b * c * 2 * n * m]

        """
        # batch_size
        b = x.size(0)
        # seq_len
        n = x.size(2)

        # x = x.permute(0, 2, 1, 3).view(b, n, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, n, 2 * self.m)

        # x = torch.cat([x_real, x_imag], dim=-1)

        out, _ = self.blstm(F.relu(self.enc(x)))

        model_output = self.fc(F.relu(out)).view(
            b, n, self.c, 2, self.m).permute(0, 2, 3, 1, 4)

        return model_output
