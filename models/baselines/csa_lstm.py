import torch
import torch.nn as nn
import torch.nn.functional as F


class CSALSTM(nn.Module):
    """cSA-based LSTM"""

    def __init__(self, input_size, num_sources, hidden_size=512):

        super().__init__()
        self.m = input_size
        self.c = num_sources
        self.lstm_r = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_i = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_r = nn.Linear(hidden_size, self.c * self.m)
        self.fc_i = nn.Linear(hidden_size, self.c * self.m)

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

        out_r, _ = self.lstm_r(x_r)
        out_i, _ = self.lstm_i(x_i)

        M_r = self.fc_r(F.relu(out_r)).view(b, self.c, n, self.m)
        M_i = self.fc_i(F.relu(out_i)).view(b, self.c, n, self.m)

        model_output = torch.stack([M_r, M_i], dim=1)

        return model_output
