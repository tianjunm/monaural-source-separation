import torch
import torch.nn as nn
import torch.nn.functional as F


class CSALSTM(nn.Module):
    """cSA-based LSTM"""

    def __init__(self, input_size, num_sources, embed_dim=1722,
                 hidden_size=512, output_dim=512, num_layers=3,
                 bidirectional=False):
        super().__init__()
        self.m = input_size
        self.s = num_sources

        self.input_r = nn.Linear(input_size, embed_dim)
        self.input_i = nn.Linear(input_size, embed_dim)

        self.lstm_r = nn.LSTM(embed_dim, hidden_size, num_layers,
                              batch_first=True, bidirectional=bidirectional)
        self.lstm_i = nn.LSTM(embed_dim, hidden_size, num_layers,
                              batch_first=True, bidirectional=bidirectional)

        input_dim = 2 * hidden_size if bidirectional else hidden_size
        self.output_r = nn.Linear(input_dim, output_dim)
        self.output_i = nn.Linear(input_dim, output_dim)

        self.fc_r = nn.Linear(output_dim, self.s * self.m)
        self.fc_i = nn.Linear(output_dim, self.s * self.m)

    def forward(self, x):
        """

        Assumes that features are represented as spectrograms.

        Args:
            x: input mixture [b * 2 * n * m]

        Returns:
            model_output: [b * s * 2 * n * m]

        """
        # batch_size
        b = x.size(0)
        # seq_len
        n = x.size(2)

        out_r, _ = self.lstm_r(torch.sigmoid(self.input_r(x[:, 0])))
        out_i, _ = self.lstm_i(torch.sigmoid(self.input_i(x[:, 1])))

        out_r = self.output_r(torch.sigmoid(out_r))
        out_i = self.output_i(torch.sigmoid(out_i))

        out_r = self.fc_r(torch.sigmoid(out_r)).view(b, self.s, n, self.m)
        out_i = self.fc_i(torch.sigmoid(out_i)).view(b, self.s, n, self.m)

        model_output = torch.stack([out_r, out_i], dim=2)

        return model_output
