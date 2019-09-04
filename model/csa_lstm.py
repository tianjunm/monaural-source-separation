import torch
import torch.nn as nn
import torch.nn.functional as F


class CSALSTM(nn.Module):
    """cSA-based LSTM"""

    def __init__(self, input_dim, num_sources, hidden_size=512):

        super().__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.lstm_real = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.lstm_imag = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc_real = nn.Linear(hidden_size, num_sources * input_dim)
        self.fc_imag = nn.Linear(hidden_size, num_sources * input_dim)


    def forward(self, x):
        batch_size = x.size()[0]

        x_real = x[:, 0]
        x_imag = x[:, 1]

        x = torch.cat([x_real, x_imag], dim=-1)

        out_real, _ = self.lstm_real(x_real)
        out_imag, _ = self.lstm_imag(x_imag)

        y_real = self.fc_real(F.relu(out_real))
        y_imag = self.fc_imag(F.relu(out_imag))

        y_real = y_real.view(batch_size, -1, self.num_sources, self.input_dim)
        y_imag = y_imag.view(batch_size, -1, self.num_sources, self.input_dim)

        y = torch.cat([y_real, y_imag], -1)

        return x.unsqueeze(2) * y
