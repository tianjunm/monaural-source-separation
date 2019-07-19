import torch
import torch.nn as nn
import torch.nn.functional as F


class SRNN(nn.Module):
    '''sRNN introduced by Huang et al. Full-RNN'''

    def __init__(self,
            input_dim,
            num_sources,
            hidden_size,
            dropout,
            num_layers=3,
            nonlinearity='relu'):

        super().__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        self.rnn = nn.RNN(input_dim, hidden_size, num_layers,
                          nonlinearity=nonlinearity,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_sources * input_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        x, _ = self.rnn(x)
        x = F.relu(self.fc(x))
        x = x.view(batch_size, -1, self.num_sources, self.input_dim)
        # smoothing
        noms = torch.norm(x, dim=-1)
        denoms = torch.sum(torch.norm(x, dim=-1), dim=-1)
        mask = noms / denoms.unsqueeze(-1)
        ys = mask.unsqueeze(-1) * x
        return ys
