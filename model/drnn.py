import torch
import torch.nn as nn
import torch.nn.functional as F


class DRNN(nn.Module):
    '''DRNN-k introduced by Huang et al.'''

    def  __init__(self,
            input_dim,
            num_sources,
            hidden_size,
            k,
            dropout,
            num_layers=3,
            nonlinearity='relu'):
        # assert(num_layers >= NUM_LAYERS)
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.num_sources = num_sources
        self._init_layers(input_dim, num_sources, hidden_size, num_layers,
                k, nonlinearity, dropout)


    def forward(self, x):
        batch_size = x.size()[0]
        for i, layer in enumerate(self.layers):
            if i == self.k - 1:
                x, _ = layer(x)
            else:
                x = F.relu(layer(x))
        x = x.view(batch_size, -1, self.num_sources, self.input_dim)
        # smoothing
        noms = torch.norm(x, dim=-1)
        denoms = torch.sum(torch.norm(x, dim=-1), dim=-1)
        mask = noms / denoms.unsqueeze(-1)
        ys = mask.unsqueeze(-1) * x
        return ys

    def _init_layers(self,
            input_dim,
            num_sources,
            hidden_size,
            num_layers,
            k,
            nonlinearity,
            dropout):

        layers = []

        if k == 1:
            # define bottom layer
            layers.append(nn.RNN(input_dim, hidden_size, num_layers=1,
                                 nonlinearity=nonlinearity, batch_first=True))
            # fill the rest
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))

        else:
            for _ in range(k - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))

            # insert rnn
            layers.append(nn.RNN(hidden_size, hidden_size, num_layers=1,
                                 nonlinearity=nonlinearity, batch_first=True))

            # fill the rest
            for _ in range(num_layers - k):
                layers.append(nn.Linear(hidden_size, hidden_size))

        # linear transformation
        layers.append(nn.Linear(hidden_size, num_sources * input_dim))

        assert(len(layers) == num_layers + 1)
        self.layers = nn.ModuleList(layers)

