import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    """DNN-based method"""

    def __init__(
            self,
            input_dim,
            num_sources,
            hidden_size=2048,
            num_layers=3):

        super().__init__()
        self._d = input_dim
        self._nsrc = num_sources
        self._h = hidden_size
        self._nl = num_layers
        self.layers = self._create_ffs()

    def _create_ffs(self):
        ffs = []
        for i in range(self._nl):
            if i == 0:
                layer = nn.Linear(self._d, self._h)

            elif i == self._nl - 1:
                layer = nn.Linear(self._h, self._nsrc * self._d)

            else:
                layer = nn.Linear(self._h, self._h)

            ffs.append(layer)

        return nn.ModuleList(ffs)

    def forward(self, x):
        mixture = x.clone().unsqueeze(2)

        for layer in self.layers:
            x = F.relu(layer(x))

        mask = x.view(mixture.size(0), -1, self._nsrc, self._d)
        pred = mixture * mask
        return pred
