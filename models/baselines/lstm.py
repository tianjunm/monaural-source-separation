import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """Jitong Chen, DeLiang Wang"""

    def __init__(
            self,
            input_dim,
            num_sources,
            hidden_size=512,
            num_layers=3):

        super().__init__()
        self._d = input_dim
        self._nsrc = num_sources
        self._h = hidden_size
        self._nl = num_layers
        self.fc = nn.Linear(self._d, self._d)
        self.reduc = nn.Linear(self._d, self._h)
        self.fc_out = nn.Linear(self._h, self._nsrc * self._d)
        self.lstms = self._create_lstms()

    def _create_lstms(self):
        lstms = []

        for i in range(self._nl):
            layer = nn.LSTM(self._h, self._h, batch_first=True)

            lstms.append(layer)

        return nn.ModuleList(lstms)

    def forward(self, x):
        bs = x.size(0)
        mixture = x.clone().unsqueeze(2)

        x = F.relu(self.fc(x))
        x = self.reduc(x)
        for layer in self.lstms:
            x, _  = layer(F.relu(x))

        mask = F.sigmoid(self.fc_out(F.relu(x)))
        pred = mixture * mask.view(bs, -1, self._nsrc, self._d)


        return pred



