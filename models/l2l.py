import torch
import torch.nn as nn
import torch.nn.functional as F


class L2LAudio(nn.Module):

    def __init__(self, input_size, num_sources, hidden_size=256,
                 num_layers=1, fc_dim=512, chan=6):
        super().__init__()
        self.m = input_size
        self.s = num_sources
        self.chan = chan

        self.kernel_dims = [(1, 7), (7, 1), (5, 5),
                            (5, 5), (5, 5), (5, 5), (5, 5)]
        self.dilation_dims = [(1, 1), (1, 1), (1, 1),
                              (2, 2), (4, 4), (8, 8), (1, 1)]

        assert(len(self.kernel_dims) == len(self.dilation_dims))

        self.num_layers = len(self.kernel_dims)
        self.convs = nn.ModuleList(self._construct_convs())
        self.bns = nn.ModuleList(self._construct_bns())

        self.blstm = nn.LSTM(8 * self.m, hidden_size, num_layers,
                             batch_first=True, bidirectional=True)

        self.fc0 = nn.Linear(2 * hidden_size, fc_dim)
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.s * 2 * self.m)

    def forward(self, x):
        """

        Assumes that features are represented as spectrograms.

        Args:
            x: input mixture [b * 2 * n * m]

        Returns:
            model_output: [b * s * 2 * n * m]

        """
        b = x.size(0)
        n = x.size(2)

        # x = x.permute(0, 1, 3, 2)
        # x = torch.cat(list(x.permute(1, 0, 2, 3)), -1)

        # dilated convolutional network
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        # [b, out_chan, n, m] --> [b, n, out_chan * m]
        # x = x.permute(0, 2, 1, 3).reshape(b, n, -1)
        x = torch.cat(list(x.permute(1, 0, 2, 3)), 2)

        # bidirectional lstm
        self.blstm.flatten_parameters()
        x, _ = self.blstm(x)
        x = F.relu(x)

        # fcs
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(b, n, self.s, 2, self.m).permute(0, 2, 3, 1, 4)

        return x

    def _construct_convs(self):
        convs = []
        for i, kernel_size in enumerate(self.kernel_dims):
            in_chan = 2 if i == 0 else self.chan
            out_chan = 8 if i == self.num_layers - 1 else self.chan

            dilation = self.dilation_dims[i]

            rpad = dilation[0] * (kernel_size[0] - 1) // 2
            cpad = dilation[1] * (kernel_size[1] - 1) // 2
            padding = [rpad, cpad]

            conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size,
                             dilation=dilation, padding=padding)
            convs.append(conv)
        return convs

    def _construct_bns(self):
        bns = []
        for i in range(self.num_layers):
            chan = 8 if i == self.num_layers - 1 else self.chan
            bn = nn.BatchNorm2d(chan)
            bns.append(bn)
        return bns
