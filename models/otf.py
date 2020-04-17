import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self, input_size, num_sources, dmodel, num_heads, num_layers,
                 hidden_size, dropout=0.5):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.m = input_size
        self.s = num_sources
        self.dmodel = dmodel

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(dmodel, dropout)
        encoder_layers = TransformerEncoderLayer(dmodel, num_heads,
                                                 hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.encoder = nn.Linear(2 * input_size, dmodel)
        self.decoder = nn.Linear(dmodel, 2 * self.s * self.m)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

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

        # if self.src_mask is None:
        #     device = x.device
        #     mask = self._generate_square_subsequent_mask(n).to(device)
        #     self.src_mask = mask

        x = x.permute(2, 0, 1, 3).reshape(n, b, -1)

        x = self.encoder(x) * math.sqrt(self.dmodel)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, self.src_mask)
        model_output = self.decoder(out).view(n, b, 2, self.s, self.m).permute(
            1, 3, 2, 0, 4)

        return model_output


class CNNTransformer(nn.Module):

    def __init__(self, input_size, num_sources, dmodel, num_heads,
                 hidden_size, num_layers, fc_dim, chan, dropout):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.m = input_size
        self.s = num_sources
        self.chan = chan

        self.kernel_dims = [(1, 7), (7, 1), (5, 5),
                            (5, 5), (5, 5), (5, 5), (5, 5)]
        self.dilation_dims = [(1, 1), (1, 1), (1, 1),
                              (2, 2), (4, 4), (8, 8), (1, 1)]

        assert(len(self.kernel_dims) == len(self.dilation_dims))

        self.num_conv_layers = len(self.kernel_dims)
        self.convs = nn.ModuleList(self._construct_convs())
        self.bns = nn.ModuleList(self._construct_bns())

        self.dmodel = dmodel

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(dmodel, dropout)
        encoder_layers = TransformerEncoderLayer(dmodel, num_heads,
                                                 hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.encoder = nn.Linear(8 * self.m, dmodel)
        # self.decoder = nn.Linear(dmodel, 2 * self.c * self.m)
        self.fc0 = nn.Linear(dmodel, fc_dim)
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.s * 2 * self.m)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def _construct_convs(self):
        convs = []
        for i, kernel_size in enumerate(self.kernel_dims):
            in_chan = 2 if i == 0 else self.chan
            out_chan = 8 if i == self.num_conv_layers - 1 else self.chan

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
        for i in range(self.num_conv_layers):
            chan = 8 if i == self.num_conv_layers - 1 else self.chan
            bn = nn.BatchNorm2d(chan)
            bns.append(bn)
        return bns

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc0.bias.data.zero_()
        self.fc0.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """

        Assumes that features are represented as spectrograms.

        Args:
            x: input mixture [b * 2 * n * m]

        Returns:
            model_output: [b * c * 2 * n * m]

        """
        b = x.size(0)
        n = x.size(2)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        x = torch.cat(list(x.permute(1, 0, 2, 3)), 2)
        # if self.src_mask is None:
        #     device = x.device
        #     mask = self._generate_square_subsequent_mask(n).to(device)
        #     self.src_mask = mask

        x = x.permute(1, 0, 2)

        x = self.encoder(x) * math.sqrt(self.dmodel)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)

        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(n, b, 2, self.s, self.m).permute(1, 3, 2, 0, 4)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
