import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# class DoubleEncoder(nn.Module):
#     "Encoder with temporal and spectral self-attention components."
#     def __init__(self, layer, layer_t, N):
#         super(DoubleEncoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.layer_ts = clones(layer_t, N)
#         self.norm = LayerNorm(layer.size)
#         self.tr = transpose

#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for i, layer in enumerate(self.layers):
#             x_t = x.clone().permute(0, 2, 1)

#             x = layer(x, mask)
#             x_t = self.layer_ts[i](x_t, mask)

#             if self.tr:
#                 x = x + x_t.permute(0, 2, 1)
#             else:
#                 x = x + x_t

#         return self.norm(x)


# class ConvEncoderLayer(nn.Module):
#     "Encoder is made up of self-attn and feed forward (defined below)"
#     def __init__(self, size, chan, c_out, d_out, ks2, self_attn, feed_forward,
#                  dropout):
#         super().__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.size = size
#         self.chan = chan
#         self.c_out = c_out
#         self.d_out = d_out
#         self.ks2 = ks2
#         self.convs = self.get_layers()
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)

#     def get_layers(self):
#         ks1 = self.size - self.d_out + self.ks2

#         conv1 = nn.Conv1d(self.chan, self.c_out, kernel_size=ks1)
#         bn1 = nn.BatchNorm1d(self.c_out)
#         conv2 = nn.ConvTranspose1d(self.c_out, self.chan, kernel_size=self.ks2)
#         bn2 = nn.BatchNorm1d(self.chan)
#         lut = nn.Linear(self.d_out, self.size)
#         return nn.Sequential(
#             conv1,
#             bn1,
#             nn.ReLU(),
#             conv2,
#             bn2,
#             nn.ReLU(),
#             lut)

#     def forward(self, x, mask):
#         "Follow Figure 1 (left) for connections."
#         x = self.convs(x)
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)


# class AAAI(nn.Module):

    
class Transformer(nn.Module):

    def __init__(self, input_size, num_sources, dmodel, num_heads, num_layers,
                 hidden_size, dropout=0.5, num_splits=2):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.m = input_size
        self.s = num_sources
        self.dmodel = dmodel
        self.ns = num_splits

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(dmodel, dropout)
        encoder_layers = TransformerEncoderLayer(dmodel, num_heads,
                                                 hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.encoder = nn.Linear(2 * input_size, self.ns * dmodel)
        self.decoder = nn.Linear(dmodel, 2 * self.s * self.m // self.ns)

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

        if self.src_mask is None:
            device = x.device
            mask = self._generate_square_subsequent_mask(
                self.ns * n).to(device)
            self.src_mask = mask

        x = x.permute(2, 0, 1, 3).reshape(n, b, -1)

        x = self.encoder(x)
        # x = self.pos_encoder(x)

        encoded_chunks = []
        for chunk in x.split(self.dmodel, dim=-1):
            encoded_chunks.append(self.pos_encoder(chunk) *
                                  math.sqrt(self.dmodel))

        x = torch.cat(encoded_chunks, dim=0)

        x = self.decoder(self.transformer_encoder(x, self.src_mask))

        x = torch.cat(x.split(n, dim=0), dim=-1).view(
            n, b, self.s, 2, self.m).permute(1, 2, 3, 0, 4)

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
