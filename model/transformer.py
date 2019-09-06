"""
MIT License

Copyright (c) 2018 Alexander Rush
Copyright (c) 2019 Tianjun Ma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


D_MODEL = 512
D_FF = 1024
H = 4


"UTILITIES"


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# TODO: harvard paper has mask = 1 everyehere, is it the same with mask=None?
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def make_model(input_dim, N=3, d_model=D_MODEL, d_ff=D_FF, h=H,
        num_sources=2, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # FIXME: experimenting with num_sources
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, input_dim), c(position)),
        nn.Sequential(Embeddings(d_model, input_dim,
            num_sources=num_sources), c(position)),
        Generator(d_model, input_dim, num_sources=num_sources))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def make_stt(
        input_dim,
        seq_len,
        stt_type="STT1",
        N=3,
        d_model=D_MODEL,
        d_ff=D_FF,
        h=H,
        num_sources=2,
        dropout=0.1,
        cr_args=None,
        res_size=None):
    """Helper: Construct a double-encoder model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    # XXX: h for attn_t is currently a fixed param
    attn_t = MultiHeadedAttention(4, seq_len)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_t = PositionwiseFeedForward(seq_len, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)
    # FIXME: experimenting with num_sources
    if stt_type == "STT1":
        model = EncoderDecoder(
            DoubleEncoder(
                EncoderLayer(d_model, c(attn), c(ff), dropout),
                EncoderLayer(seq_len, c(attn_t), c(ff_t), dropout),
                N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(
                Embeddings(d_model, input_dim, num_sources=num_sources)),
            Generator(d_model, input_dim, num_sources=num_sources))

    if stt_type == "STT1-CR":
        model = EncoderDecoder(
            DoubleEncoder(
                ConvEncoderLayer(
                    d_model,
                    seq_len,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn),
                    c(ff),
                    dropout),
                ConvEncoderLayer(
                    seq_len,
                    d_model,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn_t),
                    c(ff_t),
                    dropout),
                N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(
                Embeddings(d_model, input_dim, num_sources=num_sources),
                c(position)),
            Generator(
                d_model,
                input_dim,
                num_sources=num_sources,
                res=True,
                res_size=res_size))

    if stt_type == "STT2tp":
        model = EncoderDecoder(
            DoubleTPEncoder(
                d_model,
                ConvEncoderLayer(
                    d_model,
                    seq_len,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn),
                    c(ff),
                    dropout),
                ConvEncoderLayer(
                    d_model,
                    seq_len,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn),
                    c(ff),
                    dropout),
                N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(
                Embeddings(d_model, input_dim, num_sources=num_sources),
                c(position)),
            Generator(
                d_model,
                input_dim,
                num_sources=num_sources,
                res=True,
                res_size=res_size))

    if stt_type == "STT2sp":
        model = EncoderDecoder(
            DoubleSPEncoder(
                d_model,
                ConvEncoderLayer(
                    seq_len,
                    d_model,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn_t),
                    c(ff_t),
                    dropout),
                ConvEncoderLayer(
                    seq_len,
                    d_model,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn_t),
                    c(ff_t),
                    dropout),
                N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(
                Embeddings(d_model, input_dim, num_sources=num_sources),
                c(position)),
            Generator(
                d_model,
                input_dim,
                num_sources=num_sources,
                res=True,
                res_size=res_size))

    if stt_type == "STTtp":
        model = EncoderDecoder(
            Encoder(
                ConvEncoderLayer(
                    d_model,
                    seq_len,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn),
                    c(ff),
                    dropout),
                N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(
                Embeddings(d_model, input_dim, num_sources=num_sources),
                c(position)),
            Generator(
                d_model,
                input_dim,
                num_sources=num_sources,
                res=True,
                res_size=res_size))

    if stt_type == "STTsp":
        model = EncoderDecoder(
            TREncoder(
                d_model,
                ConvEncoderLayer(
                    seq_len,
                    d_model,
                    cr_args['c_out'],
                    cr_args['d_out'],
                    cr_args['ks2'],
                    c(attn_t),
                    c(ff_t),
                    dropout),
                N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            # nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            nn.Sequential(
                Embeddings(d_model, input_dim, num_sources=num_sources),
                c(position)),
            Generator(
                d_model,
                input_dim,
                num_sources=num_sources,
                res=True,
                res_size=res_size))


    elif stt_type == "STT2":
        model = EncoderTF(
            DoubleEncoder(
                EncoderLayer(d_model, c(attn), c(ff), dropout),
                EncoderLayer(seq_len, c(attn_t), c(ff_t), dropout),
                2 * N),
            nn.Sequential(
                Embeddings(d_model, input_dim),
                c(position)),
            Generator(d_model, input_dim, num_sources=num_sources))

    elif stt_type == "STT3":
        decoders = []
        embeddings = []
        generators = []

        for _ in range(num_sources):
            decoders.append(
                Decoder(
                    DecoderLayer(
                        d_model,
                        c(attn),
                        c(attn),
                        c(ff),
                        dropout),
                    N))
            embeddings.append(
                nn.Sequential(
                    Embeddings(d_model, input_dim, num_sources=1),
                    c(position)))
            generators.append(Generator(d_model, input_dim, num_sources=1))

        model = MultiOutputEncoderDecoder(
            # DoubleEncoder(
            #     EncoderLayer(d_model, c(attn), c(ff), dropout),
            #     EncoderLayer(seq_len, c(attn_t), c(ff_t), dropout),
            #     N),
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Encoder(EncoderLayer(seq_len, c(attn_t), c(ff_t), dropout), N),
            decoders,
            nn.Sequential(Embeddings(d_model, input_dim), c(position)),
            embeddings,
            generators)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# FIXME: strange CUDA issue: how to achieve efficient memory usage
def greedy_decoder(model, src, seq_len, num_sources, input_dim, device, 
        learn_mask=True, start_symbol=1):
    "Decode the separation results."
    memory = model.encode(src, None)
    nbatch = src.shape[0]
    ntoken = 1

    "The 'start' symbol"
    ys = torch.ones(nbatch, ntoken, num_sources * \
            input_dim).fill_(start_symbol).type_as(src.data).to(device)

    for i in range(seq_len):
        subseq_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)
        out = model.decode(memory, None, ys, subseq_mask)
        # [1 seq_len input_dim * num_sources]
        out = model.generator(src, out, learn_mask=learn_mask).view(nbatch,
                -1, num_sources * input_dim)
        ys = torch.cat([ys, out[:, -1, :].unsqueeze(1).data], dim=1)
    return ys[:, 1:]


# FIXME: very sketchy, need to fix style and notations
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(
            self,
            d_model,
            input_dim,
            num_sources=1,
            res=False,
            res_size=None):
        super(Generator, self).__init__()
        self._nsrc = num_sources
        self.res = res
        self.input_dim = input_dim,
        self.proj = nn.Linear(d_model, input_dim * num_sources)
        if self.res:
            self.fc1 = nn.Linear(d_model, res_size)
            self.fc2 = nn.Linear(res_size, input_dim * num_sources)

    def forward(self, agg, x, learn_mask=True):
        s = min(agg.shape[1], x.shape[1])

        input_dim = agg.size(2)
        # print(input_dim)
        # sep_mask = F.log_softmax(self.proj(x), dim=-1).view(-1,
        #         seq_len + 1, self._nsrc, input_dim)[:, :-1]
        if self.res:
            out = F.relu(self.fc1(x[:, :s]))
            out_res = self.fc2(out)
            out = F.relu(out_res) * out_res
            out = out.view(-1, s, self._nsrc, input_dim)
        else:
            out = self.proj(x)[:, :s]
            out = out.view(-1, s, self._nsrc, input_dim)
        # print(x.shape)
        # print(output.shape)
        # print(sep_mask.shape)
        if learn_mask:
            return agg[:, :s].unsqueeze(2) * out[:, :s]
        else:
            return out[:, :s]


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class ConvEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(
            self,
            size,
            chan,
            c_out,
            d_out,
            ks2,
            self_attn,
            feed_forward,
            dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.chan = chan
        self.c_out = c_out
        self.d_out = d_out
        self.ks2 = ks2
        self.convs = self.get_layers()
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def get_layers(self):
        ks1 = self.size - self.d_out + self.ks2

        conv1 = nn.Conv1d(self.chan, self.c_out, kernel_size=ks1)
        bn1 = nn.BatchNorm1d(self.c_out)
        conv2 = nn.ConvTranspose1d(self.c_out, self.chan, kernel_size=self.ks2)
        bn2 = nn.BatchNorm1d(self.chan)
        lut = nn.Linear(self.d_out, self.size)
        return nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(),
            conv2,
            bn2,
            nn.ReLU(),
            lut)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.convs(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, input_dim, num_sources=1):
        super(Embeddings, self).__init__()
        # self.lut = nn.Embedding(input_dim, d_model)
        # FIXME: confirm legitimacy of linear transformation
        self.lut = nn.Linear(input_dim * num_sources, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# class ConvEmbeddings(nn.Module):
#     def __init__(self, d_model, seq_len, input_dim, num_sources=1):
#         super().__init__()
#         # self.lut = nn.Embedding(input_dim, d_model)
#         self.seq_len = seq_len
#         self.input_dim = input_dim
#         self.d_model = d_model
#         self.conv1, self.bn1, self.conv2, self.bn2, self.lut = self.get_layers()

#     def get_layers(self):
#         c_out = 512
#         d_out = 128
#         ks1 = 5
#         ks2 = self.input_dim - d_out - ks1 + 2

#         conv1 = nn.Conv1d(self.seq_len, c_out, kernel_size=ks1)
#         bn1 = nn.BatchNorm1d(c_out)
#         conv2 = nn.Conv1d(c_out, self.seq_len, kernel_size=ks2)
#         bn2 = nn.BatchNorm1d(self.seq_len)
#         lut = nn.Linear(d_out, self.d_model)
#         return conv1, bn1, conv2, bn2, lut

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))

#         return self.lut(x) * math.sqrt(self.d_model)

# class GroupEmbedding(nn.Module):
#     def __init__(self, embeddings):
#         super(GroupEmbedding, self).__init__()

#     def forward(self, in_gts):
#         src_embeds = []
#         for i, in_gt in enumerate(in_gts):
#             src_embeds.append(embeddings[i](in_gt))

#         return src_embeds

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


"TRANSFORMER MODEL"
"transformer"
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.generator(src, self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask))

    def encode(self, src, src_mask):
        embedded = self.src_embed(src)
        return self.encoder(embedded, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class EncoderTF(nn.Module):
    """Encoder-only transformer"""

    def __init__(self, encoder, src_embed, generator):
        super(EncoderTF, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        embedded = self.src_embed(src)
        return self.encoder(embedded, src_mask)


class MultiOutputEncoderDecoder(nn.Module):
    """Multi-decoder transformer"""
    def __init__(self, encoder, encoder_t, decoders, src_embed, tgt_embeds, generators):
        super(MultiOutputEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.encoder_t = encoder_t
        self.decoders = nn.ModuleList(decoders)
        self.src_embed = src_embed
        self.tgt_embeds = nn.ModuleList(tgt_embeds)
        self.generators = nn.ModuleList(generators)

    def forward(self, src, tgts, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(
            src,
            self.encode(src, src_mask),
            src_mask,
            tgts,
            tgt_mask)

    def encode(self, src, src_mask):
        # embedded = self.src_embed(src)
        # return self.encoder(embedded, src_mask)
        embedded = self.src_embed(src)  # [seq_len, d_model]
        embedded_t = embedded.clone().permute(0, 2, 1)  # [d_model, seq_len]
        return self.encoder(embedded, src_mask) + \
            self.encoder_t(embedded_t, src_mask).permute(0, 2, 1)  # [seq_len, d_model]

    def decode(self, aggregate, memory, src_mask, tgts, tgt_mask):
        _, _, input_dim = aggregate.shape
        preds = []
        for i, decoder in enumerate(self.decoders):
            tgt = tgts[:, :, i * input_dim:(i + 1) * input_dim]
            tgt_embed = self.tgt_embeds[i]
            generator = self.generators[i]

            out = decoder(tgt_embed(tgt), memory, src_mask, tgt_mask)
            preds.append(generator(aggregate, out))
        return torch.cat(preds, dim=2)


class DoubleEncoderDecoder(nn.Module):
    """
    A double-encoder + decoder architecture
    """
    def __init__(self, encoder, encoder_t, decoder, src_embed, tgt_embed, generator):
        super(DoubleEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.encoder_t = encoder_t
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        embedded = self.src_embed(src)  # [seq_len, d_model]
        embedded_t = embedded.clone().permute(0, 2, 1)  # [d_model, seq_len]
        return self.encoder(embedded, src_mask) + \
            self.encoder_t(embedded_t, src_mask).permute(0, 2, 1)  # [seq_len, d_model]

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


"components"
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DoubleEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, layer_t, N, transpose=True):
        super(DoubleEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.layer_ts = clones(layer_t, N)
        self.norm = LayerNorm(layer.size)
        self.tr = transpose

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            if self.tr:
                x_t = x.clone().permute(0, 2, 1)
            else:
                x_t = x.clone()

            x = layer(x, mask)
            x_t = self.layer_ts[i](x_t, mask)

            if self.tr:
                x = x + x_t.permute(0, 2, 1)
            else:
                x = x + x_t

        return self.norm(x)


class DoubleTPEncoder(nn.Module):
    def __init__(self, d_model, layer, layer_t, N):
        super(DoubleTPEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.layer_ts = clones(layer_t, N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            x_t = x.clone()

            x = layer(x, mask)
            x_t = self.layer_ts[i](x_t, mask)

            x = x + x_t

        return self.norm(x)


class DoubleSPEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_model, layer, layer_t, N):
        super(DoubleEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.layer_ts = clones(layer_t, N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            x = x.permute(0, 2, 1)
            x_t = x.clone()

            x = layer(x, mask)
            x_t = self.layer_ts[i](x_t, mask)

            x = (x + x_t).permute(0, 2, 1)

        return self.norm(x)


class TREncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_model, layer, N):
        super(TREncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x = x.permute(0, 2, 1)
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)

        x = x.permute(0, 2, 1)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# "pytorch wrapper for skorch model object"
# class Transformer(skorch.NeuralNet):

#     # def __init__(self, *args, **kwargs):
#     #     super(kk
#     def infer(slef, Xi, yi):
#         pass

#     def train_step(self, Xi, yi):
#         pass
