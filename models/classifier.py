import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class Transformer(nn.Module):

    def __init__(self, num_outputs, input_size, input_shape, num_sources, dmodel, num_heads, num_layers,
                 hidden_size, dropout=0.5):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.m = input_size
        self.n = input_shape[1]
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
        self.decoder = nn.Linear(dmodel, 2 * input_size)
        self.dropout_layer = nn.Dropout(dropout)
        #self.pre_classify = nn.Linear(dmodel* self.n , num_outputs*4))
        self.classify = nn.Linear(2 * input_size*self.n, self.n* num_outputs)
        self.create_wts = nn.Linear(2 * input_size*self.n, self.n)
        self.num_outputs =  num_outputs
        #self.sigmoid = torch.sigmoid()
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
        # b * 84 * 64000 b*84 *1 
        b = x.size(0)
        n = x.size(2)
        if self.src_mask is None:
             device = x.device
             mask = self._generate_square_subsequent_mask(n).to(device)
             self.src_mask = mask
        x = x.permute(2, 0, 1, 3).reshape(n, b, -1)
        x = self.encoder(x) * math.sqrt(self.dmodel)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x, self.src_mask)
        decoded = self.decoder(encoded).permute(1, 0, 2)
        decoded = decoded.flatten(start_dim=1)
        x = self.dropout_layer(self.classify(decoded)).view(-1,  n, self.num_outputs)
        #  10, 84, 6: 10, 1, 6
        x_wts = self.create_wts(decoded)
        
        wts = torch.softmax(x_wts, dim = 1)
        #logging.info(torch.sum(wts, dim=1))
        wts = wts.view( wts.shape[0], wts.shape[1], 1)
        wts = wts.expand_as(x)
        mult_res = (wts*x) # 10, 84, 6
        
        return torch.sigmoid(torch.sum(mult_res, dim = 1))

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
'''
class TransformerSet(nn.Module):
    def __init__(self, num_outputs, input_size, input_shape, num_sources, dmodel, num_heads, num_layers,
                 hidden_size, dropout=0.5):

        self.all_models = nn.ModuleList()
        for i in range(num_ont_categories):
            model = TransformerModel()
            self.all_models.append(model)
        
    def forward(self, x):
        results = {}
        i = 0
        for model in self.all_models:
            x = model(x)
            results[i] 
'''
    


    