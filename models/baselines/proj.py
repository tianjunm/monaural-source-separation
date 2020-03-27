
import torch
import torch.nn as nn
import torch.nn.functional as F


class PROJ(nn.Module):

    def __init__(self, num_sources):
        super().__init__()
        self._nsrc = num_sources

    def forward(self, x):
        bs, sl, dim = x.size()
        return x.repeat(1, 1, self._nsrc).view(bs, sl, self._nsrc, dim)
