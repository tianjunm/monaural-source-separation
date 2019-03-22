import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from tensorboardX import SummaryWriter


NUM_LAYERS = 1
BATCH_SIZE = 32

# INPUT_DIM = 3
NUM_SOURCES = 2
SEQ_LEN = 173 


def reshape(x, seq_len, bs):
    x = [x[:, ts, :] for ts in range(seq_len)]
    x = torch.cat(x).view(seq_len, bs, -1)
    return x


def calc_dists(preds, gts, device):
    """
    Args:
        preds: n_sources * [seq_len, bs, input_dim]
        gts: n_sources * [seq_len, bs, input_dim]
    
    Returns:
        dists: [bs, n_sources]
    """
    n_sources = len(preds)
    assert n_sources == len(gts)

    bs = preds[0].size()[1]
    # TODO: greedy assignment
    dists = torch.zeros(bs, n_sources).to(device)

    for src_id in range(n_sources):
        pred = preds[src_id]
        gt = gts[src_id]
#         print(pred.size())
#         print(gt.size())
        for batch in range(bs):
            dist = torch.norm(torch.squeeze(pred[:, batch, :] - gt[:, batch, :], dim=1), 2)
            dists[batch, src_id] = dist
        
    return dists


class MinLoss(nn.Module):
    """Custom loss function #1

    Compare the distance from output with its closest ground truth.

    """
    def __init__(self, device):
        # nn.Module.__init__(self)
        super(MinLoss, self).__init__()
        self.device = device

    def forward(self, predictions, ground_truths):
        """
        Args:
            prediction: num_sources * [seq_len, bs, input_dim]
            ground_truths: num_sources * [bs, seq_len, input_dim] 
        Returns:
            loss: [bs,]
        """
        seq_len = predictions[0].size()[0]
        bs = predictions[0].size()[1]
        # reshape gts into seq_len, bs, input_dim
        gts = [reshape(gt, seq_len, bs) for gt in ground_truths]

        # get distance measure (bs * num_sources)
        dists = calc_dists(predictions, gts, self.device)
        
        loss = torch.sum(dists)
        
        return loss


class Baseline(nn.Module):
    """LSTM-only network

    Args:
        input_dim: the length of a vertical slice of the spectrogram
        num_sources: number of individual sources in the combined clip.
                     also the hidden dimension for the LSTM network
    """

    def __init__(
            self,
            input_dim,
            # batch_size,
            seq_len=SEQ_LEN,
            num_sources=NUM_SOURCES):
        super(Baseline, self).__init__()
        self.input_dim = input_dim
        self.num_sources = num_sources
        # self.bs = batch_size
        self.seq_len = seq_len
        self.num_layers = NUM_LAYERS
        self.lstm = nn.LSTM(input_dim, num_sources * input_dim)
        # self.encoder = nn.Linear()


    def forward(self, x):
        bs = x.size()[0]
        # x is agg
        x = reshape(x, self.seq_len, bs)
        x, _ = self.lstm(x)  # [seq_len, batch_size, num_sources]
        # x = self.decoder(x)  # [seq_len, batch_size, num_sources * input_dim]
        x = F.relu(x)
        prediction = torch.split(x, self.input_dim, dim=-1)
        return prediction 


# dummy = torch.randn(INPUT_DIM, SEQ_LEN)
# with SummaryWriter(comment='BaseLSTM') as w:
#     w.add_graph(Baseline(INPUT_DIM, BATCH_SIZE, NUM_SOURCES), dummy, True)


# # TODO: exploit other structures
# class Net2(nn.Module):
#     """Conv + LSTM
#     """

#     def __init__(self):
#         super(Net2, self).__init__()
#         self.conv = None
#         self.lstm = None

#     def forward(self, x):
#         batch_size, nrows, ncols = x.size()
#         x = x.view(-1, nrows * ncols)
#         x = F.relu(self.conv(x))
#         x = F.relu(self.lstm(x))
#         return x
