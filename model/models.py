import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import scipy.spatial.distance as ssdist


NUM_LAYERS = 1
BATCH_SIZE = 32

# INPUT_DIM = 3
NUM_SOURCES = 2
SEQ_LEN = 346


def get_orders(dists):
    '''
    args:
        dists: [bs, n_sources n_sources]
    '''
    num_batches = dists.shape[0]
    d = dists.shape[1]
    orders = np.zeros(dists.shape)
    for batch in range(num_batches):
        flattened = np.copy(dists[batch, :, :].reshape(1, -1))
        indices = np.argsort(flattened)
        flattened[:, indices] = np.arange(flattened.size)
        # print(flattened.reshape(d, -1))
        # print(flattened.reshape(d, -1).astype(int))
        orders[batch, :, :] = flattened.reshape(d, -1)
    
    return orders.astype(int)


def get_matches(orders):
    num_batches = orders.shape[0]
    d = orders.shape[1]
    mask = np.max(orders) + 1
    
    matched_pairs = np.zeros((num_batches, d))
    for i in range(d):
        for batch in range(num_batches):
            m = np.argmin(orders[batch, :, :])
            indices = np.array([(m // d), (m % d)])

            matched_pairs[batch][indices[0]] = indices[1]

            # mask row & col
            orders[batch, indices[0], :] = np.ones(d) * mask
            orders[batch, :, indices[1]] = np.ones(d) * mask
    
    return matched_pairs    


def reshape(x, seq_len, bs):
    x = [x[:, ts, :] for ts in range(seq_len)]
    x = torch.cat(x).view(seq_len, bs, -1)
    return x


def flatten(t, n_sources):
    seq_len, n_sources, input_dim = t.size()
    flattened = np.zeros((n_sources, seq_len * input_dim))
    for s in range(n_sources):
        flattened[s] = t[:, s, :].detach().numpy().flatten()
    return flattened


def get_dists(preds, gts, bs, n_sources, metric):
    dists = np.zeros((bs, n_sources, n_sources))
#     print(preds.size())
#     print(gts.size())
    for b in range(bs):
        dists[b] = ssdist.cdist(flatten(preds[b], n_sources), flatten(gts[b], n_sources), metric=metric)
        if metric == 'correlation':
            dists[b] = -dists[b]
    return dists


def get_correlation(m1, m2):
    mm1 = (m1 - m1.mean()).flatten()
    mm2 = (m2 - m2.mean()).flatten()
    
    n = torch.dot(mm1, mm2)
    d = torch.norm(mm1) * torch.norm(mm2)
    
    return -(1 - n / d)


def get_min_dist(preds, gts, device, metric):
    """
    Args:
        preds(tensor): [bs, seq_len, n_sources, input_dim]
        gts(tensor): [bs, seq_len, n_sources, input_dim]
    
    Returns:
        dists: [bs, n_sources]
    """
    bs, _, n_sources, _ = preds.size()

    # getting the distances from each prediction to all gts
    all_dists = get_dists(preds, gts, bs, n_sources, metric)
    all_orders = get_orders(all_dists)
    all_matches = get_matches(all_orders)
    dists = torch.zeros(bs, n_sources).to(device)

    for b in range(bs):
        matches = all_matches[b]
        for src_id in range(n_sources):
            pred = preds[:, :, src_id, :]
            gt_match = gts[:, :, int(matches[src_id]), :]
            # recomputing required to keep track of grads
            if metric == 'correlation':
                dist = get_correlation(pred[b], gt_match[b])
            else:
                dist = torch.norm(pred[b] - gt_match[b])
            dists[b, src_id] = dist

    return dists

def get_mse(preds, gts, device):
    bs, _, n_sources, _ = preds.size()
    dists = torch.zeros(bs, n_sources).to(device)
    
    for b in range(bs):
        for src_id in range(n_sources):
            pred = preds[:, :, src_id, :]
            gt_match = gts[:, :, src_id, :]
            dists[b, src_id] = torch.norm(pred[b] - gt_match[b])
    return dists


class MSELoss(nn.Module):
    def __init__(self, device, metric):
        super(MSELoss, self).__init__()
        self.device = device
        self.metric = metric

    def forward(self, predictions, ground_truths):
        """
        Args:
            prediction: bs, seq_len, num_sources, input_dim
            ground_truths: nbs, seq_len, num_sources, input_dim
        Returns:
            loss: [bs,]
        """
        # seq_len = predictions[0].size()[0]
        # bs = predictions[0].size()[1]
        # # reshape gts into seq_len, bs, input_dim
        # gts = [reshape(gt, seq_len, bs) for gt in ground_truths]

        # get distance measure (bs * num_sources)
        dists = calc_dists(predictions, groud_truths, self.device, self.metric)
        
        loss = torch.sum(dists)
        
        return loss


class MinLoss(nn.Module):
    """Custom loss function #1

    Compare the distance from output with its closest ground truth.

    """
    def __init__(self, device, metric):
        # nn.Module.__init__(self)
        super(MinLoss, self).__init__()
        self.device = device
        self.metric = metric

    def forward(self, predictions, ground_truths):
        """
        Args:
            prediction: bs, seq_len, num_sources, input_dim
            ground_truths: bs, seq_len, num_sources, input_dim
        Returns:
            loss: [bs,]
        """
        # seq_len = predictions[0].size()[0]
        # bs = predictions[0].size()[1]
        # # reshape gts into seq_len, bs, input_dim
        # gts = [reshape(gt, seq_len, bs) for gt in ground_truths]

        # get distance measure (bs * num_sources)
        dists = get_min_dist(predictions, ground_truth, self.device, self.metric)
        
        loss = torch.sum(dists) / bs
        
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
        x, _ = self.lstm(x)  # [seq_len, batch_size, num_sources * input_dim]
        x = F.relu(x)
        prediction = torch.split(x, self.input_dim, dim=-1)
        return prediction 


class A1(nn.Module):
    """lstm, fc; relu"""
    def __init__(self, input_dim, num_layers=NUM_LAYERS, seq_len=SEQ_LEN, num_sources=NUM_SOURCES):
        super(A1, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_sources = num_sources
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, 100, batch_first=True)
        self.fc = nn.Linear(100, num_sources * input_dim, bias=True)

    def forward(self, x):
        bs = x.size()[0]
        # x_ = reshape(x, self.seq_len, bs)
        out, _ = self.lstm(x)
        ys = self.fc(F.relu(out))
        # # prediction = torch.split(ys, self.input_dim, dim=-1)
        ys = ys.view(bs, self.seq_len, self.num_sources, self.input_dim)
        prediction = x.unsqueeze(2) * ys
        return prediction 
