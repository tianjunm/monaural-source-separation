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
SEQ_LEN = 173


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
    seq_len, _, input_dim = t.size()
    flattened = np.zeros((n_sources, seq_len * input_dim))
    for s in range(n_sources):
        flattened[s] = t[:, s, :].cpu().detach().numpy().flatten()
    return flattened


def get_dists(preds, gts, bs, n_sources, metric):
    dists = np.zeros((bs, n_sources, n_sources))
#     print(preds.size())
#     print(gts.size())
    for b in range(bs):
        dists[b] = ssdist.cdist(flatten(preds[b], n_sources),
                                flatten(gts[b], n_sources), metric=metric)
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
        gts(tensor): [bs, seq_len, n_sources, input_dim

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
            pred = preds[b, :, src_id, :]
            gt_match = gts[b, :, int(matches[src_id]), :]
            # recomputing required to keep track of grads
            if metric == 'correlation':
                dist = get_correlation(pred, gt_match)
            else:
                dist = torch.norm(pred - gt_match)

            dists[b, src_id] = dist

    return dists


class DiscrimLoss(nn.Module):
    '''Discriminative loss function introduced by Huang et al.'''
    def __init__(self, device, metric, gamma):
        super().__init__()
        self.device = device
        self.metric = metric
        self.gamma = gamma

    def forward(self, predictions, ground_truths):
        """
        Args:
            prediction: bs, seq_len, num_sources, input_dim
            ground_truths: bs, seq_len, num_sources, input_dim
        Returns:
            loss: [bs,]
        """
        bs = predictions.size()[0]
        # TODO: Huang et al. only introduced the discrim penalty
        # for tasks with 2 sources
        dists = self._calc_dists(predictions, ground_truths)
        pred_swapped = torch.Tensor(predictions.size()).to(self.device)
        pred_swapped[:, :, [1, 0]] = predictions

        penalties = self.gamma * \
            self._calc_dists(pred_swapped, ground_truths)
        sum_ = torch.sum(dists, 1) ** 2 + torch.sum(penalties, 1) ** 2

        # loss = torch.sum(dists)
        loss = torch.log(torch.sqrt(sum_.mean()))

        return loss

    def _calc_dists(self, preds, gts):
        bs, _, n_sources, _ = preds.size()
        dists = torch.zeros(bs, n_sources).to(self.device)

        for b in range(bs):
            for src_id in range(n_sources):
                pred = preds[:, :, src_id, :]
                gt_match = gts[:, :, src_id, :]
                if self.metric == 'correlation':
                    dist = get_correlation(pred[b], gt_match[b])
                else:
                    dist = torch.norm(pred[b] - gt_match[b])
                dists[b, src_id] = dist
        return dists


# class MinLoss(nn.Module):
#     pass


class GreedyLoss(nn.Module):
    """Custom loss function #1

    Compare the distance from output with its closest ground truth.

    """
    def __init__(self, device, metric, gamma, num_sources):
        # nn.Module.__init__(self)
        super(GreedyLoss, self).__init__()
        self.device = device
        self.metric = metric
        self.gamma = gamma
        self.num_sources = num_sources

    def forward(self, predictions, ground_truths):
        """
        Args:
            prediction: bs, seq_len, num_sources, input_dim
            ground_truths: bs, seq_len, num_sources, input_dim
        Returns:
            loss: [bs,]
        """
        bs = predictions.size()[0]

        # get distance measure (bs * num_sources)
        dists = get_min_dist(predictions, ground_truths, self.device,
                             self.metric)

        sum_ = torch.sum(dists, 1) / self.num_sources
        # greedy RMSE
        loss = torch.sqrt((sum_ ** 2).mean())

        return torch.log(loss)


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
        out, _ = self.lstm(x)
        ys = self.fc(F.relu(out))
        ys = ys.view(bs, self.seq_len, self.num_sources, self.input_dim)
        prediction = x.unsqueeze(2) * ys
        return prediction, ys


class B1(nn.Module):
    '''LSTM baseline with residual connection'''

    def __init__(self,
            input_dim,
            hidden_size, 
            num_layers=NUM_LAYERS,
            seq_len=SEQ_LEN,
            num_sources=NUM_SOURCES):

        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_sources = num_sources
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc0 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc1 = nn.Linear(hidden_size, num_sources * input_dim, bias=True)

    def forward(self, x):
        # x size: [bs, seq_len, input_dim]
        bs = x.size()[0]
        x_attn = self.fc0(x)
        out, _ = self.lstm(x + x_attn)
        ys = self.fc1(F.relu(out))
        ys = ys.view(bs, -1, self.num_sources, self.input_dim)
        prediction = x.unsqueeze(2) * ys
        return prediction


class LookToListenAudio(nn.Module):

    def __init__(self, input_dim, in_chan=2, chan=6,
            num_sources=NUM_SOURCES):
        super(LookToListenAudio, self).__init__()
        self.input_dim = input_dim
        # self.seq_len = seq_len
        self.num_sources = num_sources
        self.in_chan = in_chan
        self.chan = chan

        self.kernel_dims = [(1, 7), (7, 1), (5, 5),
                (5, 5), (5, 5), (5, 5), (5, 5)]
        self.dilation_dims = [(1, 1), (1, 1), (1, 1),
                (2, 2), (4, 4), (8, 8), (1, 1)]
        assert(len(self.kernel_dims) == len(self.dilation_dims))

        self.num_layers = len(self.kernel_dims)
        self.convs = nn.ModuleList(self._construct_convs())
        self.bns = nn.ModuleList(self._construct_bns())

        self.blstm = nn.LSTM(8 * self.input_dim, hidden_size=200, 
                batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(400, 600, bias=True)
        self.fc2 = nn.Linear(600, self.input_dim * self.in_chan * \
                self.num_sources, bias=True)

    def forward(self, x):
        x_in = torch.cat(list(x.permute(1, 0, 2, 3)), -1)
        # dilated convolutional network
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        # audio-visual fusion
        # [bs, seq_len, out_chan * input_dim]
        x = torch.cat(list(x.permute(1, 0, 2, 3)), 2)

        # bidirectional lstm
        x, _ = self.blstm(x)
        x = F.relu(x)

        # fcs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = x.view(x.shape[0], -1, self.num_sources,
                self.input_dim * self.in_chan)
        prediction = x_in.unsqueeze(2) * x
        return prediction

    def _construct_convs(self):
        convs = []
        for i, kernel_size in enumerate(self.kernel_dims):
            in_chan = 2 if i == 0 else self.chan
            out_chan = 8 if i == self.num_layers - 1 else self.chan

            dilation = self.dilation_dims[i]

            rpad = dilation[0] * (kernel_size[0] - 1) // 2
            cpad = dilation[1] * (kernel_size[1] - 1) // 2
            padding= [rpad, cpad]

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




