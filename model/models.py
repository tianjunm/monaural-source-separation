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

# helpers for calc_dists
def get_orders(dists):
    '''
    Args:
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


def flatten(ms, batch):
    return np.array([m[:, batch, :].detach().numpy().flatten() for m in ms])


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

    # getting the distances from each prediction to all gts
    all_dists = np.zeros((bs, n_sources, n_sources))
    
    for batch in range(bs):
        pred_flattened = flatten(preds, batch)
        gt_flattened = flatten(gts, batch)
        all_dists[batch] = ssdist.cdist(pred_flattened, gt_flattened)
    
    all_orders = get_orders(all_dists)
    all_matches = get_matches(all_orders)
    dists = torch.zeros(bs, n_sources).to(device)
    
    for batch in range(bs):
        matches = all_matches[batch]
        for src_id in range(n_sources):
            pred = preds[src_id]
            gt_match = gts[int(matches[src_id])]
            # recomputing required to keep track of grads
            dist = torch.norm(torch.squeeze(pred[:, batch, :] - gt_match[:, batch, :], dim=1), 2)
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
