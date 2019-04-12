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
    """turn input into torch tensor with dimension seq_lem * bs * input_dim"""
    x = [x[:, ts, :] for ts in range(seq_len)]
    x = torch.cat(x).view(seq_len, bs, -1)
    return x


def flatten(ms, batch):
    return np.array([m[:, batch, :].cpu().detach().numpy().flatten() for m in ms])


def calc_dists(preds, gts, device, metric):
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
   
   #  print(metric)
    for batch in range(bs):
        pred_flattened = flatten(preds, batch)
        gt_flattened = flatten(gts, batch)
        all_dists[batch] = ssdist.cdist(pred_flattened, gt_flattened, metric=metric)
    
    all_orders = get_orders(all_dists)
    # print(all_orders)
    all_matches = get_matches(all_orders)
    dists = torch.zeros(bs, n_sources).to(device)
  
    # print(all_dists)
    # print(all_matches)

    for batch in range(bs):
        matches = all_matches[batch]
        # print(matches)
        for src_id in range(n_sources):
            pred = preds[src_id]
            gt_match = gts[int(matches[src_id])]
            # recomputing norms (required to keep track of grads)
            dist = torch.norm(torch.squeeze(pred[:, batch, :] - gt_match[:, batch, :], dim=1), 2)
            dists[batch, src_id] = dist

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
    def __init__(self, device, metric='euclidean'):
        # nn.Module.__init__(self)
        super(MinLoss, self).__init__()
        self.device = device
        self.metric = metric

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
        dists = calc_dists(predictions, gts, self.device, self.metric)
        
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
            seq_len=SEQ_LEN,
            num_layers=NUM_LAYERS,
            num_sources=NUM_SOURCES):
        super(Baseline, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers 
        self.num_sources = num_sources
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, num_sources * input_dim)


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
        self.lstm = nn.LSTM(input_dim, 100) 
        self.fc = nn.Linear(100, num_sources * input_dim, bias=True)
        
    def forward(self, x):
        bs = x.size()[0]
        x = reshape(x, self.seq_len, bs)
        out, _ = self.lstm(x)
        ys = self.fc(F.relu(out))
        prediction = torch.split(ys, self.input_dim, dim=-1)
        return prediction 
