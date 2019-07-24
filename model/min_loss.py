import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# XXX: currently only support 2 or 3 sources
task2perm = {
    2: [[0, 1],[1, 0]],
    3: [[0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0]]
}

class MinLoss(nn.Module):
    '''Minimum loss function'''
    def __init__(self, device, metric, num_sources):
        super().__init__()
        self.device = device
        self.metric = metric
        self.num_sources = num_sources

    def forward(self, predictions, ground_truths):
        """
        Args:
            prediction: bs, seq_len, num_sources, input_dim
            ground_truths: bs, seq_len, num_sources, input_dim
            num_sources: 2/3/5
        Returns:
            loss: [bs,]
        """
        bs = predictions.size()[0]

        best_loss = np.inf
        perms = task2perm[self.num_sources]
        # mse = nn.MSELoss()
        for perm in perms:
            # loss = mse(predictions, ground_truths[:, :, perm])
            dists = self._calc_dists(predictions, ground_truths[:, :, perm])

            sum_ = torch.sum(dists, 1) / self.num_sources
            loss = torch.sqrt(sum_ ** 2).mean()
            best_loss = min(loss, best_loss)

        # loss = torch.sum(dists)
        # loss = torch.log(torch.sqrt(sum_.mean()))
        return torch.log(best_loss)

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
