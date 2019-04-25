"""convert audio dataset to images for training"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """custom dataset for source separation"""

    def __init__(self, root_dir, transform=None):
        """
        args:
            transform: optional
        """
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(os.listdir(self.root_dir)) - 2

    def __getitem__(self, idx):
        # data_path= self.root_dir + str(idx + 1) # aggregates
        data_path = os.path.join(self.root_dir, str(idx))

        item = {'aggregate': None, 'ground_truths': []}

        agg = np.load(os.path.join(data_path, 'agg.npy'))
        item['aggregate'] = agg

        gt_path = os.path.join(data_path, 'gt')
        for gt_name in os.listdir(gt_path):
            if gt_name.endswith('.npy'):
                gt = np.load(os.path.join(gt_path, gt_name))
                item['ground_truths'].append(gt)
        
        if self.transform:
            item = self.transform(item)
        return item


class Concat(object):
    """2-channel spectrogram to single-channel tensor
    """
    
    def __init__(self, size):
        self.size = size

    def __call__(self, item):
        transformed = {'aggregate': None, 'ground_truths': None}
        agg = self._from_complex(item['aggregate'])
        transformed['aggregate'] = agg  

        seq_len, input_dim = agg.size()
        n_sources = len(item['ground_truths'])

        gts = torch.zeros((seq_len, n_sources, input_dim))
        for s, gt in enumerate(item['ground_truths']):
            gts[:, s, :] = self._from_complex(gt)
        
        transformed['ground_truths'] = torch.FloatTensor(gts)
        return transformed 

    def _from_complex(self, m):
        # num_channels, nrows, ncols = m.shape
        nrows, ncols = self.size

        result = np.zeros((nrows * 2, ncols))

        result[:nrows, :] = m[0][:, :ncols]
        result[nrows:, :] = m[1][:, :ncols]

        return torch.t(torch.from_numpy(result)).float()
