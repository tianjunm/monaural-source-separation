"""convert audio dataset to images for training"""
import os
import numpy as np
import torch
# from scipy.io import wavfile
# import matplotlib.pyplot as plt
# from pydub.playback import play
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """custom dataset for source separation"""

    def __init__(self, root_dir, sample_rate=22050, transform=None):
        """
        args:
            transform: optional
        """
        self.transform = transform
        self.root_dir = root_dir
        self.sr = sample_rate

    def __len__(self):
        return len(os.listdir(self.root_dir)) - 1

    def __getitem__(self, idx):
        # fixme: hacky - root_dir may not have '/'

        data_path= self.root_dir + str(idx + 1) + '/' # aggregates

        item = {'aggregate': None, 'ground_truths': []}
        for filename in os.listdir(data_path):
            # print('staty')
            if filename.endswith('.npy'):
                # agg_complex = torch.from_numpy(np.load(data_path + filename))
                # agg = self._from_complex(data_path + filename) 
                agg = np.load(data_path + filename)
                item['aggregate'] = agg
            else:
                gt_path = data_path + 'gt/'
                # print(gt_path)
                for gt_name in os.listdir(gt_path):
                    # gt = torch.from_numpy(np.load(gt_path + gt_name))
                    gt = np.load(gt_path + gt_name)
                    # gt = self._from_complex(gt_path + gt_name)
                    # print(gt)
                    item['ground_truths'].append(gt)
        
        if self.transform:
            item = self.transform(item)
        return item


class Concat(object):
    """2-channel spectrogram to single-channel tensor
    """
    
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
        num_channels, nrows, ncols = m.shape

        result = np.zeros((nrows * num_channels, ncols))

        for i in range(num_channels):
            start = i * nrows;
            end = (i + 1) * nrows;
            result[start:end, :] = m[i]

        result[:nrows, :] = m[0]
        result[nrows:, :] = m[1]

        return torch.t(torch.from_numpy(result)).float()
