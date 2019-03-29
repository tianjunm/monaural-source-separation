"""convert audio dataset to images for training"""
import os
import numpy as np
import torch
# from scipy.io import wavfile
# import matplotlib.pyplot as plt
# from pydub.playback import play
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """Custom Dataset for source separation"""

    def __init__(self, root_dir, sample_rate=22050, transform=None):
        """
        Args:
            transform: Optional
        """
        self.transform = transform
        self.root_dir = root_dir
        self.sr = sample_rate

    def __len__(self):
        return len(os.listdir(self.root_dir)) - 1

    def __getitem__(self, idx):
        # FIXME: hacky - root_dir may not have '/'

        data_path= self.root_dir + str(idx + 1) + '/' # aggregates

        item = {'aggregate': None, 'ground_truths': []}
        # print(data_path)
        # print(len(os.listdir(data_path)))
        for filename in os.listdir(data_path):
            # print('staty')
            if filename.endswith('.npy'):
                # agg_complex = torch.from_numpy(np.load(data_path + filename))
                # agg = self._from_complex(data_path + filename) 
                agg = self._from_complex(data_path + filename)
                item['aggregate'] = agg
            else:
                gt_path = data_path + 'gt/'
                # print(gt_path)
                for gt_name in os.listdir(gt_path):
                    # gt = torch.from_numpy(np.load(gt_path + gt_name))
                    gt = self._from_complex(gt_path + gt_name)
                    item['ground_truths'].append(gt)

            # print(data_path)
        return item

    # FIXME: hacky, should use Transform
    def _from_complex(self, filepath):
        m = np.load(filepath)
        num_channels, nrows, ncols = m.shape

        result = np.zeros((nrows * num_channels, ncols))

        for i in range(num_channels):
            start = i * nrows;
            end = (i + 1) * nrows;
            result[start:end, :] = m[i]

        result[:nrows, :] = m[0]
        result[nrows:, :] = m[1]

        return torch.t(torch.from_numpy(result)).float()

    # def get_spectrograms(self):
        
    #     # get the size
    #     aggregates = [None] * len(all_files)
    #     ground_truths = [[]] * len(all_files)

    #     for filename in all_files:
    #         idx = int(filename.split('_')[-1].split('.npy')[0]) - 1
    #         if '-' in filename:  # spectrogram of aggregate
    #             aggregates[idx] = np.load(filename)
    #         else:
    #             ground_truths[idx].append(np.load(filename))

    #     assert len(aggregates) == len(ground_truths)
    #     return aggregates, ground_truths 
