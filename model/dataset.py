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

        for filename in os.listdir(data_path):
            if filename.endswith('.npy'):
                agg = torch.from_numpy(np.load(data_path + filename))
                item['aggregate'] = agg 
            else:
                gt_path = data_path + 'gt/'
                for gt_name in os.listdir(gt_path):
                    gt = torch.from_numpy(np.load(gt_path + gt_name))
                    item['ground_truths'].append(gt)

        return item

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
