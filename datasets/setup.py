"""Sets up different datasets."""


import json
import torch
from . import dataloaders
from . import transforms
import numpy as np
import logging

from guppy import hpy
import sys
import torch
import gc
from memory_profiler import profile


#@profile
def prepare_dataloader(dataset_spec, model_spec, dataset_split):
    """Return a loadable dataset.

    Args:
        dataset_spec
        dataset_split: indicates train, val, or test

    Returns:
        train_data, val_data
    """
    batch_size = model_spec['model']['config']['batch_size']

    if dataset_spec['name'] == 'wild-mix':

        if dataset_spec['transform'] == 'stft':
            transform = transforms.STFT(dataset_spec)

        elif dataset_spec['transform'] == 'pcm':
            transform = None
        
        elif dataset_spec['transform'] == 'wave-u-net':
            transform = transforms.WaveunetWindow(dataset_spec)
            '''
            worker_init_fn = lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
            dataset = dataloaders.WildMix(dataset_spec['config'], dataset_split,
                                      transform=transform)
            dloader = torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=(dataset_split == 'train'))
            return dloader
            '''


        dataset = dataloaders.WildMix(dataset_spec['config'], dataset_split,
                                      transform=transform)
    elif dataset_spec['name'] == 'avspeech':
        pass

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=(dataset_split == 'train'))

def classifier_prepare_dataloader(dataset_spec, model_spec, dataset_split):
    """Return a loadable dataset.

    Args:
        dataset_spec
        dataset_split: indicates train, val, or test

    Returns:
        train_data, val_data
    """
    batch_size = model_spec['model']['config']['c_batch_size']
    
    transform = transforms.STFT(dataset_spec)
    dataset = dataloaders.WildMix(dataset_spec['config'], dataset_split,transform=transform)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=(dataset_split == 'train'))


