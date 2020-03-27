"""Sets up different datasets."""


import json
import torch
from . import dataloaders
from . import transforms


def prepare_dataloader(config, dataset_split):
    """Return a loadable dataset.

    Args:
        config_path
        dataset_split: indicates train, val, or test

    Returns:
        train_data, val_data
    """
    batch_size = config['model']['config']['batch_size']

    dataset_info = config['dataset']

    if dataset_info['name'] == 'wild-mix':

        if dataset_info['transform'] == 'stft':
            transform = transforms.STFT(dataset_info)

        elif dataset_info['transform'] == 'pcm':
            transform = None

        dataset = dataloaders.WildMix(dataset_info['config'], dataset_split,
                                      transform=transform)

    elif dataset_info['name'] == 'avspeech':
        pass

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=(dataset_split == 'train'))
