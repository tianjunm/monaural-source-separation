"""Sets up different datasets."""


import json
import torch
from . import dataloaders
from . import transforms


def prepare_dataloader(dataset_spec, model_spec, dataset_split):
    """Return a loadable dataset.

    Args:
        dataset_spec
        dataset_split: indicates train, val, or test

    Returns:
        train_data, val_data
    """
    batch_size = model_spec['model']['config']['batch_size']

    if dataset_spec['transform'] == 'stft':
        transform = transforms.STFT(dataset_spec)

    elif dataset_spec['transform'] == 'pcm':
        transform = None

    if dataset_spec['name'] == 'wild-mix':

        dataset = dataloaders.WildMix(dataset_spec['config'], dataset_split,
                                      transform=transform)

    elif dataset_spec['name'] == 'timit':

        dataset = dataloaders.Timit(dataset_spec, dataset_split, transform=transform)

    elif dataset_spec['name'] == 'musdb18':

        dataset = dataloaders.Musdb18(dataset_spec['config'], dataset_split,
                                      transform=transform)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=(dataset_split == 'train'))
