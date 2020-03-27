"""Sets up different datasets."""


import json
from dataloaders import WildMix
from transforms import STFT


def prepare_dataset(config_path, dataset_split):
    """Return a loadable dataset.

    Args:
        config_path
        dataset_split: indicates train, val, or test

    Returns:
        train_data, val_data
    """

    with open(config_path) as f:
        config = json.load(f)

    dataset_info = config['dataset']

    if dataset_info['name'] == 'wild-mix':

        if dataset_info['transform'] == 'stft':
            transform = STFT(dataset_info)

        elif dataset_info['transform'] == 'pcm':
            transform = None

        dataset = WildMix(dataset_info, dataset_split, transform=transform)

    elif dataset_info['name'] == 'avspeech':
        pass

    return dataset
