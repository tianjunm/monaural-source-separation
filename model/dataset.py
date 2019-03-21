"""convert audio dataset to images for training"""
# from scipy.io import wavfile
# import matplotlib.pyplot as plt
# from pydub.playback import play
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """Custom Dataset for source separation"""

    def __init__(self, transform=None):
        """
        Args:
            transform: Optional
        """
        self.transform = transform

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return idx
