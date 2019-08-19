"""convert audio dataset to images for training"""
import os
import numpy as np
import torch
import pandas as pd
import copy
from pydub import AudioSegment as auseg
from torch.utils.data import Dataset

RAW_DATA_PATH = '/home/ubuntu/datasets/original/FSDKaggle/train'
# XXX
MIXTURE_DURATION = 2000
WINDOW_SIZE = 256
HOP_LENGTH = 256 // 4 * 3


# audio manipulation utils
def overlay(ground_truths):
    """Overlay ground truths together to create the aggregate."""
    aggregate = copy.deepcopy(ground_truths[0])
    for idx in range(1, len(ground_truths)):
        aggregate = aggregate.overlay(ground_truths[idx])
    return aggregate


def get_spect(wav, ttype):
    """Apply STFT on the wave form to get its spectrogram representation."""
    wav_arr = torch.from_numpy(np.array(wav.get_array_of_samples())).float()
    # wav_arr = torch.Tensor(wav.get_array_of_samples())
    spect = torch.stft(
        wav_arr,
        WINDOW_SIZE,
        HOP_LENGTH,
        window=torch.hann_window(256)).permute(2, 1, 0)

    if ttype == 'Concat':
        return torch.cat([spect[0], spect[1]], dim=-1)

    return spect


class MixtureDataset(Dataset):
    """This dataset loads the mixture-ground truth pairs.

    This dataset followes a set of instructions to load
    sound clips from the raw dataset and convert them to
    spectrograms.
    """

    def __init__(
            self,
            num_sources,
            data_path,
            raw_data_path=RAW_DATA_PATH,
            transform=None):
        self._nsrc = num_sources
        self._data = pd.read_csv(data_path)
        self._raw_path = raw_data_path
        self._transform = transform

    def __len__(self):
        return len(self._data) // self._nsrc

    def __getitem__(self, idx):
        # what the dataloader loads
        item = {'aggregate': None, 'ground_truths': []}

        # load [nsrc] rows from instruction
        instances = self._load_instances(idx)

        for iid in instances.index:
            item['ground_truths'].append(self._create_gt(instances, iid))

        item['aggregate'] = overlay(item['ground_truths'])

        return self._transform(item)

    def _load_instances(self, idx):
        # loading the section containing the [idx]th set of instances
        begin, end = idx * self._nsrc, (idx + 1) * self._nsrc
        instances = self._data.iloc[begin:end]
        return instances

    def _create_gt(self, instances, iid):
        # load original files according to filenames
        filename = instances.at[iid, 'filename']
        dur = instances.at[iid, 'clip_duration']
        start = instances.at[iid, 'mixture_placement']
        # start, end = parse_range(instances.at[iid, 'mixture_placement'])
        # print(filename)

        wav_path = os.path.join(self._raw_path, filename)
        wav = auseg.from_wav(wav_path)
        wav = wav[:min(len(wav), dur)]

        # follow the corresponding instructions to
        # manipulate the sound files
        pad_before = auseg.silent(start)
        pad_after = auseg.silent(MIXTURE_DURATION - len(wav) - start)

        padded_wav = pad_before + wav + pad_after
        assert len(padded_wav) == MIXTURE_DURATION

        return padded_wav


class Wav2Spect():
    """Transforms wave forms to spectrogram tensors"""

    def __init__(self, agg_type=None, enc_dec=False):
        self._ttype = agg_type
        # self._gt_type = in_gts_type
        self._ed = enc_dec

    def __call__(self, item):
        transformed = {'aggregate': None, 'ground_truths': None}
        transformed['aggregate'] = get_spect(item['aggregate'], self._ttype)

        gts = []
        for gt_wav in item['ground_truths']:
            gt_spect = get_spect(gt_wav, 'Concat')
            gts.append(gt_spect)

        transformed['ground_truths'] = torch.stack(gts, dim=-2)

        if self._ed:
            seq_len, nsrc, input_dim = transformed['ground_truths'].shape
            # ground truths preceded by start symbols
            # combine last 2 dimensions
            # if self._gt_type == "STT3":
            #    start = torch.ones(1, input_dim)
            #    transformed['ground_truths'][:, :, s]
            start = torch.ones(1, nsrc * input_dim)
            # end = torch.zeros(1, n_sources, input_dim)
            gts_reshape = transformed['ground_truths'].view(seq_len, nsrc * input_dim)
            # [nbatch, seq_len+1, n_sources * input_dim]
            tgt_in = torch.cat([start, gts_reshape])
            # [nbatch, seq_len+1, n_sources, input_dim]
            # tgt_gt = torch.cat([gts, end])
            transformed['ground_truths_in'] = tgt_in.clone()
            transformed['ground_truths_gt'] = transformed['ground_truths']
        return transformed


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
        return len(os.listdir(self.root_dir)) - 1

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


class ToTensor(object):
    "2-channel numpy spectrogram to 2-channle tensor spectrogram"

    def __init__(self, size):
        self.size = size

    def __call__(self, item):
        transformed = {'aggregate': None, 'ground_truths': None}
        agg = self._from_numpy(item['aggregate'])
        transformed['aggregate'] = agg

        n_channels, seq_len, input_dim = agg.size()
        n_sources = len(item['ground_truths'])

        gts = torch.zeros((n_channels, seq_len, n_sources, input_dim))
        for s, gt in enumerate(item['ground_truths']):
            gts[:, :, s, :] = self._from_numpy(gt)

        gts = torch.cat(list(gts), -1)
        transformed['ground_truths'] = torch.FloatTensor(gts)
        return transformed

    def _from_numpy(self, m):
        input_dim, seq_len = self.size
        result = torch.zeros((2, seq_len, input_dim))
        result[0] = torch.from_numpy(m[0, :, :seq_len].T)
        result[1] = torch.from_numpy(m[1, :, :seq_len].T)
        return result.float()


class Concat(object):
    """2-channel spectrogram to single-channel tensor
    """

    def __init__(self, size, encdec=False):
        self.size = size
        self.encdec= encdec

    def __call__(self, item):
        transformed = {'aggregate': [], 'ground_truths': [],
                'ground_truths_in': [], 'ground_truths_gt': []}
        agg = self._from_complex(item['aggregate'])
        transformed['aggregate'] = agg

        seq_len, input_dim = agg.size()
        n_sources = len(item['ground_truths'])

        gts = torch.zeros((seq_len, n_sources, input_dim))
        for s, gt in enumerate(item['ground_truths']):
            gts[:, s, :] = self._from_complex(gt)

        if (self.encdec):
            # ground truths preceded by start symbols
            # combine last 2 dimensions
            start = torch.ones(1, n_sources * input_dim)
            # end = torch.zeros(1, n_sources, input_dim)
            gts_reshape = gts.view(seq_len, n_sources * input_dim)
            # [nbatch, seq_len+1, n_sources * input_dim]
            tgt_in = torch.cat([start, gts_reshape])
            # [nbatch, seq_len+1, n_sources, input_dim]
            # tgt_gt = torch.cat([gts, end])
            transformed['ground_truths_in'] = torch.FloatTensor(tgt_in)
            transformed['ground_truths_gt'] = torch.FloatTensor(gts)

        transformed['ground_truths'] = torch.FloatTensor(gts)

        return transformed

    def _from_complex(self, m):
        # num_channels, nrows, ncols = m.shape
        nrows, ncols = self.size

        result = np.zeros((nrows * 2, ncols))

        # contatenating real and imaginary features together
        result[:nrows, :] = m[0][:, :ncols]
        result[nrows:, :] = m[1][:, :ncols]

        return torch.t(torch.from_numpy(result)).float()

