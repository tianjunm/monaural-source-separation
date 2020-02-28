"""convert audio dataset to images for training

FIXME:

Problems with current data pipeline:

- currently only randomizing start of source within mixture,
  while always cutting the source clips from the beginning 

- fetching datapoints and raw_data separately

- datapoints need to be overwritten were new classes to be introduced

- not sure how to perform dataset normalization, or if it's necessary

- dataset does not come with metadata (i.e. categories within mixture)

"""
import copy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from mmsdk import mmdatasdk

RAW_DATA_PATH = '/work/tianjunm/large_datasets/audioset_verified/audio_csd/cut/16000'

# XXX: STFT parameters, should be present in the datapoint file
SAMPLE_RATE = 16000
MIXTURE_DURATION = 2 * SAMPLE_RATE
WINDOW_SIZE = 256
HOP_LENGTH = 256 // 4 * 3


# audio manipulation utils
def overlay(ground_truths):
    """Overlay ground truths together to create the aggregate."""
    aggregate = copy.deepcopy(ground_truths[0])
    for idx in range(1, len(ground_truths)):
        aggregate += ground_truths[idx]
    return aggregate


def get_spect(wav, ttype):
    """Apply STFT on the wave form to get its spectrogram representation."""
    wav_arr = torch.from_numpy(wav).float()
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
            raw_path=RAW_DATA_PATH,
            transform=None):
        self._nsrc = num_sources
        self._datapoints = pd.read_csv(data_path)
        self._raw_path = raw_path
        self._transform = transform
        self._raw_data = self._init_compseq()


    def __len__(self):
        return len(self._datapoints) // self._nsrc

    def __getitem__(self, idx):
        # what the dataloader loads
        item = {'aggregate': None, 'ground_truths': [], 'category_names': []}

        # load [nsrc] rows from instruction
        instances = self._load_instances(idx)

        for iid in instances.index:
            item['ground_truths'].append(self._create_gt(instances, iid))
            item['category_names'].append(instances.at[iid, 'category'])

        item['aggregate'] = overlay(item['ground_truths'])

        return self._transform(item)
    
    def _init_compseq(self):
        """Loads all raw data into memory. 

        """

        files = {}
        for csd_file in os.listdir(self._raw_path):
            # category = csd_file
            files[csd_file] = os.path.join(self._raw_path, csd_file)
            dataset = mmdatasdk.mmdataset(files)

        return dataset

    def _load_instances(self, idx):
        # loading the section containing the [idx]th set of instances
        begin, end = idx * self._nsrc, (idx + 1) * self._nsrc
        instances = self._datapoints.iloc[begin:end]
        return instances

    def _create_gt(self, instances, iid):
        # load original files according to filenames

        category = instances.at[iid, 'category']
        sound_id = instances.at[iid, 'filename']
        dur = instances.at[iid, 'clip_duration']
        start = instances.at[iid, 'mixture_placement']

        wav = self._raw_data.computational_sequences[category][sound_id]['features']
        wav = np.sum(wav, axis=0) / 2

        wav = wav[:min(len(wav), dur)]

        # follow the datapoint to create ground truths
        prefix = np.zeros(start) * 1.0
        suffix = np.zeros(MIXTURE_DURATION - len(wav) - start) * 1.0

        # padded_wav = pad_before + wav + pad_after
        wav_padded = np.concatenate((prefix, wav, suffix), axis=None)
        assert len(wav_padded) == MIXTURE_DURATION

        return wav_padded 


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


# transformations


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
        transformed['category_names'] = item['category_names']

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

            # [nbatch, seq_len + 1, n_sources * input_dim]
            tgt_in = torch.cat([start, gts_reshape])

            # [nbatch, seq_len+1, n_sources, input_dim]
            # tgt_gt = torch.cat([gts, end])
            transformed['ground_truths_in'] = tgt_in.clone()
            transformed['ground_truths_gt'] = transformed['ground_truths']
        return transformed


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

