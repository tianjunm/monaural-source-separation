"""Convert between wav files and spectrograms
"""

import argparse
import json
import librosa
import numpy as np
from tqdm import tqdm
from scipy import signal

FORWARD = 'wav2spec'
BACKWARD = 'spec2wav'
COMPRESS_FACTOR = 1


def get_argument():
    """create parser for arguments"""
    parser = argparse.ArgumentParser(description='Convert \
            between wav files and spectrograms')
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--direction', type=str, default=FORWARD)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--compress_factor', type=int, default=COMPRESS_FACTOR)
    # parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    return parser.parse_args()


def griffinlim(spectrogram, n_iter=100, window='hann', n_fft=2048,
               hop_length=-1, verbose=False):
    '''audio reconstruction

    Reference: https://github.com/librosa/librosa/issues/434 by Jongwook
    '''

    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)
        rebuilt = librosa.stft(inverse, n_fft=n_fft,
                               hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, window=window)

    return inverse


def reconstruct(spect):
    reconstructed_data = griffinlim(spect)
    # TODO: reconsrucr wirh sr
    return reconstructed_data


def get_spectrogram(data, sample_rate, compress_factor=1):
    n_frames = len(data)
    out_frames = n_frames // compress_factor
    sample_rate_compressed = sample_rate // compress_factor

    data_compressed = signal.resample(data, out_frames)

    spect = np.abs(librosa.stft(data_compressed))**2
    return (spect, sample_rate_compressed)


def main():
    args = get_argument()

    # wav to spectrogram
    if args.direction == FORWARD:
        data, sr_in = librosa.load(args.filename)
        spect, sr = get_spectrogram(data, sr_in, args.compress_factor)

        # save
        filename = args.filename.split('/')[-1].split('.wav')[0]
        filepath = args.output_dir + '/' + filename
        info = {'spectrogram': spect.tolist(), 'sample_rate': sr}
        with open('{}.json'.format(filepath), 'w') as fjson:
            json.dump(info, fjson)

    # TODO: reconstruction
    # else:  # BACKWARD
    #     with open(args.filename, 'r') as fjson:
    #         info = json.load(fjson)
    #         spect = info['spectrogram']
    #         sr = info['sample_rate']
    #     wav = reconstruct(spect)


if __name__ == "__main__":
    main()
