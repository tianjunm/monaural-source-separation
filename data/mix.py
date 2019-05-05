"""Mixer for dataset generation"""
import logging
import argparse
import random
import os
import errno
import csv
import numpy as np
import scipy.io.wavfile
import scipy.signal
from pydub import AudioSegment
from tqdm import tqdm

# DATA_DIRECTORY = "./raw/"
# NUM_CLASSES = 100
# CLASS_MAX_SIZE = 17

SILENT = True 
SEC_LEN = 1000


def get_arguments():
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation dataset')
    # parser.add_argument('--id', type=int)
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--num_sources', type=int, default=3)
    parser.add_argument('--aggregate_duration', type=int, required=True)
    parser.add_argument('--num_examples', type=int, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)

    # optional arguments
    parser.add_argument('--selected_classes', nargs='*')
    parser.add_argument('--selection_range', type=int, default=1)
    # parser.add_argument('--wav_ids', nargs='*')
    parser.add_argument('--ground_truth_durations', type=float, nargs='*')

    return parser.parse_args()


def log(msg, newline=False):
    if not SILENT:
        if newline:
            print(msg, end='')
        else:
            print(msg)


def is_empty(path):
    return ([f for f in os.listdir(path)
             if not (f.startswith('.') or ('csv' in f))] == [])


def select(upperbound, num_samples):
    if num_samples > upperbound:
        raise Exception('uppebound is smaller than number of samples!')

    return random.sample(range(upperbound), num_samples)


def get_pathname(class_name, wav_id):
    '''get pathname for file'''
    pathname = DATA_DIRECTORY + class_name + "/{}.wav".format(wav_id)
    return pathname


class Mixer():
    '''Mixer creates combinations of sound files.

    Mixer object mixes individual tracks from given library, with different
    levels of randomization.

    :attribute agg_dur: length of generated mixture
    :attribute num_sources: number of sound sources in the mixture
    :attribute wav_ids: ids of .wav files within each class
    :attribute gt_durations: durations for each independent source
    :attribute intervals: periods in which sound clips appear in the mixture
    :attribute wav_files: AudioSegment objects of raw wav file of each source
    :attribute ground_truths: AudioSegment objects of cropped wav of each source
    :attribute aggregate: AudioSegment object the aggregate

    '''

    def __init__(self, num_sources, aggregate_duration, ground_truth_durations):
        self.num_sources = num_sources
        self.agg_dur = aggregate_duration
        self.gt_durations = ground_truth_durations

        self.reset()

    # PUBLIC METHODS

    def mix(self):
        """Combing multiple tracks into one

        Mixes sound files from different sources and returns the mixture
        and ground truths.

        :param all_classes: a list of names of available classes of sound sources
        :param selected_classes: types of sound sources to appear in the aggregate
        :param duration: length of the final output

        :return intervals: list of starting and end times for each source
        :return ground_truths: list of individual clips of each source
        :return aggregate: the combined clip

        """

        # fill self.intervals
        self._generate_intervals()

        # fill self.ground_truths
        assert(len(self.wav_files) == len(self.gt_durations))
        self._create_ground_truths()

        # fill self.agregate
        self._overlay_clips()

    def reset(self):
        self.wav_files = []
        self.intervals = []
        self.ground_truths = None
        self.aggregate = None


    def export_results(self, dataset_dir, i, metadata):
        '''Export agg and gt into root directory of dataset

        :param dataset_dir: root directory of the entire dataset
        :param i: the ith generated sample in this dataset
        :param filenames: file names of wav files from which gts are extracted

        '''
        # self._save_metadata(out_name, out_path, all_classes)
        self._save_media(dataset_dir, metadata, i)
        # self._save_config(out_name, out_path)

    # PRIVATE METHODS

    def _save_metadata(self, dataset_dir, metadata):
        meta_path = '{}/meta.txt'.format(dataset_dir)
        create_dir(meta_path)
        with open(meta_path, 'w+') as meta:
            for filename, src in metadata:
                meta.write('{} {}\n'.format(filename, src))


    # def _save_metadata(self, out_name, out_path, all_classes):
    #     meta_path = out_path + "/" + "{}.csv".format(out_name)
    #     create_dir(meta_path)

    #     log("Saving metadata...", newline=True)
    #     with open(meta_path, 'w+') as meta:
    #         meta_fields = ["start time", "end time", "class_name"]
    #         writer = csv.DictWriter(meta, fieldnames=meta_fields)
    #         writer.writeheader()
    #         for i, interval in enumerate(self.intervals):
    #             start, end = interval
    #             class_id = int(self.selected_classes[i])
    #             writer.writerow({
    #                 "start time": start / SEC_LEN,
    #                 "end time": end / SEC_LEN,
    #                 "class_name": all_classes[class_id]
    #             })
    #     log("saved!")


    def _get_spect(self, wav_path):
        '''
        :param wav_path: path to saved wav file 
        :return spect: spectrogram with real and complex components
        '''

        # sr = w.frame_rate
        # data = np.frombuffer(w.raw_data, dtype=np.int8)[0::2]

        sr, data = scipy.io.wavfile.read(wav_path)
        _, _, spect_raw = scipy.signal.stft(data, fs=sr, boundary='zeros',
                padded=True, nfft=256)

        nrows, ncols = spect_raw.shape

        # 2 channels
        spect_sep = np.zeros((2, nrows, ncols))
        spect_sep[0, :, :] = spect_raw.real
        spect_sep[1, :, :] = spect_raw.imag

        return spect_sep

    def _save_wav_spect(self, output_dir, w, filename):
        '''
        :param output_dir: filename to save
        :param w: wav
        :param filename: filename within output directory
        '''

        # audio
        audio_path = output_dir + '{}.wav'.format(filename)
        create_dir(audio_path)
        w.export(audio_path, format="wav")

        # spect
        spect_path = output_dir + filename 
        spect = self._get_spect(audio_path)
        create_dir(spect_path)
        np.save(spect_path, spect)

    def _save_media(self, dataset_dir, metadata, i):
        '''
        :param dataset_dir: root directory of the entire dataset
        :param i: the ith generated sample in this dataset
        '''

        media_dir = dataset_dir + '/{}/'.format(str(i))

        # save text modality
        self._save_metadata(media_dir, metadata)

        # aggregate
        log("Saving combined clip...", newline=True)
        self._save_wav_spect(media_dir, self.aggregate, 'agg')
        log("saved!")

        # ground truths
        log("Saving ground truths...", newline=True)
        gt_dir = media_dir + 'gt/'
        for i in range(len(self.ground_truths)):
            gt = self.ground_truths[i]
            self._save_wav_spect(gt_dir, gt, str(i))
        log("saved!")

    # def _save_config(self, out_name, out_path):
    #     """
    #     Space-separated argument list corresponding to:
    #     --num_sources, --duration, --selected_classes, --wav_ids,
    #     --intervals, --out_path
    #     """
    #     log("Saving configs...", newline=True)
    #     config_path = out_path + "/" + "{}.txt".format(out_name)
    #     create_dir(config_path)
    #     content = ""

    #     selected_ids = [str(cid) for cid in self.selected_classes]
    #     wav_ids = [str(wid) for wid in self.wav_ids]
    #     content += (str(self.num_sources) + " ")
    #     content += (str(self.agg_dur) + " ")
    #     content += (" ".join(selected_ids) + " ")
    #     content += (" ".join(wav_ids) + " ")
    #     # content += (" ".join(self.intervals) + " ")
    #     content += (out_path + "\n")

    #     with open(config_path, 'w') as file_path:
    #         file_path.write(content)

    #     log("saved!")

    # def _select_classes(self, all_classes):
    #     selected = []
    #     while len(selected) < self.num_sources:
    #         i = random.randint(0, len(all_classes) - 1)
    #         path = DATA_DIRECTORY + all_classes[i]
    #         if (i not in selected) and (not is_empty(path)):
    #             selected.append(i)
    #     self.selected_classes = selected

    def _generate_intervals(self):
        """Get the intervals where each wav appear in the combination

        First get the duration of each source, then randomize the interval
        where each source appear in the mixture.
        """

        n_frames = self.agg_dur * SEC_LEN
        durations = self.gt_durations

        if durations is None:
            # randomize durations for each source
            dur_lo = SEC_LEN  # one sec
            dur_hi = n_frames
            length = self.num_sources
            durations = [random.randint(dur_lo, dur_hi) for i in range(length)]

        intervals = []
        for duration_in_sec in durations:
            duration = duration_in_sec * SEC_LEN
            start_lo = 0
            start_hi = n_frames - duration
            start = random.randint(start_lo, start_hi)
            end = start + duration
            intervals.append((start, end))

        self.intervals = intervals
        assert(len(self.intervals) == len(self.wav_files))

    # def _get_wav_files(self, all_classes):
    #     # if specific sound clip IDs are not given
    #     if self.wav_ids is None:
    #         self.wav_ids = []
    #         for class_id in self.selected_classes:
    #             class_name = all_classes[int(class_id)]
    #             while True:
    #                 try:
    #                     i = random.randint(0, CLASS_MAX_SIZE - 1)
    #                     wav_path = get_pathname(class_name, i)
    #                     wav_file = AudioSegment.from_wav(wav_path)
    #                     break  # found clip, go to next category
    #                 except OSError:
    #                     pass
    #             self.wav_ids.append(i)
    #             self.wav_files.append(wav_file)

    #     # sound clip IDs are given
    #     else:
    #         for i, class_id in enumerate(self.selected_classes):
    #             class_name = all_classes[int(class_id)]
    #             wav_id = self.wav_ids[i]
    #             wav_path = get_pathname(class_name, wav_id)
    #             log(wav_path)
    #             wav_file = AudioSegment.from_wav(wav_path)
    #             self.wav_files.append(wav_file)

    def _create_ground_truths(self):
        '''Based on intervals, crop wav files and store them as ground truths

        '''
        n_frames = self.agg_dur * SEC_LEN
        ground_truths = []

        for i, interval in enumerate(self.intervals):
            wav = self.wav_files[i]
            start, end = interval
            pad_before = AudioSegment.silent(start)
            pad_after = AudioSegment.silent(n_frames - end)
            dur = int(end - start)
            wav = wav[:dur]

            ground_truth = pad_before + wav + pad_after
            # set to mono
            ground_truth = ground_truth.set_channels(1)
            ground_truths.append(ground_truth)

        self.ground_truths = ground_truths

    def _overlay_clips(self):
        aggregate = None
        for i, clip in enumerate(self.ground_truths):
            if i == 0:
                aggregate = clip
            else:
                aggregate = aggregate.overlay(clip)

        aggregate = aggregate.set_channels(1)

        self.aggregate = aggregate


def create_dir(outdir):
    """creates directory if it does not exist"""
    if not os.path.exists(os.path.dirname(outdir)):
        try:
            os.makedirs(os.path.dirname(outdir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# def get_classes(classes):
#     """gather all available classes"""
#     all_classes = [name for name in os.listdir(path)]
#     return all_classes

def get_filenames(raw_data_dir, classes):
    all_classes = {}
    for src in classes:
        with open(raw_data_dir + '/meta/{}.txt'.format(src), 'r') as f:
            all_classes[src] = f.readlines()
    return all_classes


def main():
    args = get_arguments()
    all_filenames = get_filenames(args.raw_data_dir, args.selected_classes)

    mixer = Mixer(
            args.num_sources,
            args.aggregate_duration,
            args.ground_truth_durations)
    
    for i in tqdm(range(args.num_examples), ncols=100):

        # filename and source class of the components in the aggregate
        metadata = []

        # select waves from each class
        for src in args.selected_classes:
            while True:
                if args.selection_range == -1:
                    wav_id = random.randint(0, len(all_filenames[src]))
                else:
                    wav_id = random.randint(0, args.selection_range - 1)
                filename = all_filenames[src][wav_id].strip()
                wav_path = os.path.join(args.raw_data_dir, filename)
                wav_file = AudioSegment.from_wav(wav_path)
                if len(wav_file) >= args.aggregate_duration * SEC_LEN:
                    break
            metadata.append((filename, src))
            mixer.wav_files.append(wav_file)
        
        assert(len(mixer.wav_files) == len(args.selected_classes))
        # print(mixer.wav_files)
        # print(args.num_sources)
        assert(len(mixer.wav_files) == args.num_sources)

        mixer.mix()
        mixer.export_results(args.dataset_dir, i, metadata)
        mixer.reset()


if __name__ == "__main__":
    main()

