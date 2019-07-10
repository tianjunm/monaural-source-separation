"""Mixer for dataset generation"""
import logging
import argparse
import random
import os
import errno
import csv
import json
import numpy as np
import scipy.io.wavfile
import scipy.signal
from pydub import AudioSegment
from tqdm.auto import tqdm


SILENT = True 
SEC_LEN = 1000
MIN_LEN = 300  # shortest clip is 300ms
COMPRESS_FACTOR = 2
WINDOW_SIZE = 256
OVERLAP_LEN = WINDOW_SIZE // 4
CATEGORY_CAPACITY = 60  # the category with least number of files has 59 files
TRAIN_TEST_SPLIT = 45  # first 45 files for train, the rest for test
CATEGORY_COUNT = 42  # number of distinct categories available
MAX_SIZE = 1e9


def get_arguments():
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation dataset')
    # arguments 
    parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--num_sources', type=int, default=3)
    parser.add_argument('--aggregate_duration', type=int, required=True)
    parser.add_argument('--num_examples', type=int, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--ground_truth_durations', type=float, nargs='*')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE)
    parser.add_argument('--overlap_len', type=int, default=OVERLAP_LEN)

    # options
    parser.add_argument("--test_dataset", dest="dataset_type", const="test", 
            action="store_const", default="train")
    # optional arguments
    # parser.add_argument('--selection_range', type=int, default=1)
    # parser.add_argument('--wav_ids', nargs='*')

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


def select_classes():
    selected = []
    while len(selected) < self.num_sources:
        i = random.randint(0, len(all_classes) - 1)
        path = + all_classes[i]
        if (i not in selected) and (not is_empty(path)):
            selected.append(i)
    self.selected_classes = selected


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

    def __init__(self, num_sources, aggregate_duration,
            ground_truth_durations, window_size, overlap_len):
        self.num_sources = num_sources
        self.agg_dur = aggregate_duration
        self.gt_durations = ground_truth_durations
        
        self.window_size = window_size
        self.overlap_len = overlap_len
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
        # self._generate_intervals()

        # fill self.ground_truths
        # assert(len(self.wav_files) == len(self.gt_durations))
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
        # extra return step for saving information regarding shape of spect
        return self.spect_shape

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

        _, data = scipy.io.wavfile.read(wav_path)
        sr, data = self._compress(data)

        _, _, spect_raw = scipy.signal.stft(
                data,
                fs=sr,
                nperseg=self.window_size,
                noverlap=self.overlap_len)
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

        # saving spect shape information for pytorch dataloader 
        # e.g. [2, 129, 231] - [nchan, freq_range, seq_len] 
        _, fr, sl = spect.shape
        self.spect_shape = (fr, sl)

        create_dir(spect_path)
        np.save(spect_path, spect)

    def _save_media(self, dataset_dir, metadata, i):
        '''
s        :param dataset_dir: root directory of the entire dataset
        :param i: the ith generated sample in this dataset
        '''

        media_dir = dataset_dir + '/{}/'.format(str(i))

        # save text modality
        self._save_metadata(media_dir, metadata)

        # aggregate
        log("Saving combined clip...", newline=True)
        self._save_wav_spect(media_dir, self.aggregate, 'agg')
        log("saved!")

        # ground truthss
        log("Saving ground truths...", newline=True)
        gt_dir = media_dir + 'gt/'
        for i in range(len(self.ground_truths)):
            gt = self.ground_truths[i]
            self._save_wav_spect(gt_dir, gt, str(i))
        log("saved!")

    # def _save_spect_shape(self, dataset_dir, spect_shape):
    #     spect_shape = {}
    #     spect_shape['freq_range'] = min(
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

    # def _generate_intervals(self):
    #     """Get the intervals where each wav appear in the combination

    #     First get the duration of each source, then randomize the interval
    #     where each source appear in the mixture.
    #     """

    #     n_frames = self.agg_dur * SEC_LEN
    #     durations = self.gt_durations

    #     if durations is None:
    #         # randomize durations for each source
    #         dur_lo = SEC_LEN  # one sec
    #         dur_hi = n_frames
    #         length = self.num_sources
    #         durations = [random.randint(dur_lo, dur_hi) for i in range(length)]

    #     intervals = []
    #     for duration_in_sec in durations:
    #         duration = duration_in_sec * SEC_LEN
    #         start_lo = 0
    #         start_hi = n_frames - duration
    #         start = random.randint(start_lo, start_hi)
    #         end = start + duration
    #         intervals.append((start, end))

    #     self.intervals = intervals

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
        "wav_files --> segments within mixture"
        n_frames = self.agg_dur * SEC_LEN
        ground_truths = []

#         for i, interval in enumerate(self.intervals):
#             wav = self.wav_files[i]
#             start, end = interval
#             pad_before = AudioSegment.silent(start)
#             pad_after = AudioSegment.silent(n_frames - end)
#             dur = int(end - start)
#             wav = wav[:dur]

#             ground_truth = pad_before + wav + pad_after
#             # set to mono
#             # ground_truth = ground_truth.set_channels(1)
#             ground_truths.append(ground_truth)

        for i, wav in enumerate(self.wav_files):
            if self.gt_durations is None:
                duration = min(len(wav), random.randint(MIN_LEN,
                    self.agg_dur * SEC_LEN))
            else:
                duration = min(len(wav), self.gt_durations[i] * SEC_LEN)

            start_lo = 0
            start_hi = n_frames - duration
            start = random.randint(start_lo, start_hi)
            end = start + duration
            wav = wav[:duration]
            pad_before = AudioSegment.silent(start)
        # select waves from each class
        # for src in args.selected_classes:
            pad_after = AudioSegment.silent(n_frames - end)

            ground_truth = pad_before + wav + pad_after
            assert(len(ground_truth) == n_frames)
            ground_truths.append(ground_truth)

        self.ground_truths = ground_truths
        assert(len(self.ground_truths) == len(self.wav_files))
            
    def _overlay_clips(self):
        aggregate = None
        for i, clip in enumerate(self.ground_truths):
            if i == 0:
                aggregate = clip
            else:
                aggregate = aggregate.overlay(clip)

        self.aggregate = aggregate

    def _compress(self, data):
        "make subsequent spectrogram space-efficient"
        tgt_len = len(data) // COMPRESS_FACTOR 
        data_resamp = scipy.signal.resample(data, tgt_len)
        sr_resamp = tgt_len // self.agg_dur 
        return sr_resamp, data_resamp

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

def get_filename(metadata_path, category, file_id):
    all_files = []
    with open(metadata_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1].strip() == category.strip() and row[2] == "1":
                all_files.append(row[0])

    return all_files[file_id] 


def get_categories(metadata_path):
    all_categories = set() 
    excluded = [
            "Bass_drum",
            "Burping_or_eructation",
            "Bus",
            "Cello",
            "Chime",
            "Double_bass",
            "Drawer_open_or_close",
            "Fireworks",
            "Fart",
            "Hi-hat",
            "Gong",
            "Glockenspiel",
            "Harmonica", 
            "Microwave_oven",
            "Scissors",
            "Squeak",
            "Telephone",
            "label"]
    with open(metadata_path, newline='') as f:
        reader = csv.reader(f)
        row_count = 0
        for row in reader:
            if row[1] not in excluded:
                all_categories.add(row[1])
        assert(len(all_categories) + len(excluded) - 1 == CATEGORY_COUNT)
    return list(all_categories)


def get_filename(metadata_path, category, file_id):
    all_files = []
    with open(metadata_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1].strip() == category.strip() and row[2] == "1":
                all_files.append(row[0])
    
    return all_files[file_id] 


def export_spect_shape(dataset_dir, spect_shape):
    with open("data_spec.json", "w") as f:
        json.dump(spect_shape, f) 


def main():
    args = get_arguments()
    
    # if not args.selected_classes:
        # select waves from each class
        # for src in args.selected_classes:
    #     args.selected_classes = select_classes()
    # all_filenames = get_filenames(args.raw_data_dir, args.selected_classes)

    all_categories = get_categories(args.metadata_path) 
    mixer = Mixer(
            args.num_sources,
            args.aggregate_duration,
            args.ground_truth_durations,
            args.window_size,
            args.overlap_len)
    
    spect_shape = {}
    spect_shape['freq_range'] = MAX_SIZE
    spect_shape['seq_len'] = MAX_SIZE

    for i in tqdm(range(args.num_examples)):

        # filename and source class of the components in the aggregate
        metadata = []

        # select waves from each class
        # for src in args.selected_classes:
        #     while True:
        #         if args.selection_range == -1:
        #             wav_id = random.randint(0, len(all_filenames[src]))
        #         else:
        #             wav_id = random.randint(0, args.selection_range - 1)
        #         filename = all_filenames[src][wav_id].strip()
        #         wav_path = os.path.join(args.raw_data_dir, filename)
        #         wav_file = AudioSegment.from_wav(wav_path)
        #         if len(wav_file) >= args.aggregate_duration * SEC_LEN:
        #             break
        #     metadata.append((filename, src))
        #     mixer.wav_files.append(wav_file)
    
        class_ids = random.sample(range(0, len(all_categories) - 1),
                args.num_sources) 

        # each source comes from a different category
        for class_id in class_ids: 
            # choose a file within chosen category
            # while True:
            if args.dataset_type == "train":
                file_id = random.randint(0, TRAIN_TEST_SPLIT - 1) 
            else:
                file_id = random.randint(TRAIN_TEST_SPLIT,
                        CATEGORY_CAPACITY - 1)

            filename = get_filename(args.metadata_path, 
                    all_categories[class_id], file_id)

            wav_path = os.path.join(args.raw_data_dir, filename)
            wav_file = AudioSegment.from_wav(wav_path)
                # if len(wav_file) >= args.aggregate_duration * SEC_LEN:
                #     break
            metadata.append((filename, all_categories[class_id]))

            mixer.wav_files.append(wav_file)

        # assert(len(mixer.wav_files) == len(args.selected_classes))
        # # print(args.selected_classes)
        # print(mixer.wav_files)
        # print(args.num_sources)
        assert(len(mixer.wav_files) == args.num_sources)

        # FIXME: handle clips with short duration
        mixer.mix()
        fr, sl = mixer.export_results(args.dataset_dir, i, metadata)
        mixer.reset()
        
        # handling potentially differing output shape of STFT
        # using the smallest spect dimension as the unified dimension for all
        spect_shape['freq_range'] = min(fr, spect_shape['freq_range'])
        spect_shape['seq_len'] = min(sl, spect_shape['seq_len'])

    export_spect_shape(args.dataset_dir, spect_shape) 


if __name__ == "__main__":
    main()

