"""
This script generates a complete dataset based on the given configuration.
The generated dataset contains train, validation, and test sets.
"""

import logging
import argparse
import random
import os
import errno
import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.signal
from pydub import AudioSegment
from tqdm.auto import tqdm


FRAMES_PER_SEC = 1000

# COMPRESS_FACTOR = 2
# WINDOW_SIZE = 256
# OVERLAP_LEN = WINDOW_SIZE // 4
# CATEGORY_CAPACITY = 60  # the category with least number of files has 59 files
# CATEGORY_CAPACITY = 25  # the category with least number of files has 59 files
# TRAIN_TEST_SPLIT = 45  # first 45 files for train, the rest for test
# TRAIN_TEST_SPLIT = 15  # first 45 files for train, the rest for test
# CATEGORY_COUNT = 42  # number of distinct categories available
# # CATEGORY_COUNT = 42  # number of distinct categories available
# CATEGORY_COUNT = 41  # number of distinct categories available
# MAX_SIZE = np.inf
DATASET_ROOT = '/home/ubuntu/datasets/processed/datagen'

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

def get_arguments():
    """parsing arguments"""
    parser = argparse.ArgumentParser(description='Dataset creation script')
    # arguments
    # parser.add_argument('--raw_data_dir', type=str, required=True)
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--num_categories', type=int, required=True)
    parser.add_argument('--num_sources', type=int, default=2)
    parser.add_argument('--max_clip_duration', type=int, required=True)
    parser.add_argument('--mixture_duration', type=int, required=True)
    parser.add_argument('--dataset_path', type=str, default=DATASET_ROOT)
    # parser.add_argument('--window_size', type=int, default=WINDOW_SIZE)
    # parser.add_argument('--overlap_len', type=int, default=OVERLAP_LEN)

    # options
    # parser.add_argument("--test_dataset", dest="dataset_type", const="test",
    #                     action="store_const", default="train")
    # optional arguments
    # parser.add_argument('--selection_range', type=int, default=1)
    # parser.add_argument('--wav_ids', nargs='*')

    return parser.parse_args()

class Mixer():
    """Mixer creates combinations of sound files.

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

    """

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
        """Export agg and gt into root directory of dataset

        :param dataset_dir: root directory of the entire dataset
        :param i: the ith generated sample in this dataset
        :param filenames: file names of wav files from which gts are extracted

        """
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
        """
        :param wav_path: path to saved wav file 
        :return spect: spectrogram with real and complex components
        """

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
        """
        :param output_dir: filename to save
        :param w: wav
        :param filename: filename within output directory
        """

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
        """
        :param dataset_dir: root directory of the entire dataset
        :param i: the ith generated sample in this dataset
        """

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

    # def get_wav_files(self, all_classes):
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


def sample_categories(metadata, ncat, ds_type=1):
    """Samples ncat categories from all categories presented in metadata."""
    all_cats = metadata.label.unique()

    if ds_type == 1:
        ids = random.sample(range(len(all_cats)), ncat)

    elif ds_type == 2:
        ids = [random.randint(0, len(all_cats) - 1)] * ncat

    else:
        ids = []
        for _ in range(ncat):
            ids.append(random.randint(0, len(all_cats) - 1))

    return all_cats[ids]


def extract_metadata(metadata_path):
    """Extracts verified instances from the raw dataset

    """
    raw_metadata = pd.read_csv(metadata_path)
    header = ['fname', 'label']
    metadata = raw_metadata[raw_metadata.manually_verified == 1][header]
    return metadata


def select_data(metadata, ncat):
    """Selects [num_categories] categories.

    This functions returns the set of categories that the to-be-generated
    dataset will choose from. The size of the set will be [num_categories].
    """

    # manually selected categories to avoid
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
        "Telephone"]

    # keep rows not of categories specified in the excluded list
    filtered = metadata[~metadata.label.isin(excluded)]

    # select [num_categories] categories from all categories
    selected_cats = sample_categories(filtered, ncat)

    # keep rows only of the selected categories
    selected = filtered[filtered.label.isin(selected_cats)]

    return selected


class Dataset():
    """A class that represents train/val/test instance collection

    """

    def __init__(self, dataset_path):
        self.reset()
        self._dest_root = dataset_path

    def reset(self):
        """Sets the dataset empty"""
        self._dataset = {
            'filename': [],
            'clip_duration': [],
            'mixture_placement': [],
            'category': []
            }

    def insert(self, info):
        """Inserts a row into the result dataset."""
        self._dataset['filename'].append(info['filename'])
        self._dataset['clip_duration'].append(info['duration'])
        self._dataset['mixture_placement'].append(info['start_time'])
        self._dataset['category'].append(info['category'])

    def export(self, ds_name, nsrc, ncat, ds_type=1):
        """Saves the dataset to a csv file."""
        # prefix = '{}-s_{}-c'.format(nsrc, ncat)
        prefix = f"t{ds_type}-{nsrc}s-{ncat}c"
        dest = os.path.join(self._dest_root, prefix, '{}.csv'.format(ds_name))
        create_dir(dest)
        pd.DataFrame(self._dataset).to_csv(dest, index=False)


class DataGenerator(object):
    """A class used to generate the dataset

    """

    def __init__(
            self,
            num_sources,
            train_val_test_split,
            selected_data,
            max_clip_duration,
            mixture_duration,
            dataset_path,
            num_categories,
            dataset_type=1,
            types_to_generate=['train', 'val', 'test']):

        self._ds_type = dataset_type
        self._nsrc = num_sources
        self._split = train_val_test_split
        self._data = selected_data
        self._cdur = max_clip_duration
        self._mdur = mixture_duration
        self._dtypes = types_to_generate
        self._ncat = num_categories

        self._dataset = Dataset(dataset_path)

    # public
    def generate(self):
        """Generate the dataset for the corresponding dataset_type."""
        for dtype in self._dtypes:
            self._gen(dtype)

    # private
    def _gen(self, dtype):
        logging.info('generating %s set...', dtype)

        # size of the train/val/test dataset to be generated
        dsize = self._split['sizes'][dtype]

        for _ in tqdm(range(dsize)):
            # choose [nsrc] distinct, same, or mixed
            # categories from the given dataframe
            cats = sample_categories(self._data, self._nsrc, self._ds_type)

            # for each chosen category for this training instance,
            # choose a sound file from the sounds of that category
            # within the allowed range specific to each dataset_type
            for cat in cats:
                chosen = self._choose_sound_file(cat, dtype)
                # specifies how instances for the dataset will be extracted
                # from the raw files
                info = self._pick_and_drop(chosen)
                self._dataset.insert(info)

        self._dataset.export(dtype, self._nsrc, self._ncat, self._ds_type)
        self._dataset.reset()

        logging.info('finished generating the %s set!', dtype)

    def _choose_sound_file(self, cat, dtype):
        # getting the range of allowed sound clips to choose from
        start, end = self._split['ranges'][dtype]
        allowed_data = self._data[self._data.label == cat][start:end]
        return allowed_data.sample(1)

    def _pick_and_drop(self, chosen):
        info = {}

        info['filename'] = chosen.fname.to_string(index=False).strip()
        info['category'] = chosen.label.to_string(index=False).strip()

        # randomize the duration of the snippet from the original data
        clip_dur = random.randint(FRAMES_PER_SEC, self._cdur * FRAMES_PER_SEC)

        # randomize the placement of clip in the mixture
        start_lo = 0
        start_hi = self._mdur * FRAMES_PER_SEC - clip_dur
        start = random.randint(start_lo, start_hi)

        info['duration'] = clip_dur
        # info['placement'] = [start, end]
        info['start_time'] = start

        return info


def define_dataset_split(num_sources):
    """Specifications of the train/val/test sets to be generated.


    This function sepcifies the sizes of train/val/test sets. Beyond the sizes
    of different datasets, different types of dataset should contain sets of
    data exclusive to others to guarantee fair evaluation of the generalization
    capability of the trained model. This function also specifies how these
    datasets divide the raw data (file_selection_range).
    """

    train_scale = 10000
    val_scale = 1000
    test_scale = 500

    return {
        # sizes of train/val/test datasets
        'sizes': {
            'train': num_sources * train_scale,
            'val': num_sources * val_scale,
            'test': num_sources * test_scale
            },

        # ranges within each category of the raw dataset from which
        # train/val/test datasets draw their data from
        'ranges': {
            'train': (0, 40),
            'val': (40, 50),
            'test': (50, 60)
            },
        }


def main():
    args = get_arguments()
    logging.info('preparing dataset generation...')

    # datasets are generated based on the raw dataset, whose metadata will be
    # contained in the following dataframe
    metadata = extract_metadata(args.metadata_path)

    # given the number of categories to appear in the dataset,
    # we randomly select them from all categories in the raw dataset
    selected_data = select_data(metadata, args.num_categories)

    # specification of how to split files within each category
    # of the raw dataset among train, val, and test sets
    train_val_test_split = define_dataset_split(args.num_sources)

    generator = DataGenerator(
        args.num_sources,
        train_val_test_split,
        selected_data,
        args.max_clip_duration,
        args.mixture_duration,
        args.dataset_path,
        args.num_categories,
        args.dataset_type)
        # types_to_generate=['test'])

    generator.generate()

    logging.info('finished generating the dataset!')


def deprecated():
    # for i in tqdm(range(args.num_examples)):

    #     # FIXME: what are these
    #     # filename and source class of the components in the aggregate
    #     metadata = []

    #     # select waves from each class
    #     # for src in args.selected_classes:
    #     #     while True:
    #     #         if args.selection_range == -1:
    #     #             wav_id = random.randint(0, len(all_filenames[src]))
    #     #         else:
    #     #             wav_id = random.randint(0, args.selection_range - 1)
    #     #         filename = all_filenames[src][wav_id].strip()
    #     #         wav_path = os.path.join(args.raw_data_dir, filename)
    #     #         wav_file = AudioSegment.from_wav(wav_path)
    #     #         if len(wav_file) >= args.aggregate_duration * SEC_LEN:
    #     #             break
    #     #     metadata.append((filename, src))
    #     #     mixer.wav_files.append(wav_file)

    #     # TODO: given num_categories, choose randomly
    #     class_ids = random.sample(range(0, len(all_categories) - 1),
    #                               args.num_sources)


    #     # each source comes from a different category
    #     for class_id in class_ids:
    #         # choose a file within chosen category
    #         # TODO: add validation
    #         if args.dataset_type == "train":
    #             file_id = random.randint(0, TRAIN_TEST_SPLIT - 1)
    #         else:
    #             file_id = random.randint(TRAIN_TEST_SPLIT,
    #                                      CATEGORY_CAPACITY - 1)

    #         filename = get_filename(args.metadata_path,
    #                                 all_categories[class_id], file_id)

    #         wav_path = os.path.join(args.raw_data_dir, filename)
    #         wav_file = AudioSegment.from_wav(wav_path)
    #             # if len(wav_file) >= args.aggregate_duration * SEC_LEN:
    #             #     break
    #         metadata.append((filename, all_categories[class_id]))

    #         mixer.wav_files.append(wav_file)

    #     assert len(mixer.wav_files) == args.num_sources

    #     mixer.mix()
    #     fr, sl = mixer.export_results(args.dataset_dir, i, metadata)
    #     mixer.reset()

    #     # TODO: keep or remove
    #     # handling potentially differing output shape of STFT
    #     # using the smallest spect dimension as the unified dimension for all
    #     spect_shape['freq_range'] = min(fr, spect_shape['freq_range'])
    #     spect_shape['seq_len'] = min(sl, spect_shape['seq_len'])

    # export_spect_shape(args.dataset_dir, spect_shape)
    pass


if __name__ == "__main__":
    main()

