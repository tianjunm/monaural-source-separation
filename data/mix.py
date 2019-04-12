"""Mixer for dataset generation"""
# import logging
import argparse
import random
import os
import errno
import csv
from pydub import AudioSegment


SILENT = True 
DATA_DIRECTORY = "./raw/"
NUM_CLASSES = 100
# FIXME: current dataset is a mini version
CLASS_MAX_SIZE = 17
SEC_LEN = 1000


def log(msg, newline=False):
    if not SILENT:
        if newline:
            print(msg, end='')
        else:
            print(msg)


def is_empty(path):
    """is_empty"""
    return ([f for f in os.listdir(path)
             if not (f.startswith('.') or ('csv' in f))] == [])


def select(upperbound, num_samples):
    """select num_samples random numbers within upperbound"""
    if num_samples > upperbound:
        raise Exception("uppebound is smaller than number of samples!")

    return random.sample(range(upperbound), num_samples)


def get_pathname(class_name, wav_id):
    """get pathname for file"""
    pathname = DATA_DIRECTORY + class_name + "/{}.wav".format(wav_id)
    return pathname


def get_arguments():
    """get arguments"""
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation')
    parser.add_argument('--id', type=int)
    parser.add_argument('--num_sources', type=int, default=3, required=True)
    parser.add_argument('--duration', type=int, required=True)

    # optional arguments
    parser.add_argument('--selected_classes', nargs='*')
    parser.add_argument('--wav_ids', nargs='*')
    parser.add_argument('--source_durations', type=float, nargs='*')
    parser.add_argument('--out_path', type=str, default='./test')

    return parser.parse_args()


class Mixer():
    """Mixer creates combinations of sound files.

    Mixer object mixes individual tracks from given library, with different
    levels of randomization.

    Attributes:
        mixture_duration:
            length of generated mixture
        all_classes (:obj:`list` of :obj:`str`):
            all names of sound sources
        selected_classes:
            types of sound sources to be combined
        wav_ids:
            ids of .wav files within each class in selected_classes
        source_durations:
            durations for each independent source
        intervals:
            times for each sound clip to appear in the mixture
        wav_files:
            .wav clips for overlay
        ground_truths:
            list of individual clips of each source
        aggregate:
            the combined clip
    """

    def __init__(self, args):
        self.num_sources = args.num_sources
        self.mixture_duration = args.duration
        self.selected_classes = args.selected_classes
        self.wav_ids = args.wav_ids
        self.source_durations = args.source_durations
        self.wav_files = []
        self.ground_truths = None
        self.aggregate = None

    # PUBLIC METHODS

    def mix(self, all_classes):
        """Combing multiple tracks into one

        Mixes sound files from different sources and returns the mixture and
        ground truths.

        Args:
            all_classes: a list of names of available classes of sound sources
            selected_classes: a subset of all_classes represented by indices,
                              randomly chosen
            duration: length of the final output
        Returns:
            intervals: list of starting and end times for each source
            ground_truths: list of individual clips of each source
            aggregate: the combined clip
        """
        if not self.selected_classes:
            self._select_classes(all_classes)

        # self.intervals
        self._generate_intervals()

        # self.wav_files
        # self._get_wav_files(all_classes)
        # FIXME: temporary, change later
        honk = AudioSegment.from_wav('honk.wav')
        bird = AudioSegment.from_wav('bird.wav')
        self.wav_files = [honk, bird]

        # self.ground_truths
        self._create_ground_truths()

        # self.agregate
        self._overlay_clips()

    # TODO: allow resetting different fields
    def reset(self, criteria):
        """reset the values for given field and randomize"""

    def export_results(self, out_path, all_classes, iter_id):
        '''export all related results
        Args:
            iter_id: ith iteration
        '''
        # sample: 1-2-3
        out_name = "-".join([str(i) for i in self.selected_classes])
        suffix = str(iter_id)
        # self._save_metadata(out_name, out_path, all_classes)
        self._save_media(out_name, out_path, suffix)
        # self._save_config(out_name, out_path)

    # PRIVATE METHODS

    def _save_metadata(self, out_name, out_path, all_classes):
        meta_path = out_path + "/" + "{}.csv".format(out_name)
        create_dir(meta_path)

        log("Saving metadata...", newline=True)
        with open(meta_path, 'w+') as meta:
            meta_fields = ["start time", "end time", "class_name"]
            writer = csv.DictWriter(meta, fieldnames=meta_fields)
            writer.writeheader()
            for i, interval in enumerate(self.intervals):
                start, end = interval
                class_id = int(self.selected_classes[i])
                writer.writerow({
                    "start time": start / SEC_LEN,
                    "end time": end / SEC_LEN,
                    "class_name": all_classes[class_id]
                })
        log("saved!")

    def _save_media(self, out_name, out_path, suffix):
        # combined clip
        clip_path = out_path + "/" + "{}_{}.wav".format(out_name, suffix)
        create_dir(clip_path)

        log("Saving combined clip...", newline=True)
        log(clip_path)
        self.aggregate.export(clip_path, format="wav")
        log("saved!")

        # ground truths
        log("Saving ground truths...", newline=True)
        for i in range(len(self.ground_truths)):
            ground_truth = self.ground_truths[i]
            filename = self.selected_classes[i]
            # ground_truth_path = out_path + "/{}/".format(out_name)\
            #     + "{}.wav".format(filename)
            ground_truth_path = '{}/{}_{}.wav'.format(out_path, filename,
                                                      suffix)
            log(ground_truth_path)
            create_dir(ground_truth_path)
            ground_truth.export(ground_truth_path, format="wav")
        log("saved!")

    def _save_config(self, out_name, out_path):
        """
        Space-separated argument list corresponding to:
        --num_sources, --duration, --selected_classes, --wav_ids,
        --intervals, --out_path
        """
        log("Saving configs...", newline=True)
        config_path = out_path + "/" + "{}.txt".format(out_name)
        create_dir(config_path)
        content = ""

        selected_ids = [str(cid) for cid in self.selected_classes]
        wav_ids = [str(wid) for wid in self.wav_ids]
        content += (str(self.num_sources) + " ")
        content += (str(self.mixture_duration) + " ")
        content += (" ".join(selected_ids) + " ")
        content += (" ".join(wav_ids) + " ")
        # content += (" ".join(self.intervals) + " ")
        content += (out_path + "\n")

        with open(config_path, 'w') as file_path:
            file_path.write(content)

        log("saved!")

    # FIXME: currently only search for non-empty folders
    def _select_classes(self, all_classes):
        selected = []
        while len(selected) < self.num_sources:
            i = random.randint(0, len(all_classes) - 1)
            path = DATA_DIRECTORY + all_classes[i]
            if (i not in selected) and (not is_empty(path)):
                selected.append(i)
        self.selected_classes = selected

    def _generate_intervals(self):
        """Get the intervals where each wav appear in the combination
        First get the duration of each source, then randomize the interval
        where each source appear in the mixture.
        """

        n_frames = self.mixture_duration * SEC_LEN
        durations = self.source_durations

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

    # FIXME: function has confusing structure
    def _get_wav_files(self, all_classes):
        # if specific sound clip IDs are not given
        if self.wav_ids is None:
            self.wav_ids = []
            for class_id in self.selected_classes:
                class_name = all_classes[int(class_id)]
                while True:
                    try:
                        i = random.randint(0, CLASS_MAX_SIZE - 1)
                        wav_path = get_pathname(class_name, i)
                        wav_file = AudioSegment.from_wav(wav_path)
                        break  # found clip, go to next category
                    except OSError:
                        pass
                self.wav_ids.append(i)
                self.wav_files.append(wav_file)

        # sound clip IDs are given
        else:
            for i, class_id in enumerate(self.selected_classes):
                class_name = all_classes[int(class_id)]
                wav_id = self.wav_ids[i]
                wav_path = get_pathname(class_name, wav_id)
                log(wav_path)
                wav_file = AudioSegment.from_wav(wav_path)
                self.wav_files.append(wav_file)

    def _create_ground_truths(self):
        n_frames = self.mixture_duration * SEC_LEN
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


def get_classes(path):
    """gather all available classes"""
    all_classes = [name for name in os.listdir(path)]
    return all_classes


def main():
    """main"""
    args = get_arguments()

    # all_classes = get_classes(DATA_DIRECTORY)
    all_classes = []

    mixer = Mixer(args)
    mixer.mix(all_classes)
    mixer.export_results(args.out_path, all_classes, args.id)


if __name__ == "__main__":
    main()
