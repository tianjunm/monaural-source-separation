"""Mixer for dataset generation"""
# import logging
import argparse
import random
import os
import errno
import csv
from pydub import AudioSegment


DATA_DIRECTORY = "./raw/"
NUM_CLASSES = 100
# FIXME: current dataset is a mini version
CLASS_MAX_SIZE = 17
SEC_LEN = 1000


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
    parser.add_argument('--num_sources', type=int, default=3, required=True)
    parser.add_argument('--duration', type=int, required=True)

    # optional arguments
    parser.add_argument('--selected_classes', nargs='*')
    parser.add_argument('--wav_ids', nargs='*')
    parser.add_argument('--intervals', nargs='*')
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
        self.intervals = args.intervals
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

        if not self.intervals:
            self._generate_intervals()

        # automatically handles empty self.wav_paths
        self._get_wav_files(all_classes)

        self._create_ground_truths()
        self._overlay_clips()

    # TODO: allow resetting different fields
    def reset(self, criteria):
        """reset the values for given field and randomize"""

    def export_results(self, out_path, all_classes):
        """export all related results"""
        # print(self.selected_classes)
        out_name = "-".join([str(i) for i in self.selected_classes])
        self._save_metadata(out_name, out_path, all_classes)
        self._save_media(out_name, out_path)
        self._save_config(out_name, out_path)

    # PRIVATE METHODS

    def _save_metadata(self, out_name, out_path, all_classes):
        meta_path = out_path + "/" + "{}.csv".format(out_name)
        create_dir(meta_path)

        print("Saving metadata...", end="")
        with open(meta_path, 'w+') as meta:
            meta_fields = ["start time", "end time", "class_name"]
            writer = csv.DictWriter(meta, fieldnames=meta_fields)
            writer.writeheader()
            for i, interval in enumerate(self.intervals):
                start, end = interval
                class_id = int(self.selected_classes[i])
                writer.writerow({
                    "start time": start,
                    "end time": end,
                    "class_name": all_classes[class_id]
                })
        print("saved!")

    def _save_media(self, out_name, out_path):
        # combined clip
        clip_path = out_path + "/" + "{}.wav".format(out_name)
        create_dir(clip_path)

        print("Saving combined clip...", end="")
        self.aggregate.export(clip_path, format="wav")
        print("saved!")

        # ground truths
        print("Saving ground truths...", end="")
        for i in range(len(self.ground_truths)):
            ground_truth = self.ground_truths[i]
            filename = self.selected_classes[i]
            ground_truth_path = out_path + "/{}/".format(out_name)\
                + "{}.wav".format(filename)
            create_dir(ground_truth_path)
            ground_truth.export(ground_truth_path, format="wav")
        print("saved!")

    def _save_config(self, out_name, out_path):
        """
        Space-separated argument list corresponding to:
        --num_sources, --duration, --selected_classes, --wav_ids,
        --intervals, --out_path
        """
        print("Saving configs...", end="")
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

        print("saved!")

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

        Randomly creates intervals for n sound source.

        """

        # randomize the starting times for each source
        starts = select(self.mixture_duration, self.num_sources)
        starts.sort()

        upperbounds = [random.randint(1, 10) for i in range(self.num_sources)]

        intervals = []
        for i, start in enumerate(starts):
            end = min(start + upperbounds[i], self.mixture_duration)
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
                print(wav_path)
                wav_file = AudioSegment.from_wav(wav_path)
                self.wav_files.append(wav_file)

    def _create_ground_truths(self):
        ground_truths = []

        for i, interval in enumerate(self.intervals):
            wav = self.wav_files[i]
            start, end = interval
            pad_before = AudioSegment.silent(start * SEC_LEN)
            pad_after = AudioSegment.silent((self.mixture_duration -
                                             end) * SEC_LEN)
            dur = end - start
            wav = wav[:dur * SEC_LEN]

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

    all_classes = get_classes(DATA_DIRECTORY)

    mixer = Mixer(args)
    mixer.mix(all_classes)
    mixer.export_results(args.out_path, all_classes)


if __name__ == "__main__":
    main()
