"""Mixer for dataset generation"""
import logging
import argparse
import random
import os
import errno
import csv
from pydub import AudioSegment


DATA_DIRECTORY = "./raw/"
NUM_CLASSES = 100
CLASS_MAX_SIZE = 100
SEC_LEN = 1000


def is_empty(path):
    """is_empty"""
    return ([f for f in os.listdir(path)
             if not (f.startswith('.') or ('csv' in f))] == [])


def select_classes(all_sources, num_sources):
    """a """
    selected = []
    while len(selected) < num_sources:
        try:
            i = random.randint(0, len(all_sources))
            path = DATA_DIRECTORY + all_sources[i]
            if (i not in selected) and (not is_empty(path)):
                selected.append(i)
        except Exception as err:
            logging.exception(err)
    return selected


def select(upperbound, num_samples):
    """select num_samples random numbers within upperbound"""
    if num_samples > upperbound:
        raise Exception("uppebound is smaller than number of samples!")

    return random.sample(range(upperbound), num_samples)


# needs
def get_arguments():
    """get arguments"""
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation')
    parser.add_argument('--num_sources', type=int, default=3, required=True)
    parser.add_argument('--duration', type=int, required=True)

    # optional arguments
    parser.add_argument('--selected_classes', nargs='*')
    parser.add_argument('--wav_paths', nargs='*')
    parser.add_argument('--intervals', nargs='*')
    parser.add_argument('--out_path', type=str, default='./test')

    return parser.parse_args()


def overlay_clips(clips):
    """"""
    aggregate = None
    for i in range(len(clips)):
        if i == 0:
            aggregate = clips[i]
        else:
            aggregate = aggregate.overlay(clips[i])

    aggregate = aggregate.set_channels(1)
    return aggregate


def get_intervals(num_sources, duration):
    """Get the intervals where each wav appear in the combination
    Randomly creates intervals for n sound source.
    """
    # randomize the starting times for each source
    starts = select(duration, num_sources)
    starts.sort()

    upperbounds = [random.randint(1, 10) for i in range(num_sources)]

    intervals = []
    for i, start in enumerate(starts):
        end = min(start + upperbounds[i], duration)
        intervals.append((start, end))

    return intervals


def get_wav_files(all_classes, selected_classes):
    wav_files = []
    for (class_id, wav_id) in selected_classes:
        class_name = all_classes[class_id]

        # if pre-selected df
        if wav_id != -1:
            wav_path = DATA_DIRECTORY + class_name + "/{}.wav".format(wav_id)
        # otherwise, randomly choose one from the given category
        while True:
            try:
                i = random.randint(0, CLASS_MAX_SIZE - 1)
                wav_path = DATA_DIRECTORY + class_name + "/{}.wav".format(i)
                wav_file = AudioSegment.from_wav(wav_path)
                print(wav_path)
                break
            except(Exception) as err:
                print(err)
                continue

        wav_files.append(wav_file)

    return wav_files


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
        wav_files:
            specific sound clip within selected classes
        intervals:
            start + end times for each sound source to appear in the mixture
        ground_truths:
            list of individual clips of each source
        aggregate:
            the combined clip
    """

    def __init__(self, args):
        self.num_sources = args.num_sources
        self.mixture_duration = args.duration
        self.selected_classes = args.selected_classes
        self.wav_paths = args.wav_paths
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
        pass

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
                class_id = self.selected_classes[i]
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

    # TODO: implement me!
    def _save_config(self, out_name, out_path):
        """
        Space-separated argument list corresponding to:
        --num_sources, --duration, --selected_classes, --wav_files,
        --intervals, --out_path
        """
        print("Saving configs...", end="")
        config_path = out_path + "/" + "{}.txt".format(out_name)
        create_dir(config_path)
        content = ""

        content += (str(self.num_sources) + " ")
        content += (str(self.mixture_duration) + " ")
        content += (" ".join(self.selected_classes) + " ")
        content += (" ".join(self.wav_paths) + " ")
        content += (" ".join(self.intervals) + " ")
        content += (out_path)
        print("saved!")

    # FIXME: currently only search for non-empty folders
    def _select_classes(self, all_classes):
        selected = []
        while len(selected) < self.num_sources:
            i = random.randint(0, len(all_classes))
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

    def _get_wav_files(self, all_classes):
        # if specific sound clips are given
        if self.wav_paths is not None:
            for wav_path in self.wav_paths:
                wav_file = AudioSegment.from_wav(wav_path)
                self.wav_files.append(wav_file)

        # otherwise, randomly select available clips
        else:
            for class_id in self.selected_classes:
                class_name = all_classes[class_id]
                while True:
                    try:
                        i = random.randint(0, CLASS_MAX_SIZE - 1)
                        wav_path = DATA_DIRECTORY + \
                            class_name + "/{}.wav".format(i)
                        wav_file = AudioSegment.from_wav(wav_path)
                        break  # found clip, go to next category
                    except():
                        continue
                self.wav_paths.append(wav_path)
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
        for i in range(len(self.ground_truths)):
            if i == 0:
                aggregate = self.ground_truths[i]
            else:
                aggregate = aggregate.overlay(self.ground_truths[i])

        aggregate = aggregate.set_channels(1)

        self.aggregate = aggregate


def mix(all_classes, selected_classes, duration):

    num_sources = len(selected_classes)

    intervals = get_intervals(num_sources, duration)
    wav_files = get_wav_files(all_classes, selected_classes)
    ground_truths = get_ground_truths(wav_files, intervals, duration)
    aggregate = overlay_clips(ground_truths)

    return (intervals, ground_truths, aggregate)


def get_ground_truths(wav_files, intervals, duration):
    ground_truths = []

    for i, interval in enumerate(intervals):
        wav = wav_files[i]
        start, end = interval
        pad_before = AudioSegment.silent(start * 1000)
        pad_after = AudioSegment.silent((duration - end) * 1000)
        dur = end - start
        wav = w[:dur * 1000]

        ground_truth = pad_before + w + pad_after
        # set to mono
        ground_truth = ground_truth.set_channels(1)
        ground_truths.append(ground_truth)

    return ground_truths


def create_dir(outdir):
    if not os.path.exists(os.path.dirname(outdir)):
        try:
            os.makedirs(os.path.dirname(outdir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def export(all_sources, selected_classes, info, out_path):
    intervals, ground_truths, aggregate = info
    print(selected_classes)
    out_name = "".join([str(i) for i in selected_classes])

    # metadata
    meta_path = out_path + "/" + "{}.csv".format(out_name)
    create_dir(meta_path)
    with open(meta_path, 'w+') as meta:
        meta_fields = ["start time", "end time", "class_name"]
        writer = csv.DictWriter(meta, fieldnames=meta_fields)
        writer.writeheader()
        for i, interval in enumerate(intervals):
            start, end = interval
            class_id = selected_classes[i]
            writer.writerow({
                "start time": start,
                "end time": end,
                "class_name": all_sources[class_id]
            })

    # combined clip
    clip_path = out_path + "/" + "{}.wav".format(out_name)
    create_dir(clip_path)
    aggregate.export(clip_path, format="wav")

    # ground truths
    for i, ground_truth in enumerate(ground_truths):
        filename = selected_classes[i]
        ground_truth_path = out_path + "/{}/".format(out_name) + \
            "{}.wav".format(filename)
        create_dir(ground_truth_path)
        ground_truth.export(ground_truth_path, format="wav")


def get_classes(path):
    """gather all available classes"""
    all_classes = [name for name in os.listdir(path)]
    return all_classes


def main():
    """main"""
    args = get_arguments()

    all_classes = get_classes(DATA_DIRECTORY)

    mixer = Mixer(args)

    # info = mix(all_sources, selected_classes, args.duration)
    mixer.mix(all_classes)
    mixer.export_results(args.out_path, all_classes)
    # export(all_sources, selected_classes, info, args.out_path)


if __name__ == "__main__":
    main()
