import argparse
from pydub import AudioSegment
import random
import os
import errno
import csv


DATA_DIRECTORY = "./raw/"
NUM_CLASSES = 100
CLASS_MAX_SIZE = 100
SEC_LEN = 1000


def is_empty(path):
    return ([f for f in os.listdir(path)
             if not (f.startswith('.') or ('csv' in f))] == [])


def select_classes(all_sources, num_sources):
    selected = []
    while (len(selected) < num_sources):
        try:
            i = random.randint(0, len(all_sources))
            path = DATA_DIRECTORY + all_sources[i]
            if (i not in selected) and (not is_empty(path)):
                selected.append(i)
        except Exception as e:
            print(e)
    return selected


def select(n, num_samples):
    if (num_samples > n):
        raise Exception("n is smaller than number of samples to select!")

    return random.sample(range(n), num_samples)


def get_classes(path, args):
    """Get audio sources
    """
    all_sources = [name for name in os.listdir(path)]
    if (args.selected_classes):
        selected_classes = args.selected_classes
        return (all_sources, selected_classes)

    num_sources = args.num_sources

    # selected_classes = select(len(all_sources), num_sources)
    selected_classes = select_classes(all_sources, num_sources)
    print(selected_classes)
    for c in selected_classes:
        print(all_sources[c])
    return (all_sources, selected_classes)


# needs
def get_arguments():
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation')
    parser.add_argument('--num_class', type=int, default=3, required=True)
    parser.add_argument('--duration', type=int, required=True)

    # optional arguments
    parser.add_argument('--selected_classes', nargs='+')
    parser.add_argument('--intervals', nargs='+')
    parser.add_argument('--out_path', type=str, default='./test')

    return parser.parse_args()


def overlay_clips(clips):
    aggregate = None
    for i in range(len(clips)):
        if (i == 0):
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
        if (wav_id != -1):
            wav_path = DATA_DIRECTORY + class_name + "/{}.wav".format(wav_id)
        # otherwise, randomly choose one from the given category
        while True:
            try:
                i = random.randint(0, CLASS_MAX_SIZE - 1)
                wav_path = DATA_DIRECTORY + class_name + "/{}.wav".format(i)
                wav_file = AudioSegment.from_wav(wav_path)
                print(wav_path)
                break
            except(Exception) as e:
                print(e)
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

    def __init__(
            self,
            num_sources,
            mixture_duration,
            all_classes,
            selected_classes=None,
            wav_files=None,
            intervals=None):
        self.num_sources = num_sources
        self.mixture_duration = mixture_duration
        self.all_classes = all_classes
        self.selected_classes = selected_classes
        self.wav_files = wav_files
        self.intervals = intervals

    # public methods
    def mix(self):
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
            self._select_classes()

        if not self.intervals:
            self._generate_intervals()

        if not self.wav_files:
            self._get_wav_files()

        self._create_ground_truths()
        self._overlay_clips()

    def export_results(self, out_path):
        # intervals, ground_truths, aggregate = info
        print(self.selected_classes)
        out_name = "-".join([str(i) for i in self.selected_classes])

        # metadata
        meta_path = out_path + "/" + "{}.csv".format(out_name)
        create_dir(meta_path)
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
                    "class_name": self.all_classes[class_id]
                })

        # combined clip
        clip_path = out_path + "/" + "{}.wav".format(out_name)
        create_dir(clip_path)
        self.aggregate.export(clip_path, format="wav")

        # ground truths
        for i in range(len(self.ground_truths)):
            ground_truth = self.ground_truths[i]
            filename = self.selected_classes[i]
            ground_truth_path = out_path + "/{}/".format(out_name) + \
                "{}.wav".format(filename)
            create_dir(ground_truth_path)
            ground_truth.export(ground_truth_path, format="wav")

    # private methods
    # FIXME: currently only search for non-empty folders
    def _select_classes(self):
        selected = []
        while (len(selected) < self.num_sources):
            try:
                i = random.randint(0, len(self.all_classes))
                path = DATA_DIRECTORY + self.all_classes[i]
                if (i not in selected) and (not is_empty(path)):
                    selected.append(i)
            except Exception as e:
                print(e)
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

    def _get_wav_files(self):
        wav_files = []
        for class_id in self.selected_classes:
            class_name = self.all_classes[class_id]

            while True:
                try:
                    i = random.randint(0, CLASS_MAX_SIZE - 1)
                    wav_path = DATA_DIRECTORY + \
                        class_name + "/{}.wav".format(i)
                    wav_file = AudioSegment.from_wav(wav_path)
                    print(wav_path)
                    break
                except(Exception) as e:
                    print(e)
                    continue

            wav_files.append(wav_file)

        self.wav_files = wav_files

    def _create_ground_truths(self):
        ground_truths = []

        for i, interval in enumerate(self.intervals):
            w = self.wav_files[i]
            start, end = interval
            pad_before = AudioSegment.silent(start * SEC_LEN)
            pad_after = AudioSegment.silent((self.mixture_duration -
                                             end) * SEC_LEN)
            dur = end - start
            w = w[:dur * SEC_LEN]

            ground_truth = pad_before + w + pad_after
            # set to mono
            ground_truth = ground_truth.set_channels(1)
            ground_truths.append(ground_truth)

        self.ground_truths = ground_truths

    def _overlay_clips(self):
        aggregate = None
        for i in range(len(self.ground_truths)):
            if (i == 0):
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
        w = wav_files[i]
        start, end = interval
        pad_before = AudioSegment.silent(start * 1000)
        pad_after = AudioSegment.silent((duration - end) * 1000)
        dur = end - start
        w = w[:dur * 1000]

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
    for i in range(len(ground_truths)):
        ground_truth = ground_truths[i]
        filename = selected_classes[i]
        ground_truth_path = out_path + "/{}/".format(out_name) + \
            "{}.wav".format(filename)
        create_dir(ground_truth_path)
        ground_truth.export(ground_truth_path, format="wav")


def main():
    args = get_arguments()

    all_sources, selected_classes = get_classes(DATA_DIRECTORY, args)

    info = mix(all_sources, selected_classes, args.duration)

    export(all_sources, selected_classes, info, args.out_path)


if __name__ == "__main__":
    main()
