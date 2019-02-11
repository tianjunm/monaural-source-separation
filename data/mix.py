import argparse
from pydub import AudioSegment
import random
import os
import errno
import csv


DATA_DIRECTORY = "./raw/"
NUM_CLASSES = 100
CLASS_MAX_SIZE = 100


def select(n, num_samples):
    if (num_samples > n):
        raise Exception("n is smaller than number of samples to select!")

    return random.sample(range(n), num_samples)


def get_classes(path, num_sources):
    all_sources = [name for name in os.listdir(path)]
    # selected_classes = select(len(all_sources), num_sources)
    selected_classes = [0, 20, 35]
    for c in selected_classes:
        print (all_sources[c])
    print (selected_classes)
    return (all_sources, selected_classes)


# needs
def get_arguments():
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation')
    parser.add_argument('--num_class', type=int, default=3, required=True)
    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--out_path', type=str)

    return parser.parse_args()


def overlay_clips(clips):
    aggregate = None
    for i in range(len(clips)):
        if (i == 0):
            aggregate = clips[i]
        else:
            aggregate = aggregate.overlay(clips[i])

    return aggregate


def mix(all_classes, selected_classes, duration):
    # background = AudioSegment.silent(duration=duration)
    num_sources = len(selected_classes)

    # randomize the starting times for each source
    starts = select(duration, num_sources)
    starts.sort()

    upperbounds = [random.randint(1, 10) for i in range(num_sources)]

    intervals = []
    for i, start in enumerate(starts):
        end = min(start + upperbounds[i], duration)
        intervals.append((start, end))

    wav_paths = []
    for class_id in selected_classes:
        class_name = all_classes[class_id]
        for i in range(CLASS_MAX_SIZE):
            try:
                wav_path = DATA_DIRECTORY + class_name + "/{}.wav".format(i)
                break
            except(Exception):
                continue

        wav_paths.append(wav_path)

    print (upperbounds)
    print (intervals)

    ground_truths = []
    for i, interval in enumerate(intervals):
        w = AudioSegment.from_wav(wav_paths[i])
        start, end = interval
        pad_before = AudioSegment.silent(start * 1000)
        pad_after = AudioSegment.silent((duration - end) * 1000)
        dur = end - start
        w = w[:dur * 1000]

        ground_truth = pad_before + w + pad_after
        ground_truths.append(ground_truth)

    aggregate = overlay_clips(ground_truths)
    return (intervals, ground_truths, aggregate)


def create_dir(outdir):
    if not os.path.exists(os.path.dirname(outdir)):
        try:
            os.makedirs(os.path.dirname(outdir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def export(all_sources, selected_classes, info, out_path):
    intervals, ground_truths, aggregate = info
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

    all_sources, selected_classes = get_classes(DATA_DIRECTORY,
                                                args.num_class)

    info = mix(all_sources, selected_classes, args.duration)

    export(all_sources, selected_classes, info, args.out_path)


if __name__ == "__main__":
    main()
