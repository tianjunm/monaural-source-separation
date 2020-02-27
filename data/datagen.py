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
from tqdm.auto import tqdm
from mmsdk import mmdatasdk


SAMPLE_RATE = 16000

RAW_DATA_PATH = "/work/tianjunm/large_datasets/audioset_verified/audio_csd/cut/16000"
OUTPUT_PATH = "/work/tianjunm/dataset/processed/datagen"

INTERCLASS = 0
INTRACLASS = 1
HYBRID = 2


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def get_arguments():
    """parsing arguments"""
    parser = argparse.ArgumentParser(description='Dataset creation script')

    parser.add_argument('--raw_data_path', type=str, default=RAW_DATA_PATH)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--num_sources', type=int, default=2)
    parser.add_argument('--mixing_type', type=int, default=INTERCLASS)
    parser.add_argument('--max_source_duration', type=float, required=True)
    parser.add_argument('--mixture_duration', type=int, required=True)
    parser.add_argument('--output_path', type=str, default=OUTPUT_PATH)

    return parser.parse_args()


def create_dir(outdir):
    """creates directory if it does not exist"""
    if not os.path.exists(os.path.dirname(outdir)):
        try:
            os.makedirs(os.path.dirname(outdir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def sample_categories(population, k, mixing_type=INTERCLASS):
    """Samples ncat categories from all categories presented in metadata.
    
    Args:
        population: a list of classes to sample from
        k: num_to_sample
        mixing_type: 

    Returns:
        selected_classes
    """

    if mixing_type == INTERCLASS:
        selected_classes = random.sample(population, k)

    elif mixing_type == INTRACLASS:
        selected_classes = [random.choice(population)] * k

    elif mixing_type == HYBRID:
        selected_classes = [random.choice(population) for _ in k]

    return selected_classes

def extract_metadata(metadata_path):
    """Extracts verified instances from the raw dataset

    Args:
        metadata_path: necessary path for extracting classes
    
    Returns:
        metadata: dataframe that contains 
    
    """
    raw_metadata = pd.read_csv(metadata_path)
    header = ['fname', 'label']
    metadata = raw_metadata[raw_metadata.manually_verified == 1][header]
    return metadata


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

    def insert(self, datapoint):
        """Inserts a row into the result dataset."""
        self._dataset['filename'].append(datapoint['filename'])
        self._dataset['clip_duration'].append(datapoint['duration'])
        self._dataset['mixture_placement'].append(datapoint['start_time'])
        self._dataset['category'].append(datapoint['category'])

    def export(self, dataset_type, mixing_type, num_sources, num_classes):
        """Saves the dataset to a csv file."""
        description = f"t{mixing_type}-{num_sources}s-{num_classes}c"
        output_filename = os.path.join(self._dest_root, description, f"{dataset_type}.csv")

        create_dir(output_filename)
    
        pd.DataFrame(self._dataset).to_csv(output_filename, index=False)


class DataGenerator(object):
    """A class used to generate the dataset

    Attributes:
        mixing_type: inter, intra, or hybrid
        num_sources: 
        raw_data: a dictionary containing all raw data
        selected_classes: 
        split: XXX: self defined way of splitting data into train-val-test
        max_source_duration: max duration of single source within mixture
        mixture_duration: duration of mixtures in the generated dataset
        output_path: 
        types_to_generate=['train', 'val', 'test']):

    """

    def __init__(
            self,
            mixing_type,
            num_sources,
            raw_data,
            selected_classes,
            split,
            max_source_duration,
            mixture_duration,
            output_path):

        self._mtype = mixing_type
        self._nsrc = num_sources
        self._raw = raw_data
        self._classes = selected_classes
        self._split = split
        self._sdur = max_source_duration
        self._mdur = mixture_duration

        self._dataset = Dataset(output_path)

    # public
    def generate(self):
        """Generate the dataset for the corresponding dataset_type.

        """
        for dataset_type in ['train', 'val', 'test']:
            self._gen(dataset_type)

    # private
    def _gen(self, dataset_type):
        logging.info('generating %s set...', dataset_type)

        # size of the train/val/test dataset to be generated
        num_datapoints = self._split['sizes'][dataset_type]

        for _ in tqdm(range(num_datapoints)):
            # choose the nsrc classes from selected classes
            datapoint_classes = sample_categories(
                population=self._classes,
                k=self._nsrc,
                mixing_type=self._mtype)

            # for each chosen category for this training instance,
            # choose a sound file from the sounds of a given class 
            for c in datapoint_classes:
                fid = self._choose_id(c, dataset_type)
                datapoint = self._create_datapoint(c, fid)
                self._dataset.insert(datapoint)

        self._dataset.export(
            dataset_type=dataset_type,
            mixing_type=self._mtype,
            num_sources=self._nsrc,
            num_classes=len(self._classes))

        logging.info('finished generating the %s set!', dataset_type)
        self._dataset.reset()

    def _choose_id(self, sound_class, dataset_type):
        # getting the range of allowed sound clips to choose from
        lo_frac, hi_frac = self._split['ranges'][dataset_type]

        # allowed files depending on whether we are building train/val/test
        all_ids = sorted(self._raw[sound_class].keys())

        n = len(all_ids)
        lo = int(lo_frac * n)
        hi = int(hi_frac * n)
        chosen_id = random.choice(all_ids[lo:hi])

        return chosen_id

    def _create_datapoint(self, sound_class, fid):
        datapoint = {}

        # info['filename'] = chosen.fname.to_string(index=False).strip()
        # info['category'] = chosen.label.to_string(index=False).strip()

        datapoint['filename'] = fid
        datapoint['category'] = sound_class

        # randomize the duration of the snippet from the original data
        clip_dur = random.randint(1 * SAMPLE_RATE, self._sdur * SAMPLE_RATE)

        # randomize the placement of clip in the mixture
        start_lo = 0
        start_hi = self._mdur * SAMPLE_RATE - clip_dur
        start = random.randint(start_lo, start_hi)

        datapoint['duration'] = clip_dur
        datapoint['start_time'] = start

        return datapoint 


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
            'train': (0, 0.8),
            'val': (0.8, 0.9),
            'test': (0.9, 1)
            },
        }


def load_raw_data(raw_data_path):
    """Loads all raw data to the memory

    Args:
        raw_data_path: location on disk
    
    Returns:
        raw_data: raw data in [whatever] format
    """
    # raw_data_path = '/work/tianjunm/large_datasets/audioset_verified/audio_csd/cut/16000'

    dataset_dict = {}

    for csd in os.listdir(raw_data_path):
        dataset_dict[csd] = os.path.join(raw_data_path, csd)
    
    raw_data = mmdatasdk.mmdataset(dataset_dict)
    return raw_data


def main():
    args = get_arguments()
    logging.info('preparing dataset generation...')

    raw_data = load_raw_data(args.raw_data_path)

    # select the given number of classes to be in the dataset from all available classes
    selected_classes = sample_categories(raw_data.keys(), args.num_classes)    

    # specification of how to split files within each category
    # of the raw dataset among train, val, and test sets
    split = define_dataset_split(args.num_sources)

    generator = DataGenerator(
        mixing_type=args.mixing_type,
        num_sources=args.num_sources,
        raw_data=raw_data,
        selected_classes=selected_classes,
        split=split,
        max_source_duration=args.max_source_duration,
        mixture_duration=args.mixture_duration,
        output_path=args.output_path)

    generator.generate()

    logging.info('finished generating the dataset!')


if __name__ == "__main__":
    main()

