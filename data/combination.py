import sys
import argparse
from pydub import AudioSegment


DATA_DIRECTORY = "./raw_clips"

# needs 
def get_arguments():
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation')
    # parser.add_argument('--num_classes', type=int, default=BATCH_SIZE,
    #                     help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    # parser.add_argument('--comb_out_path')

    # parser.add_argument('--')

def loader():
    return


def main():
    # get_arguments()
    sound1 = AudioSegment.from_file("raw_clips/bird/0.wav")
    sound2 = AudioSegment.from_file("raw_clips/gun/0.wav")

    combined = sound1.overlay(sound2)

    combined.export("./combined.wav", format='wav')

if __name__ == "__main__":
    main()
