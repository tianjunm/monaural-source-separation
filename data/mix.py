import argparse
from pydub import AudioSegment


DATA_DIRECTORY = "./raw_clips"

V_N = 1
MV_N = 2
MV_MN = 3


# needs
def get_arguments():
    parser = argparse.ArgumentParser(description='Data generator for audio \
        separation')
    parser.add_argument('--s_level', type=int, default=1, required=True)
    parser.add_argument('--v_file', nargs='+', required=True)
    parser.add_argument('--n_file', nargs='+', required=True)
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--out_dir', type=str)

    return parser.parse_args()


def overlay_clips(args):
    if (args.s_level == V_N):
        v1 = AudioSegment.from_file(args.v_file[0])
        n1 = AudioSegment.from_file(args.n_file[0])

    combined = v1.overlay(n1)
    combined.export("{}/{}.wavk".format(args.out_dir, args.id), format='wav')


def main():
    args = get_arguments()
    overlay_clips(args)


if __name__ == "__main__":
    main()
