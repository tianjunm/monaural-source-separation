import torch
import argparse
from model import Net1, loss_fn
import torch.optim as optim


BATCH_SIZE = 1
LEARNING_RATE = 1e-3
OUTPUT_DIR = './'


def get_argument():
    parser = argparse.ArgumentParser(description='Separator network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--ouput_dir', type=str, default=OUTPUT_DIR)
    # parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    return parser


def main():
    args = get_argument()

    # load data
    input_dim = 0

    # create network
    net = Net1(
        input_dim=input_dim,
        batch_size=args.batch_size,
        num_sources=args.num_sources)

    criterion = loss_fn()
    optimizer = optim.SGD(net.parameters(), lr=args.learni)


if __name__ == '__main__':
    main()
