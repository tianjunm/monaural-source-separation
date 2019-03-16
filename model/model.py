import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
from tensorboardX import SummaryWriter


NUM_LAYERS = 1
BATCH_SIZE = 32

# FIXME: examples rn!!
INPUT_DIM = 3
NUM_SOURCES = 2
SEQ_LEN = 10


# TODO: outpus dimension
# FIXME: currently a greedy approach
def loss_fn(outputs, labels):
    """Closest distance loss function

    Calulates the distance from output to each of the ground truths, and
    the loss is the min of all distances

    Args:
        outputs: (w*h) * n * batch_size, model outputs
        labels: (w*h) * n, ground truths
    """
    batch_size, length, n = outputs.size()  # length = w * h
    loss = 0

    # TODO: reshape into 1-d tensors
    # ground_truths = labels.repeat(batch_size, 1, 1)
    for i in range(n):
        predicted = outputs[:, i].unsqueeze(1)
        dists = (predicted - labels).norm(dim=2)
        loss += dists.min()
    return loss


class Net1(nn.Module):
    """LSTM-only network

    Args:
        input_dim: the length of a vertical slice of the spectrogram
        num_sources: number of individual sources in the combined clip.
                     also the hidden dimension for the LSTM network
    """

    def __init__(self, input_dim, batch_size, num_sources, num_layers=1):
        super(Net1, self).__init__()
        self.num_sources = num_sources
        self.batch_size = batch_size
        self.num_layers = NUM_LAYERS
        self.lstm = nn.LSTM(input_dim, num_sources)
        # self.hidden = self.init_hidden()
        # self.num_directions = 1
        # self.mapping = nn.Linear(

    # TODO: input dimension! what should the non-linearity be?
    def forward(self, x):
        x = x.view(SEQ_LEN, 1, -1)
        lstm_out, _ = self.lstm(x)
        # predicted = F.log_softmax(lstm_out)
        return lstm_out


dummy = torch.randn(INPUT_DIM, SEQ_LEN)
with SummaryWriter(comment='BaseLSTM') as w:
    w.add_graph(Net1(INPUT_DIM, BATCH_SIZE, NUM_SOURCES), dummy, True)


# # TODO: exploit other structures
# class Net2(nn.Module):
#     """Conv + LSTM
#     """

#     def __init__(self):
#         super(Net2, self).__init__()
#         self.conv = None
#         self.lstm = None

#     def forward(self, x):
#         batch_size, nrows, ncols = x.size()
#         x = x.view(-1, nrows * ncols)
#         x = F.relu(self.conv(x))
#         x = F.relu(self.lstm(x))
#         return x
