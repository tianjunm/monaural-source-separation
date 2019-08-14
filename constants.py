"""Custom constants"""


import os
import numpy as np


ROOT_DIR = "/home/ubuntu/"
TRAIN_SCALE = 1e4
DATASET_PATH_PREFIX = os.path.join(ROOT_DIR, "datasets/processed/mixer/")
RESULT_PATH_PREFIX = os.path.join(ROOT_DIR, "/home/ubuntu/experiment_logs/results")
DATASET_PATH = '/home/ubuntu/datasets/processed/datagen/'
TBLOG_PATH = os.path.join(ROOT_DIR, "/home/ubuntu/experiment_logs/tb_logs")
RESULT_FILENAME = 'results_new.csv'

# default parameters
NUM_SOURCES = 2
MAX_EPOCHS = 500
LOG_FREQ = 10
CHECKPOINT_FREQ = 1

MAX_LOSS = np.inf
UM_CONFIGS = 10
NUM_TRIALS = 1
TOLERANCE = 1000  # for early stopping

FIELD_NAMES = [
    'id',
    'model',
    'metric',
    'loss_fn',
    'stop_epoch',
    'max_epoch',
    'lr',
    'optim',
    'batch_size',
    'dropout',
    'momentum',
    'beta1',
    'beta2',
    'epsilon',
    'hidden_size',
    'in_chan',
    'chan',
    'N',
    'h',
    'd_model',
    'd_ff',
    'gamma',
    'best_val_loss',
    'experiment_path']

# dataset related
NFFT = 256
N_FREQ = NFFT // 2 + 1
