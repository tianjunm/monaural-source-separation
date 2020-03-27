"""Custom constants"""


import os
import numpy as np


ROOT = "/work/tianjunm/monaural-source-separation"
TRAIN_SCALE = 10000
# DATASET_PATH_PREFIX = os.path.join(ROOT_DIR, "dataset/processed/mixer/")
# RESULT_PATH_PREFIX = os.path.join(ROOT, "experiment/snapshots")
DATASET_PATH = os.path.join(ROOT, 'dataset/processed/datagen/')

SNAPSHOTS_PATH = os.path.join(ROOT, 'experiments/snapshots')

# TBLOG_PATH = os.path.join(ROOT_DIR, "home/ubuntu/experiment_logs/tb_logs")
RESULT_FILENAME = 'results_new.csv'

VISUALIZATION_PATH = os.path.join(ROOT_DIR, 'visualizations')
VDATA_PATH = os.path.join(VISUALIZATION_PATH, 'data')
VRESULT_PATH = os.path.join(VISUALIZATION_PATH, 'results')

# default parameters
NUM_SOURCES = 2
MAX_EPOCHS = 500
LOG_FREQ = 10
CHECKPOINT_FREQ = 1

MAX_LOSS = np.inf
NUM_CONFIGS = 10
NUM_TRIALS = 1
TOLERANCE = 15 # for early stopping

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
    'c_out',
    'd_out',
    'ks2',
    'res_size',
    'best_val_loss',
    'experiment_title']

# dataset related
NFFT = 256
N_FREQ = NFFT // 2 + 1

INTERCLASS = 0
INTRACLASS = 1
HYBRID = 2