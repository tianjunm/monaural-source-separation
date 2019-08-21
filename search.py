"""Script for performing random search"""


import argparse
import csv
import logging
import os
import random
import time
import torch
# from tensorboardX import SummaryWriter as tensorboard_writer
import train
import utils
import constants as const


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def get_argument():
    """context for searching"""
    parser = argparse.ArgumentParser(description='Setting up an experiment')

    # dataset specification
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--num_sources', type=int)
    parser.add_argument('--category_range', type=int)

    # checkpointing, allowing two options:
    # 1. continue from the last seen epoch of the last seen config
    # 2. startover from the last seen config
    parser.add_argument("--load_checkpoint", type=bool)
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--cont', type=bool, default=False)

    # single configuration
    # 1. repeat a particular config
    # 2. custom config
    parser.add_argument('--config_path', type=str)

    # others
    parser.add_argument('--num_configs', type=int, default=const.NUM_CONFIGS)
    parser.add_argument('--add_csv_header', type=bool, default=False)
    parser.add_argument(
        '--result_filename',
        type=str,
        default=const.RESULT_FILENAME)

    return parser.parse_args()


def get_device(gpu_id):
    """get cuda device object"""
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    return torch.device('cpu')


def get_experiment_path(setup):
    """Returns the root path for saving the result of given experiment."""

    ds_type = setup['dataset_type']
    nsrc = setup['num_sources']
    ncat = setup['category_range']
    metric = setup['metric']
    m_type = setup['model_type']
    eid = setup['experiment_id']

    # e.g. t1-2s-5c_euclidean_DETF_20190101
    return (
        f"t{ds_type}-{nsrc}s-{ncat}c_"
        f"{metric}_{m_type}_{eid}"
        )


def write_header():
    """Creates header in the result csv file."""
    csv_path = os.path.join(const.RESULT_PATH_PREFIX, const.RESULT_FILENAME)
    utils.make_dir(csv_path)
    with open(csv_path, "a+") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=const.FIELD_NAMES)
        csv_writer.writeheader()


def fetch_progress(experiment_path):
    """Continue from where the experiment was left off."""

    progress_log = os.path.abspath(os.path.join(
        const.RESULT_PATH_PREFIX,
        experiment_path,
        'progress.tar'))

    start_config_id = torch.load(progress_log)['curr_id']

    return start_config_id


def get_param_grid():
    """All possible values each parameter can choose from."""

    grid = {
        'max_epochs': [300, 400, 500],
        'lrs': [0.001, 0.0005, 0.003],
        'optims': ['Adam'],
        'loss_fns': ['Greedy'],
        'batch_sizes': [4, 16, 64],
        'dropouts': [0.1, 0.3, 0.5],
        'momentums': [0.0, 0.5, 0.9],
        'beta1s': [0.9, 0.95, 0.99],
        'beta2s': [0.98, 0.99, 0.999],
        'epsilons': [1e-8, 1e-9],
        'hidden_sizes': [32, 128, 512],
        'in_chans': [4, 16],
        'chans': [4, 16],
        'Ns': [4, 5, 6],
        'hs': [4, 8],
        'd_models': [128, 256],
        'd_ffs': [128, 256],
        'gammas': [0.01, 0.05, 0.1],
        }

    return grid


def select_config(config_id, param_grid):
    """Randomly select a configuration from parameter grid"""
    config = {}

    config['id'] = config_id
    for param in param_grid.keys():
        config[param[:-1]] = random.choice(param_grid[param])

    return config


def save_config(experiment_path, config_id, config):
    """Saves the configuration"""
    config_path = os.path.abspath(os.path.join(
        const.RESULT_PATH_PREFIX,
        experiment_path,
        str(config_id),
        "config.tar"))

    progress_log = os.path.abspath(os.path.join(
        const.RESULT_PATH_PREFIX,
        experiment_path,
        'progress.tar'))

    utils.make_dir(config_path)
    utils.make_dir(progress_log)

    # save config information
    torch.save(config, config_path)

    # keep track of overall progress
    torch.save({"curr_id": config_id}, progress_log)


def load_config(config_path):
    """Loads the configuration"""
    config = torch.load(config_path)
    return config


def run_config(
        experiment_path,
        experiment_setup,
        config,
        start_trial=0,
        load_checkpoint=False):
    """Train with the given configuration."""

    trainer = train.Trainer(
        setup=experiment_setup,
        config=config,
        experiment_path=experiment_path,
        load_checkpoint=load_checkpoint)

    # run the experiment [num_trials] times
    for trial_id in range(start_trial, const.NUM_TRIALS):
        logging.info(
            "running configuration #%d [%d/%d]...",
            config['id'],
            trial_id + 1,
            const.NUM_TRIALS)

        # picking up unfinished trial
        # if start_trial != 0 and start_trial == trial_id:
        trainer.fit()
        # else:
        #     # trainer.run(trial_id, record)
        #     trainer.run(trial_id)

        logging.info("trial %d finished!", trial_id + 1)

    time.sleep(5)


def run_configs(
        experiment_path,
        experiment_setup,
        param_grid,
        start_config_id=0,
        has_checkpoint=False):
    """Run the random search experiment and log the results."""

    num_configs = experiment_setup['num_configs']
    assert start_config_id < num_configs


    logging.debug("has checkpoint: %s", has_checkpoint)
    if has_checkpoint:
        logging.debug("has checkpoint")
        config = load_config(os.path.join(
            const.RESULT_PATH_PREFIX,
            experiment_path,
            str(start_config_id),
            "config.tar"))

        run_config(
            experiment_path,
            experiment_setup,
            config,
            load_checkpoint=True)

        # start fresh with the next config
        start_config_id += 1


    for config_id in range(start_config_id, num_configs):
        # randomly select parameters for config
        config = select_config(config_id, param_grid)

        logging.info("configuration #%s:", config_id)
        logging.info(config)

        save_config(experiment_path, config_id, config)

        run_config(experiment_path, experiment_setup, config)


def main():
    args = get_argument()

    # unique id for each experiment assigned according to experiment init time
    if args.experiment_id is None:
        experiment_id = time.strftime("%y%m%d", time.gmtime())
    else:
        experiment_id = args.experiment_id

    experiment_setup = {
        'device': get_device(args.gpu_id),
        'dataset_type': args.dataset_type,
        'num_sources': args.num_sources,
        'category_range': args.category_range,
        'metric': args.metric,
        'model_type': args.model_type,
        'experiment_id': experiment_id,
        'num_configs': args.num_configs
        }

    experiment_path = get_experiment_path(experiment_setup)
    logging.info("GPU: %s, experiment: %s", experiment_setup['device'],
                 experiment_path)

    # parameter grid for random search
    param_grid = get_param_grid()

    # running a specific configuration alone
    if args.config_path:
        logging.info("running the following config:")
        config = load_config(args.config_path)
        logging.info(config)

        run_config(experiment_path, experiment_setup, config)

    # loading from checkpoint to finish the rest of [num_configs]
    elif args.load_checkpoint:
        logging.info("resuming from interrupted experiment...")

        # the unfinished config if it exists
        start_config_id = fetch_progress(experiment_path)
        logging.info("found partial results")

        run_configs(
            experiment_path,
            experiment_setup,
            param_grid,
            start_config_id,
            has_checkpoint=args.cont)

    # no checkpoint, start fresh
    else:
        logging.info("initiating new experiment...")

        if args.add_csv_header:
            write_header()

        run_configs(experiment_path, experiment_setup, param_grid)

    logging.info("experiment done!")


if __name__ == '__main__':
    main()
