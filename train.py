"""
This script trains the models for monaural source separation.

Contributors:
    Tianjun Ma (tianjunm@cs.cmu.edu)

"""

import argparse
import csv
import copy
import logging
import time
import os
import torch
import torch.nn as nn
import torch.optim

import models.setup
import datasets.setup
import loss_functions.setup


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)


class Experiment():
    def __init__(self, data, snapshot_path, loss_function):
        self._train_data, self._val_data = data
        self._loss_fn = loss_function

    def run(self): 
        start_epoch, min_loss = self._load_checkpoint()

        for epoch in range(start_epoch, min_loss):
            self._step()

        self._post_result(epoch, min_loss)

    def _step(self):

        train_loss = self._train_step(epoch)
        val_loss = self._val_step()

        if val_loss < min_loss:
            es_counter = 0

            # save best loss of current unfinished trial
            min_loss = val_loss
            logging.info("saving best model...")

            self._save_model(epoch, train_loss, val_loss,
                                best_model=self._model)

            logging.info("finished saving current best model")
        else:
            es_counter += 1

            if es_counter >= const.TOLERANCE:
                es_done = True
                logging.info(
                    "val loss does not decrease for more than "
                    "%d epochs, training will be terminated",
                    const.TOLERANCE)

        logging.info(
            "[%2d/%2d] %s: epoch %3d, [%5d/%5d] "
            "train loss: %.2f, val loss: %.2f, duration: %.0fs "
            "(early stopping: %d/%d)",
            self._config['id'] + 1,
            self._nconf,
            self._etitle,
            epoch + 1,
            self._ds_size,
            self._ds_size,
            train_loss,
            val_loss,
            time.time() - end,
            es_counter,
            const.TOLERANCE)

        # self._log_tb(losses, epoch, trial_id)
        if (epoch + 1) % const.CHECKPOINT_FREQ == 0:
            logging.info("saving snapshot...")
            self._save_model(
                epoch,
                train_loss,
                val_loss,
                min_loss=min_loss)
            logging.info("finished saving snapshots")

    def _train_step(self):
        self._model.train()

        for batch_idx, batch in enumerate(self._train_data):
            pass

        return loss.item()
    
    def _val_step(self):
        running_loss = 0.0
        iters = 0

        self._model.eval()

        with torch.no_grad():
            for batch in self._val_data:
                iters += 1
                loss = self._loss_fn(batch)
                running_loss += loss.item()

        return running_loss / iters

    def _save_snapshot(self):
        torch.save()
    
    def _load_snapshot(self):
        snapshot = torch.load()

        start_epoch = snapshot['epoch'] + 1 
        min_loss = snapshot['min_loss']
        return start_epoch, min_loss
    
    def _post_result(self, epoch, min_loss):
        csv_path = self._log_path

    def _sync_tensorboard(self):
        legend = ""


def get_arguments():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--config_path', type=str)
    parser.add_argument('--early_stopping_limit', type=int, default=0)
    # parser.add_argument('--snapshot_path', type=str)

    return parser.parse_args()


def main():
    args = get_arguments()

    train_dataloader = datasets.setup.prepare_dataset(args.config_path,
                                                      'train')

    val_dataloader = datasets.setup.prepare_dataset(args.config_path,
                                                    'val')

    model = models.setup.prepare_model(args.config_path,
                                       train_dataloader.input_shape)

    loss_fn = loss_functions.setup.prepare_loss_fn(args.config_path)

    snapshot_path = utils.io.resolve_path('snapshot', args)

    # tensorboard_path = utils.experiment.get_path('tensorboard', args)

    experiment = Experiment(model=model,
                            datasets=datasets,
                            loss_fn=loss_fn,
                            snapshot_path=snapshot_path,
                            early_stopping_limit=args.early_stopping_limit)

    experiment.run()


if __name__ == '__main__':
    main()