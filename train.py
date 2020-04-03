"""
This script trains the models for monaural source separation.

Contributors:
    Tianjun Ma (tianjunm@cs.cmu.edu)

"""

import argparse
import csv
# import copy
import json
import logging
import time
import os
import torch
import torch.nn as nn
import torch.optim

import models.setup
import datasets.setup
import loss_functions.setup
import utils.io
import utils.hardware


PATH_TO_RESULTS = '/results/tianjunm/monaural-source-separation/experiments'


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)


class Experiment():
    def __init__(self,
                 model,
                 model_name,
                 train_data,
                 val_data,
                 loss_fn,
                 optim,
                 max_epochs,
                 save_path_prefix,
                 config_id,
                 load_path=None):

        self.identifier = time.strftime("%y%m%d", time.gmtime())
        self.device = utils.hardware.get_device()

        self._model = model
        self._model_name = model_name
        self._train_data = train_data
        self._val_data = val_data
        self._loss_fn = loss_fn
        self._optim = optim
        self._max_epochs = max_epochs
        self._save_path_prefix = save_path_prefix
        self._config_id = config_id
        self._load_path = load_path

    def run(self, record_path, checkpoint_freq=50, early_stopping_limit=None):
        logging.info(f'starts experiment: {self._save_path_prefix}')
        start_epoch, min_loss = self._load_snapshot()

        counter = 0
        for epoch in range(start_epoch, self._max_epochs):
            if (early_stopping_limit is not None and
               counter >= early_stopping_limit):
                break

            train_loss = self._train_step()
            val_loss = self._val_step()

            if min_loss is None or val_loss < min_loss:
                counter = 0
                min_loss = val_loss
                self._save_snapshot(epoch, val_loss, 'best')
            else:
                counter += 1

            if (epoch + 1) % checkpoint_freq == 0:
                self._save_snapshot(epoch, val_loss, 'checkpoint')

            logging.info('epoch %3d, train loss: %.2f, val loss: %.2f',
                         epoch, train_loss, val_loss)

        self._save_record(record_path, epoch, min_loss)

    def _train_step(self):
        self._model.train()
        running_loss = 0.0
        iters = 0

        for i, batch in enumerate(self._train_data):
            self._optim.zero_grad()

            model_input = batch['model_input'].to(self.device)
            ground_truths = batch['ground_truths'].to(self.device)
            model_output = self._model(model_input)
            loss = self._loss_fn(model_input, model_output, ground_truths)
            loss.backward()
            self._optim.step()

            running_loss += loss.item()
            iters += 1

        return running_loss / iters

    def _val_step(self):
        running_loss = 0.0
        iters = 0

        self._model.eval()

        with torch.no_grad():
            for batch in self._val_data:
                model_input = batch['model_input'].to(self.device)
                ground_truths = batch['ground_truths'].to(self.device)
                model_output = self._model(model_input)
                loss = self._loss_fn(model_input, model_output, ground_truths)
                running_loss += loss.item()
                iters += 1

        return running_loss / iters

    def _save_snapshot(self, epoch, loss, category):
        snapshot = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optim_state_dict': self._optim.state_dict(),
            'loss': loss
        }

        save_path = os.path.join(self._save_path_prefix, self.identifier,
                                 f"{category}")

        utils.io.make_dir(save_path)
        torch.save(snapshot, save_path)

    def _load_snapshot(self):
        if self._load_path is None:
            return 0, None

        snapshot = torch.load(self._load_path, map_location=self.device)

        self._model.load_state_dict(snapshot['model_state_dict'])
        self._optim.load_state_dict(snapshot['optim_state_dict'])
        start_epoch = snapshot['epoch'] + 1
        min_loss = snapshot['loss']

        return start_epoch, min_loss

    def _save_record(self, record_path, epoch, loss):
        snapshot_path = os.path.join(self._save_path_prefix, self.identifier,
                                     'best.tar')

        record = {
            'model': self._model_name,
            'config_id': self._config_id,
            'experiment_id': self.identifier,
            'loss': loss,
            'snapshot_path': snapshot_path
        }

        utils.io.make_dir(record_path)
        with open(record_path, 'a+') as f:
            csv_writer = csv.DictWriter(f, fieldnames=record.keys())
            csv_writer.writerow(record)

    # def _sync_tensorboard(self):
    #     legend = ""


def get_arguments():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--config_path', type=str)
    parser.add_argument('--early_stopping_limit', type=int)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--checkpoint_load_path', type=str)

    return parser.parse_args()


def prepare_optimizer(model, config):
    optim_name = config['optimizer']['name']
    if optim_name == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=config['optimizer']['config']['lr'],
                                 betas=config['optimizer']['config']['betas'],
                                 eps=config['optimizer']['config']['epsilon'])

    return optim


def prepare_save_path(config):
    save_path_prefix = os.path.join(PATH_TO_RESULTS,
                                    'snapshots',
                                    config['model']['name'],
                                    config['id'])

    return save_path_prefix


def prepare_record_path(config):
    dataset_name = config['dataset']['name']
    mix_method = config['dataset']['config']['mix_method']
    nsrc = config['dataset']['config']['num_sources']
    ncls = config['dataset']['config']['num_classes']
    filename = f't{mix_method}-{nsrc}s-{ncls}c.csv'

    record_path = os.path.join(PATH_TO_RESULTS,
                               'records',
                               dataset_name,
                               filename)

    return record_path


def main():
    args = get_arguments()

    with open(args.config_path) as f:
        config = json.load(f)

    train_dataloader = datasets.setup.prepare_dataloader(config, 'train')

    val_dataloader = datasets.setup.prepare_dataloader(config, 'val')

    input_shape = train_dataloader.dataset.input_shape
    model = models.setup.prepare_model(config, input_shape)
    model = model.to(utils.hardware.get_device())

    loss_fn = loss_functions.setup.prepare_loss_fn(config)

    optimizer = prepare_optimizer(model, config)

    # tensorboard_path = utils.experiment.get_path('tensorboard', args)
    save_path_prefix = prepare_save_path(config)
    record_path = prepare_record_path(config)

    logging.info('finished setting up training')

    experiment = Experiment(model=model,
                            model_name=config['model']['name'],
                            train_data=train_dataloader,
                            val_data=val_dataloader,
                            loss_fn=loss_fn,
                            optim=optimizer,
                            max_epochs=config['model']['config']['max_epoch'],
                            save_path_prefix=save_path_prefix,
                            config_id=config['id'],
                            load_path=args.checkpoint_load_path)

    experiment.run(record_path, args.checkpoint_freq,
                   args.early_stopping_limit)


if __name__ == '__main__':
    main()
