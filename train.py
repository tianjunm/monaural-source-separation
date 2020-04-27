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
                 train_data,
                 val_data,
                 loss_fn,
                 optim,
                 max_epochs,
                 save_path_prefix,
                 seq2seq=False,
                 load_path=None):

        self.identifier = time.strftime("%y%m%d%H%M%S", time.gmtime())
        self.device = utils.hardware.get_device()

        self._model = model
        self._train_data = train_data
        self._val_data = val_data
        self._loss_fn = loss_fn
        self._optim = optim
        self._max_epochs = max_epochs
        self._save_path_prefix = save_path_prefix
        self._seq2seq = seq2seq
        self._load_path = load_path

    def run(self, record_path, record_template, checkpoint_freq=50,
            early_stopping_limit=None, no_tgt=False):
        logging.info(f'starts experiment: {self._save_path_prefix}')
        start_epoch, min_val_loss = self._load_snapshot()

        counter = 0
        min_train_loss = None

        train_losses = []
        val_losses = []

        for epoch in range(start_epoch, self._max_epochs):
            if (early_stopping_limit is not None and
               counter >= early_stopping_limit):
                break

            train_loss = self._train_step(no_tgt)
            val_loss = self._val_step(no_tgt)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if min_train_loss is None or train_loss < min_train_loss:
                min_train_loss = train_loss

            if min_val_loss is None or val_loss < min_val_loss:
                counter = 0
                min_val_loss = val_loss
                self._save_snapshot(epoch, val_loss, 'best',
                                    train_losses, val_losses)
            else:
                counter += 1

            if (epoch + 1) % checkpoint_freq == 0:
                self._save_snapshot(epoch, min_val_loss, 'checkpoint',
                                    train_losses, val_losses)

            if early_stopping_limit is not None:
                logging.info('epoch %3d, train loss: %.2f, val loss: %.2f | '
                             '%2d/%2d', epoch, train_loss, val_loss,
                             counter, early_stopping_limit)
            else:
                logging.info('epoch %3d, train loss: %.2f, val loss: %.2f',
                             epoch, train_loss, val_loss)

        self._save_record(record_path, record_template, min_train_loss,
                          min_val_loss)

    def _train_step(self, no_tgt):
        self._model.train()
        running_loss = 0.0
        iters = 0

        for i, batch in enumerate(self._train_data):
            self._optim.zero_grad()

            model_input = batch['model_input'].to(self.device)
            ground_truths = batch['ground_truths'].to(self.device)

            if self._seq2seq:
                model_output = self._model(model_input, ground_truths)
                # prediction = self._decode(model_input, ground_truths)
                # loss = self._loss_fn(prediction, ground_truths[:, 1:])
                loss = self._loss_fn(model_input, model_output,
                                     ground_truths[:, 1:])
            else:
                model_output = self._model(model_input)
                loss = self._loss_fn(model_input, model_output, ground_truths)

            loss.backward()
            self._optim.step()

            running_loss += loss.item()
            iters += 1

        # print(i)
        return running_loss / iters

    def _val_step(self, no_tgt):
        running_loss = 0.0
        iters = 0

        self._model.eval()

        with torch.no_grad():
            for batch in self._val_data:
                model_input = batch['model_input'].to(self.device)
                ground_truths = batch['ground_truths'].to(self.device)

                if self._seq2seq:
                    model_output = self._model(model_input, ground_truths)
                    loss = self._loss_fn(model_input, model_output,
                                         ground_truths[:, 1:])
                else:
                    model_output = self._model(model_input)
                    loss = self._loss_fn(model_input, model_output,
                                         ground_truths)

                running_loss += loss.item()
                iters += 1

        return running_loss / iters

    def _save_snapshot(self, epoch, loss, category, train_losses, val_losses):
        snapshot = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optim_state_dict': self._optim.state_dict(),
            'loss': loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        save_path = os.path.join(self._save_path_prefix,
                                 self.identifier,
                                 f"{category}.tar")

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

    def _save_record(self, record_path, record_template, train_loss, val_loss):
        snapshot_path = os.path.join(self._save_path_prefix, self.identifier,
                                     'best.tar')

        record_template['experiment_id'] = self.identifier
        record_template['train_loss'] = train_loss
        record_template['val_loss'] = val_loss
        record_template['snapshot_path'] = snapshot_path

        utils.io.make_dir(record_path)
        with open(record_path, 'a+') as f:
            csv_writer = csv.DictWriter(f, fieldnames=record_template.keys())
            csv_writer.writerow(record_template)

    # TODO: temporary, will clean up soon
    def _decode(self, model_input, ground_truths):
        b = model_input.size(0)
        n = model_input.size(1)

        ys = torch.ones(b, n + 1, ground_truths.size(-1)).to(self.device)

        src = model_input.unsqueeze(2)

        running_loss = 0.0
        for i in range(n):
            out = self._model(model_input, ys) * src
            print('------------------------------------------------------------')
            print(torch.cuda.max_memory_allocated(self.device))
            loss = self._loss_fn(out[:, i].unsqueeze(1),
                                  ground_truths[:, i + 1].unsqueeze(1))
            # ys = torch.cat((ys, out.view(
            #     b, i + 1, -1)[:, -1].unsqueeze(1)), dim=1)
            ys[:, 1:i + 2] = out[:, :i + 1].view(b, i + 1, -1)
            print(torch.cuda.max_memory_allocated(self.device))
            print('------------------------------------------------------------')

        # return ys[:, 1:]
        return loss / n

    # def _sync_tensorboard(self):
    #     legend = ""


def prepare_optimizer(model, model_spec):
    optim_name = model_spec['optimizer']['name']
    optim_config = model_spec['optimizer']['config']

    if optim_name == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=optim_config['lr'],
                                 betas=optim_config['betas'],
                                 eps=optim_config['epsilon'])

    elif optim_name == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(),
                                    lr=optim_config['lr'])

    return optim


def prepare_save_path_prefix(dataset_spec, model_spec):
    did = dataset_spec['id']
    dataset_name = dataset_spec['name']
    if dataset_name == 'wild-mix':
        mix_method = dataset_spec['config']['mix_method']
        nsrc = dataset_spec['config']['num_sources']
        ncls = dataset_spec['config']['num_classes']
        dataset_config = f'{dataset_name}-t{mix_method}-{nsrc}s-{ncls}c'

    elif dataset_name == 'timit':
        dataset_config = 'timit'

    mid = model_spec['id']
    model_name = model_spec['model']['name']

    filename = f'{model_name}-{mid}-{dataset_config}-{did}'

    save_path_prefix = os.path.join(PATH_TO_RESULTS, 'snapshots', filename)

    return save_path_prefix


def prepare_record_template(dataset_spec, model_spec):
    did = dataset_spec['id']
    dataset_name = dataset_spec['name']
    if dataset_name == 'wild-mix':
        mix_method = dataset_spec['config']['mix_method']
        nsrc = dataset_spec['config']['num_sources']
        ncls = dataset_spec['config']['num_classes']
        dataset_config = f'{dataset_name}-t{mix_method}-{nsrc}s-{ncls}c'

    elif dataset_name == 'timit':
        dataset_config = 'timit'

    mid = model_spec['id']
    model_name = model_spec['model']['name']

    template = {
        'dataset_config': dataset_config,
        'dataset_id': did,
        'model': model_name,
        'mid': mid,
        'experiment_id': None,
        'train_loss': None,
        'val_loss': None,
        'snapshot_path': None
    }

    return template


def prepare_record_path(dataset_spec):
    dataset_name = dataset_spec['name']
    filename = 'records.csv'

    record_path = os.path.join(PATH_TO_RESULTS, 'records', dataset_name,
                               filename)

    return record_path


def get_arguments():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--dataset_spec', type=str)
    parser.add_argument('--model_spec', type=str)
    parser.add_argument('--early_stopping_limit', type=int)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--checkpoint_load_path', type=str)
    # parser.add_argument('--no_pit', type=bool, nargs='?', const=True,
    #                     default=False)

    return parser.parse_args()


def main():
    args = get_arguments()

    with open(args.dataset_spec) as df, open(args.model_spec) as mf:
        dataset_spec = json.load(df)
        model_spec = json.load(mf)

    train_dataloader = datasets.setup.prepare_dataloader(
        dataset_spec, model_spec, 'train')

    val_dataloader = datasets.setup.prepare_dataloader(
        dataset_spec, model_spec, 'val')

    input_shape = train_dataloader.dataset.input_shape
    model = models.setup.prepare_model(dataset_spec, model_spec, input_shape)
    model = model.to(utils.hardware.get_device())
    # print(model.device)

    loss_fn = loss_functions.setup.prepare_loss_fn(dataset_spec, model_spec)

    optimizer = prepare_optimizer(model, model_spec)

    max_epochs = model_spec['model']['config']['max_epoch']

    # tensorboard_path = utils.experiment.get_path('tensorboard', args)
    save_path_prefix = prepare_save_path_prefix(dataset_spec, model_spec)
    record_path = prepare_record_path(dataset_spec)

    logging.info('finished setting up training')

    seq2seq = (model_spec['model']['name'] in ['Transformer', 'STF', 'STT'])

    # if seq2seq:
    #     loss_fn = loss_functions.loss_implementation.NoMask(
    #         dataset_spec['config'])

    experiment = Experiment(model=model,
                            train_data=train_dataloader,
                            val_data=val_dataloader,
                            loss_fn=loss_fn,
                            optim=optimizer,
                            max_epochs=max_epochs,
                            save_path_prefix=save_path_prefix,
                            seq2seq=seq2seq,
                            load_path=args.checkpoint_load_path)

    record_template = prepare_record_template(dataset_spec, model_spec)

    # TODO just a test
    # no_tgt = (model_spec['id'] == 'sample2')
    no_tgt = False

    experiment.run(record_path, record_template, args.checkpoint_freq,
                   args.early_stopping_limit, no_tgt)


if __name__ == '__main__':
    main()
