"""
This script trains the models for monaural source separation.

Contributors:
    Tianjun Ma (tianjunm@cs.cmu.edu)
    Shreya Bali (sbali@andrew.cs.cmu.edu)

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
import pandas as pd

import models.setup
import datasets.setup
import loss_functions.setup
import utils.io
import utils.hardware
import loss_functions.loss_implementation as c_loss_implementation

import gc


from guppy import hpy
import sys
import torch
import gc
from memory_profiler import profile

import torch
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass


PATH_TO_RESULTS = '/work/sbali/monaural-source-separation/results'
ALL_CATS = '/work/sbali/monaural-source-separation/all_cats.csv'


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)

#@profile
def one_hot_embedding(labels, num_classes=6):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels]


class Experiment():
    #@profile
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 loss_fn,
                 optim,
                 max_epochs,
                 save_path_prefix,
                 seq2seq=False,
                 waveunet=False,
                 weighted_waveunet=False,
                 classifier=None,
                 train_classifier=False,
                 load_path=None):
        print("RUNNING")
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
        self._waveunet = waveunet
        self._weighted_waveunet = weighted_waveunet
        self._load_path = load_path
        self._classifier = classifier
        self._train_classifier = train_classifier
        if train_classifier :
            df = pd.read_csv(ALL_CATS, index_col='class')
            self._allcats = pd.read_csv(ALL_CATS, index_col='class').to_dict()['maincategory']
            self._num_top_cats = len(df['maincategory'].unique())
    
    #@profile
    def run(self, record_path, record_template, checkpoint_freq=50, 
            early_stopping_limit=None, no_pit=False, no_tgt=False):
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
            train_loss = self._train_step(no_pit, no_tgt)
            val_loss = self._val_step(no_pit, no_tgt)

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
    
    #@profile
    def __get_categories__(self, batch, model_input):
        categories = batch['component_info']
        one_hot_labels = torch.zeros((model_input.shape[0], self._num_top_cats))
        ones = torch.ones(one_hot_labels.shape)
        for cat_num in range(len(categories)):
            labels = list(map(lambda x: self._allcats[x[:-4]], list(categories[cat_num])))
            one_hot_label_i = list(map(lambda x: one_hot_embedding(x, self._num_top_cats), labels))
            one_hot_label_i = torch.stack(one_hot_label_i, dim=0)
            one_hot_labels += one_hot_label_i
            one_hot_labels = torch.min(one_hot_labels, ones)
        return one_hot_labels

    #@profile
    def _calculate_accuracy(self, model_output, one_hot_labels, t):
        output = model_output.cpu()
        labels = one_hot_labels.cpu()
        topk = output > 0.5
        fin_vals = torch.zeros(one_hot_labels.shape).cpu()
        fin_vals[topk] = 1 
        accuracy_p = (fin_vals == labels) * (fin_vals == 1)
        total_vals = labels==1
        #torch.sum((fin_vals == labels) * (fin_vals == 1), dim = 1) == torch.sum( labels== 1, dim = 1)
        accuracy = len(output[accuracy_p])/(len(labels[total_vals]))
        logging.info(t + "ACC "+ str(accuracy))
    
    #@profile
    def _train_step(self, no_pit, no_tgt):
        self._model.train()
        running_loss = 0.0
        iters = 0
        logging.info("START TRAIN STEP")
        for i, batch in enumerate(self._train_data):
            logging.info(i)
            self._optim.zero_grad()
            model_input = batch['model_input'].to(self.device)
            if self._train_classifier: 
                one_hot_labels = self.__get_categories__(batch, model_input)
                one_hot_labels = one_hot_labels.to(self.device)
                model_output = self._model(model_input)
                loss = self._loss_fn(model_output, one_hot_labels) 
                self._calculate_accuracy(model_output, one_hot_labels, "T")
            else:
                ground_truths = batch['ground_truths'].to(self.device)

            if self._seq2seq:
                model_output = self._model(model_input, ground_truths,
                                           no_tgt=no_tgt)
                loss = self._loss_fn(model_input, model_output,
                                     ground_truths[:, 1:])
            elif self._waveunet:
                model_input = model_input.reshape(model_input.shape[0], 1, model_input.shape[1])
                model_output = self._model(model_input)
                loss = self._loss_fn(model_output, ground_truths)

            elif self._weighted_waveunet:
                model_input = model_input.reshape(model_input.shape[0], 1, model_input.shape[1])
                model_output = self._model(model_input, one_hot_labels)
                loss = self._loss_fn(stft_input, model_output, ground_truths)
            
            elif not self._train_classifier:
                model_output = self._model(model_input)
                loss = self._loss_fn(model_input, model_output, ground_truths)

            loss.backward()
            self._optim.step()
            running_loss += loss.item()
            logging.info("TRAIN " + str(loss.item()))
            torch.cuda.empty_cache() 
            iters += 1

        logging.info("TRAIN", running_loss / iters)
        # print(i)
        return running_loss / iters
    
    #@profile
    def _val_step(self, no_pit, no_tgt):
        running_loss = 0.0
        iters = 0

        self._model.eval()

        with torch.no_grad():
            for batch in self._val_data:
                model_input = batch['model_input'].to(self.device)

                if self._train_classifier: 
                    one_hot_labels = self.__get_categories__(batch, model_input)
                    one_hot_labels = one_hot_labels.to(self.device)
                    model_output = self._model(model_input)
                    loss = self._loss_fn(model_output, one_hot_labels) 
                    self._calculate_accuracy(model_output, one_hot_labels, "V")
                else:
                    ground_truths = batch['ground_truths'].to(self.device)

                if self._seq2seq:
                    model_output = self._model(model_input, ground_truths,
                                               no_tgt=no_tgt)
                    loss = self._loss_fn(model_input, model_output,
                                         ground_truths[:, 1:])
                elif self._waveunet:
                    model_input = model_input.reshape(model_input.shape[0], 1, model_input.shape[1])
                    model_output = self._model(model_input)
                    loss = self._loss_fn(model_output, ground_truths)
 
                elif self._weighted_waveunet:
                    with torch.no_grad():
                        labels = self._classifier(stft_input)
                    model_input = model_input.reshape(model_input.shape[0], 1, model_input.shape[1]).float()
                    model_output = self._model(model_input, labels).float()
                    loss = self._loss_fn(stft_input, model_output, ground_truths)

                elif not self._train_classifier:
                    model_output = self._model(model_input)
                    loss = self._loss_fn(model_input, model_output, ground_truths)

                logging.info("VAL " + str(loss.item()))
                running_loss += loss.item()
                torch.cuda.empty_cache() 
                iters += 1
        logging.info("VAL", running_loss/iters)
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


def prepare_save_path_prefix(dataset_spec, model_spec, classifier=False):
    did = dataset_spec['id']
    dataset_name = dataset_spec['name']
    if dataset_name == 'wild-mix':
        mix_method = dataset_spec['config']['mix_method']
        nsrc = dataset_spec['config']['num_sources']
        ncls = dataset_spec['config']['num_classes']
        dataset_config = f'{dataset_name}-t{mix_method}-{nsrc}s-{ncls}c'

    mid = model_spec['id']
    model_name = model_spec['model']['name']
    filename = f'{model_name}-{mid}-{dataset_config}-{did}'
    if classifier : 
        save_path_prefix = os.path.join(PATH_TO_RESULTS, 'snapshots', "classifier"+filename)
    else :
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


def prepare_record_path(dataset_spec, classifier=True):
    dataset_name = dataset_spec['name']
    filename = 'records.csv'

    record_path = os.path.join(PATH_TO_RESULTS, 'records', dataset_name,
                               "classifier"+filename)

    return record_path


def get_arguments():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--dataset_spec', type=str)
    parser.add_argument('--model_spec', type=str)
    parser.add_argument('--early_stopping_limit', type=int)
    parser.add_argument('--checkpoint_freq', type=int)
    parser.add_argument('--checkpoint_load_path', type=str)
    parser.add_argument('--no_pit', type=bool, nargs='?', const=True, default=False)

    return parser.parse_args()

def prepare_classifier(dataset_spec, model_spec, checkpoint_load_path, args):
    print("ENTERED")
    train_dataloader = datasets.setup.classifier_prepare_dataloader(
        dataset_spec, model_spec, 'train')
    val_dataloader = datasets.setup.classifier_prepare_dataloader(
        dataset_spec, model_spec, 'val')

    input_shape = train_dataloader.dataset.input_shape
    
    model = models.setup.classifier_prepare_model(dataset_spec, model_spec, input_shape)
    model = model.to(utils.hardware.get_device())
    # print(model.device)

    loss_fn =  loss_functions.setup.classifier_prepare_loss_fn()
   
    optimizer = prepare_optimizer(model, model_spec)
    max_epochs = model_spec['model']['config']['max_epoch']
    # tensorboard_path = utils.experiment.get_path('tensorboard', args)
    save_path_prefix = prepare_save_path_prefix(dataset_spec, model_spec, classifier=True)
    record_path = prepare_record_path(dataset_spec, classifier=True)
    logging.info('finished setting up training')
    experiment = Experiment(model=model,
                            train_data=train_dataloader,
                            val_data=val_dataloader,
                            loss_fn=loss_fn,
                            optim=optimizer,
                            max_epochs=max_epochs,
                            save_path_prefix=save_path_prefix,
                            load_path=checkpoint_load_path,
                            train_classifier=True)
    record_template = prepare_record_template(dataset_spec, model_spec)
    no_tgt = (model_spec['id'] == 'sample2')
    print(args.no_pit)
    experiment.run(record_path, record_template, args.checkpoint_freq,
                   args.early_stopping_limit, args.no_pit, no_tgt)
    
    print("EXITED")
    return model

    
#@profile
def main():
    args = get_arguments()

    with open(args.dataset_spec) as df, open(args.model_spec) as mf:
        dataset_spec = json.load(df)
        model_spec = json.load(mf)
    seq2seq = (model_spec['model']['name'] in ['STT'])
    waveunet = (model_spec['model']['name'] in ['WAVE-U-NET'])
    weighted_waveunet = (model_spec['model']['name'] in ['W-WAVE-U-NET'])
    classifier = None 

    if waveunet or weighted_waveunet:
        if weighted_waveunet:
            classifier = prepare_classifier(dataset_spec, model_spec, args.checkpoint_load_path, args)
        model = models.setup.prepare_model(dataset_spec, model_spec)
        dataset_spec['agg_len'] = model.shapes["input_frames"]
        dataset_spec['gt_start'] = model.shapes["output_start_frame"]
        dataset_spec['gt_end'] = model.shapes["output_end_frame"]
        model = model.to(utils.hardware.get_device())

    train_dataloader = datasets.setup.prepare_dataloader(
        dataset_spec, model_spec, 'train')

    val_dataloader = datasets.setup.prepare_dataloader(
        dataset_spec, model_spec, 'val')

    if not (waveunet or weighted_waveunet):
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

    experiment = Experiment(model=model,
                            train_data=train_dataloader,
                            val_data=val_dataloader,
                            loss_fn=loss_fn,
                            optim=optimizer,
                            max_epochs=max_epochs,
                            save_path_prefix=save_path_prefix,
                            seq2seq=seq2seq,
                            load_path=args.checkpoint_load_path,
                            classifier=classifier,
                            waveunet=waveunet,
                            weighted_waveunet=weighted_waveunet)

    record_template = prepare_record_template(dataset_spec, model_spec)

    # TODO just a test
    no_tgt = (model_spec['id'] == 'sample2')

    print(args.no_pit)
    experiment.run(record_path, record_template, args.checkpoint_freq,
                   args.early_stopping_limit, args.no_pit, no_tgt)


if __name__ == '__main__':
    main()
