"""train.py

"""
import argparse
import copy
import csv
import errno
import json
import logging
import os
import random
import re
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter as tensorboard_writer
import model.dataset as custom_dataset
import model.models as custom_models
import model.transformer as custom_transformer


# default parameters
NUM_SOURCES = 2
MAX_EPOCHS = 500 
LOG_FREQ = 10
CHECKPOINT_FREQ = 1 
ROOT_DIR = "/home/ubuntu/"
RESULT_PATH_PREFIX = os.path.join(ROOT_DIR, "multimodal-listener/results")
DATASET_PATH_PREFIX = os.path.join(ROOT_DIR, "datasets/processed/mixer/")
TBLOG_PATH = os.path.join(ROOT_DIR, "multimodal-listener/tb_logs")

LOSS_MAX = 1e9
MAX_EVAL = 10 
NUM_TRIALS = 1

fieldnames = ['model', 'metric', 'task', 'loss_fn', 'trial',
    'max_epoch', 'lr', 'optim', 'batch_size', 'dropout',
    'momentum', 'beta1', 'beta2', 'epsilon', 'hidden_size', 'in_chan',
    'chan', 'N', 'h', 'd_model', 'd_ff', 'best_val_loss',
    'best_model_path']

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def get_argument():
    """context for searching"""
    parser = argparse.ArgumentParser(description='Setting up an experiment')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--task', type=str) 
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--metric', type=str)

    parser.add_argument("--resume_from_checkpoint",
            dest="resume", const=True, 
            action="store_const", default=False)
    parser.add_argument('--time', type=str, default=None)
    return parser.parse_args()


def get_device(gpu_id):
    """get cuda device object"""
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    return torch.device('cpu')


def make_dir(path):
    """creates directory if it does not exist"""
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def save_configs(time_info, all_configs, task, metric, model_type):
    path = os.path.abspath(os.path.join(
            RESULT_PATH_PREFIX,
            "{}_{}_{}_{}/".format(task, metric, model_type, time_info),
            "config.tar"))
    make_dir(path)
    
    # e.g. results/2s-euclidean-LSTM-190712/configs.tar
    torch.save({"all_configs": all_configs}, path)


def fetch_progress(task, metric, model_type, time_info):
    # e.g. results/2s-euclidean-LSTM-190712/
    setup_path = os.path.join(RESULT_PATH_PREFIX,
        "{}_{}_{}_{}".format(task, metric, model_type, time_info))
  
    all_configs = torch.load(os.path.join(setup_path,
        "config.tar"))['all_configs']

    # XXX: determine which config to pick up
    config_progress = -1    
    for name in os.listdir(setup_path):
        if "config" in name and os.path.isdir(os.path.join(setup_path, name)):
            config_progress += 1
    
    assert(config_progress >= 0)

    # determine which trial to pick up
    # XXX: ignoring trial information
    # trial_progress = -1
    # config_path = os.path.join(setup_path, "config_{}".format(config_progress))
    # for name in os.listdir(config_path):
    #     if os.path.isdir(os.path.join(config_path, name)):
    #         trial_progress +=1

    # assert(trial_progress >= 0)
         
    # results/setup_name/config_n/trial{}/snapshots/checkpoint.tar
    rec_info = {
            'all_configs': all_configs[config_progress:],
            # 'start_trial': trial_progress,
            'checkpoint_path': os.path.join(
                setup_path,
                "config_{}".format(config_progress),
                # "trial_{}".format(start_trial),
                "snapshots/checkpoint.tar")
            }
    
    return rec_info

class Trainer():
    """Trainer object that runs through a single experiment"""
    def __init__(self, task, metric, model_type,
        config, device, tb_writer, time_info, rec_info):
        self.time_info = time_info
        self.tb_writer = tb_writer
        self.rec_info = rec_info
        self.device = device
        self.task = task
        self.metric = metric
        self.model_type = model_type 

        self.dataset_path_prefix = DATASET_PATH_PREFIX 
        self.config = config
        # self.dataloader = self._init_dataloader()
        # self.model = self._init_model()
        self.dataloader, self.model = self._init_model()
        self.optim = self._init_optim()
        self.criterion = self._init_criterion()
        self.start_epoch = 0

        # fill start_epoch, model, optim
        self._load_checkpoint()
        
    def run(self, trial_id, best_val_loss=LOSS_MAX):
        """train the specified model for [max_epoch] epochs"""
        logging.debug(self.device)
        best_model = None
        losses = {}
        losses['curr_best_loss'] = best_val_loss
        # self.config['max_epoch'] = 50
        for epoch in range(self.start_epoch, self.config['max_epoch']):
            end = time.time()
            train_loss = self.train(epoch)
            val_loss = self.validate(epoch)
           
            losses['train'] = train_loss
            losses['val'] = val_loss

            # save model with the best performance
            if val_loss < losses['curr_best_loss']:
                best_model = copy.deepcopy(self.model)
                # save best loss of current unfinished trial
                losses['curr_best_loss'] = val_loss
            
            logging.info("epoch {}, [{}/{}] train loss: {:.2f}, "
                    "val loss: {:.2f}, duration: {:.0f}s".format(
                epoch + 1,
                self.task[2:-2], 
                self.task[2:-2], 
                losses['train'],
                losses['val'],
                time.time() - end))

            self._log_tb(losses, epoch, trial_id)

            if (epoch + 1) % CHECKPOINT_FREQ == 0:
                logging.info("saving snapshot...")
                self._save_model(epoch, losses, trial_id=trial_id)
                if best_model is not None:
                    logging.info("saving best model...")
                    best_path = self._save_model(epoch, losses, 
                            best_model=best_model, trial_id=trial_id)
                logging.info("finished saving snapshots")


        # TODO: use google api
        self._log_sheet(trial_id, losses['curr_best_loss'], best_path)

    def _log_sheet(self, trial_id, best_val_loss, best_path):
        csv_path = os.path.join(RESULT_PATH_PREFIX, "results.csv")
        with open(csv_path, 'a+') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            content = copy.deepcopy(self.config)
            del content['id']
            content['model'] = self.model_type
            content['metric'] = self.metric
            content['task'] = self.task
            content['trial'] = trial_id + 1
            content['best_val_loss'] = best_val_loss
            content['best_model_path'] = best_path 
            csv_writer.writerow(content)

    def train(self, epoch):
        self.model.train()
        # train_loss, test_loss = 0.0, 0.0
        for batch_idx, batch in enumerate(self.dataloader['train']):
            logging.debug(batch_idx + 1)
            logging.debug(len(self.dataloader['train']) // LOG_FREQ)
            self.optim.zero_grad()
            
            loss = self._compute_loss(batch)
            
            loss.backward()
            self.optim.step()
           
            if (batch_idx + 1) % (len(self.dataloader['train']) // \
                    LOG_FREQ) == 0:
                logging.info("epoch {}, [{:5d}/{}] train loss: {:.2f}".format(
                    epoch + 1,
                    batch_idx * self.config['batch_size'], 
                    self.task[2:-2],
                    loss.item()))
        return loss.item()

    def validate(self, batch):
        running_loss = 0.0
        iters = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader['test']):
                iters += 1
                loss = self._compute_loss(batch)
                running_loss += loss.item()

        return running_loss / iters
   
    def _compute_loss(self, batch):
        aggregate = batch['aggregate'].to(self.device)
        if self.model_type == "VTF":
            ground_truths_in = batch['ground_truths_in'].to(self.device)
            ground_truths = batch['ground_truths_gt'].to(self.device)

            mask_size = ground_truths_in.shape[1]
            subseq_mask = custom_transformer.subsequent_mask(
                    mask_size).to(self.device)
            out = self.model(aggregate, ground_truths_in, None, subseq_mask)
            prediction = self.model.generator(aggregate, out)
            loss = self.criterion(prediction, ground_truths)
        else:
            ground_truths = batch['ground_truths'].to(self.device)
            prediction = self.model(aggregate)
            loss = self.criterion(prediction, ground_truths)
        
        return loss

    def _log_tb(self, losses, epoch, trial_id):
        legend_prefix = "{}_{}_{}_{}".format(
                self.model_type,
                self.task,
                self.config['id'],
                trial_id)
        self.tb_writer.add_scalars("data/{}/loss".format(self.metric),
                {
                    "{}_train".format(legend_prefix): losses['train'],
                    "{}_val".format(legend_prefix): losses['val'],
                }, epoch)

    # XXX: currently ignoring trial information
    def _save_model(self, epoch, losses, best_model=None, trial_id=0):
        model_name = "checkpoint.tar" if best_model is None else "best.tar"
        # e.g. results/setup_name/config_0/snapshots/checkpoint.tar
        # e.g. results/setup_name/config_0/snapshots/best.tar
        model_path = os.path.join(
                RESULT_PATH_PREFIX,
                "{}_{}_{}_{}".format(self.task, self.metric,
                    self.model_type, self.time_info),
                "config_{}".format(self.config['id']),
                # "trial_{}".format(trial_id),
                "snapshots",
                model_name)

        make_dir(model_path)

        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'train_loss': losses['train'],
            'val_loss': losses['train'],
            'curr_best_loss': losses['curr_best_loss'],
            }, model_path)
 
        return model_path

    def _init_criterion(self):
        if self.config['loss_fn'] == "Greedy":
            criterion = custom_models.GreedyLoss(self.device, self.metric)
        elif self.config['loss_fn'] == "Min":
            criterion = custom_models.MinLoss(self.device, self.metric)
        else:
            logging.error("loss function {}"
                    "is not supported".format(self.config['loss_fn']))
       
        return criterion

    def _init_optim(self):
        if self.config['optim'] == "SGD":
            optim = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.config['lr'],
                    momentum=self.config['momentum'])
        elif self.config['optim'] == "Adam":
            optim = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config['lr'],
                    betas=(self.config['beta1'], self.config['beta2']),
                    eps=self.config['epsilon'])
        else:
            logging.error("optimizer {}"
                    "is not supported".format(self.config['loss_fn']))
        return optim
    
    def _init_model(self):
        """create dataloader and model"""

        def get_loader(dataset_type):
            """get loader based on train/test spec"""

            dataset_path = os.path.join(self.dataset_path_prefix,
                    "{}/{}".format(self.task, dataset_type))

            dataset = custom_dataset.SignalDataset(root_dir=dataset_path,
                    transform=transform)

            dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=self.config['batch_size'],
                    shuffle=True)
            
            return dataloader

        # load spectrogram spec
        with open (os.path.join(self.dataset_path_prefix, self.task, "train",
            "data_spec.json"), 'r') as reader:
            spect_shape = json.load(reader)
        
        num_sources = int(self.task[0])
        # load dataloader and model 
        if (self.model_type == "LSTM"):
            transform = custom_dataset.Concat(
                size=(spect_shape['freq_range'], spect_shape['seq_len']))
            model = custom_models.B1(
                    input_dim=spect_shape['freq_range'] * 2,
                    seq_len=spect_shape['seq_len'],
                    num_sources=num_sources).to(self.device)

        elif (self.model_type == "GAB"):
            transform = custom_dataset.ToTensor(
                size=(spect_shape['freq_range'], spect_shape['seq_len']))
            model = custom_models.LookToListenAudio(
                    input_dim=spect_shape['freq_range'],
                    seq_len=spect_shape['seq_len'],
                    num_sources=num_sources).to(self.device)

        elif (self.model_type == "VTF"):
            transform = custom_dataset.Concat(
                size=(spect_shape['freq_range'], spect_shape['seq_len']),
                encdec=True)
            model = custom_transformer.make_model(
                    input_dim=spect_shape['freq_range'] * 2,
                    N=self.config['N'],
                    d_model=self.config['d_model'],
                    d_ff=self.config['d_ff'],
                    h=self.config['h'],
                    num_sources=num_sources,
                    dropout=self.config['dropout']).to(self.device)

        else:
            logging.error("model type {}"
                    "is not supported".format(self.model_type))

        dataloader = {}
        dataloader['train'] = get_loader("train")
        dataloader['test'] = get_loader("test")


        logging.info("finished setting up {}".format(self.model_type))
        return dataloader, model

    def _load_checkpoint(self):

        # load checkpoint if resuming operations
        if self.rec_info is not None:
            # given trial, pick up model snapshot  
            logging.info("loading checkpoint from configuration # {}".format(
                self.config['id']))
            checkpoint = torch.load(self.rec_info['checkpoint_path'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.curr_best_loss = checkpoint['curr_best_loss']
            logging.info("checkpoint loaded!")


def select_configs():
    all_configs = []
    max_eval = MAX_EVAL 

    max_epochs = [300, 400, 500]
    lrs = [0.001, 0.005, 0.01]
    optims = ['SGD', 'Adam', 'Adam']
    # TODO: implement Min
    # loss_fns = ['Greedy', 'Min']
    loss_fns = ['Greedy']
    batch_sizes = [16, 64, 128]
    dropouts = [0.1, 0.3, 0.5]
    momentums = [0.0, 0.5, 0.9]
    beta1s = [0.9, 0.95, 0.99]
    beta2s = [0.98, 0.99, 0.999]
    epsilons = [1e-8, 1e-9]

    # for LSTM
    hidden_sizes = [32, 128, 512]

    # for google baseline
    in_chans = [4, 16]
    chans = [4, 16, 64]

    # for vanilla transformer
    Ns = [2, 3, 4]
    hs = [2, 4]
    d_models = [64, 128, 256]
    d_ffs = [64, 128, 256]

    all_params = {
        'max_epochs': max_epochs,
        'lrs': lrs,
        'optims': optims,
        'loss_fns': loss_fns,
        'batch_sizes': batch_sizes,
        'dropouts': dropouts,
        'momentums': momentums,
        'beta1s': beta1s,
        'beta2s': beta2s,
        'epsilons': epsilons,
        'hidden_sizes': hidden_sizes,
        'in_chans': in_chans,
        'chans': chans,
        'Ns': Ns,
        'hs': hs,
        'd_models': d_models,
        'd_ffs': d_ffs
        }
    
    for i in range(max_eval):
        config = {}
        config['id'] = i
        for param in all_params.keys():
            config[param[:-1]] = random.choice(all_params[param])
        all_configs.append(config)
    return all_configs 


def main():
    args = get_argument()
    device = get_device(args.gpu_id)
    
    # TODO: inter face to google sheets
    # record = Record()
   
    # csv_path = "/media/bighdd7/tianjunm/multimodal-listener/results"

    logging.info("GPU: {}, Task: {}, Metric: {}, Model: {}".format(device, 
        args.task, args.metric, args.model_type))

    if args.resume:
        logging.info("resuming from interrupted experiment...")
        time_info = args.time
        # information for recovery
        rec_info = fetch_progress(args.task, args.metric, args.model_type,
                time_info)
        all_configs = rec_info['all_configs']
        # start_trial = rec_info['start_trial']
        logging.info("found partial results")
    else:
        logging.info("initiating new experiment...")
        time_info = time.strftime("%y%m%d", time.gmtime())

        # csv_path = os.path.join(RESULT_PATH_PREFIX, "results.csv")
        # make_dir(csv_path)
        # with open(csv_path, "a+") as csvfile:
        #     csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     csv_writer.writeheader()

        rec_info = None
        all_configs = select_configs()
        # start_trial = 0
        save_configs(time_info, all_configs, args.task,
            args.metric, args.model_type)
        assert(len(all_configs) == MAX_EVAL) 

    start_trial = 0
    # tensorboard writer
    tb_writer = tensorboard_writer(logdir=TBLOG_PATH)
    # hyperparameter search with random search 
    for config in all_configs:
        logging.info("configuration #{}:".format(config['id']))
        logging.info(config)
        trainer = Trainer(args.task, args.metric, args.model_type, config,
                device, tb_writer, time_info, rec_info)

        # run the experiment [num_trials] times 
        for trial_id in range(start_trial, NUM_TRIALS):
            logging.info("running configuration #{} [{}/{}]...".format(
                config['id'],
                trial_id + 1,
                NUM_TRIALS))

            # picking up unfinished trial
            if start_trial != 0 and start_trial == trial_id:
                trainer.run(trial_id, self.curr_best_loss)
            else:
                # trainer.run(trial_id, record)
                trainer.run(trial_id)

            logging.info("trial {} finished!".format(trial_id + 1))

        # won't need to recover again after finishing recovery
        rec_info = None

    logging.info("experiment done!")
    logging.info("all stats uploaded to {}".format(record.url))
   
    tb_writer.close()


if __name__ == '__main__':
    main()
