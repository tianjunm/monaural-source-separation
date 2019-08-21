import argparse
import csv
import copy
import logging
import time
import os
import torch
import torch.optim
import utils
import constants as const

# FIXME: module rename
import model.dataset as custom_dataset
import model.models as custom_models
import model.transformer as custom_transformer
import model.srnn as huang_srnn
import model.drnn as huang_drnn
import model.csa_lstm as csa_lstm
from model.min_loss import MinLoss


logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)


class Trainer():
    """Trainer object that runs through a single experiment"""
    def __init__(self, setup, config, experiment_path, load_checkpoint=False):

        self._device = setup['device']
        self._ds_type = setup['dataset_type']
        self._nsrc = setup['num_sources']
        self._ncat = setup['category_range']
        self._metric = setup['metric']
        self._m_type = setup['model_type']
        self._eid = setup['experiment_id']
        self._nconf = setup['num_configs']

        self._config = config
        self._epath = experiment_path
        self._cp = load_checkpoint

        self._ds_size = self._nsrc * const.TRAIN_SCALE
        # self.tb_writer = tb_writer
        # self.task = task

        self._dl, self._model = self._init_model()
        self._optim = self._init_optim()
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optim, 'min')
        self._criterion = self._init_criterion()

        self.best_path = None
        # fill start_epoch, model, optim
        # self._load_checkpoint()

    def fit(self):
        """Train the specified model for [max_epoch] epochs."""
        logging.debug(self._device)

        # TODO: es_counter will not be restored
        es_counter = 0  # keeping track of early stopping
        es_done = False  # indicator of early stopping

        # self._load_()
        if not self._cp:
            start_epoch, min_loss = 0, const.MAX_LOSS
        else:
            start_epoch, min_loss = self._load_cp()

        for epoch in range(start_epoch, self._config['max_epoch']):
            if es_done:
                break

            end = time.time()
            train_loss = self.train(epoch)
            val_loss = self.validate()

            # self.scheduler.step(val_loss)

            # save model with the best performance
            if val_loss < min_loss:
                es_counter = 0

                # save best loss of current unfinished trial
                min_loss = val_loss
                logging.info("saving best model...")

                self._save_model(
                    epoch,
                    train_loss,
                    val_loss,
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
                self._epath,
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

        self._log_sheet(epoch, min_loss)

    def _log_sheet(self, epoch, best_val_loss):
        csv_path = os.path.join(const.RESULT_PATH_PREFIX, const.RESULT_FILENAME)
        with open(csv_path, 'a+') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=const.FIELD_NAMES)
            content = copy.deepcopy(self._config)
            content['model'] = self._m_type
            content['metric'] = self._metric
            content['stop_epoch'] = epoch
            content['best_val_loss'] = best_val_loss
            content['experiment_path'] = self._epath
            csv_writer.writerow(content)

    def train(self, epoch):
        "Trains the model for one epoch."
        self._model.train()
        for batch_idx, batch in enumerate(self._dl['train']):
            self._optim.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            self._optim.step()

            log_size = (self._ds_size // \
                        self._config['batch_size'] // const.LOG_FREQ)

            if (batch_idx + 1) % log_size == 0:
                logging.info(
                    "[%2d/%2d] %s: epoch %3d, [%5d/%5d] "
                    "train loss: %.2f",
                    self._config['id'] + 1,
                    self._nconf,
                    self._epath,
                    epoch + 1,
                    batch_idx * self._config['batch_size'],
                    self._ds_size,
                    loss.item())

        return loss.item()

    def validate(self):
        "Returns validation loss on the validation set."
        running_loss = 0.0
        iters = 0
        self._model.eval()
        with torch.no_grad():
            for batch in self._dl['test']:
                iters += 1
                loss = self._compute_loss(batch)
                running_loss += loss.item()

        return running_loss / iters

    def _compute_loss(self, batch):
        aggregate = batch['aggregate'].to(self._device)
        if self._m_type in ["SRNN", "DRNN", "LSTM", "CSALSTM", "GAB"]:
            ground_truths = batch['ground_truths'].to(self._device)
            prediction = self._model(aggregate)
            loss = self._criterion(prediction, ground_truths)

        # STT1, STT2, STT3
        else:
            in_gts = batch['ground_truths_in'].to(self._device)
            cmp_gts = batch['ground_truths_gt'].to(self._device)

            mask_size = in_gts.shape[1]
            subseq_mask = custom_transformer.subsequent_mask(
                mask_size).to(self._device)

            if self._m_type in ["VTF", "STT1", "STT2"]:
                out = self._model(aggregate, in_gts, None, subseq_mask)
                prediction = self._model.generator(aggregate, out)
            else:  # STT3
                prediction = self._model(
                    aggregate,
                    in_gts,
                    None,
                    subseq_mask)
            loss = self._criterion(prediction, cmp_gts)
        return loss

    def _log_tb(self, losses, epoch, trial_id):
        legend_prefix = "{}_{}_{}_{}_{}".format(
                self._m_type,
                self.task,
                self._config['id'],
            self.catcap,
                trial_id)
        self.tb_writer.add_scalars("data/{}/loss".format(self.metric),
                {
                    "{}_train".format(legend_prefix): losses['train'],
                    "{}_val".format(legend_prefix): losses['val'],
                }, epoch)

    def _save_model(
            self,
            epoch,
            train_loss,
            val_loss,
            min_loss=const.MAX_LOSS,
            best_model=None):

        if best_model is None:
            model_name = "checkpoint.tar"
        else:
            model_name = "best.tar"
            min_loss = val_loss

        # e.g. results/setup_name/config_0/snapshots/checkpoint.tar
        # e.g. results/setup_name/config_0/snapshots/best.tar
        model_path = os.path.join(
            const.RESULT_PATH_PREFIX,
            self._epath,
            str(self._config['id']),
            "snapshots",
            model_name)

        utils.make_dir(model_path)

        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optim_state_dict': self._optim.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'min_loss': min_loss,
            }, model_path)

        return model_path

    def _init_criterion(self):
        gamma = 0
        if self._m_type == "DRNN" or self._m_type == "SRNN":
            gamma = self._config['gamma']

        if self._config['loss_fn'] == "Greedy":
            criterion = custom_models.GreedyLoss(
                self._device,
                self._metric,
                self._nsrc)

        elif self._config['loss_fn'] == "Min":
            criterion = MinLoss(
                self._device,
                self._metric,
                self._nsrc)

        elif self._config['loss_fn'] == "Discrim":
            criterion = custom_models.DiscrimLoss(
                self._device,
                self._metric,
                gamma)

        else:
            logging.error(
                "loss function %s is not supported",
                self._config['loss_fn'])


        return criterion

    def _init_optim(self):
        if self._config['optim'] == "SGD":
            optim = torch.optim.SGD(
                self._model.parameters(),
                lr=self._config['lr'],
                momentum=self._config['momentum'])

        elif self._config['optim'] == "Adam":
            optim = torch.optim.Adam(
                self._model.parameters(),
                lr=self._config['lr'],
                betas=(self._config['beta1'], self._config['beta2']),
                eps=self._config['epsilon'])

        else:
            logging.error(
                "optimizer %s is not supported",
                self._config['loss_fn'])

        return optim

    def _init_model(self):
        """create dataloader and model"""

        def get_loader(ds_name):
            """get loader based on train/test spec"""

            data_path = os.path.join(
                const.DATASET_PATH,
                self._epath.split('_')[0],
                f"{ds_name}.csv")
            logging.debug("%s", data_path)

            dataset = custom_dataset.MixtureDataset(
                num_sources=self._nsrc,
                data_path=data_path,
                transform=transform)

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self._config['batch_size'],
                shuffle=True)

            return dataloader

        if self._m_type == "CSALSTM":
            transform = custom_dataset.Wav2Spect()
            model = csa_lstm.CSALSTM(
                input_dim=const.N_FREQ,
                num_sources=self._nsrc,
                hidden_size=self._config['hidden_size']).to(self._device)

        elif self._m_type == "LSTM":
            transform = custom_dataset.Wav2Spect('Concat')
            model = custom_models.B1(
                input_dim=const.N_FREQ * 2,
                hidden_size=self._config['hidden_size'],
                num_sources=self._nsrc).to(self._device)

        elif self._m_type == "GAB":
            # transform = custom_dataset.ToTensor(
            #     size=(spect_shape['freq_range'], spect_shape['seq_len']))
            transform = custom_dataset.Wav2Spect()
            model = custom_models.LookToListenAudio(
                input_dim=const.N_FREQ,
                chan=self._config['chan'],
                num_sources=self._nsrc).to(self._device)

        elif self._m_type == "VTF":
            # transform = custom_dataset.Concat(
            #     size=(spect_shape['freq_range'], spect_shape['seq_len']),
            #     encdec=True)
            transform = custom_dataset.Wav2Spect('Concat', enc_dec=True)
            model = custom_transformer.make_model(
                input_dim=const.N_FREQ * 2,
                N=self._config['N'],
                d_model=self._config['d_model'],
                d_ff=self._config['d_ff'],
                h=self._config['h'],
                num_sources=self._nsrc,
                dropout=self._config['dropout']).to(self._device)

        # FIXME: seq_len is temporary
        elif self._m_type in ["STT1", "STT2", "STT3"]:
            # if self._m_type == "STT3":
            #     transform = custom_dataset.Wav2Spect('Separate', enc_dec=True)
            # else:
            transform = custom_dataset.Wav2Spect('Concat', enc_dec=True)

            model = custom_transformer.make_stt(
                input_dim=const.N_FREQ * 2,
                seq_len=460,
                stt_type=self._m_type,
                N=self._config['N'],
                d_model=self._config['d_model'],
                d_ff=self._config['d_ff'],
                h=self._config['h'],
                num_sources=self._nsrc,
                dropout=self._config['dropout']).to(self._device)

        elif self._m_type == "SRNN":
            # self._config['loss_fn'] = 'Discrim'
            self._config['optim'] = 'SGD'
            transform = custom_dataset.Wav2Spect('Concat')
            model = huang_srnn.SRNN(
                input_dim=const.N_FREQ * 2,
                num_sources=self._nsrc,
                hidden_size=self._config['hidden_size'],
                dropout=self._config['dropout']).to(self._device)

        elif self._m_type == "DRNN":
            # self._config['loss_fn'] = 'Discrim'
            # self._config['optim'] = 'SGD'
            transform = custom_dataset.Wav2Spect('Concat')
            model = huang_drnn.DRNN(
                input_dim=const.N_FREQ * 2,
                num_sources=self._nsrc,
                hidden_size=self._config['hidden_size'],
                k=1,
                dropout=self._config['dropout']).to(self._device)

        else:
            logging.error("%s is not supported", self._m_type)

        dataloader = {}
        dataloader['train'] = get_loader("train")
        dataloader['test'] = get_loader("val")

        logging.info(
            "%s using %s as criterion and %s as optimizer",
            self._m_type,
            self._config['loss_fn'],
            self._config['optim'])

        return dataloader, model

    def _load_cp(self):
        config_id = self._config['id']

        logging.info("loading checkpoint from configuration #%d...", config_id)

        checkpoint_path = os.path.join(
            const.RESULT_PATH_PREFIX,
            self._epath,
            str(config_id),
            "snapshots",
            "checkpoint.tar")

        # to continue training
        cp = torch.load(
            checkpoint_path,
            map_location=self._device)

        self._model.load_state_dict(cp['model_state_dict'])
        self._optim.load_state_dict(cp['optim_state_dict'])

        start_epoch = cp['epoch'] + 1
        min_loss = cp['min_loss']

        logging.info("checkpoint loaded")

        return start_epoch, min_loss


# if __name__ == '__main__':
#     main()
