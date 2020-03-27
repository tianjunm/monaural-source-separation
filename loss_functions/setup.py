"""Sets up loss function / criterion for calculating loss."""


import json
from . import loss_implementation


def prepare_loss_fn(config):
       
    model_name = config['model']['name']

    if model_name == "cSA-LSTM":
        loss_fn = loss_implementation.CSALoss()
    # elif model_nme ==implementation
    #     loss_fn = NoOp()

    return loss_fn


# config_path = "/work/tianjunm/monaural-source-separation/experiments/hyperparameter/csa_lstm/000.json"