"""Sets up loss function / criterion for calculating loss."""


import json
import loss_functions


def prepare_loss_fn(config_path):
    with open(config_path) as f:
        config = json.load(f)

    model_name = config['model']['name']

    if model_name == "cSA-LSTM":
        loss_fn = loss_functions.CSALoss()
    # elif model_nme == "STT":
    #     loss_fn = NoOp()

    return loss_fn


# config_path = "/work/tianjunm/monaural-source-separation/experiments/hyperparameter/csa_lstm/000.json"