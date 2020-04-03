"""Sets up the model to be trained."""


import json
from . import csa_lstm, blstm
# from stt import stt_aaai


def prepare_model(config, input_shape):
    model_name = config['model']['name']

    if model_name == 'cSA-LSTM':
        input_size = input_shape[-1]
        num_sources = config['dataset']['config']['num_sources']
        hidden_size = config['model']['config']['hidden_size']

        model = csa_lstm.CSALSTM(input_size, num_sources, hidden_size)

    if model_name == 'BLSTM':
        input_size = input_shape[-1]
        num_sources = config['dataset']['config']['num_sources']
        dmodel = config['model']['config']['dmodel']
        hidden_size = config['model']['config']['hidden_size']

        model = blstm.BLSTM(input_size, num_sources, dmodel, hidden_size)

    return model
