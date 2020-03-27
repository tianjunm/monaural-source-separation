"""Sets up the model to be trained."""


import json
from baselines import csa_lstm
from stt import stt_aaai


def prepare_model(config_path, dataset_spec):
    with open(config_path) as f:
        config = json.load(f)

    model_name = config['model']['name']

    if model_name == 'cSA-LSTM':
        input_size = dataset_spec['input_shape'][-1]
        num_sources = config['dataset']['num_sources']
        hidden_size = config['model']['hidden_size']

        model = csa_lstm.CSALSTM(input_size, num_sources, hidden_size)

    return model
