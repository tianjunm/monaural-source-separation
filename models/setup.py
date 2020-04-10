"""Sets up the model to be trained."""


import json
from . import csa_lstm, blstm, l2l, otf, stt
# from stt import stt_aaai


def get_input_size(input_shape):
    # TODO: currently the algorithm assumes we are only dealing with B2NM
    return input_shape[-1]


def prepare_model(dataset_spec, model_spec, input_shape):
    dataset_config = dataset_spec['config']

    model_name = model_spec['model']['name']
    model_config = model_spec['model']['config']

    if model_name == 'cSA-LSTM':
        input_size = get_input_size(input_shape)
        num_sources = dataset_config['num_sources']
        embed_dim = model_config['embed_dim']
        output_dim = model_config['output_dim']
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']

        model = csa_lstm.CSALSTM(input_size, num_sources, embed_dim,
                                 hidden_size, output_dim, num_layers)

    elif model_name == 'cSA-BLSTM':
        # input_size = get_input_size(input_shape)
        # dmodel = model_config['dmodel']
        # hidden_size = model_config['hidden_size']
        # num_sources = dataset_config['num_sources']

        # model = blstm.BLSTM(input_size, num_sources, dmodel, hidden_size)
        input_size = get_input_size(input_shape)
        num_sources = dataset_config['num_sources']
        embed_dim = model_config['embed_dim']
        output_dim = model_config['output_dim']
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']

        model = csa_lstm.CSALSTM(input_size, num_sources, embed_dim,
                                 hidden_size, output_dim, num_layers, True)

    elif model_name == 'SingleBLSTM':
        input_size = get_input_size(input_shape)
        dmodel = model_config['dmodel']
        hidden_size = model_config['hidden_size']
        num_sources = dataset_config['num_sources']

        model = blstm.SingleBLSTM(input_size, num_sources, dmodel, hidden_size)

    elif model_name == 'L2L':
        input_size = get_input_size(input_shape)
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']
        fc_dim = model_config['fc_dim']
        chan = model_config['chan']

        num_sources = dataset_config['num_sources']
        model = l2l.L2LAudio(input_size, num_sources, hidden_size, num_layers,
                             fc_dim, chan)

    elif model_name == 'Transformer':
        input_size = get_input_size(input_shape)
        num_heads = model_config['num_heads']
        num_layers = model_config['num_layers']
        dmodel = model_config['dmodel']
        hidden_size = model_config['hidden_size']
        dropout = model_config['dropout']
        num_sources = dataset_config['num_sources']

        model = otf.Transformer(input_size, num_sources, dmodel, num_heads,
                                num_layers, hidden_size, dropout)

    elif model_name == 'STT':
        input_size = get_input_size(input_shape)
        num_heads = model_config['num_heads']
        num_layers = model_config['num_layers']
        dmodel = model_config['dmodel']
        hidden_size = model_config['hidden_size']
        dropout = model_config['dropout']
        num_splits = model_config['num_splits']
        num_sources = dataset_config['num_sources']

        model = stt.Transformer(input_size, num_sources, dmodel, num_heads,
                                num_layers, hidden_size, dropout, num_splits)

    return model
