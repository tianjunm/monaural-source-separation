"""Sets up the model to be trained."""


import json
from . import csa_lstm, blstm, l2l, otf, stt_aaai, waveunet
# from stt import stt_aaai


def get_input_size(dataset_spec, input_shape):
    if dataset_spec['config']['input_dimensions'] == 'B2NM':
        return input_shape[-1]
    elif dataset_spec['config']['input_dimensions'] == 'BN(2M)':
        return input_shape[-1] // 2


def prepare_model(dataset_spec, model_spec, input_shape = None):
    dataset_config = dataset_spec['config']

    model_name = model_spec['model']['name']
    model_config = model_spec['model']['config']

    if model_name == 'cSA-LSTM':
        input_size = get_input_size(dataset_spec, input_shape)
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
        input_size = get_input_size(dataset_spec, input_shape)
        num_sources = dataset_config['num_sources']
        embed_dim = model_config['embed_dim']
        output_dim = model_config['output_dim']
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']

        model = csa_lstm.CSALSTM(input_size, num_sources, embed_dim,
                                 hidden_size, output_dim, num_layers, True)

    elif model_name == 'SingleBLSTM':
        input_size = get_input_size(dataset_spec, input_shape)
        dmodel = model_config['dmodel']
        hidden_size = model_config['hidden_size']
        num_sources = dataset_config['num_sources']

        model = blstm.SingleBLSTM(input_size, num_sources, dmodel, hidden_size)

    elif model_name == 'L2L':
        input_size = get_input_size(dataset_spec, input_shape)
        hidden_size = model_config['hidden_size']
        num_layers = model_config['num_layers']
        fc_dim = model_config['fc_dim']
        chan = model_config['chan']

        num_sources = dataset_config['num_sources']
        model = l2l.L2LAudio(input_size, num_sources, hidden_size, num_layers,
                             fc_dim, chan)

    elif model_name == 'Transformer':
        input_size = get_input_size(dataset_spec, input_shape)
        num_heads = model_config['num_heads']
        num_layers = model_config['num_layers']
        dmodel = model_config['dmodel']
        hidden_size = model_config['hidden_size']
        # dff = model_config['dff']
        dropout = model_config['dropout']
        num_sources = dataset_config['num_sources']

        model = otf.Transformer(input_size, num_sources, dmodel, num_heads,
                                num_layers, hidden_size, dropout)
        # model = stt_aaai.make_OTF(2 * input_size, num_heads, dmodel,
        #                           hidden_size, num_heads, num_sources, dropout)

    elif model_name == 'CNNTransformer':
        input_size = get_input_size(dataset_spec, input_shape)
        num_heads = model_config['num_heads']
        num_layers = model_config['num_layers']
        dmodel = model_config['dmodel']
        hidden_size = model_config['hidden_size']
        fc_dim = model_config['fc_dim']
        chan = model_config['chan']
        dropout = model_config['dropout']
        num_sources = dataset_config['num_sources']

        model = otf.CNNTransformer(input_size, num_sources, dmodel, num_heads,
                                   hidden_size, num_layers, fc_dim,
                                   chan, dropout)

    elif model_name == 'STT':
        input_size = get_input_size(dataset_spec, input_shape)
        num_heads = model_config['num_heads']
        num_layers = model_config['num_layers']
        dmodel = model_config['dmodel']
        hidden_size = model_config['hidden_size']
        dropout = model_config['dropout']
        # num_splits = model_config['num_splits']
        c_out = model_config['c_out']
        d_out = model_config['d_out']
        ks2 = model_config['ks2']
        res_size = model_config['res_size']
        num_sources = dataset_config['num_sources']

        # model = stt.Transformer(input_size, num_sources, dmodel, num_heads,
        #                         num_layers, hidden_size, dropout, num_splits)
        model = stt_aaai.make_STT(2 * input_size, N=num_heads, d_model=dmodel,
                                  d_ff=hidden_size, h=num_heads, h_t=2,
                                  num_sources=num_sources, dropout=dropout,
                                  c_out=c_out, d_out=d_out, ks2=ks2,
                                  res_size=res_size)
    elif model_name=='WAVE-U-NET':
        layer_channels = [model_config['channels']*2**i for i in range(0, model_config['levels'])]
        in_channels = dataset_config['in_channels']
        num_cats = dataset_config['num_sources']
        # ALL CATS???, target_output_size, 
        kernel_size =  model_config['kernel_size']
        pre_depth = model_config['pre_depth']
        post_depth = model_config['post_depth']
        depth = model_config['depth']
        stride = model_config['stride']
        stride_change = model_config['stride_change']
        target_output_size = dataset_config['mixture_duration']* dataset_config["sample_rate"]
        model = waveunet.WaveUNet(in_channels, layer_channels, in_channels, 
                num_cats, kernel_size, target_output_size, pre_depth, 
                post_depth, depth, stride, stride_change)

    return model
