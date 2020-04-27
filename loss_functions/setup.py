"""Sets up loss function / criterion for calculating loss."""


import json
from . import loss_implementation
import torch.nn as nn


def prepare_loss_fn(dataset_spec, model_spec):

    loss_function = model_spec['loss_function']
    waveunet = (model_spec['model']['name'] in ['WAVE-U-NET'])
    if waveunet:
        loss_fn = nn.MSELoss()
    elif loss_function == 'CSALoss':
        loss_fn = loss_implementation.CSALoss(dataset_spec['config'], waveunet)

    elif loss_function == 'Difference':
        loss_fn = loss_implementation.Difference(dataset_spec['config'])

    return loss_fn


def classifier_prepare_loss_fn():
    return loss_implementation.CatLoss()
