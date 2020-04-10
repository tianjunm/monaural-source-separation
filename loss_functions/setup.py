"""Sets up loss function / criterion for calculating loss."""


import json
from . import loss_implementation


def prepare_loss_fn(model_spec):

    loss_function = model_spec['loss_function']

    if loss_function == 'CSALoss':
        loss_fn = loss_implementation.CSALoss()

    elif loss_function == 'Difference':
        loss_fn = loss_implementation.Difference()

    return loss_fn
