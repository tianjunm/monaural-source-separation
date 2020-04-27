"""Sets up loss function / criterion for calculating loss."""


import json
from . import loss_implementation


def prepare_loss_fn(dataset_spec, model_spec, no_pit=False):

    loss_function = model_spec['loss_function']

    if loss_function == 'CSALoss':
        loss_fn = loss_implementation.CSALoss(dataset_spec['config'])

    elif loss_function == 'CSALoss-NP':  # no permutation invariant training
        loss_fn = loss_implementation.CSALoss(dataset_spec['config'],
                                              no_pit=True)

    elif loss_function == 'Difference':
        loss_fn = loss_implementation.Difference(dataset_spec['config'])

    return loss_fn
