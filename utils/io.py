"""Utility functions are defined here."""
import errno
import os


def make_dir(path):
    """creates directory if it does not exist"""
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_experiment_title(setup):
    """Gets a name for the experiment given setup.

    e.g. t1-2s-5c_euclidean_DETF_20190101

    """

    ds_type = setup['mixing_type']
    nsrc = setup['num_sources']
    ncat = setup['num_classes']
    metric = setup['metric']
    m_type = setup['model_type']
    eid = setup['experiment_id']

    return (f"t{ds_type}-{nsrc}s-{ncat}c_"
            f"{metric}_{m_type}_{eid}")

def timer():
    pass

def early_stopping():
    pass
