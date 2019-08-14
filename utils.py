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
