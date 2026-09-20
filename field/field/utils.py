"""Miscellaneous utils."""
import glob
import os
from pathlib import Path
import re
import subprocess
import signal
import fnmatch
from contextlib import contextmanager
import numpy as np


@contextmanager
def _dummy_with():
    """Dummy statement."""
    yield

def recursive_insensitive_glob(path, pattern, return_relative=False):
    """Find files matching pattern ignoring case-style."""
    found = []
    reg_expr = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    for root, _, files in os.walk(path, topdown=True):
        for f in files:
            if re.match(reg_expr, f):
                f_path = os.path.join(root, f)
                found.append(
                    f_path if not return_relative else os.path.relpath(f_path, start=path)
                )
    return found

def get_single_path(dir_path, filename, logger=None):
    """Find a file withihn the directory. Raise error if multiple files found."""
    files = recursive_insensitive_glob(dir_path, filename)
    if not files:
        if logger is not None:
            logger.warning("{} file was not found.".format(filename))
        return None
    if len(files) > 1:
        raise ValueError('Directory {} contains multiple {} files.'.format(dir_path, filename))
    return files[0]

def rolling_window(a, strides):
    """Rolling window without overlays."""
    strides = np.asarray(strides)
    output_shape = tuple((np.array(a.shape) - strides)//strides + 1) + tuple(strides)
    output_strides = tuple(strides * np.asarray(a.strides)) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=output_shape, strides=output_strides)

def get_multout_paths(path_to_results, basename, extension_regexp=r'X\d+'):
    r"""Searches for multout files in a RESULT folder near .DATA file.

    Parameters
    ----------
    path_to_results: str
        Path to the folder with precomputed results of hydrodynamical simulation.
    basename: str
        Model Name.
    extension_regexp: str, optional
        Regexp to match a file extension, by default r'X\d+'.

    Returns
    -------
    paths: list or None
        List of the paths found. None else.
    """
    multout_paths = []
    for filename in glob.iglob(os.path.join(path_to_results, '**', basename+'.*'), recursive=True):
        _, ext = os.path.splitext(filename)
        if re.fullmatch(extension_regexp, ext[1:]):
            multout_paths.append(filename)
    return sorted(multout_paths) if multout_paths else None
