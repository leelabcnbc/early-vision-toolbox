# coding=utf-8
"""utilities"""

from __future__ import division, print_function, absolute_import
import numpy as np

def make_2d_input_matrix(imgs):
    # make flat, if not ndarray or at least 3d.
    if (not isinstance(imgs, np.ndarray)) or imgs.ndim >= 3:
        imgs_array = np.array([im.ravel() for im in imgs])
    else:
        imgs_array = np.atleast_2d(imgs)

    return imgs_array

def transpose_c_and_f(w):
    """ convert row (C) major and column (F) major arrangement of input features back and forth.

    Parameters
    ----------
    w

    Returns
    -------

    """
    n_filter, filtersizesq = w.shape
    filtersize = np.int(np.sqrt(filtersizesq))
    assert filtersize ** 2 == filtersizesq, "filter must be originally square!"
    w = w.reshape(n_filter, filtersize, filtersize)
    w = np.transpose(w, (0, 2, 1))  # change between row and column.
    w = w.reshape(n_filter, filtersizesq)
    return w