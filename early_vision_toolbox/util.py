# coding=utf-8
"""utilities"""

from __future__ import division, print_function, absolute_import
import numpy as np
import h5py


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


class HDF5Iter(object):
    def __init__(self, filename, datasetlist):
        self.filename = filename
        self.datasetlist = datasetlist

    def __iter__(self):
        def fetch_one_dataset(filename, datasetname):
            f = h5py.File(filename, 'r')
            result = f[datasetname][:]
            f.close()
            return result

        return (fetch_one_dataset(self.filename, dataset) for dataset in self.datasetlist)

    def __len__(self):
        return len(self.datasetlist)


def make_hdf5_iter_class(filename, datasetlist):
    return HDF5Iter(filename, datasetlist)


def normalize_vector_inplace(x):
    """
    this inplace should be understood as best effort, not guaranteed.
    :param x:
    :return:
    """
    # be a little careful. actually I'm not sure if this will make sure it's really inplace.
    assert type(x) == np.ndarray, "must use vanilla np.ndarray to support inplace operation!"
    x -= x.mean()
    a = x.std()
    if a != 0:
        x /= a
    return x
