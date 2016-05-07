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