"""remake of DiCarlo's v1like model <https://github.com/npinto/v1s>"""

from __future__ import division, print_function, absolute_import
import numpy as np
import scipy as sp
import imagen as ig
from joblib import Parallel, delayed
from numpy.linalg import norm

def fastnorm(x):
    """ Fast Euclidean Norm (L2)
    This version should be faster than numpy.linalg.norm if
    the dot function uses blas.
    Inputs:
      x -- numpy array
    Output:
      L2 norm from 1d representation of x

    """
    xv = x.ravel()
    return sp.dot(xv, xv) ** (1 / 2.)


def gabor2d_legacy(gsw, gsh, gx0, gy0, wfreq, worient, wphase, shape):
    """ Generate a gabor 2d array

    Inputs:
      gsw -- standard deviation of the gaussian envelope (width)
      gsh -- standard deviation of the gaussian envelope (height)
      gx0 -- x indice of center of the gaussian envelope
      gy0 -- y indice of center of the gaussian envelope
      wfreq -- frequency of the 2d wave
      worient -- orientation of the 2d wave
      wphase -- phase of the 2d wave
      shape -- shape tuple (height, width)
    Outputs:
      gabor -- 2d gabor with zero-mean and unit-variance
    """

    height, width = shape
    y, x = sp.mgrid[0:height, 0:width]

    X = x * sp.cos(worient) * wfreq   # this is wrong. Should be better x = x-width/2 to make the phase easier to use.
    Y = y * sp.sin(worient) * wfreq   # this is wrong. Should be better y = y-width/2 to make the phase easier to use.

    env = sp.exp(-.5 * (((x - gx0) ** 2. / gsw ** 2.) + ((y - gy0) ** 2. / gsh ** 2.)))
    wave = sp.exp(1j * (2 * sp.pi * (X + Y) + wphase))  # X+Y is rotated X or Y in imagen.
    gabor = sp.real(env * wave)
    gabor -= gabor.mean()
    gabor /= fastnorm(gabor)

    return gabor


def gabor2d(sigma, wfreq, worient, wphase, size, normalize=True):
    """specific gabor function with (roughly) compatible interface with old V1 like model.
    It will produce roughly the same image, except for orientation rotation direction, and phase.

    Parameters
    ----------
    sigma : float
        sigma of gaussian envelope in pixels.
    wfreq : float
        frequency of cosine. in the unit of cycle / per pixel.
    worient : float
        orientation. in radian
    wphase : float
        phase. zero phase means having maximum value in the central of the image.
    size : int
        size of patch. final result would be size x size.
    normalize : bool
        whether remove DC and make unit variance. by default True.

    Returns
    -------
    a 2d ndarray of shape ``(size, size)``
    """
    gabor = ig.Gabor(frequency=wfreq*size, xdensity=size, ydensity=size, size=2*sigma/size,
                     orientation=worient, phase=wphase)()
    if normalize:
        gabor -= gabor.mean()
        gabor /= norm(gabor)

    return gabor
