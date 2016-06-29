"""functions for usually used statistics in early visual cortex research"""

from __future__ import division, unicode_literals, print_function, absolute_import
import numpy as np
from numpy.fft import fftshift, fft2
from scipy.signal import hamming
from skimage.color import rgb2gray


def sparsity_measure_rolls(x, normalize=True, clip=True):
    """ return the sparsity measure.

    Parameters
    ----------
    x : numpy.ndarray
        (implicitly flattened) vector to compute sparsity. it will be copied and converted to ``np.float64``.
    normalize : bool
        whether divide by ``1.0-1.0/x.size`` (``True``) or ``1.0`` (``False``).
    clip : bool
        whether make negative values zero or not.

    Returns
    -------
    a single ``np.float64`` number (usually) between 0 and 1 for sparsity. Larger means more sparse.

    Notes
    -----
    #. `clip` set to ``True`` is consistent with [2]_ when dealing with "response sparsity"
       (defined as ``raw response - baseline response``, and in [2]_,
       negative values obtained in this way are clipped to 0).
    #. `normalize` is ``True`` in [1]_  and ``False`` in [2]_.


    References
    ----------

    .. [1] Vinje, W. E., & Gallant, J. L. (2000). Sparse Coding and Decorrelation in
       Primary Visual Cortex During Natural Vision. Science 287(5456), 1273-1276.
       http://doi.org/10.1126/science.287.5456.1273
    .. [2] Rolls, E. T., & Tovee, M. J. (1995). Sparseness of the neuronal representation of stimuli in
       the primate temporal visual cortex. Journal of Neurophysiology, 73(2), 713-726.

    """

    # this should return a copy
    x_new = x.astype(np.float64)
    assert not np.may_share_memory(x_new, x)
    # clip
    if clip:
        x_new[x_new < 0] = 0

    # compute sparsity.
    divisor = np.mean(x_new * x_new)
    if divisor == 0:
        sparsity_raw = 1.0
        # print('hit!')
        # print(np.all(x==0))
    else:
        sparsity_raw = ((np.mean(x_new)) ** 2) / divisor

    if normalize:
        norm_factor = 1.0 - 1.0 / x_new.size
    else:
        norm_factor = 1.0

    return (1.0 - sparsity_raw) / norm_factor


def expoenential_fit(x, remove_zero=False, clip_lower=0.0, clip_upper=100.0, check=True):
    """return scale parameter of an exponential distribution.

    Parameters
    ----------
    x
    remove_zero : bool
        removing zero or not. By default, keep all nonnegative elements, rather than strictly positive elements.
            this option can be useful to show the abundance of zero in exponential fit.
    clip_lower : float
        lower percentile
    clip_upper : float
        upper percentile
    Returns
    -------
    the scale parameter.
    """
    x = x.ravel().astype(np.float64)
    if remove_zero:
        x = x[x > 0]
    else:
        x = x[x >= 0]
    if x.size == 0:
        if check:
            raise AssertionError('you must have things left!')
        else:
            return np.float64(0)
    x.clip(min=np.percentile(x, clip_lower), max=np.percentile(x, clip_upper))
    scale_this = x.mean()
    if check:
        assert scale_this > 0, "scale must be greater than zero"
    return scale_this


def power_spectrum(im, window=True, remove_dc=True):
    assert np.all(np.isfinite(im)), 'invalid numbers in input!'
    assert im.ndim == 2 or im.ndim == 3
    flag_2d = False
    if im.ndim == 2:
        flag_2d = True
        im = im[np.newaxis, :, :]
    n, h, w = im.shape
    assert w == h
    if window:  # add hamming window.
        # create hamming window, by dot product of 2 1-d windows.
        hamming_1d = hamming(w, sym=False)  # `periodic` in MATLAB equivalent.
        hamming_2d = np.dot(hamming_1d[:, np.newaxis], hamming_1d[np.newaxis, :])
        im = im * hamming_2d  # using broadcasting
    if remove_dc:
        im = im - np.mean(im, axis=(1, 2), keepdims=True)
    fft_result = fft2(im)
    raw_spectrum = abs(fftshift(fft_result, axes=(1, 2))) ** 2
    assert raw_spectrum.shape == im.shape
    if flag_2d:  # get back 2d
        assert n == 1
        raw_spectrum = raw_spectrum[0]
    return raw_spectrum


def power_spectrum_batch(ims, remove_dc=True, unit_std=False, pars_dict_inner=None):
    if pars_dict_inner is None:
        pars_dict_inner = dict()
    # compute spectral signature!!!
    ims_bw = np.asarray([rgb2gray(im) for im in ims])
    # then remove dc.
    assert ims_bw.ndim == 3
    if remove_dc:
        ims_bw = ims_bw - np.mean(ims_bw, axis=(1, 2), keepdims=True)
    if unit_std:
        ims_bw = ims_bw / np.std(ims_bw, axis=(1, 2), keepdims=True)
    n_im, h, w = ims_bw.shape
    assert h == w
    return power_spectrum(ims_bw, **pars_dict_inner)
