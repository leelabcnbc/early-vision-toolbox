"""remake of DiCarlo's v1like model <https://github.com/npinto/v1s>"""

from __future__ import division, print_function, absolute_import
import numpy as np
import scipy as sp
import scipy.signal
import imagen as ig
from joblib import Parallel, delayed
from numpy.linalg import norm
from copy import deepcopy
from skimage.color import rgb2gray
from scipy.ndimage.filters import uniform_filter
from skimage.transform import resize
from early_vision_toolbox.util import normalize_vector_inplace
from PIL import Image
from skimage import img_as_float
from .legacy import v1s_funcs, v1s_math
from scipy.misc import fromimage

conv = sp.signal.fftconvolve


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
    gabor = ig.Gabor(frequency=wfreq * size, xdensity=size, ydensity=size, size=2 * sigma / size,
                     orientation=worient, phase=wphase)()
    if normalize:
        gabor -= gabor.mean()
        gabor /= norm(gabor)

    return gabor


def default_pars():
    """default parameter for v1like model. here I copied from simple_plus.

    Returns
    -------

    """
    norients = 16
    orients = [o * np.pi / norients for o in range(norients)]
    divfreqs = [2, 3, 4, 6, 11, 18]
    freqs = [1. / n for n in divfreqs]
    phases = [0]

    # this is something new.
    # there are 6 steps. and I can turn on/off these steps. potentially.
    steps = {'preproc_resize',
             'preproc_lowpass',
             'normin',
             'filter', 'activ',
             'normout', 'dimr'}

    # dict with all representation parameters
    representation = {

        # - preprocessing
        # prepare images before processing
        'preproc': {
            # resize input images by keeping aspect ratio and fix the biggest edge
            'max_edge': 150,
            # kernel size of the box low pass filter
            'lsum_ksize': 3,
        },

        # - input local normalization
        # local zero-mean, unit-magnitude
        'normin': {
            # kernel shape of the local normalization
            'kshape': (3, 3),
            # magnitude threshold
            # if the vector's length is below, it doesn't get resized
            'threshold': 1.0,
        },

        # - linear filtering
        'filter': {
            # kernel shape of the gabors
            'kshape': (43, 43),
            # list of orientations
            'orients': orients,
            # list of frequencies
            'freqs': freqs,
            # list of phases
            'phases': phases,
            # threshold (variance explained) for the separable convolution
            'sep_threshold': .9,
        },

        # - simple non-linear activation
        'activ': {
            # minimum output (clamp)
            'minout': 0,
            # maximum output (clamp)
            'maxout': 1,
        },

        # - output local normalization
        'normout': {
            # kernel shape of the local normalization
            'kshape': (3, 3),
            # magnitude threshold
            # if the vector's length is below, it doesn't get resized
            'threshold': 1.0,
        },

        # - dimension reduction
        'dimr': {
            # kernel size of the local sum (2d slice)
            'lsum_ksize': 17,
            # fixed output shape (only the first 2 dimensions, y and x)
            'outshape': (30, 30),
        },
    }

    featsel = {
        # Include representation output ? True or False
        'output': True,

        # Include grayscale values ? None or (height, width)
        'input_gray': (100, 100),
        # Include color histograms ? None or nbins per color
        'input_colorhists': None,
        # Include input norm histograms ? None or (division, nfeatures)
        'normin_hists': None,
        # Include filter output histograms ? None or (division, nfeatures)
        'filter_hists': None,
        # Include activation output histograms ? None or (division, nfeatures)
        'activ_hists': (2, 10000),
        # Include output norm histograms ? None or (division, nfeatures)
        'normout_hists': (1, 10000),
        # Include representation output histograms ? None or (division, nfeatures)
        'dimr_hists': (1, 10000),
    }

    return deepcopy({'steps': steps,
                     'representation': representation,
                     'featsel': featsel})


def _part_generate_repr(img, steps, params, featsel, legacy=True):
    """ rewrite of samed named function in V1 like model.

    Parameters
    ----------
    img
    steps
    params
    featsel

    Returns
    -------

    """

    assert img.ndim == 2 or img.ndim == 3, "must be gray or RGB"

    if 'preproc_resize' in steps:
        img = _preproc_resize(img, params['preproc']['max_edge'], legacy)

    response = img_as_float(img, force_copy=True)
    # convert image into gray scale, 2 dim array.
    if response.ndim == 3:
        if not legacy:
            response = rgb2gray(response)
        else:  # use original formula.
            response = 0.2989 * response[:, :, 0] + 0.5870 * response[:, :, 1] + 0.1140 * response[:, :, 2]

    assert response.ndim == 2, "must be two channels!"

    if 'preproc_lowpass' in steps:
        response = _preproc_lowpass(response, params['preproc']['lsum_ksize'], legacy)

    if 'normin' in steps:
        response = _normin(response, params['normin'], legacy)

    if 'filter' in steps:
        pass

    if 'activ' in steps:
        pass

    if 'normout' in steps:
        pass

    if 'dimr' in steps:
        pass

    return response


def _preproc_resize(im, max_edge, legacy=False, order=1):
    """

    Parameters
    ----------
    im
    max_edge
    legacy
    order

    Returns
    -------

    Notes
    -----
    skimage and PIL can result in pretty different interpolation results.
    seems that skimage can give sharper results, but PIL give a result more similar to that in MATLAB.
    looks to me there's a pre smoothing in MATLAB and PIL
    see <https://github.com/scikit-image/scikit-image/issues/1035>
    and <http://nickyguides.digital-digest.com/bilinear-vs-bicubic.htm>

    for convenience, I simply use bilinear by default (order = 1)
    """
    # -- resize so that the biggest edge is max_edge (keep aspect ratio)

    if legacy:
        assert im.dtype == np.uint8, "only support uint8 for legacy!"

    ih, iw = im.shape[:2]
    if iw > ih:
        new_iw = max_edge
        new_ih = int(round(1. * max_edge * ih / iw))
    else:
        new_iw = int(round(1. * max_edge * iw / ih))
        new_ih = max_edge
    if not legacy:
        return resize(im, (new_ih, new_iw), order=order, preserve_range=True,
                      mode='edge')
    else:
        # check <https://github.com/python-pillow/Pillow/blob/master/Tests/test_numpy.py>
        # on how to do conversion between Image and ndarray
        # well I think the documentation of Pillow is bad (maybe I don't read carefully).
        # also, I haven't tested fromarray for floating point images.
        im = Image.fromarray(im)
        im = im.resize((new_iw, new_ih), Image.BICUBIC)
        return fromimage(im)


def _preproc_lowpass(im, lsum_ksize, legacy=False):
    mode = 'same'
    if lsum_ksize is not None:
        if legacy:
            k = np.ones((lsum_ksize), 'f') / lsum_ksize
            im = conv(conv(im, k[np.newaxis, :], mode), k[:, np.newaxis], mode)
        else:
            im = uniform_filter(im, size=lsum_ksize, mode='constant')

    normalize_vector_inplace(im)
    return im


def _normin(im, params, legacy=False):
    if legacy:
        return v1s_funcs.v1s_norm(im[:, :, np.newaxis], **params)[:, :, 0]
    else:
        return local_normalization(im, **params)


def local_normalization(im, kshape, threshold):
    # a more efficient version of input normalization.
    kh, kw = kshape
    ksize = kh*kw
    eps = 1e-5
    assert im.ndim == 2
    assert kh % 2 == 1 and kw % 2 == 1, "kernel must have odd shape"

    h_slice = slice(kh // 2, -(kh // 2))
    w_slice = slice(kw // 2, -(kw // 2))

    # first compute local mean
    local_mean = uniform_filter(im, size=(kh, kw), mode='constant')[h_slice, w_slice]
    im_without_mean = im[h_slice, w_slice] - local_mean  # this is ``hnum`` in the original implementation.

    # then compute local divisor
    # actually, this divisor is not the sum of square of ``im_without_mean``,
    # that is, it's not $\sum_{i \in N(c)} (x_i-\avg{x_i})^2), where N(c) is some neighborhood of location c, and
    # $\avg{x_i}$ is the local mean with center at i.
    # instead, it's $\sum_{i \in N(c)} (x_i-\avg{x_c})^2). so there's a different offset mean subtracted,
    # when contributing to the normalization of different locations.
    # the above is explanation of line ``val = (hssq - (hsum**2.)/size)`` in the original code.
    # I don't know whether this is intended or not. But if we want to stick to the first definition,
    # then we need more padding, in order to compute $\avg{x_i}$ when i is at the border (which requires some padding)

    hssq = ksize*uniform_filter(np.square(im), size=(kh, kw), mode='constant')[h_slice, w_slice]
    hssq_normed = hssq - ksize*np.square(local_mean)
    np.putmask(hssq_normed, hssq_normed < 0, 0)
    h_div = np.sqrt(hssq_normed) + eps
    np.putmask(h_div, h_div < (threshold + eps), 1)
    return im_without_mean/h_div
