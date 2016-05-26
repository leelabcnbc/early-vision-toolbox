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
from .legacy import v1s_misc, v1s_funcs
from scipy.misc import fromimage
from itertools import product

conv = sp.signal.fftconvolve


class V1Like(object):
    def __init__(self, pars_to_update):
        pass
        self._filt_l = None

    def _get_gabor_filters(self, params, legacy=False):
        if legacy:
            filt_l, _ = v1s_misc._get_gabor_filters(params)
        else:
            fh, fw = params['kshape']
            orients = params['orients']
            freqs = params['freqs']
            phases = params['phases']
            assert fh == fw, "filter kernel must be square!"

            nf = len(orients) * len(freqs) * len(phases)
            fbshape = nf, fh, fw
            sigma = fh / 5.
            filt_l = [gabor2d(sigma, freq, orient,
                              phase, fh) for freq, orient, phase in product(freqs, orients, phases)]
            filt_l = np.array(filt_l)
            assert filt_l.shape == fbshape

        self._filt_l = filt_l


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
            # should be set to 1 or bigger when debugging.
            'sep_threshold': .9,
            'max_component': 100000,
            'fix_bug': True,  # whether fixing separated convolution bug.
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


def _part_generate_repr(img, steps, params, featsel, filt_l, legacy=True):
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
    response = response.astype(np.float32, copy=False)

    if 'preproc_lowpass' in steps:
        response = _preproc_lowpass(response, params['preproc']['lsum_ksize'], legacy)

    if 'normin' in steps:
        response = _normin(response, params['normin'], legacy)

    if 'filter' in steps:
        response = _filter(response, filt_l, legacy)

    # make sure it's 3d.
    assert response.ndim == 3, "must have a 3d response array"

    if 'activ' in steps:
        response = response.clip(params['activ']['minout'], params['activ']['maxout'])

    if 'normout' in steps:
        response = _normin(response, params['normout'], legacy)

    if 'dimr' in steps:
        pass

    # handle additional features.

    return response


def _filter(im, filt_l, legacy=False):
    assert im.dtype == np.float32
    if legacy:
        result = v1s_funcs.v1s_filter(im, filt_l)
    else:
        # do filtering using one line.
        # DON't reverse the order of filters to get same result as in the original implementation.
        # not reversing filt_l is correct behavior.
        result_padded = conv(im[np.newaxis, :, :], filt_l, mode='full').astype(np.float32)
        _, current_h, current_w = result_padded.shape
        new_h, new_w = im.shape

        h_slice = slice((current_h - new_h) // 2, (current_h - new_h) // 2 + new_h)
        w_slice = slice((current_w - new_w) // 2, (current_w - new_w) // 2 + new_w)
        result = result_padded.transpose((1, 2, 0))[h_slice, w_slice]

        # this is the one used to select the shape in scipy.signal.fftconvolve
        # this is true for scipy 0.17.0

        # def _centered(arr, newsize):
        #     # Return the center newsize portion of the array.
        #     newsize = asarray(newsize)
        #     currsize = array(arr.shape)
        #     startind = (currsize - newsize) // 2
        #     endind = startind + newsize
        #     myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
        #     return arr[tuple(myslice)]
    assert result.dtype == np.float32
    return result


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
    assert im.dtype == np.float32
    mode = 'same'
    if lsum_ksize is not None:
        if legacy:
            k = np.ones(lsum_ksize, 'f') / lsum_ksize
            im = conv(conv(im, k[np.newaxis, :], mode), k[:, np.newaxis], mode).astype(np.float32)
        else:
            im = uniform_filter(im, size=lsum_ksize, mode='constant')

    normalize_vector_inplace(im)
    assert im.dtype == np.float32
    return im


def _normin(im, params, legacy=False):
    assert im.dtype == np.float32
    if legacy:
        result = v1s_funcs.v1s_norm(im[:, :, np.newaxis], **params)[:, :, 0]
    else:
        result = local_normalization(im, **params)
    assert result.dtype == np.float32
    return result


def local_normalization(im, kshape, threshold):
    # a more efficient version of input normalization.
    kh, kw = kshape
    ksize = kh * kw
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

    hssq = ksize * uniform_filter(np.square(im), size=(kh, kw), mode='constant')[h_slice, w_slice]
    hssq_normed = hssq - ksize * np.square(local_mean)
    np.putmask(hssq_normed, hssq_normed < 0, 0)
    h_div = np.sqrt(hssq_normed) + eps
    np.putmask(h_div, h_div < (threshold + eps), 1)
    return im_without_mean / h_div


def _dimr(im, lsum_ksize, outshape, legacy=False):
    assert im.dtype == np.float32
    assert im.ndim == 3
    if legacy:
        result = v1s_funcs.v1s_dimr(im, lsum_ksize, outshape)
    else:
        assert lsum_ksize % 2 == 1, "must be odd size!"
        filtered_im = lsum_ksize * lsum_ksize * uniform_filter(im, size=(lsum_ksize, lsum_ksize, 1),
                                                               mode='constant')
        inh, inw = filtered_im.shape[:2]
        outh, outw = outshape
        hslice = np.round(np.linspace(0, inh-1, outh)).astype(np.int16)
        wslice = np.round(np.linspace(0, inw-1, outw)).astype(np.int16)
        result = filtered_im[np.ix_(hslice,wslice)]
    assert result.dtype == np.float32 and result.ndim == 3
    return result
