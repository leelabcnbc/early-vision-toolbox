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
import scipy.misc
from scipy.misc import fromimage
from itertools import product
from functools import partial

conv = sp.signal.fftconvolve


# conv = sp.signal.convolve
# conv_spatial = sp.signal.convolve

class V1Like(object):
    def __init__(self, pars_baseline='simple', pars_to_update=None, n_jobs=4, legacy=False, debug=False):
        self._filt_l = None
        self.pars = default_pars(pars_baseline)
        # then update.

        # then get gabor.
        self._filt_l = _get_gabor_filters(self.pars['representation']['filter'], legacy)
        self._n_jobs = n_jobs
        self.legacy = legacy
        self.debug = debug
        self.func_to_apply = partial(_part_generate_repr, steps=self.pars['steps'],
                                     params=self.pars['representation'],
                                     featsel=self.pars['featsel'],
                                     filt_l=self._filt_l, legacy=legacy, debug=debug)

    def reload_filters(self, filt_l):
        self._filt_l = filt_l
        self.func_to_apply = partial(_part_generate_repr, steps=self.pars['steps'],
                                     params=self.pars['representation'],
                                     featsel=self.pars['featsel'],
                                     filt_l=self._filt_l, legacy=self.legacy, debug=self.debug)

    def fit(self, X, y):
        """defined to match sklearn Pipeline protocol.

        Parameters
        ----------
        X

        Returns
        -------

        """
        return self   # as required by sklearn.

    def transform(self, X, n_jobs=None):
        """

        Parameters
        ----------
        X : iterable of images.

        Returns
        -------

        """
        print("\nworking on problem of size {}".format(len(X)))
        if n_jobs is None:
            _n_jobs = self._n_jobs
        else:
            _n_jobs = n_jobs
        result = Parallel(n_jobs=_n_jobs, verbose=5, max_nbytes=None)(delayed(self.func_to_apply)(x) for x in X)
        result = np.array(result)
        assert result.ndim == 2 and result.dtype == np.float32
        return result


def _get_gabor_filters(params, legacy=False):
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
    return filt_l


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


def default_pars(type='simple_plus'):
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
            'fix_bug': False,  # whether fixing separated convolution bug.
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

    if type == 'simple_plusplus_2nd_scale':
        representation['preproc']['max_edge'] = 75

    if type == 'simple_plus':
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
    elif type == 'simple':
        featsel = {
            # Include representation output ? True or False
            'output': True,

            # Include grayscale values ? None or (height, width)
            'input_gray': None,
            # Include color histograms ? None or nbins per color
            'input_colorhists': None,
            # Include input norm histograms ? None or (division, nfeatures)
            'normin_hists': None,
            # Include filter output histograms ? None or (division, nfeatures)
            'filter_hists': None,
            # Include activation output histograms ? None or (division, nfeatures)
            'activ_hists': None,
            # Include output norm histograms ? None or (division, nfeatures)
            'normout_hists': None,
            # Include representation output histograms ? None or (division, nfeatures)
            'dimr_hists': None,
        }
    elif type == 'simple_plusplus_2nd_scale':
        featsel = {
            # Include representation output ? True or False
            'output': True,

            # Include grayscale values ? None or (height, width)
            'input_gray': (37, 37),
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
    else:
        raise NotImplementedError('not supported pars type!')

    return deepcopy({'steps': steps,
                     'representation': representation,
                     'featsel': featsel})


def _part_generate_repr(img, steps, params, featsel, filt_l, legacy=True, debug=False):
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
    orig_imga = img.copy()
    if debug:
        print("orig_imga, mean {}, std {}".format(orig_imga.mean(), orig_imga.std()))
    # convert image into gray scale, 2 dim array.
    response = img
    if response.ndim == 3:
        response = img.astype(np.float32) / 255.0
        response = 0.2989 * response[:, :, 0] + 0.5870 * response[:, :, 1] + 0.1140 * response[:, :, 2]

    assert response.ndim == 2, "must be two channels!"
    response = response.astype(np.float32, copy=True)
    if debug:
        print("imga0, mean {}, std {}".format(response.mean(), response.std()))

    if 'preproc_lowpass' in steps:
        response = _preproc_lowpass(response, params['preproc']['lsum_ksize'], legacy)
    imga0 = response.copy()
    if debug:
        print("imga0 normalized, mean {}, std {}".format(imga0.mean(), imga0.std()))

    if 'normin' in steps:
        response = _normin(response, params['normin'], legacy)
    imga1 = response.copy()
    if debug:
        print("imga1, shape{}, mean {}, std {}".format(imga1.shape, imga1.mean(), imga1.std()))

    if 'filter' in steps:
        response = _filter(response, filt_l, legacy)
    imga2 = response.copy()
    if debug:
        print("imga2, shape {}, mean {}, std {}".format(imga2.shape, imga2.mean(), imga2.std()))

    # make sure it's 3d.
    assert response.ndim == 3, "must have a 3d response array"

    if 'activ' in steps:
        response = response.clip(params['activ']['minout'], params['activ']['maxout'])
    imga3 = response.copy()
    if debug:
        print("imga3, shape {}, mean {}, std {}".format(imga3.shape, imga3.mean(), imga3.std()))

    if 'normout' in steps:
        response = _normin(response, params['normout'], legacy)
    imga4 = response.copy()
    if debug:
        print("imga4, shape {}, mean {}, std {}".format(imga4.shape, imga4.mean(), imga4.std()))

    if 'dimr' in steps:
        response = _dimr(response, params['dimr']['lsum_ksize'], params['dimr']['outshape'], legacy)
    if debug:
        print("output, shape {}, mean {}, std {}".format(response.shape, response.mean(), response.std()))
    images = {'imga0': imga0,
              'imga1': imga1,
              'imga2': imga2,
              'imga3': imga3,
              'imga4': imga4,
              'orig_imga': orig_imga}

    # handle additional features.
    # pure legacy functions.
    fvector = handle_feature_selection(response, images, featsel)
    return fvector


def handle_feature_selection(output, images, featsel):
    feat_l = []

    # include representation output ?
    f_output = featsel['output']
    if f_output:
        feat_l.append(output.ravel())

    # include grayscale values ?
    f_input_gray = featsel['input_gray']
    if f_input_gray is not None:
        shape = f_input_gray
        feat_l.append(scipy.misc.imresize(images['imga0'], shape).ravel().astype(np.float32, copy=False))

    # include color histograms ?
    f_input_colorhists = featsel['input_colorhists']
    if f_input_colorhists is not None:
        nbins = f_input_colorhists
        colorhists = np.empty((3, nbins), np.float32)
        orig_imga = images['orig_imga']
        if orig_imga.ndim == 3:
            for d in range(3):
                h = np.histogram(orig_imga[:, :, d].ravel(),
                                 bins=nbins,
                                 range=[0, 255])
                binvals = h[0].astype(np.float32)
                colorhists[d] = binvals
        else:
            h = np.histogram(orig_imga[:, :].ravel(),
                             bins=nbins,
                             range=[0, 255])
            binvals = h[0].astype(np.float32)
            colorhists[:] = binvals  # use broadcasting.
        feat_l.append(colorhists.ravel())

    # include input norm histograms ?
    f_normin_hists = featsel['normin_hists']
    if f_normin_hists is not None:
        division, nfeatures = f_normin_hists
        feat_l.append(v1s_funcs.rephists(images['imga1'], division, nfeatures))

    # include filter output histograms ?
    f_filter_hists = featsel['filter_hists']
    if f_filter_hists is not None:
        division, nfeatures = f_filter_hists
        feat_l.append(v1s_funcs.rephists(images['imga2'], division, nfeatures))

    # include activation output histograms ?
    f_activ_hists = featsel['activ_hists']
    if f_activ_hists is not None:
        division, nfeatures = f_activ_hists
        feat_l.append(v1s_funcs.rephists(images['imga3'], division, nfeatures))

    # include output norm histograms ?
    f_normout_hists = featsel['normout_hists']
    if f_normout_hists is not None:
        division, nfeatures = f_normout_hists
        feat_l.append(v1s_funcs.rephists(images['imga4'], division, nfeatures))

    # include representation output histograms ?
    f_dimr_hists = featsel['dimr_hists']
    if f_dimr_hists is not None:
        division, nfeatures = f_dimr_hists
        feat_l.append(v1s_funcs.rephists(output, division, nfeatures))

    # -- done !
    fvector = np.concatenate(feat_l)
    assert fvector.ndim == 1 and fvector.dtype == np.float32
    return fvector


def _filter(im, filt_l, legacy=False):
    if not legacy:
        assert im.dtype == np.float32
    assert im.ndim == 2
    if legacy:
        result = v1s_funcs.v1s_filter(im, filt_l)
    else:
        # do filtering using one line.
        # DON't reverse the order of filters to get same result as in the original implementation.
        # not reversing filt_l is correct behavior.
        result_padded = conv(im[np.newaxis, :, :], filt_l, mode='full')
        _, current_h, current_w = result_padded.shape
        new_h, new_w = im.shape

        h_slice = slice((current_h - new_h) // 2, (current_h - new_h) // 2 + new_h)
        w_slice = slice((current_w - new_w) // 2, (current_w - new_w) // 2 + new_w)
        result = result_padded.transpose((1, 2, 0))[h_slice, w_slice].astype(np.float32)

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
    if not legacy:
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
    if not legacy:
        assert im.dtype == np.float32
    mode = 'same'
    if lsum_ksize is not None:
        if legacy:
            k = np.ones(lsum_ksize, 'f') / lsum_ksize
            im = conv(conv(im, k[np.newaxis, :], mode), k[:, np.newaxis], mode)
        else:
            im = uniform_filter(im, size=lsum_ksize, mode='constant')

    normalize_vector_inplace(im)
    if not legacy:
        assert im.dtype == np.float32
    return im


def _normin(im, params, legacy=False):
    if not legacy:
        assert im.dtype == np.float32
    if legacy:
        assert im.ndim == 2 or im.ndim == 3
        if im.ndim == 2:
            result = v1s_funcs.v1s_norm(im[:, :, np.newaxis], **params)[:, :, 0]
        else:
            result = v1s_funcs.v1s_norm(im, **params)
    else:
        result = local_normalization(im, **params)
    if not legacy:
        assert result.dtype == np.float32
    return result


def local_normalization(im, kshape, threshold):
    # a more efficient version of input normalization.
    # perhaps this is across channel normalization.
    kh, kw = kshape
    eps = 1e-5
    assert im.ndim == 2 or im.ndim == 3
    assert kh % 2 == 1 and kw % 2 == 1, "kernel must have odd shape"

    if im.ndim == 2:
        original_2d = True
        im = im[:, :, np.newaxis]
    else:
        original_2d = False
    assert im.ndim == 3
    depth = im.shape[2]
    ksize = kh * kw * depth
    h_slice = slice(kh // 2, -(kh // 2))
    w_slice = slice(kw // 2, -(kw // 2))
    d_slice = slice(depth // 2, depth // 2 + 1)  # this always works, for both even and odd depth.

    # TODO: add an option to do per channel or across channel normalization.
    # first compute local mean (on 3d stuff).

    # local mean kernel
    local_mean_kernel = np.ones((kh, kw, depth), dtype=im.dtype) / ksize
    # local_mean = conv(im, local_mean_kernel, mode='valid')  # it's 3D.

    local_mean = uniform_filter(im, size=(kh, kw, depth), mode='constant')[h_slice, w_slice, d_slice]

    assert local_mean.ndim == 3
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

    local_sum_kernel = np.ones((kh, kw, depth), dtype=im.dtype)

    # hssq = conv(np.square(im), local_sum_kernel, mode='valid')
    # uniform filter is faster.
    hssq = ksize * uniform_filter(np.square(im), size=(kh, kw, depth), mode='constant')[h_slice, w_slice, d_slice]
    hssq_normed = hssq - ksize * np.square(local_mean)
    np.putmask(hssq_normed, hssq_normed < 0, 0)
    h_div = np.sqrt(hssq_normed) + eps
    np.putmask(h_div, h_div < (threshold + eps), 1)
    result = im_without_mean / h_div
    if original_2d:
        result = result[:, :, 0]
    return result.astype(np.float32, copy=False)


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
        hslice = np.round(np.linspace(0, inh - 1, outh)).astype(np.int16)
        wslice = np.round(np.linspace(0, inw - 1, outw)).astype(np.int16)
        result = filtered_im[np.ix_(hslice, wslice)]
    assert result.dtype == np.float32 and result.ndim == 3
    return result
