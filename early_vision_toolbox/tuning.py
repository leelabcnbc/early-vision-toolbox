# coding=utf-8
"""functions for measuring tuning of model neurons as in real experiments"""

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.linalg import norm
import imagen as ig
from .interface import NeuronBank
from .util import make_2d_input_matrix


class LinearSquareNeuronBank(NeuronBank):
    """
    NeuronBank with linear response computing methods and square RF size
    """

    def __init__(self, w):
        super(LinearSquareNeuronBank, self).__init__()
        w = np.atleast_2d(w)
        assert len(w.shape) == 2
        n_filter, filtersizesq = w.shape
        filtersize = np.int(np.sqrt(filtersizesq))
        assert filtersize ** 2 == filtersizesq, "filter must be originally square!"
        if w.dtype is not np.float64:
            w = w.astype(np.float64)  # use float64
        self.w = w
        self.__filtersize = filtersize

    @property
    def n_neuron(self):
        """ number of neurons

        Returns
        -------

        """
        return self.w.shape[0]

    @property
    def rf_size(self):
        """ size of RF

        Returns
        -------

        """
        return self.__filtersize, self.__filtersize

    def predict(self, imgs, neuron_idx=None):
        """ get neuron response to images

        Parameters
        ----------
        imgs

        Returns
        -------

        """
        imgs_array = make_2d_input_matrix(imgs)
        if neuron_idx is None:
            response = (np.dot(self.w, imgs_array.T))
        else:
            response = (np.dot(self.w[neuron_idx:(neuron_idx + 1), :], imgs_array.T))
        return response


_default_parameters = {
    'orientation': np.arange(0, 50, dtype=np.float64) / 50 * np.pi,  # in radian
    'frequency': np.arange(0, 50, dtype=np.float64) / 50,
    # this will be multipled by half the image size. in cycle/image
    'phase': np.arange(0, 20, dtype=np.float64) / 20 * 2 * np.pi  # in radian
}


def create_linear_square_neuron_bank(w):
    assert isinstance(w, NeuronBank) or isinstance(w, np.ndarray)

    if isinstance(w, np.ndarray):
        w = LinearSquareNeuronBank(w)

    if isinstance(w, LinearSquareNeuronBank):
        linear = True
    else:
        linear = False

    return w, linear


def quadrature_gratings(freq_or_pair_list, filtersize, legacy=False):
    freq_or_pair_list_sin = ((f, o, 0.0) for (f, o) in freq_or_pair_list)
    freq_or_pair_list_cos = ((f, o, np.pi / 2) for (f, o) in freq_or_pair_list)
    grating_sin = sine_gratings_batch(freq_or_pair_list_sin, filtersize, legacy)
    grating_cos = sine_gratings_batch(freq_or_pair_list_cos, filtersize, legacy)
    return grating_sin, grating_cos


def sine_gratings_batch(freq_or_phase_triplet_list, filtersize, legacy=False):
    # https://github.com/ioam/imagen/blob/d451081ad903f81e89959403c755f20ebff4f2f8/imagen/patterngenerator.py#L246
    # this line deals with orientation, actually, it rotates all points by MINUS orientation, since those are the points
    # that should be sampled from the rectangular grid, if finished stuff is rotated counterclock wise by orientation
    # also check https://en.wikipedia.org/wiki/Rotation_matrix
    if not legacy:
        gratings = [ig.SineGrating(phase=p, frequency=f, orientation=o, xdensity=filtersize,
                                   ydensity=filtersize)() - 0.5 for (f, o, p) in freq_or_phase_triplet_list]
        for g in gratings:
            g_norm = norm(g, 'fro')
            if g_norm != 0:
                g /= g_norm
    else:
        # this one is the old implementation. the points are not sampled not as in the imagen implementation. But this
        # should not matter, since we only need two gratings of 90 phase difference.
        # in the code, there's
        #
        # %rotate x and y values according to the desired orientation
        # zm=sin(orvalue).*xm+cos(orvalue).*ym;
        #
        # this is correct, since ym=[-Ym], where Y is in the usual cooridinate system (Y up, X right), rather than the
        # matlab system (Y down, X right). Replace y with -y in the reverse rotation formula in
        # https://github.com/ioam/imagen/blob/d451081ad903f81e89959403c755f20ebff4f2f8/imagen/patterngenerator.py#L246,
        # and then take the whole stuff to be its negative to transform back the coordinate system, we get
        # ``zm=sin(orvalue).*xm+cos(orvalue).*ym;`` exactly.
        gratings = [creategratings_legacy(patchsize=filtersize, freqvalue=f, orvalue=o, phasevalue=p)
                    for (f, o, p) in freq_or_phase_triplet_list]


    return gratings


def check_square_filter_size(w):
    filtersize, filtersize_ = w.rf_size
    assert filtersize == filtersize_, "filter must be originally square!"
    # assert filtersize % 2 == 0, "even size of filter for convenience!"
    return filtersize


def find_optimal_paras_rf(w, freqvalues=None, orvalues=None, phasevalues=None, legacy=False):
    """

    Parameters
    ----------
    w : numpy.ndarray or a NeuronBank object
        if a 2d array, then each row representing a flattened filter, in row major order. and response will be evaluated
        in a linear fasion. Otherwise, it must be a NeuronBank object.

    freqvalues, orvalues, phasevalues : array_like or None, optional
        (implicitly flattened) 1d array of frequency, orientation, and phase values to test.
        By default, they have values as used in the demo program (``figures.m``) of the original code (see Notes):

        .. code-block:: matlab

            %set number of different values for the grating parameters used in
            %computing tuning curves and optimal parameters
            freqno=50; %how many frequencies
            orno=50; %how many orientations
            phaseno=20; %how many phases
            %compute the used values for the orientation angles and frequencies
            orvalues=[0:orno-1]/orno*pi;
            freqvalues=[0:freqno-1]/freqno*patchsize/2;
            phasevalues=[0:phaseno-1]/phaseno*2*pi;
    legacy: bool, optional
        whether returning exactly the same optx and opty as in the original implementation
        (which I believe is problematic). Default ``False``
        also, this will use the original implementation for computing grating.


    Returns
    -------
    a numpy structured array with 5 columns, each row for a filter. It has the following ``dtype``.

    .. code-block:: python

        [('optx', np.float64),  #  gravity center of x (column)
         ('opty', np.float64),  #  gravity center of y (ro)
         ('optfreq', np.float64),  #  optimal frequency
         ('optor', np.float64),    #  optimal orientation.
         ('optphase', np.float64)] #  optimal phase

    Notes
    -----
    The linear part of this function is basically a port of ``findoptimalparas.m`` from code for [1]_.


    References
    ----------
    .. [1] Hyvärinen, A., Hurri, J., & Hoyer, P. O. (2009).
       Natural Image Statistics: A Probabilistic Approach to Early Computational Vision. (1st ed., Vol. 39).
       Springer Publishing Company, Incorporated. Retrieved from http://www.naturalimagestatistics.net/
       http://dx.doi.org/10.1007/978-1-84882-491-1

    """
    w, linear = create_linear_square_neuron_bank(w)

    n_filter = w.n_neuron
    filtersize = check_square_filter_size(w)

    if freqvalues is None:
        freqvalues = _default_parameters['frequency'] * filtersize / 2
    else:
        freqvalues = np.array(freqvalues).astype(np.float64).ravel()

    if orvalues is None:
        orvalues = _default_parameters['orientation']
    else:
        orvalues = np.array(orvalues).astype(np.float64).ravel()

    if phasevalues is None:
        phasevalues = _default_parameters['phase']
    else:
        phasevalues = np.array(phasevalues).astype(np.float64).ravel()

    # create a bunch of filters using imagen, of different freq and ori
    freq_or_pair_list = [(f, o) for f in freqvalues for o in orvalues]
    grating_sin, grating_cos = quadrature_gratings(freq_or_pair_list, filtersize, legacy=legacy)
    response = (w.predict(grating_sin)) ** 2 + (w.predict(grating_cos)) ** 2
    response_max = np.argmax(response, axis=1)

    # create the result array
    result_dtype = [('optx', np.float64),  # gravity center of x (column)
                    ('opty', np.float64),  # gravity center of y (ro)
                    ('optfreq', np.float64),  # optimal frequency
                    ('optor', np.float64),  # optimal orientation.
                    ('optphase', np.float64)]  # optimal phase
    result = np.zeros(shape=(n_filter,), dtype=result_dtype)

    for row_idx, max_idx in enumerate(response_max):
        optfreq, optor = freq_or_pair_list[max_idx]
        result[row_idx]['optfreq'] = optfreq
        result[row_idx]['optor'] = optor
        grating_phase = sine_gratings_batch([(optfreq, optor, p) for p in phasevalues], filtersize=filtersize,
                                            legacy=legacy)
        response_phase = w.predict(grating_phase, neuron_idx=row_idx).ravel()
        result[row_idx]['optphase'] = phasevalues[np.argmax(response_phase)]

        if linear:
            # then work on 'optx' and 'opty'
            filter_this = (w.w[row_idx].reshape(filtersize, filtersize)) ** 2
            filter_this = filter_this / filter_this.sum()  # so filter_this add to 1.
            # compute the gravity center along x and along y.
            if legacy:  # this one can't cover 1.
                grid_points = np.arange(0, filtersize, dtype=np.float64) / filtersize
            else:
                grid_points = np.linspace(0, 1, filtersize, dtype=np.float64)
            opt_x = np.sum(filter_this * grid_points[np.newaxis, :])
            opt_y = np.sum(filter_this * grid_points[:, np.newaxis])
            result[row_idx]['optx'] = opt_x
            result[row_idx]['opty'] = opt_y
        else:
            result[row_idx]['optx'] = np.nan
            result[row_idx]['opty'] = np.nan

    return result


def creategratings_legacy(patchsize, freqvalue, orvalue, phasevalue):
    x = np.arange(0, patchsize, dtype=np.float64) / patchsize
    xm, ym = np.meshgrid(x, x)

    zm = np.sin(orvalue) * xm + np.cos(orvalue) * ym
    grating2d = np.sin(zm * freqvalue * 2 * np.pi + phasevalue)
    grating2d /= (norm(grating2d, 'fro') + .00001)
    return grating2d.ravel()


def freq_tuning_curve(w, freqvalues_test=None, **kwargs):
    """ find the frequency tuning of a bunch of neurons.

    Parameters
    ----------
    w
    freqvalues_test : array_like
    kwargs

    Returns
    -------
    a 2d numpy ndarray, each row being the response of neuron to freqvalues_probe

    """
    w, linear = create_linear_square_neuron_bank(w)
    filtersize = check_square_filter_size(w)

    if freqvalues_test is None:
        freqvalues_test = _default_parameters['frequency'] * filtersize / 2
    else:
        freqvalues_test = np.array(freqvalues_test).astype(np.float64).ravel()

    best_tuning = find_optimal_paras_rf(w, **kwargs)
    result = np.zeros(shape=(w.n_neuron, freqvalues_test.size), dtype=np.float64)

    for row_idx, opt_params in enumerate(best_tuning):
        freq_or_pair_list = [(f, opt_params['optor']) for f in freqvalues_test]
        grating_sin, grating_cos = quadrature_gratings(freq_or_pair_list, filtersize=filtersize)
        response = (w.predict(grating_sin, neuron_idx=row_idx)) ** 2 + (w.predict(grating_cos, neuron_idx=row_idx)) ** 2
        result[row_idx, :] = np.sqrt(response.ravel())

    return result, freqvalues_test


def phase_tuning_curve(w, phasevalues_test=None, **kwargs):
    """ find the phase tuning of a bunch of neurons.

    Parameters
    ----------
    w
    freqvalues_probe : array_like
    kwargs

    Returns
    -------
    a 2d numpy ndarray, each row being the response of neuron to freqvalues_probe

    """

    w, linear = create_linear_square_neuron_bank(w)
    filtersize = check_square_filter_size(w)

    if phasevalues_test is None:
        phasevalues_test = _default_parameters['phase']
    else:
        phasevalues_test = np.array(phasevalues_test).astype(np.float64).ravel()

    best_tuning = find_optimal_paras_rf(w, **kwargs)
    result = np.zeros(shape=(w.n_neuron, phasevalues_test.size), dtype=np.float64)

    for row_idx, opt_params in enumerate(best_tuning):
        freq_or_phase_triplet_list = [(opt_params['optfreq'], opt_params['optor'], p) for p in phasevalues_test]
        gratings = sine_gratings_batch(freq_or_phase_triplet_list, filtersize)
        result[row_idx, :] = w.predict(gratings, neuron_idx=row_idx).ravel()

    return result, phasevalues_test


def ori_tuning_curve(w, orvalues_test=None, **kwargs):
    """ find the orientation tuning of a bunch of neurons.

    Parameters
    ----------
    w
    orvalues_test : array_like
    kwargs

    Returns
    -------
    a 2d numpy ndarray, each row being the response of neuron to freqvalues_probe

    """
    w, linear = create_linear_square_neuron_bank(w)

    if orvalues_test is None:
        orvalues_test = _default_parameters['orientation']
    else:
        orvalues_test = np.array(orvalues_test).astype(np.float64).ravel()

    filtersize = check_square_filter_size(w)
    best_tuning = find_optimal_paras_rf(w, **kwargs)
    result = np.zeros(shape=(w.n_neuron, orvalues_test.size), dtype=np.float64)

    for row_idx, opt_params in enumerate(best_tuning):
        freq_or_pair_list = [(opt_params['optfreq'], o) for o in orvalues_test]
        grating_sin, grating_cos = quadrature_gratings(freq_or_pair_list, filtersize=filtersize)
        response = (w.predict(grating_sin, neuron_idx=row_idx)) ** 2 + (w.predict(grating_cos, neuron_idx=row_idx)) ** 2
        result[row_idx, :] = np.sqrt(response.ravel())

    return result, orvalues_test
