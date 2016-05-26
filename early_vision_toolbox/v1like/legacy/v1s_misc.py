from __future__ import division, print_function, absolute_import
from numpy import newaxis
from .v1s_math import fastsvd, gabor2d
import numpy as np

def _get_gabor_filters(params):
    """ Return a Gabor filterbank (generate it if needed)

    Inputs:
      params -- filters parameters (dict)

    Outputs:
      filt_l -- filterbank (list)

    """
    # -- get parameters
    fh, fw = params['kshape']
    orients = params['orients']
    freqs = params['freqs']
    phases = params['phases']
    nf = len(orients) * len(freqs) * len(phases)
    fbshape = nf, fh, fw
    gsw = fw / 5.
    gsh = fw / 5.
    xc = fw / 2
    yc = fh / 2
    filt_l = []
    filt_l_raw = []
    fix_bug = params['fix_bug']
    # -- build the filterbank
    for freq in freqs:
        for orient in orients:
            for phase in phases:
                # create 2d gabor
                filt = gabor2d(gsw, gsh,
                               xc, yc,
                               freq, orient, phase,
                               (fw, fh))
                filt_l_raw.append(filt)
                # vectors for separable convolution
                U, S, V = fastsvd(filt)
                tot = 0
                vectors = []
                idx = 0
                if fix_bug:
                    _S = S.copy()
                else:
                    _S = S
                S **= 2.
                # idx is the next one to work on.
                while tot <= params['sep_threshold'] and idx < min(params['max_component'], fh, fw):
                    row = (U[:, idx] * _S[idx])[:, newaxis]
                    col = (V[idx, :])[newaxis, :]
                    vectors += [(row, col)]
                    tot += S[idx] / S.sum()
                    idx += 1

                filt_l += [vectors]

    # second argument for debugging purpose.
    return filt_l, np.array(filt_l_raw)
