# coding=utf-8
"""preprocessing pipelines for (single channel) images"""

from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from functools import partial
from copy import deepcopy
from .util import make_2d_input_matrix
from numpy.fft import fft2, ifft2, fftshift, ifftshift

FunctionTransformer = partial(FunctionTransformer, validate=False)  # turn off all validation.


def check_valid_steps(steps):
    pass


def get_patch_location(crows, ccols, patchsize):
    row_index = np.floor(np.arange(crows - patchsize / 2.0, crows + patchsize / 2.0))
    col_index = np.floor(np.arange(ccols - patchsize / 2.0, ccols + patchsize / 2.0))
    row_index = row_index.astype(np.int)
    col_index = col_index.astype(np.int)

    return row_index, col_index


def extract_image_patch(img, row_index, col_index):
    rowsthis, colsthis = img.shape  # this will implicitly check shape is 2-tuple.
    assert np.all(np.logical_and(row_index >= 0, row_index < rowsthis))
    assert np.all(np.logical_and(col_index >= 0, col_index < colsthis))
    return img[np.ix_(row_index, col_index)]


def get_grid_portions(images, row_grid, col_grid, patchsize, offset='c'):
    new_image_list = []
    for img in images:
        rowsthis, colsthis = img.shape  # this will implicitly check shape is 2-tuple.
        # for new image (canvas), use integer to index.
        if offset == 'c':
            crows = rowsthis / 2.0
            ccols = colsthis / 2.0
        else:
            raise NotImplementedError('such offset {} not implemented!'.format(offset))

        for r_pos, c_pos in zip(row_grid, col_grid):
            row_index, col_index = get_patch_location(crows + r_pos, ccols + c_pos, patchsize)
            new_image_list.append(extract_image_patch(img, row_index, col_index))

    return np.array(new_image_list)  # return as a 3d array.


def whiten_olsh_lee(images, f_0=None, central_clip=(None, None)):
    new_image_list = []
    for image in images:
        new_image_list.append(whiten_olsh_lee_inner(image, f_0, central_clip))
    return np.array(new_image_list)  # return as a 3d array.


def whiten_olsh_lee_inner(image, f_0=None, central_clip=(None, None)):
    height, width = image.shape
    assert height % 2 == 0 and width % 2 == 0, "image must have even size!"
    image = image - image.mean()
    std_im = image.std(ddof=1)
    assert std_im != 0, "constant image unsupported!"
    image /= std_im

    fx, fy = np.meshgrid(np.arange(-height / 2, height / 2), np.arange(-width / 2, width / 2), indexing='ij')
    rho = np.sqrt(fx * fx + fy * fy)
    if f_0 is None:
        f_0 = 0.4 * (height + width) / 2
    filt = rho * np.exp(-((rho / f_0) ** 4))

    im_f = fft2(image)

    fft_filtered_old = im_f * fftshift(filt)
    fft_filtered_old = fftshift(fft_filtered_old)
    if central_clip != (None, None):
        fft_filtered_old = fft_filtered_old[height / 2 - central_clip[0] / 2:height / 2 + central_clip[0] / 2,
                           width / 2 - central_clip[1] / 2:width / 2 + central_clip[1] / 2]
    im_out = np.real(ifft2(ifftshift(fft_filtered_old)))
    std_im_out = im_out.std(ddof=1)
    return im_out/std_im_out



def step_transformer_dispatch(step, step_pars):
    if step == 'sampling':
        sampling_type = step_pars['type']
        patchsize = step_pars['patchsize']
        assert patchsize is not None
        if sampling_type == 'grid':
            grid_origin = step_pars['grid_origin']
            grid_spacing = step_pars['grid_spacing']
            grid_order = step_pars['grid_order']  # should be 'C' or 'F'
            row_grid_num, col_grid_num = step_pars['grid_gridsize']
            row_grid = np.linspace(0, (row_grid_num - 1) * grid_spacing, row_grid_num)
            col_grid = np.linspace(0, (col_grid_num - 1) * grid_spacing, col_grid_num)
            if grid_origin == 'center':
                row_grid -= row_grid.mean()
                col_grid -= col_grid.mean()
                row_grid, col_grid = np.meshgrid(row_grid, col_grid, indexing='ij')
                row_grid = row_grid.ravel(order=grid_order)
                col_grid = col_grid.ravel(order=grid_order)
                return FunctionTransformer(partial(get_grid_portions, row_grid=row_grid,
                                                   col_grid=col_grid,
                                                   patchsize=patchsize,
                                                   offset='c'))
            else:
                raise NotImplementedError("type {} not supported!".format(grid_origin))

        elif sampling_type == 'clip':
            clip_origin = step_pars['clip_origin']
            clip_offset = step_pars['clip_offset']
            if clip_origin == 'center':
                return FunctionTransformer(partial(get_grid_portions, row_grid=[clip_offset[0]],
                                                   col_grid=[clip_offset[1]],
                                                   patchsize=patchsize,
                                                   offset='c'))
            else:
                raise NotImplementedError("type {} not supported!".format(clip_origin))
        else:
            raise NotImplementedError("type {} not supported!".format(sampling_type))
    elif step == 'removeDC':
        return FunctionTransformer(lambda x: x - np.mean(x, axis=1, keepdims=True))
    elif step == 'unitVar':
        # to be exactly the same as in adam coate's stuff.
        return FunctionTransformer(
            lambda x: x / np.sqrt(np.var(x, axis=1, ddof=step_pars['ddof'],
                                         keepdims=True) + step_pars['epsilon']))
    elif step == 'flattening':
        return FunctionTransformer(make_2d_input_matrix)
    elif step == 'oneOverFWhitening':
        return FunctionTransformer(partial(whiten_olsh_lee, f_0=step_pars['f_0'],
                                           central_clip=step_pars['central_clip']))
    else:
        raise NotImplementedError('step {} is not implemented yet'.format(step))


def bw_image_preprocessing_pipeline(steps=None, pars=None):
    """ image preprocessing pipeline for black and white (single channel) images.

    Parameters
    ----------
    steps : iterable
        an iterable of strings, specifying the steps to perfrom
    pars : dict
        a dict mapping from step name to their parameters.

    Returns
    -------
    a scikit learn Pipeline object ready to use.

    """
    canonical_order = ['normalizeRange', 'gammaCorrection', 'logTransform', 'oneOverFWhitening',
                       'sampling', 'flattening', 'removeDC', 'unitVar', 'PCA', 'ZCA']
    __step_set = frozenset(canonical_order)

    if steps is None:
        # default steps are those described as "canonical preprocessing" in the Natural Image Statistics book.
        steps = {'sampling', 'flattening', 'removeDC', 'PCA'}
    default_pars = {'sampling': {'type': 'random',  # 'all', or 'grid', or 'clip'
                                 'patchsize': None,  # for everything
                                 # params for 'random'
                                 'random_numpatch': None,  # only for random
                                 'random_buff': 4,  # only for random
                                 'random_pixelshiftx': 0,
                                 'random_seed': 0,
                                 # params for clip
                                 'clip_origin': 'center',  # or a 2-tuple specifying row and column.
                                 #
                                 'clip_offset': (0, 0),  # or a 2-tuple specifying row and column, relative to
                                 # clip origin.
                                 # this can be combined with another function generating origin positions of different
                                 # centers on the grid.
                                 'clip_random': False,  # TODO implement random selection for clip mode.
                                 'clip_random_numpatch': None,
                                 'clip_random_maxjitter': None,  # the maximum jitter size +/- maxjitter around center.
                                 'grid_origin': 'center',
                                 'grid_spacing': None,  # pixels between centers.
                                 'grid_gridsize': (None, None),  # a 2-tuple specifying how many
                                 'grid_order': 'C'
                                 },
                    'PCA': {'epsilon': 0.1,
                            'n_components': None  # keep all components.
                            },
                    'unitVar': {'epsilon': 10, 'ddof': 1},
                    'ZCA': {},
                    'oneOverFWhitening': {'f_0': None,  # cut off frequency, in cycle / image. 0.4*mean(H, W) by default
                                          'central_clip': (None, None)
                                          # clip the central central_clip[0] x central_clip[1] part in the frequency
                                          # domain. by default, don't do anything.
                                          },
                    'removeDC': {},
                    'gammaCorrection': {},
                    'normalizeRange': {},
                    'flattening': {}
                    }

    if pars is None:
        pars = default_pars

    steps = frozenset(steps)
    assert steps <= __step_set, "there are undefined operations!"
    assert frozenset(pars.keys()) <= steps, "you can't define pars for steps not in the pipeline!"
    # make sure this combination of steps is OK.
    check_valid_steps(steps)
    # construct a pars with only relevant steps.
    real_pars = {key: default_pars[key] for key in steps}
    for key in pars:
        real_pars[key].update(pars[key])

    # now let's first implement two things for Tang's data and LCA alaska snow.
    # 1. clip and grid sampling.
    # 2. removeDC and unitVar
    pipeline_step_list = []

    for candidate_step in canonical_order:
        if candidate_step in steps:
            pipeline_step_list.append((candidate_step,
                                       step_transformer_dispatch(candidate_step, real_pars[candidate_step])))

    return Pipeline(pipeline_step_list), deepcopy(real_pars)
