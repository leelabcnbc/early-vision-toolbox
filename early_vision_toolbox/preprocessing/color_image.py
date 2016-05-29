from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from functools import partial
from copy import deepcopy
from skimage.transform import resize, rescale
from skimage.color import gray2rgb
from skimage import img_as_float

FunctionTransformer = partial(FunctionTransformer, validate=False)  # turn off all validation.


def whole_image_preprocessing_pipeline_check_valid_steps(steps):
    pass


def whole_image_normalize_format(imagelist):
    # make sure that everything is 3 channel, thus (most possibly) RGB stuff.
    imagelist_new = []
    for im in imagelist:
        # first convert dtype
        uint8_flag = im.dtype == np.uint8
        float_flag = im.dtype == np.float64
        assert uint8_flag or float_flag
        im_type_correct = img_as_float(im) if uint8_flag else im
        rgb_flag = (im_type_correct.ndim == 3 and im_type_correct.shape[2] == 3)
        gray_flag = (im_type_correct.ndim == 2)
        assert rgb_flag or gray_flag
        im_new = im_type_correct if rgb_flag else gray2rgb(im_type_correct)

        # make sure everything is correct.
        assert im_new.dtype == np.float64 and im_new.ndim == 3 and im_new.shape[2] == 3
        imagelist_new.append(im_new)

    return imagelist_new


def whole_image_step_transformer_dispatch(step, step_pars):
    if step == 'normalize_format':
        return FunctionTransformer(whole_image_normalize_format)
    if step == 'rescale':
        return FunctionTransformer(
            lambda x: [rescale(im, scale=step_pars['imscale'], order=step_pars['order'], mode='edge') for im in x])
    elif step == 'resize':
        return FunctionTransformer(
            lambda x: [resize(im, output_shape=step_pars['imsize'], order=step_pars['order'], mode='edge') for im in x])
    elif step == 'putInCanvas':
        return FunctionTransformer(partial(enlarge_canvas_dataset, canvascolor=step_pars['canvascolor'],
                                           rows=step_pars['canvassize'][0], cols=step_pars['canvassize'][1],
                                           jitter=step_pars['jitter'], jittermaxpixel=step_pars['jitermaxpixel'],
                                           jitterrandstate=step_pars['jitterrandseed'],
                                           external_jitter_list=step_pars['external_jitter_list'],
                                           strict=step_pars['strict'],
                                           crows=step_pars['crows'], ccols=step_pars['ccols']))
    else:
        raise NotImplementedError('step {} is not implemented yet'.format(step))


def whole_image_preprocessing_pipeline(steps=None, pars=None):
    """a pipeline to preprocess whole, large (color) images. primarily dealing with Caffe.
    this is a rewrite of ``preprocess_dataset.py`` in the original RSA_Research_2016 project.

    :param steps:
    :param pars:
    :return:
    """

    canonical_order = ['normalize_format', 'resize', 'rescale', 'putInCanvas', 'specialEffects']
    __step_set = frozenset(canonical_order)

    if steps is None:
        steps = {'normalize_format', 'rescale', 'putInCanvas'}
    default_pars = {'normalize_format': {},
                    'rescale': {'imscale': 1.0,
                                'order': 1},  # interpolation order. 1 means bilinear.
                    'resize': {'imsize': (150, 150),  # default size in V1 like model
                               'order': 1},  # interpolation order. 1 means bilinear.
                    'putInCanvas': {'canvascolor': (0.5, 0.5, 0.5),  # gray color by default.
                                    'jitter': False, # no jitter
                                    'jittermaxpixel': 0,  # trivial jitter.
                                    'jitterrandseed': None,
                                    'canvassize': (227, 227),  # default size for AlexNet.
                                    'external_jitter_list': None,  # explicitly provide jitter parameters.
                                    'strict': True,  # check top left corner of image match canvas color.
                                    'crows': None,
                                    'ccols': None,  # center of the patch. better not change it. you need to check the
                                    # actual code to understand their behaviour completely.
                                    },
                    'specialEffects': {'type': 'circular_window'},
                    # TODO finish this circular window effect used in Tang's data
                    }

    if pars is None:
        pars = default_pars

    steps = frozenset(steps)
    assert steps <= __step_set, "there are undefined operations!"
    assert frozenset(pars.keys()) <= steps, "you can't define pars for steps not in the pipeline!"
    # make sure this combination of steps is OK.
    whole_image_preprocessing_pipeline_check_valid_steps(steps)
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
                                       whole_image_step_transformer_dispatch(candidate_step,
                                                                             real_pars[candidate_step])))

    return Pipeline(pipeline_step_list), deepcopy(real_pars)


def enlarge_canvas_dataset(imagelist, canvascolor, rows, cols=None, crows=None, ccols=None, strict=False,
                           jitter=False, jittermaxpixel=None, jitterrandstate=None, external_jitter_list=None):
    """place the image into a larger canvas.
    :param debug_flag:
    :param jitterrandstate:
    :param jittermaxpixel:
    :param jitter:
    :param external_jitter_list:
    :param strict:
    :param imagelist:
    :param canvascolor: the color of the canvas color, in float.
    :param rows:
    :param cols:
    :param crows:
    :param ccols:
    :return:
    """

    canvascolor = np.array(canvascolor)
    assert canvascolor.ndim <= 1
    assert canvascolor.size == 1 or canvascolor.size == 3

    # shape canvasColor into a 1D 3-element array.
    if canvascolor.size == 1:
        canvascolor = np.tile(canvascolor, (3,))
    canvascolor = np.reshape(canvascolor, (3,))

    if cols is None:
        cols = rows

    image_template = np.empty((rows, cols, 3), dtype=np.float64)
    image_template[:] = canvascolor

    # compute image center
    if crows is None and ccols is None:
        crows = rows / 2.0
        ccols = cols / 2.0

    assert crows is not None and ccols is not None  # either specify all or specify none.

    # compute jitter
    # add jitter if necessary.
    jitterlist = np.zeros((len(imagelist), 2))
    if jitter:
        if external_jitter_list is not None:
            assert external_jitter_list.shape == (len(imagelist), 2)  # shape match.
            jitterlist = external_jitter_list.copy()
            assert np.array_equal(jitterlist, jitterlist.round())  # check all to be integers.
        else:  # generate a jitter list.
            if jitterrandstate is None:
                jitterrandstate = np.random.RandomState(None)  # initialize a random state.
            elif isinstance(jitterrandstate, int):
                jitterrandstate = np.random.RandomState(jitterrandstate)  # initialize a random state.
            else:
                assert isinstance(jitterrandstate, np.random.RandomState), "you must give me int, None, or RandomState"
            jitterlist = jitterrandstate.randint(-jittermaxpixel, jittermaxpixel + 1,
                                                 (len(imagelist), 2))  # generate different row and col jitter!

            # let's generate a random set of -1 and +1 to flip jitter's sign.
            # jitterlist = jitterlist*jitterrandstate.choice([-1,1],jitterlist.shape)
            print("max r jitter:", jitterlist[:, 0].max(), "min r jitter: ", jitterlist[:, 0].min(),
                  "mean r jitter:", jitterlist[:, 0].mean(),
                  "max c jitter:", jitterlist[:, 1].max(), "min c jitter: ", jitterlist[:, 1].min(),
                  "mean c jitter:", jitterlist[:, 1].mean())  # give people some statistics on jitters.

    imagelistnew = [None] * len(imagelist)
    for idx, image in enumerate(imagelist):
        imagelistnew[idx] = image_template.copy()
        assert image.dtype.type is np.float64  # can preprocess image to float if necessary in the future.
        # put the image into the center of this new image.
        rowsthis, colsthis = image.shape[0], image.shape[1]

        # assert rowsthis <= rows and colsthis <= cols

        jitterthisr = jitterlist[idx, 0]
        jitterthisc = jitterlist[idx, 1]

        # for new image (canvas), use integer to index.
        row_index = np.floor(np.arange(crows - rowsthis / 2.0 + jitterthisr, crows + rowsthis / 2.0 + jitterthisr))
        col_index = np.floor(np.arange(ccols - colsthis / 2.0 + jitterthisc, ccols + colsthis / 2.0 + jitterthisc))
        row_index = row_index.astype(np.int)
        col_index = col_index.astype(np.int)

        assert row_index.size == rowsthis
        assert col_index.size == colsthis

        # for original image, use boolean to index.
        row_index_fix = np.logical_and(row_index >= 0, row_index < rows)
        col_index_fix = np.logical_and(col_index >= 0, col_index < cols)

        imagelistnew[idx][np.ix_(row_index[row_index_fix], col_index[col_index_fix])] = \
            image[np.ix_(row_index_fix, col_index_fix)]
        # check trivial case: being the same size.
        if rowsthis == rows and colsthis == cols and jitterthisr == 0 and jitterthisc == 0:
            assert np.array_equal(imagelistnew[idx], image)
            # print("being the same!")
        if (image[0, 0, :] != canvascolor).any():
            # make sure top left corner match. This is just some sanity checking, and it's not applicable in many cases.
            if strict:  # make sure top left corner match.
                raise Exception("image color doesn't match!")

    return imagelistnew
