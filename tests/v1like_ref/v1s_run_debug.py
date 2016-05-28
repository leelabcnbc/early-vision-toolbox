#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" v1s_run module
Build and evaluate a simple V1-like model using sets of images belong to an
arbitrary number of categories (e.g. the Caltech 101 dataset)

"""

import sys
import os
import scipy.io as sio
import numpy as np
from v1s import V1S

# from v1s_math import fastsvd
from v1s_funcs import get_image


# -----------------------------------------------------------------------------


def main(param_fname, input_fname_list, output_fname):
    # -- get parameters
    param_path = os.path.abspath(param_fname)
    print "Parameters file:", param_path
    v1s_params = {}
    execfile(param_path, {}, v1s_params)

    model = v1s_params['model']
    # pca_threshold = v1s_params['pca_threshold']

    v1s = V1S(**{})  # create an instance.

    num_file = len(input_fname_list)
    train_fvectors = [None] * num_file
    image_list = []
    image_list_2 = []
    for i_file in range(num_file):
        im_after_resize = get_image(input_fname_list[i_file], v1s_params['model'][0][0]['preproc']['max_edge'])
        image_list.append(im_after_resize)
        if len(v1s_params['model']) > 1:
            assert len(v1s_params['model']) == 2
            im_after_resize_2 = get_image(input_fname_list[i_file], v1s_params['model'][1][0]['preproc']['max_edge'])
            image_list_2.append(im_after_resize_2)
        train_fvectors[i_file] = v1s.generate_repr(input_fname_list[i_file], model)
        assert train_fvectors[i_file].ndim == 1
        print "done", input_fname_list[i_file]

    train_fvectors = np.vstack(train_fvectors)
    images_after_resize = np.array(image_list)
    if image_list_2:
        images_after_resize_2 = np.array(image_list_2)
    else:
        images_after_resize_2 = np.nan

    # don't do it now. rather, we can choose to do it or not later.
    # ok, let's do sphere and PCA.
    # v_sub = train_fvectors.mean(axis=0)
    # train_fvectors -= v_sub
    # v_div = train_fvectors.std(axis=0)
    # np.putmask(v_div, v_div==0, 1)
    # train_fvectors /= v_div

    # don't do PCA. do it later. I found that RDM is not invariant under PCA.
    # nvectors, vsize = train_fvectors.shape
    #
    # if nvectors < vsize:
    #     print "pca...",
    #     print train_fvectors.shape, "=>",
    #     U,S,V = fastsvd(train_fvectors)
    #     eigvectors = V.T
    #     i = tot = 0
    #     S **= 2.
    #     while (tot <= pca_threshold) and (i < S.size):
    #         tot += S[i]/S.sum()
    #         i += 1
    #     eigvectors = eigvectors[:, :i+1]
    #     train_fvectors = np.dot(train_fvectors, eigvectors)
    #     print train_fvectors.shape

    sio.savemat(output_fname, {'feature_matrix': train_fvectors,
                               'param_fname': param_fname,
                               'input_fname_list': input_fname_list,
                               'images_after_resize': images_after_resize,
                               'images_after_resize_2': images_after_resize_2})

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        mod_fname = sys.argv[1]
        input_fname = sys.argv[2]
        output_fname_outer = sys.argv[3]
    except IndexError:
        progname = sys.argv[0]
        print "Usage: %s <parameter_file> <path_to_images file> <output file>" % progname
        print "Example:"
        print "  %s params_simple_plus.py imagelist.txt output.mat" % progname
        sys.exit(1)

    with open(input_fname) as fin:  # read the file name
        input_fname_list_outer = fin.read().splitlines()

    main(mod_fname, input_fname_list_outer, output_fname_outer)
