"""module for extracting features using convolution neural networks"""
from __future__ import division, print_function, absolute_import
import os.path
import caffe
caffe_root = os.path.normpath(os.path.split(caffe.__file__)[0])
assert os.path.isabs(caffe_root)
dir_dict = {'caffe_models': os.path.normpath(os.path.join(caffe_root,
                                                          '..', '..',
                                                          'models'))}