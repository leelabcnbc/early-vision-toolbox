"""save definitions of common CNN networks, split by layers"""
from __future__ import division, print_function, absolute_import
from collections import OrderedDict
from pkgutil import get_data
from early_vision_toolbox import root_package_spec
from . import dir_dict
import os.path


def get_prototxt(filename):
    return get_data(root_package_spec + '.cnn', 'caffe_protofiles/' + filename)


def caffe_deploy_proto_by_layers(prototxt_text, num_layer_eat_dict, sep=None, base_length=1):
    assert isinstance(num_layer_eat_dict, OrderedDict)
    if sep is None:
        sep = 'layer {'
    split_result = prototxt_text.split(sep)
    net_base = sep.join(split_result[:base_length])
    split_result = split_result[base_length:]

    result = OrderedDict()
    for layer, layer_to_eat in num_layer_eat_dict.items():
        assert len(split_result) >= layer_to_eat
        result_this = sep + sep.join(split_result[:layer_to_eat])
        result[layer] = result_this
        split_result = split_result[layer_to_eat:]
    return result, net_base


# currently, 4 fields.
# 1. name of proto
# 2. layer eat dict
# 3. sep for prototxt
# 4. location of binary model file.
# later, I may need to rewrite this as dict instead of tuple for readability.
net_info_dict = {
    'caffenet': {
        'prototxt_path': 'caffenet_deploy.prototxt',
        'layer_eat_dict': [('conv1', 2),
                           ('pool1', 1),
                           ('norm1', 1),
                           ('conv2', 2),
                           ('pool2', 1),
                           ('norm2', 1),
                           ('conv3', 2),
                           ('conv4', 2),
                           ('conv5', 2),
                           ('pool5', 1),
                           ('fc6', 3),
                           ('fc7', 3),
                           ('fc8', 1),
                           ('prob', 1)],
        'sep': None,
        'caffemodel_path': os.path.join(dir_dict['caffe_models'], 'bvlc_reference_caffenet',
                                        'bvlc_reference_caffenet.caffemodel'),
        'input_size': (227, 227),
        'last_non_fc_layer': 'pool5',
    },
    'alexnet': {
        'prototxt_path': 'alexnet_deploy.prototxt',
        'layer_eat_dict': [('conv1', 2),
                           ('norm1', 1),
                           ('pool1', 1),
                           ('conv2', 2),
                           ('norm2', 1),
                           ('pool2', 1),
                           ('conv3', 2),
                           ('conv4', 2),
                           ('conv5', 2),
                           ('pool5', 1),
                           ('fc6', 3),
                           ('fc7', 3),
                           ('fc8', 1),
                           ('prob', 1)],
        'sep': None,
        'caffemodel_path': os.path.join(dir_dict['caffe_models'], 'bvlc_alexnet', 'bvlc_alexnet.caffemodel'),
        'input_size': (227, 227),
        'last_non_fc_layer': 'pool5',
    },
    'vgg16': {
        'prototxt_path': 'VGG_ILSVRC_16_layers_deploy.prototxt',
        'layer_eat_dict': [('conv1_1', 2),
                           ('conv1_2', 2),
                           ('pool1', 1),
                           ('conv2_1', 2),
                           ('conv2_2', 2),
                           ('pool2', 1),
                           ('conv3_1', 2),
                           ('conv3_2', 2),
                           ('conv3_3', 2),
                           ('pool3', 1),
                           ('conv4_1', 2),
                           ('conv4_2', 2),
                           ('conv4_3', 2),
                           ('pool4', 1),
                           ('conv5_1', 2),
                           ('conv5_2', 2),
                           ('conv5_3', 2),
                           ('pool5', 1),
                           ('fc6', 3),
                           ('fc7', 3),
                           ('fc8', 1),
                           ('prob', 1)],
        'sep': 'layers {',
        'caffemodel_path': os.path.join(dir_dict['caffe_models'], '211839e770f7b538e2d8',
                                        'VGG_ILSVRC_16_layers.caffemodel'),
        'input_size': (224, 224),
        'last_non_fc_layer': 'pool5',
    },
    'vgg19': {
        'prototxt_path': 'VGG_ILSVRC_19_layers_deploy.prototxt',
        'layer_eat_dict': [('conv1_1', 2),
                           ('conv1_2', 2),
                           ('pool1', 1),
                           ('conv2_1', 2),
                           ('conv2_2', 2),
                           ('pool2', 1),
                           ('conv3_1', 2),
                           ('conv3_2', 2),
                           ('conv3_3', 2),
                           ('conv3_4', 2),
                           ('pool3', 1),
                           ('conv4_1', 2),
                           ('conv4_2', 2),
                           ('conv4_3', 2),
                           ('conv4_4', 2),
                           ('pool4', 1),
                           ('conv5_1', 2),
                           ('conv5_2', 2),
                           ('conv5_3', 2),
                           ('conv5_4', 2),
                           ('pool5', 1),
                           ('fc6', 3),
                           ('fc7', 3),
                           ('fc8', 1),
                           ('prob', 1)],
        'sep': 'layers {',
        'caffemodel_path': os.path.join(dir_dict['caffe_models'], '3785162f95cd2d5fee77',
                                        'VGG_ILSVRC_19_layers.caffemodel'),
        'input_size': (224, 224),
        'last_non_fc_layer': 'pool5',
    }
}

# use deepcopy to make sure the correct reference is used.
caffe_deploy_proto_predefined = {key: caffe_deploy_proto_by_layers(get_prototxt(value['prototxt_path']),
                                                                   OrderedDict(value['layer_eat_dict']),
                                                                   sep=value['sep']) for key, value in
                                 net_info_dict.items()}



# where the real binary model files are.
# these files are downloaded using methods described in http://caffe.berkeleyvision.org/model_zoo.html.
# that is, for bundled models in the caffe repo, just run
# scripts/download_model_binary.py <dirname>, where <dirname> can be models/bvlc_alexnet, etc.
# caffenet. SHA1 4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46
# alexnet. SHA1 9116a64c0fbe4459d18f4bb6b56d647b63920377


# for model zoo models (VGG), first ./scripts/download_model_from_gist.sh <gist_id> should be done to create
# the directory. Although in principle I can then do scripts/download_model_binary.py to download it, but for the case
# of VGG, it doesn't work, maybe due to broken readme.md.

# So I simply used wget (or aria2c, etc.) to download them directly
# for VGG16, it's http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
# SHA1: 9363e1f6d65f7dba68c4f27a1e62105cdf6c4e24
# for VGG19, it's http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
# SHA1: 239785e7862442717d831f682bb824055e51e9ba
