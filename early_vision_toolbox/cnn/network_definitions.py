"""save definitions of common CNN networks, split by layers"""
from __future__ import division, print_function, absolute_import
from collections import OrderedDict
from pkgutil import get_data
from early_vision_toolbox import root_package_spec
import caffe
from tempfile import NamedTemporaryFile
from os import remove


def get_prototxt(filename):
    return get_data(root_package_spec + '.cnn', 'caffe_protofiles/' + filename)


def caffe_deploy_proto_by_layers(prototxt_text, num_layer_eat_dict, base_length=1, sep='layer {'):
    assert isinstance(num_layer_eat_dict, OrderedDict)
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


def alexnet_proto_by_layers():
    """split ``deploy.prototxt`` for alexnet by layers. prototxt from Caffe RC3

    Returns
    -------

    """
    prototxt_text = get_prototxt('alexnet_deploy.prototxt')
    num_layer_eat_dict = OrderedDict(
        [('conv1', 2),
         ('norm1', 1),
         ('pool1', 1),
         ('conv2', 2),
         ('norm2', 1),
         ('pool2', 1),
         ('conv3', 2),
         ('conv4', 2),
         ('conv5', 2),
         ('pool5', 1)]
    )
    return caffe_deploy_proto_by_layers(prototxt_text, num_layer_eat_dict)


def caffenet_proto_by_layers():
    """split ``deploy.prototxt`` for caffenet by layers. prototxt from Caffe RC3

    Returns
    -------

    """
    prototxt_text = get_prototxt('caffenet_deploy.prototxt')
    num_layer_eat_dict = OrderedDict(
        [('conv1', 2),
         ('pool1', 1),
         ('norm1', 1),
         ('conv2', 2),
         ('pool2', 1),
         ('norm2', 1),
         ('conv3', 2),
         ('conv4', 2),
         ('conv5', 2),
         ('pool5', 1)]
    )
    return caffe_deploy_proto_by_layers(prototxt_text, num_layer_eat_dict)


def vgg16_proto_by_layers():
    """split ``VGG_ILSVRC_16_layers_deploy.prototxt`` by layers.
    prototxt from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md (link from Caffe model zoo)
    retrieved at June 6, 2016
    Returns
    -------

    """
    prototxt_text = get_prototxt('VGG_ILSVRC_16_layers_deploy.prototxt')
    num_layer_eat_dict = OrderedDict(
        [('conv1_1', 2),
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
         ('pool5', 1)]
    )
    return caffe_deploy_proto_by_layers(prototxt_text, num_layer_eat_dict, sep='layers {')


def vgg19_proto_by_layers():
    """split ``VGG_ILSVRC_16_layers_deploy.prototxt`` by layers.
    prototxt from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md (link from Caffe model zoo)
    retrieved at June 6, 2016
    Returns
    -------

    """
    prototxt_text = get_prototxt('VGG_ILSVRC_19_layers_deploy.prototxt')
    num_layer_eat_dict = OrderedDict(
        [('conv1_1', 2),
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
         ('pool5', 1)]
    )
    return caffe_deploy_proto_by_layers(prototxt_text, num_layer_eat_dict, sep='layers {')


caffe_deploy_proto_predefined = {
    'caffenet': caffenet_proto_by_layers,
    'alexnet': alexnet_proto_by_layers,
    'vgg16': vgg16_proto_by_layers,
    'vgg19': vgg19_proto_by_layers
}


def test():
    for key, func in caffe_deploy_proto_predefined.items():
        result, net_base = func()
        print('{}, ==BASE=='.format(key))
        print(net_base)
        for layer in result:
            print('==' + layer, '==')
            print(result[layer])
    # a = create_empty_caffe_net('vgg16', 'pool5')
    # a.blobs['data'].reshape(1,3,212,212)
    # a.reshape()
    # for layer_name, blob in a.blobs.iteritems():
    #     print(layer_name + '\t' + str(blob.data.shape))

def create_empty_caffe_net(name_or_proto, last_layer):
    """ create an uninitialized caffe model.

    :param name_or_proto:
    :param last_layer:
    :return:
    """
    if isinstance(name_or_proto, str):  # predefined names
        assert name_or_proto in caffe_deploy_proto_predefined, "model {} is not defined!".format(name_or_proto)
        result, net_base = caffe_deploy_proto_predefined[name_or_proto]()
    else:
        # you must provide two parts as those predefined files
        result, net_base = name_or_proto

    assert last_layer in result, "output layer {} not defined!".format(last_layer)

    prototxt_list = [net_base]
    done = False
    for layer in result:
        if not done:
            prototxt_list.append(result[layer])
            if layer == last_layer:
                done = True
        else:
            break
    prototxt = ''.join(prototxt_list)
    # create a temporary file... fuck Caffe
    f_temp_file = NamedTemporaryFile(delete=False)
    f_temp_file.write(prototxt)
    f_temp_file.close()
    # create the net.
    created_net = caffe.Net(f_temp_file.name, caffe.TEST)
    # remove the temporary file
    remove(f_temp_file.name)
    return created_net


if __name__ == '__main__':
    test()
