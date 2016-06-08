from __future__ import division, absolute_import, print_function
import caffe
from tempfile import NamedTemporaryFile
from os import remove
from .network_definitions import caffe_deploy_proto_predefined, net_info_dict, get_prototxt


def create_empty_net(name_or_proto, last_layer=None):
    """ create an uninitialized caffe model.

    :param name_or_proto:
    :param last_layer:
    :return:
    """
    if isinstance(name_or_proto, str):  # predefined names
        assert name_or_proto in caffe_deploy_proto_predefined, "model {} is not defined!".format(name_or_proto)
        result, net_base = caffe_deploy_proto_predefined[name_or_proto]
    else:
        # you must provide two parts as those predefined files
        result, net_base = name_or_proto

    if last_layer is None:
        last_layer = list(result.keys())[-1]
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


def create_predefined_net(name):
    """ create one example CNN in caffe examples.

    :param name:
    :return:
    """
    assert name in net_info_dict, "model {} is not defined!".format(name)
    model_weight_name = net_info_dict[name]['caffemodel_path']
    model_file_name = net_info_dict[name]['prototxt_path']
    f_temp_file = NamedTemporaryFile(delete=False)
    f_temp_file.write(get_prototxt(model_file_name))
    f_temp_file.close()
    # create the net.
    created_net = caffe.Net(f_temp_file.name, model_weight_name, caffe.TEST)
    # remove the temporary file
    remove(f_temp_file.name)
    return created_net


def fill_weights(src_name_or_net, dest_net):
    """fill up weight data from one net to another

    :param src_name_or_net:
    :param dest_net:
    :return:
    """
    if isinstance(src_name_or_net, str):
        src_net = create_predefined_net(src_name_or_net)
    else:
        src_net = src_name_or_net

    # then fill up weights from src to dest.

    assert set(dest_net.params.keys()) <= set(src_net.params.keys()), "some layers non existent in src net!"

    # changing value in place is safe.
    for layer_name, param in dest_net.params.iteritems():
        assert len(param) == len(src_net.params[layer_name])
        for idx in range(len(param)):
            param[idx].data[...] = src_net.params[layer_name][idx].data
    del src_net
    return dest_net
