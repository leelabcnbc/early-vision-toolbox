from __future__ import division, absolute_import, print_function
import caffe
from tempfile import NamedTemporaryFile
from os import remove
from .network_definitions import caffe_deploy_proto_predefined, net_info_dict, get_prototxt
from copy import deepcopy
import numpy as np
from . import dir_dict
import os.path
from collections import defaultdict


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


# load the mean ImageNet image (as distributed with Caffe) for subtraction
imagenet_mu = np.load(os.path.join(dir_dict['caffe_repo_root'],
                                   'python/caffe/imagenet/ilsvrc_2012_mean.npy')).mean(1).mean(1)


def create_transformer(input_blob, input_blob_shape, scale=255, mu=None):
    if mu is None:
        mu = imagenet_mu  # default mean for most of Caffe models.
    # get transformer
    # use deep copy to avoid tricky bugs for reference.
    transformer = caffe.io.Transformer({input_blob: deepcopy(input_blob_shape)})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', deepcopy(mu))  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', scale)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    return transformer


def transform_batch(transformer, data, input_blob):
    return np.asarray([transformer.preprocess(input_blob, x) for x in data])


def get_slice_dict(slice_dict, blobs_to_extract, slice_tlbr):
    # then, compute the slice dict
    if slice_dict is None:
        slice_dict = defaultdict(lambda: ((None, None), (None, None)))

    slice_dict_real = dict()

    for blob_name_to_read in blobs_to_extract:
        slice_exp_1, slice_exp_2 = slice_dict[blob_name_to_read]
        if slice_tlbr:
            slice_r = slice(slice_exp_1[0], slice_exp_2[0])
            slice_c = slice(slice_exp_1[1], slice_exp_2[1])
        else:
            slice_r = slice(slice_exp_1[0], slice_exp_1[1])
            slice_c = slice(slice_exp_2[0], slice_exp_2[1])
        slice_dict_real[blob_name_to_read] = slice_r, slice_c
    return slice_dict_real


def reshape_blobs(net, input_blobs, batch_size):
    # reshape the net for input blobs.
    for in_blob in input_blobs:
        shape_old = net.blobs[in_blob].data.shape
        assert len(shape_old) == 4
        if shape_old[0] != batch_size:
            print('do reshape!')
            net.blobs[in_blob].reshape(batch_size, *shape_old[1:])


def extract_features(net, data_this_caffe, input_blobs=None,
                     blobs_to_extract=None, batch_size=50, slice_dict=None):
    if input_blobs is None:
        input_blobs = net.inputs[0]  # only take one, as in caffe's classifier.

    multi_input_flag = isinstance(data_this_caffe, list) and isinstance(input_blobs, list)
    single_input_flag = isinstance(data_this_caffe, np.ndarray) and isinstance(input_blobs, str)

    assert multi_input_flag or single_input_flag

    if single_input_flag:
        data_this_caffe = [data_this_caffe]
        input_blobs = [input_blobs]

    if blobs_to_extract is None:
        blobs_to_extract = set(net.blobs.keys()) - set(net.inputs)
    #print('blobs to extract: {}'.format(blobs_to_extract))

    # check each data has same size
    num_image = len(data_this_caffe[0])
    for data_this_caffe_this in data_this_caffe:
        assert len(data_this_caffe_this) == num_image

    reshape_blobs(net, input_blobs, batch_size)

    feature_dict = defaultdict(list)
    # then do the actual computation
    for startidx in range(0, num_image, batch_size):
        slice_this_time = slice(startidx, min(num_image, startidx + batch_size))
        # print("getting features for stimuli {0} to {1}, total {2}".format(
        #     slice_this_time.start + 1, slice_this_time.stop, num_image))
        slice_out_this_time = slice(0, slice_this_time.stop - slice_this_time.start)

        # set data.
        for idx, in_blob in enumerate(input_blobs):
            net.blobs[in_blob].data[slice_out_this_time] = data_this_caffe[idx][slice_this_time]

        # then forward.
        net.forward()

        for blob in blobs_to_extract:
            slice_r, slice_c = slice_dict[blob]
            blob_raw = net.blobs[blob].data[slice_out_this_time]
            assert blob_raw.ndim == 2 or blob_raw.ndim == 4
            if blob_raw.ndim == 2:
                blob_raw = blob_raw[:, :, np.newaxis, np.newaxis]
            # this copy is important... otherwise there can be issues.
            data_this_to_use = blob_raw[:, :, slice_r, slice_c].copy()
            feature_dict[blob].append(data_this_to_use)

    for blob_out in feature_dict:
        feature_dict[blob_out] = np.concatenate(feature_dict[blob_out], axis=0)

    return feature_dict
