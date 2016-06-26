"""module to compute gist features for a bunch of files, using default paramters"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.io import loadmat, savemat
from tempfile import NamedTemporaryFile
from skimage import img_as_ubyte
import os
from pipes import quote
from early_vision_toolbox import root_path
from early_vision_toolbox.util import run_matlab_script_with_exeception_handling


def get_gist(images, param=None):
    param_to_use = {
        #'imageSize': np.array([256, 256]),
        'orientationsPerScale': np.array([8, 8, 8, 8]),
        'numberBlocks': 4,
        'fc_prefilt': 4
    }

    if param is not None:
        param_to_use.update(param)

    # write those images as cell array of uint8 images
    # I don't check whether is 1 channel, or 3 channel. Just leave everything to that gist function itself.
    # you can't do np.array() directly, as sometimes the shape won't be correct.
    images_array = np.empty((len(images),), dtype=np.object_)
    for idx, im in enumerate(images):
        images_array[idx] = img_as_ubyte(im)
    assert images_array.shape == (len(images),)
    input_file = NamedTemporaryFile(suffix='.mat', delete=False)
    input_name = input_file.name
    input_file.close()
    output_file = NamedTemporaryFile(suffix='.mat', delete=False)
    output_name = output_file.name
    output_file.close()
    # so I don't need to quote them, and this should make passing raw string to matlab ok.
    assert input_name == quote(input_name) and "'" not in input_name
    assert output_name == quote(output_name) and "'" not in output_name
    # I don't need to delete this output file, as MATLAB will try to overwrite it automatically.

    # what files should be added to matlab path.
    gist_file_path = os.path.join(root_path, 'gist', 'matlab')

    # then let's save the input mat.
    param_to_use.update(images=images_array)
    try:
        print('saving input mat to {}'.format(input_name))
        savemat(input_name, param_to_use)
        print('saving done.')
        # then call the correct script.
        # here I assume that input and output name don't need quotes.
        script_to_call = """
        addpath(genpath('{gist_file_path}'));
        inputMat = load('{input_name}');
        n = numel(inputMat.images);
        gist_array = cell(n,1);
        if isfield(inputMat, 'imageSize')
            param.imageSize = double(inputMat.imageSize(:)');
        end
        param.orientationsPerScale = double(inputMat.orientationsPerScale(:)');
        param.numberBlocks = double(inputMat.numberBlocks);
        param.fc_prefilt = double(inputMat.fc_prefilt);
        images = inputMat.images(:);
        disp(n);
        disp(size(gist_array));
        disp(size(images));
        parfor i = 1:n
            [gist_array{{i}}, ~] = LMgist(images{{i}},'',param);
        end
        gist_array = cat(1, gist_array{{:}});
        save('{output_name}', 'gist_array');
        """.format(gist_file_path=gist_file_path,
                   input_name=input_name,
                   output_name=output_name)
        run_matlab_script_with_exeception_handling(script_to_call)
        result_mat = loadmat(output_name)
        return result_mat['gist_array']
    finally:
        finally_get_gist(input_name, output_name)


def finally_get_gist(input_name, output_name):
    assert os.path.exists(input_name)
    os.remove(input_name)
    assert os.path.exists(output_name)
    os.remove(output_name)
