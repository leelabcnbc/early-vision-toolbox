#!/usr/bin/env bash

# first, install an empty Python 2 environment. Use 2 for compatibility with other 3rd party packages.
conda create -y --name early-vision-toolbox python=2
# then, activate environment.
. activate early-vision-toolbox
# then install everything needed.
conda install -y nomkl scikit-learn numpy scipy matplotlib jupyter ipython scikit-image h5py nose
# great package for plotting.
pip install imagen