#!/usr/bin/env bash

# this will only work on the CNBC cluster.

. activate early-vision-toolbox   # for binaries like protc
. ~/DevOps/env_scripts/add_cudnn_v4.sh   # cuDNN
. ~/DevOps/env_scripts/add_cuda_lib.sh   # cuda runtime
. ~/DevOps/env_scripts/add_caffe_rc3_lib.sh   # libraries like HDF5, gflags, which are now not available system-wide.
# openblas. this MUST be put in the last, since `add_caffe_rc3_lib.sh` has a folder containing openblas as well,
# yet Caffe is linked to the one specified in `add_openblas.sh`
. ~/DevOps/env_scripts/add_openblas.sh
. ~/DevOps/env_scripts/add_caffe_rc3_python.sh
