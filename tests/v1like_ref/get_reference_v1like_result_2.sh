#!/usr/bin/env bash

# remember, this file must be ran under the dicarlo_v1like conda env.
# currently, it can only run the leelab gpu computer.
# copy this file, sample_dataset.txt, and all png files in this folder to the folder of original v1like repository.
# (with my modifications, so it has `v1s_run_rsa_research.py` in it.
# TODO: make a Docker environment for dicarlo_v1like

. activate dicarlo_v1like_reference
python v1s_run_debug.py params_simple.py sample_dataset_2.txt reference_2_v1like_result.mat
python v1s_run_debug.py params_simple_plus.py sample_dataset_2.txt reference_2_v1like_result_plus.mat
