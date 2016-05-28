#!/usr/bin/env bash

# first, install an empty Python 2 environment. Use 2 for compatibility with other 3rd party packages.
conda create -y --name dicarlo_v1like_reference python=2
# then, activate environment.
. activate dicarlo_v1like_reference
# then install everything needed.
conda install -y nomkl numpy=1.11.0 scipy=0.17.0 nose pillow=3.2.0
# then manually install pyml.