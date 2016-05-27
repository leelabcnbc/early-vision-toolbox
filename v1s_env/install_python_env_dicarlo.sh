#!/usr/bin/env bash

# first, install an empty Python 2 environment. Use 2 for compatibility with other 3rd party packages.
conda create -y --name dicarlo_v1like_reference python=2
# then, activate environment.
. activate dicarlo_v1like_reference
# then install everything needed.
conda install -y nomkl numpy=1.11.0 scipy=0.17.0 nose pillow=3.2.0 openblas=0.2.14
# then manually install pyml.
rm -rf PyML-0.7.13.3
tar -xvzf PyML-0.7.13.3.tar.gz
cd PyML-0.7.13.3 && python setup.py build && python setup.py install && cd ..
