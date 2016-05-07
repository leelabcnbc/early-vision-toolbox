#!/usr/bin/env bash

# remove spams dir
rm -rf spams-matlab
tar -xvzf spams-matlab-v2.5-svn2014-07-04.tar.gz

# compile it
matlab -r "cd spams-matlab; compile; exit;"