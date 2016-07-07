# early-vision-toolbox
a collection of models and analysis methods for early visual areas.

## TODOs

- [ ] write some scripts to automatically download 4 reference caffe models (maybe this should be done in the caffe script).
- [ ] make tests to CI! (Arish?)

## how to install

I assume you are using CNBC cluster, and you are in root directory of this repo.

* install caffe following the instruction in lab wiki.
* run `~/DevOps/shell_scripts/conda/envs/early-vision-toolbox.sh`
* download all caffe models following instructions in `early_vision_toolbox/cnn/network_definitions.py`.
* then run `. ./setup_env_variables.sh` to setup environment variables to properly run Caffe related functions.
* `cd` to `tests`, and run `./test_master.sh`. It will take around 10 minutes and everything should be fine.
