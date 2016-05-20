rng(0,'twister');

original_images = randn(256,256,20);
new_images = preprocessing.whiten_olsh_lee(original_images);

%% save result
save_file_name = 'one_over_f_whitening_ref.hdf5';
file_root_group = '/testcases_matlab';
h5create(save_file_name, [file_root_group, '/original_images'], size(original_images), 'DataType', 'double');
h5write(save_file_name, [file_root_group, '/original_images'], original_images);
h5create(save_file_name, [file_root_group, '/new_images'], size(new_images), 'DataType', 'double');
h5write(save_file_name, [file_root_group, '/new_images'], new_images);
