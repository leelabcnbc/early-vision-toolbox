rng(0,'twister');

%% first, load models
W = randn(16*16, 512);
W=W./repmat(sqrt(sum(W.^2)),[size(W,1) 1]);
lambda_list = linspace(0.1,1,10);
cost_list = zeros(numel(lambda_list),1);
%% get images
images = randn(16*16,100);
%images=images./repmat(sqrt(sum(images.^2)),[size(images,1) 1]);

%% then load spams
spamsDir = 'spams-matlab';
addpath(fullfile(spamsDir,'test_release'));
addpath(fullfile(spamsDir,'src_release'));
addpath(fullfile(spamsDir,'build'));

%% compute response
responseSubArray = cell(numel(lambda_list),1);
for iLambda = 1:numel(lambda_list)
    tic;
    lambda = lambda_list(iLambda);
    responseSub = mexLasso(images, W, struct('mode',2,'lambda',lambda));
    responseSub = full(responseSub); % this is crucial!
    costSPAMs = 0.5 * sum(sum((images - W*responseSub).^2)) + ...
        lambda*sum(sum(abs(responseSub)));
    responseSubArray{iLambda} = responseSub;
    assert(all(~isnan(responseSub(:))));
    cost_list(iLambda) = costSPAMs;
    toc;
end

disp(cost_list);

%% save result
save_file_name = 'sparse_coding_ref.hdf5';
file_root_group = '/spams/mexLasso/mode2';
h5create(save_file_name, [file_root_group, '/W'], size(W), 'DataType', 'double');
h5write(save_file_name, [file_root_group, '/W'], W);
h5create(save_file_name, [file_root_group, '/images'], size(images), 'DataType', 'double');
h5write(save_file_name, [file_root_group, '/images'], images);
h5writeatt(save_file_name, file_root_group, 'lambda_list', lambda_list(:));
h5writeatt(save_file_name, file_root_group, 'cost_list', cost_list(:));
for iLambda = 1:numel(lambda_list)
    response_to_write = responseSubArray{iLambda};
    h5create(save_file_name, [file_root_group, '/response/', int2str(iLambda)], size(response_to_write), 'DataType', 'double');
    h5write(save_file_name, [file_root_group, '/response/', int2str(iLambda)], response_to_write);
end
