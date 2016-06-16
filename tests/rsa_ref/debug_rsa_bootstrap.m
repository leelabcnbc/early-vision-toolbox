function debug_rsa_bootstrap()

rng(0,'twister');
addpath(genpath(fullfile(pwd, 'rsatoolbox')));
% first, generate some random rdm matrices.
nCase = 10;
nRDM = 5;
nRDMCand = 12;
nSample = 25;
nRandomisations = 250;
% create 5 random feature matrices, of size 100 x 200.
feature_matrix_all = randn(nSample, 200, nRDM, nCase);
feature_matrix_cand_all = randn(nSample, 200, nRDMCand, nCase);
rdm_stack_all = zeros(nSample,nSample,nRDM, nCase);
cand_rdm_stack_all = zeros(nSample,nSample,nRDMCand, nCase);
index_matrix_array_subject = zeros(nRandomisations, nRDM, nCase);
index_matrix_array_condition = zeros(nRandomisations, nSample, nCase);
p_value_array = zeros(nRDMCand, nRDMCand, nCase);
bootstrap_e_array = zeros(nRDMCand, nCase);
similarity_array = zeros(nRDMCand, nRandomisations, nCase);
% generate boolean matrix for shuffling or not.
shuffleOptionMatrix = false(nCase,2);
for iCase = 1:nCase
    resampleSubjects = rand() > 0.5;
    if resampleSubjects
        resampleConditions = rand() > 0.5;
    else
        resampleConditions = true;
    end
    shuffleOptionMatrix(iCase,:) = [resampleSubjects, resampleConditions];
end
assert(isequal(size(unique(shuffleOptionMatrix,'rows')), [3,2]));
for iCase = 1:nCase
    rdm_stack = get_one_rdm_stack(feature_matrix_all(:,:,:,iCase));
    rdm_stack_all(:,:,:,iCase) = rdm_stack;
    meanCandRDMs = get_one_rdm_stack(feature_matrix_cand_all(:,:,:,iCase));
    cand_rdm_stack_all(:,:,:,iCase) = meanCandRDMs;
    % specifically check single RDM
    
    if rem(iCase,2) == 0
        nRDM_this = 1;
    else
        nRDM_this = nRDM;
    end
    rdm_stack = rdm_stack(:,:,1:nRDM_this);
    userOptions = struct();
    userOptions.corrType = 'Spearman';
    userOptions.nBootstrap = nRandomisations;
    userOptions.resampleSubjects = shuffleOptionMatrix(iCase,1);
    userOptions.resampleConditions = shuffleOptionMatrix(iCase,2);
    [~, bootstrapEs, pairwisePs, bootstrapRs, resampledSubjectIs, resampledConditionIs] = bootstrapRDMs_mod(rdm_stack, meanCandRDMs, userOptions, false);
    index_matrix_array_subject(:,1:nRDM_this,iCase) = resampledSubjectIs;
    index_matrix_array_condition(:,:,iCase) = resampledConditionIs;
    bootstrap_e_array(:, iCase) = bootstrapEs;
    p_value_array(:,:,iCase) = pairwisePs;
    similarity_array(:, :, iCase) = bootstrapRs;
    [~, bootstrapEs2, pairwisePs2, bootstrapRs2, ~, ~] = bootstrapRDMs_mod(rdm_stack, meanCandRDMs, userOptions, true, resampledSubjectIs, resampledConditionIs);
    pairwisePs_valid = pairwisePs(isfinite(pairwisePs));
    pairwisePs2_valid = pairwisePs2(isfinite(pairwisePs2));
    fprintf('diff in std: %f\n', max(abs(bootstrapEs(:)-bootstrapEs2(:))));
    fprintf('diff in R: %f\n', max(abs(bootstrapRs(:)-bootstrapRs2(:))));
    fprintf('diff in p: %f\n', max(abs(pairwisePs_valid(:)-pairwisePs2_valid(:))));
end

save('debug_rsa_bootstrap.mat', 'rdm_stack_all', 'cand_rdm_stack_all', ...
    'index_matrix_array_subject', 'index_matrix_array_condition', ...
    'bootstrap_e_array', 'p_value_array', 'shuffleOptionMatrix', ...
    'similarity_array');

end

function rdm_stack = get_one_rdm_stack(feature_matrix_all_this_case)
[nSample, ~, nRDM] = size(feature_matrix_all_this_case);
rdm_stack = zeros(nSample,nSample,nRDM);
for iFeatureMatrix = 1:nRDM
    rdmThis = squareform(pdist(feature_matrix_all_this_case(:,:,iFeatureMatrix), ...
        'correlation'));
    rdm_stack(:,:,iFeatureMatrix) = rdmThis;
end


end

% a modified version that can return resampling indices.

function [realRs bootstrapEs pairwisePs bootstrapRs, resampledSubjectIs, resampledConditionIs] = ...
    bootstrapRDMs_mod(bootstrappableReferenceRDMs, candRDMs, userOptions, turn_nan, resampledSubjectIs2, resampledConditionIs2)
% [realRs bootstrapEs pairwisePs bootstrapRs] = ...
%                        bootstrapRDMs(bootstrappableReferenceRDMs, ...
%                                                candRDMs, ...
%                                                userOptions ...
%                                                )
%
%        bootstrappableReferenceRDMs --- The RDMs to bootstrap.
%                bootstrappableReferenceRDMs should be a [nConditions nConditions
%                nSubjects]-sized matrix of stacked squareform RDMs.
%
%        candRDMs --- The RDMs to test against.
%                testRDM should be an [nConditions nConditions nCandRDMs]-sized
%                matrix where each leaf is one RDM to be tested.
%
%        userOptions --- The options struct.
%                userOptions.corrType
%                        A string descriptive of the distance measure to be
%                        used. Defaults to 'Spearman'.
%                userOptions.nBootstrap
%                        How many bootstrap resamplings shoule be performed?
%                        Defaults to 1000.
%                userOptions.resampleSubjects
%                        Boolean. If true, subjects will be bootstrap resampled.
%                        Defaults to false.
%                userOptions.resampleConditions
%                        Boolean. If true, conditions will be resampled.
%                        Defaults to true.
%
%        realRs
%                The true RDM correlations between the average of the
%                bootstrappableReferenceRDMs and the testRDM.
%
%        bootstrapStdE
%                The bootstrap standard error.
%
% Cai Wingfield 6-2010, 7-2010
%__________________________________________________________________________
% Copyright (C) 2010 Medical Research Council

% Sort out defaults
userOptions = setIfUnset(userOptions, 'nBootstrap', 1000);
userOptions = setIfUnset(userOptions, 'resampleSubjects', false);
userOptions = setIfUnset(userOptions, 'resampleConditions', true);
userOptions = setIfUnset(userOptions, 'distanceMeasure', 'Spearman');

% Constants
nConditions = size(bootstrappableReferenceRDMs, 1);
nSubjects = size(bootstrappableReferenceRDMs, 3);

nCandRDMs = size(candRDMs, 3);

if ~(size(bootstrappableReferenceRDMs, 1) == size(candRDMs, 1))
    error('bootstrapRDMComparison:DifferentSizedRDMs', 'Two RDMs being compared are of different sizes. This is incompatible\nwith bootstrap methods!');
end%if

averageReferenceRDM = sum(bootstrappableReferenceRDMs, 3) ./ nSubjects;

% Decide what to say
if userOptions.resampleSubjects
    if userOptions.resampleConditions
        message = 'subjects and conditions';
    else
        message = 'subjects';
    end%if
else
    if userOptions.resampleConditions
        message = 'conditions';
    else
        message = 'nothing';
        warning('(!) You''ve gotta resample something, else the bar graph won''t mean anything!');
    end%if
end%if

fprintf(['Resampling ' message ' ' num2str(userOptions.nBootstrap) ' times']);

tic; %1

% Come up with the random samples (with replacement)

if nargin == 6
    disp('use existing index!');
    resampledSubjectIs = resampledSubjectIs2;
    resampledConditionIs = resampledConditionIs2;
else
    if userOptions.resampleSubjects
        resampledSubjectIs = ceil(nSubjects * rand(userOptions.nBootstrap, nSubjects));
    else
        resampledSubjectIs = repmat(1:nSubjects, userOptions.nBootstrap, 1); % change this to be more efficient.
    end%if:resampleSubjects

    if userOptions.resampleConditions
        resampledConditionIs = ceil(nConditions * rand(userOptions.nBootstrap, nConditions));
    else
        resampledConditionIs = repmat(1:nConditions, userOptions.nBootstrap, 1);
    end%if:resampleConditions
end

% Preallocation
realRs = nan(nCandRDMs, 1);
bootstrapRs = nan(nCandRDMs, userOptions.nBootstrap);
% bootstrapEs = nan(nCandRDMs, 1);
pairwisePs = nan(nCandRDMs, nCandRDMs);
% replace the diagonals for each instance of the candidate RDMs with
% NaN entries
% why doing this? maybe just want to remove some artifact?
% I don't like this, as this isn't necessary in my understanding,
% also, p value don't change at all in my experiment, although R changes.
if turn_nan
    for subI = 1:size(bootstrappableReferenceRDMs,3)
        temp = bootstrappableReferenceRDMs(:,:,subI);
        temp(logical(eye(size(temp,1)))) = nan;
        bootstrappableReferenceRDMs(:,:,subI) = temp;
    end
end
disp('no nan');
% Bootstrap
n = 0; k=0;

fprintf('\n');
for candRDMI = 1:nCandRDMs
    for b = 1:userOptions.nBootstrap
        n = n + 1;
        localReferenceRDMs = bootstrappableReferenceRDMs(resampledConditionIs(b,:),resampledConditionIs(b,:),resampledSubjectIs(b,:));
        localTestRDM = candRDMs(resampledConditionIs(b,:), resampledConditionIs(b,:), candRDMI);
        
        averageBootstrappedRDM = mean(localReferenceRDMs, 3);
        
        bootstrapRs(candRDMI, b) = corr(vectorizeRDMs(averageBootstrappedRDM)',vectorizeRDMs(localTestRDM)','type',userOptions.distanceMeasure,'rows','pairwise');
%         if mod(n,floor(userOptions.nBootstrap*nCandRDMs/100))==0
%             fprintf('%d%% ',floor(100*n/(userOptions.nBootstrap*nCandRDMs)));
%             if mod(n,floor(userOptions.nBootstrap*nCandRDMs/10))==0, fprintf('\n'); end;
%         end
    end%for:b
end%for:candRDMI
fprintf('\n');

bootstrapEs = std(bootstrapRs, 0, 2);
k=0;
for candRDMI = 1:nCandRDMs
%     if isequal(userOptions.RDMcorrelationType,'Kendall_taua')
%         realRs(candRDMI)=rankCorr_Kendall_taua(vectorizeRDMs(averageReferenceRDM)',vectorizeRDMs(candRDMs(:,:,candRDMI))');
%     else
%         realRs(candRDMI) = corr(vectorizeRDMs(averageReferenceRDM)',vectorizeRDMs(candRDMs(:,:,candRDMI))','type',userOptions.distanceMeasure,'rows','pairwise');
%     end
    for candRDMJ = 1:nCandRDMs
        if candRDMI == candRDMJ
            pairwisePs(candRDMI, candRDMJ) = nan;
        else
            ijDifferences = bootstrapRs(candRDMI, :) - bootstrapRs(candRDMJ, :);
            pairwisePs(candRDMI, candRDMJ) = 2*min([numel(find(ijDifferences < 0)),numel(find(ijDifferences > 0))] / userOptions.nBootstrap);
        end%if:diagonal
    end%for:candRDMJ
end%for:candRDMI

t = toc;%1

fprintf([': [' num2str(ceil(t)) 's]\n']);

end%function:bootstrapRDMComparisons