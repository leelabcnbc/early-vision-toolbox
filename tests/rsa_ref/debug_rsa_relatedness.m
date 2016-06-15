function debug_rsa_relatedness()

rng(0,'twister');
addpath(genpath(fullfile(pwd, 'rsatoolbox')));
% first, generate some random rdm matrices.
nCase = 10;
nRDM = 5;
nRDMCand = 12;
nSample = 25;
nRandomisations = 100;
% create 5 random feature matrices, of size 100 x 200.
feature_matrix_all = randn(nSample, 200, nRDM, nCase);
feature_matrix_cand_all = randn(nSample, 200, nRDMCand, nCase);
rdm_stack_all = zeros(nSample,nSample,nRDM, nCase);
cand_rdm_stack_all = zeros(nSample,nSample,nRDMCand, nCase);
index_matrix_array = zeros(nSample, nRandomisations, nCase);
p_value_array = zeros(nRDMCand, nCase);
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
    
    cand2refSims = zeros(nRDM_this, nRDMCand);
    for candI = 1:nRDMCand
        for subI = 1:nRDM_this
            cand2refSims(subI,candI)=corr(vectorizeRDMs(meanCandRDMs(:,:,candI))',vectorizeRDMs(rdm_stack(:,:,subI))','type','spearman','rows','pairwise');
        end
    end
    y = mean(cand2refSims,1);
%     disp(y);
    [p_values, index_matrix] = do_one_randomisation(mean(rdm_stack,3), meanCandRDMs, nRandomisations, y);
    index_matrix_array(:,:,iCase) = index_matrix;
    p_value_array(:, iCase) = p_values;
end

save('debug_rsa_relatedness.mat', 'rdm_stack_all', 'cand_rdm_stack_all', 'index_matrix_array', 'p_value_array');

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

function [p_values, index_matrix] = do_one_randomisation(meanRefRDM, meanCandRDMs, nRandomisations, y)
[n,n_2]=size(meanRefRDM);
assert(n==n_2);
nRDM = size(meanCandRDMs,3);
rdms = zeros(nRDM, n*(n-1)/2);
for rdmI = 1:nRDM
    % do the randomisation test, also keep the randomistion correltions
    % in a separte matrix
    rdms(rdmI,:) = vectorizeRDM(meanCandRDMs(:,:,rdmI));
end
% make space for null-distribution of correlations
rs_null=nan(nRandomisations,nRDM);

index_matrix = zeros(n, nRandomisations);
for randomisationI=1:nRandomisations
    randomIndexSeq = randomPermutation(n);
    index_matrix(:, randomisationI) = randomIndexSeq;
    rdmA_rand_vec=vectorizeRDM(meanRefRDM(randomIndexSeq,randomIndexSeq));
    rs_null(randomisationI,:)=corr(rdmA_rand_vec',rdms','type','spearman','rows','pairwise');
end % randomisationI
% p-values from the randomisation test
p_values = zeros(nRDM, 1);
for candI = 1:nRDM
    p_values(candI) = 1 - relRankIn_includeValue_lowerBound(rs_null(:,candI),y(candI)); % conservative
end
% disp(rs_null(1,:));
end