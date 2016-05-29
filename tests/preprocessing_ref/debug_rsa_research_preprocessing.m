% save examplar result on preprocessing of images for AlexNet and
% SparseDBN in the original RSA_research (no 2016) repo.

% some required files to run this script are ignored. To get them, run 
% `sparsedbn_corentin_data_07302015.py` and
% `alexnet_corentin_08072015.py` under `rsa_python_scripts`,
% and put the generated `_meta.mat` files under `results` and the generated
% `.mat` files under `features` into `rsa_research_data/alexnet` or 
% `rsa_research_data/sparsedbn`.

% basically, I will take the first 2 images of each case (in total 18 cases)
% (3 scales x [ec, ac, ex] x 2 random seeds), 36 images in total as the
% reference result that the current early vision toolbox should achieve.

groupNames = {'alexnet', 'sparsedbn'};
subDirNames = groupNames;
metaFileNames = {'AlexNet_corentin_08072015_meta.mat', ...
    'sparseDBN_corentin_data_07302015_meta.mat'};

typeList = {'ec', 'ac', 'ex'};
sizeList = {'11', '22', '33'};

file_to_save = 'rsa_research_preprocessing_ref_alexnet_sparsedbn.hdf5';

for iGroup = 1:numel(groupNames)
    groupThis = groupNames{iGroup};
    metaDataThis = metaFileNames{iGroup};
    subDirThis = fullfile('rsa_research_data', subDirNames{iGroup});
    metaDataMat = load(fullfile(subDirThis,metaDataThis));
    % disp(metaDataMat.mat_list_list); % this is [njitter x ntype x nsize]
    for iType = 1:numel(typeList)
        typeThis = typeList{iType};
        for iSize = 1:numel(sizeList)
            sizeThis = sizeList{iSize};
            subGroupThis = ['/', groupThis, '/', typeThis, '/', sizeThis];
            disp(subGroupThis);
            metaFile1 = metaDataMat.mat_list_list{1,iType,iSize};
            metaFile2 = metaDataMat.mat_list_list{end,iType,iSize};
            metaFile1Mat = load(fullfile(subDirThis, metaFile1), 'imagelist', 'jitterrandseed');
            metaFile2Mat = load(fullfile(subDirThis, metaFile2), 'imagelist', 'jitterrandseed');
            datasetName1 = int2str(metaFile1Mat.jitterrandseed);
            datasetName2 = int2str(metaFile2Mat.jitterrandseed);
            datasetName1Full = [subGroupThis, '/', datasetName1];
            datasetName2Full = [subGroupThis, '/', datasetName2];
            disp(datasetName1Full);
            disp(datasetName2Full);
            dataset1 = metaFile1Mat.imagelist(:,:,1:2);
            dataset2 = metaFile2Mat.imagelist(:,:,1:2);
            h5create(file_to_save, datasetName1Full, size(dataset1));
            h5create(file_to_save, datasetName2Full, size(dataset2));
            h5write(file_to_save, datasetName1Full, dataset1);
            h5write(file_to_save, datasetName2Full, dataset2);
            h5writeatt(file_to_save, datasetName1Full, 'jitterrandseed', metaFile1Mat.jitterrandseed);
            h5writeatt(file_to_save, datasetName2Full, 'jitterrandseed', metaFile2Mat.jitterrandseed);
        end
    end
end