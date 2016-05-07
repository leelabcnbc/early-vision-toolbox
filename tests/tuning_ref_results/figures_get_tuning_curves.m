%Main file for generating basic experiments for the book
%"Natural Image Statistics" by Hyvarinen, Hurri, and Hoyer.

%To launch this m-file just call it without any input arguments

%SET GLOBAL VARIABLES

%This is path where the figures are saved. Default: same directory.
global figurepath
figurepath='';



%The following give basic parameters used in the experiments

%sample size, i.e. how many image patches. Book value: 50000
samplesize=50000;
%patchsize in most experiments. Book value: 32
patchsize=32;
%Number of features or weight vectors in one column in the big plots
%Book value: 16
plotcols=16;
%Number of features computed, i.e. PCA dimension in big experiments
%Book value: plotcols*16, or 256
rdim=plotcols*16;

%Choose "small" value which determines when the change in estimate is so small
%that algorithm can be stopped.
%This is related to the proportional change allowed for the features
%Book value: 1e-4, i.e. accuracy must be of the order of 0.01%
global convergencecriterion
convergencecriterion=1e-4;

%define default colormap
colormap('gray')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTE BASIC ICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%to save memory, get rid of some things computed in previous section(s)
clear X XnoDC Xnorm

writeline('------------------------------------')
writeline('Starting ICA section...')
writeline('------------------------------------')

%initialize random number generators to get same results each time
initializerandomseeds;

%Sample data and preprocess
writeline('Sampling data')
X=sampleimages(samplesize,patchsize);
writeline('Removing DC component')
X=removeDC(X);
writeline('Doing PCA and whitening data')
[V,E,D]=pca(X);
Z=V(1:rdim,:)*X;


writeline('Starting complete ICA. ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This computes the main results for chapters 6 and 7. (See below for chapter 9)

W=ica(Z,rdim);
%transform back to original space from whitened space
Wica = W*V(1:rdim,:);
%Compute A using pseudoinverse (inverting canonical preprocessing is tricky)
Aica=pinv(Wica);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANALYZE ICA RESULTS (Section 6.4.2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('Analyzing tuning of ICA features')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set number of different values for the grating parameters used in
%computing tuning curves and optimal parameters
freqno=50; %how many frequencies
orno=50; %how many orientations
phaseno=20; %how many phases
%compute the used values for the orientation angles and frequencies
orvalues=[0:orno-1]/orno*pi;
freqvalues=[0:freqno-1]/freqno*patchsize/2;
phasevalues=[0:phaseno-1]/phaseno*2*pi;

%initialize optimal values
ica_optx=zeros(rdim,1);
ica_opty=zeros(rdim,1);
ica_optfreq=zeros(rdim,1);
ica_optor=zeros(rdim,1);
ica_optphase=zeros(rdim,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ANALYZE TUNING FOR ALL SIMPLE CELLS
%i is index of simple cell
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:rdim;
    
    writenum(rdim-i)
    
    %find optimal parameters for the i-th linear feature estimated by ICA
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [optxtmp,optytmp,optfreqtmp,optortmp,optphasetmp]=findoptimalparas(Wica(i,:),freqvalues,orvalues,phasevalues);
    
    ica_optx(i)=optxtmp;
    ica_opty(i)=optytmp;
    ica_optfreq(i)=optfreqtmp;
    ica_optor(i)=optortmp;
    ica_optphase(i)=optphasetmp;
    
end %for i loop through simple cells



% then beginning save the data.

fprintf('\n');

file_to_save = 'NIS_results.hdf5';
group_root = '/ICA/tuning';
h5create(file_to_save, [group_root, '/Wica'], size(Wica), 'DataType', 'double');
h5write(file_to_save, [group_root, '/Wica'], Wica);
h5writeatt(file_to_save, group_root, 'ica_optx', ica_optx);
h5writeatt(file_to_save, group_root, 'ica_opty', ica_opty);
h5writeatt(file_to_save, group_root, 'ica_optfreq', ica_optfreq);
h5writeatt(file_to_save, group_root, 'ica_optor', ica_optor);
h5writeatt(file_to_save, group_root, 'ica_optphase', ica_optphase);


%% plot neurons

for i=1:10
    
    %compute responses when phase is changed for an ICA feature
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    phaseresponse=zeros(1,phaseno);
    
    for phaseindex=1:phaseno
        
        %create new grating with many phases
        grating=creategratings(patchsize,ica_optfreq(i),ica_optor(i),phasevalues(phaseindex));
        
        %compute response
        phaseresponse(phaseindex)=Wica(i,:)*grating;
        
    end %for phaseindex
    
    %normalize
    phaseresponse=phaseresponse/max(abs(phaseresponse));
    
    %compute responses when freq is changed for an ICA feature
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %note: this is responses to "drifting gratings",
    %i.e. we cannot use optimal phase but have to recompute optimal phase
    %separately. in practice, this is done by computing fourier amplitude
    %for given frequency and orientation
    
    freqresponse=zeros(1,freqno);
    
    for freqindex=1:freqno
        
        %create new grating with many freqs
        singrating=creategratings(patchsize,freqvalues(freqindex),ica_optor(i),0);
        cosgrating=creategratings(patchsize,freqvalues(freqindex),ica_optor(i),pi/2);
        
        %compute response
        sinresponse=Wica(i,:)*singrating;
        cosresponse=Wica(i,:)*cosgrating;
        freqresponse(freqindex)=sqrt(sinresponse^2+cosresponse^2);
        
    end %for freqindex
    
    %compute responses when orientation is changed for an ICA feature
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    orresponse=zeros(1,orno);
    
    for orindex=1:orno
        
        %create new grating with many phases
        singrating=creategratings(patchsize,ica_optfreq(i),orvalues(orindex),0);
        cosgrating=creategratings(patchsize,ica_optfreq(i),orvalues(orindex),pi/2);
        
        %compute response
        sinresponse=Wica(i,:)*singrating;
        cosresponse=Wica(i,:)*cosgrating;
        orresponse(orindex)=sqrt(sinresponse^2+cosresponse^2);
        
    end %for orindex
    
    %plot and save results for the first simple cells estimated by ICA
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if i<=10
        
        
        %plot phase tuning curve
        plot_withbigfont(phasevalues,phaseresponse)
        axis([min(phasevalues),max(phasevalues),-1,1]);
        print('-deps',[figurepath,'icasel' num2str(i) 'c.eps']),
        
        %plot freq tuning curve
        plot_withbigfont(freqvalues,freqresponse)
        print('-deps',[figurepath,'icasel' num2str(i) 'a.eps']),
        
        %plot orientation tuning curve
        plot_withbigfont(orvalues,orresponse);
        print('-deps',[figurepath,'icasel' num2str(i) 'b.eps']);
        
    end %of if
    
    
end %for i loop through simple cells

