function IMAGES = whiten_olsh_lee(images_in)
% WHITEN_OLSH_LEE ...
%
%   a wrapper of Honglak Lee's 1/f whiten procedure...
%
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 19-Mar-2014 21:39:33 $
%% DEVELOPED : 8.1.0.604 (R2013a)
%% FILENAME  : whiten_olsh_lee.m

if ( ~iscell(images_in) )% for a matrix case!!!
    
    num_images = size(images_in,3);
    
    % if (size(images_in,1)~=size(images_in,2)) % better be square
    %     warning('width and height are not equal');
    % end
    
    M = num_images;
    
    IMAGES = zeros(size(images_in));
    
    for i = 1:M
        IMAGES(:,:,i) = whiten_olsh_lee_inner(images_in(:,:,i));
    end
    
else
    num_images = numel(images_in);
    M = num_images;
    IMAGES = cell(size(images_in));
    
    for i = 1:M
        IMAGES{i} = whiten_olsh_lee_inner( images_in{i} );
    end
end

end


function im_out = whiten_olsh_lee_inner(im)

if size(im,3)>1
    im = rgb2gray(im);
    error('this can''t happen!');
end

im = double(im);
% normalize first
im = im - mean(im(:));
im = im./std(im(:));

N1 = size(im, 1);
N2 = size(im, 2);

% make sure they are even
assert(rem(N1,2)==0);
assert(rem(N2,2)==0);

[fx, fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
rho=sqrt(fx.*fx+fy.*fy)';

f_0=0.4*mean([N1,N2]);
filt=rho.*exp(-(rho/f_0).^4);

If=fft2(im);
imw=real(ifft2(If.*fftshift(filt))); 
% take real, although the complex part should be very very small. 
% since fft/ifft is linear operation, mean of new image is also zero.

im_out = imw/std(imw(:)); % 0.1 is the same factor as in make-your-own-images

end


% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [whiten_olsh_lee.m] ======
