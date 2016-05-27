"""generate examples to compare my V1 reference and the original one"""
from __future__ import division, print_function, absolute_import
from skimage.data import astronaut
from skimage.io import imsave
from sklearn.feature_extraction.image import extract_patches_2d
from os.path import exists

def main():
    im = astronaut()
    patches = extract_patches_2d(im, (96, 96), 10, random_state=0)
    # save these files on disk
    file_names = ['sample_{}.png'.format(x) for x in range(10)]
    for idx, filename in enumerate(file_names):
        patch_this = patches[idx]
        if not exists(filename):
            imsave(filename, patch_this)


if __name__ == '__main__':
    main()
