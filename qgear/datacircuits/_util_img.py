#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats


def convert_max_val(im, mv_out, mv_in=256):
    '''Convert the maximum intensity value of the image im by rescaling the
    data.

    Args:
        im:
            image data
        mv_out: int
            maximum non-inclusive intensity value after rescaling
        mv_in: int (256)
            maximum non-inclusive intensity value after rescaling
    '''
    im = np.round(im.astype('float64') * ((mv_out - 1) / (mv_in - 1)))
    return im.astype('uint8' if mv_out <= 256 else 'uint16')


def l1_distance(img_in, img_recovered, mode='relative'):
    '''L1 distance between input and recovered image.'''
    img_in = img_in.astype('float32')
    img_recovered = img_recovered.astype('float32')
    dist = np.sum(np.abs(img_in - img_recovered))
    if mode == 'relative':
        return dist / np.sum(np.abs(img_in))
    return dist


def l2_distance(img_in, img_recovered, mode='relative'):
    '''L2 distance between input and recovered image.'''
    img_in = img_in.astype('float32')
    img_recovered = img_recovered.astype('float32')
    dist = np.sum(np.square(img_in - img_recovered))
    if mode == 'relative':
        return dist / np.sum(np.square(img_in))
    return dist


def wasserstein_distance(img_in, img_recovered):
    return stats.wasserstein_distance(
        np.ravel(img_in), np.ravel(img_recovered)
    )