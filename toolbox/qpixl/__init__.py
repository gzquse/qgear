#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QPIXL Library"""
from ._version import __version__
from . import frqi, qcrank
from ._util import rescale_data_to_angles, rescale_angles_to_fdata, rescale_angles_to_idata
from ._util_img import (
    convert_max_val,
    l1_distance,
    l2_distance,
    wasserstein_distance
)

__all__ = [
    'rescale_data_to_angles',
    'rescale_angles_to_data',
    'frqi',
    'qcrank',
    'convert_max_val',
    'l1_distance',
    'l2_distance',
    'wasserstein_distance'
]


__author__ = '''Daan Camps, Jan Balewski, Albert Musaelian'''
__maintainer__ = 'Daan Camps'
__email__ = 'daancamps@gmail.com'
__license__ = 'see LICENSE file'
__copyright__ = '''see COPYRIGHT file'''
__version__ = __version__
