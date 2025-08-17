#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data Encoder Circuits Library"""
from ._version import __version__
from . import frqi
from . import qcrank
from ._util import rescale_data_to_angles, rescale_angles_to_fdata
from ._util_img import (
    convert_max_val,
    l1_distance,
    l2_distance,
    wasserstein_distance
)

__all__ = [
    'rescale_data_to_angles',
    'rescale_angles_to_fdata',
    'frqi',
    'qcrank',
    'ParametricQCrankV2',
    'convert_max_val',
    'l1_distance',
    'l2_distance',
    'wasserstein_distance'
]


__author__ = '''Daan Camps, Jan Balewski'''
__maintainer__ = 'Daan Camps, Jan Balewski'
__email__ = 'daancamps@gmail.com'
__license__ = 'see LICENSE file'
__copyright__ = '''see COPYRIGHT file'''
__version__ = __version__