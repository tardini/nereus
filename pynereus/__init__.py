#!/usr/bin/env python

"""NEutron REactions Utility Software
"""

import os, logging

__author__  = 'Giovanni Tardini (Tel. +49 89 3299-1898)'
__version__ = '0.0.5'
__date__    = '15.01.2025'

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
hnd.setLevel(level=logging.INFO)
logger = logging.getLogger('nereus')
logger.addHandler(hnd)
logger.setLevel(logging.INFO)
logger.propagate = False

nereus_home = os.path.dirname(os.path.realpath(__file__))

logger.info('Using version %s', __version__)
logger.info('NEREUS home %s', nereus_home)

from .nereus_gui import *
