import os
import numpy as np

nrespDir = os.path.dirname(os.path.realpath(__file__))

# Variable types

flt_typ = np.float64
int_typ = np.int32

# Detector settings

f_detector = '%s/inc/detectorAUG.json'   %nrespDir
f_in_light = '%s/inc/light_func_jet.dat' %nrespDir
f_poly     = '%s/inc/poly_yield.json'    %nrespDir

# Run settings

nmc = 1e6
distr = 'gauss'
En_wid_frac = .01
Ebin_MeVee = .005
