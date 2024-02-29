import os
import numpy as np

nrespDir = os.path.dirname(os.path.realpath(__file__))

# Variable types

flt_typ = np.float32
int_typ = np.int32

# Detector settings

dist  = 2016.5000
theta = 0.0000
DG =   5.2400
RG =   2.9100
DSZ =  5.0800
RSZ =  2.5400
DL =   2.0000
dens_lg  =  1.1800
dens_sc  =  0.8740
alpha_lg =  1.1000
alpha_sc =  1.2120
f_in_light = '%s/inc/light_func_jet.dat' %nrespDir

# Run settings

nmc = 1e5
distr = 'gauss'
En_wid_frac = .01
Ebin_MeVee = .005

# Polynomial coefficients for light yield

DLT0 =  0.0000
DLT1 =  0.0000
FLT1 =  0.0277
FLT2 =  1.7490
ENAL1 = 6.7600
GLT0 = -0.6366
GLT1 =  0.2100
GLT2 =  0.0000
RLT1 =  0.0100
SLT1 =  0.0097
