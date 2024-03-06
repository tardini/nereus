import os

nrespDir = os.path.dirname(os.path.realpath(__file__))

# Detector settings

nresp_set = {'f_detector': '%s/inc/detectorAUG.json' %nrespDir,
    'f_in_light': '%s/inc/light_func_jet.dat' %nrespDir,
    'nmc': 1e5, 'distr': 'gauss', 'En_wid_frac': 0.01, 'Ebin_MeVee': 0.005}
