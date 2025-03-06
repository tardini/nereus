import numpy as np
from .reactions import reaction
from .constants import alpha

# ControlRoom/source/reactivities.cpp
#
# neutrons/paper/bosch_nf92.pdf
# Bosch-Hale, table VII
# Ti validity range (keV)
#   DT    : 0.2-100
#   D3He  : 0.5-190
#   DDn3He: 0.2-100
#   DDpt  : 0.2-100


def react(T_keV, reac_lbl):
    '''Input:
        Ti: reactant temperature (scalar or 1d array) in keV
    '''

# Reduced mass, MeV -> keV

    reac = reaction[reac_lbl]
    coeff = reac.coeff_reac
    mu_c2 = 1e3*(reac.in1.m * reac.in2.m)/(reac.in1.m + reac.in2.m)

    Bg = np.pi * alpha * reac.in1.Z * reac.in2.Z * np.sqrt(2 * mu_c2)
    theta = T_keV/(  1 - \
        (T_keV * (coeff[1] + T_keV * (coeff[3] + T_keV * coeff[5]))) / \
        (1 + T_keV * (coeff[2] + T_keV * (coeff[4] + T_keV * coeff[6])))  \
    )
    csi = (Bg**2/(4*theta))**0.333
    react = coeff[0] * theta * np.sqrt(csi/(mu_c2 * T_keV**3)) * np.exp(-3*csi)

    return react
