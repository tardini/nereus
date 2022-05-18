import numpy as np
import constants as con

# ControlRoom/source/reactivities.cpp
# Bosch-Hale, table VII
# Ti validity ranfe (keV)
#   DT    : 0.2-100
#   D3He  : 0.5-190
#   DDn3He: 0.2-100
#   DDpt  : 0.2-100

coeff_reac = {}
coeff_reac['dt']     = [1.17302e-9 , 1.51361e-2, 7.51886e-2, 4.60643e-3,  1.35000e-2, -1.06750e-4, 1.36600e-5]
coeff_reac['ddpt']   = [5.43360e-12, 5.85778e-3, 7.68222e-3,          0, -2.96400e-6,           0,          0]
coeff_reac['ddn3he'] = [5.65718e-12, 3.41267e-3, 1.99167e-3,          0,  1.05060e-5,           0,          0]
coeff_reac['d3he']   = [5.51036e-10, 6.41918e-3, -202896e-3, -1.9108e-5,  1.35776e-4,           0,          0]

mDc2_keV = 1e3*con.mDc2
mTc2_keV = 1e3*con.mTc2
mHe3c2_keV = 1e3*con.mHe3c2

def react(T, m1_c2=con.mDc2, m2_c2=con.mDc2, Z1=con.Z_D, Z2=con.Z_D, coeff=coeff_reac['ddn3he']):
    '''Input:
        Ti: reactant temperature (scalar or 1d array)
    '''

# Reduced mass, MeV -> keV

    mu_c2 = 1e3*(m1_c2 * m2_c2)/(m1_c2 + m2_c2)

    Bg = np.pi * con.alpha * Z1 * Z2 * np.sqrt(2 * mu_c2)
    theta = T/(  1 - \
        (T * (coeff[1] + T * (coeff[3] + T * coeff[5]))) / \
        (1 + T * (coeff[2] + T * (coeff[4] + T * coeff[4])))  \
    )
    csi = (Bg**2/(4*theta))**0.333
    react = coeff[0] * theta * np.sqrt(csi/(mu_c2 * T**3)) * np.exp(-3*csi)

    return react
