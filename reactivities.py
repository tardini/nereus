import numpy as np
import constants as con

# ControlRoom/source/reactivities.cpp
# Bosch-Hale, table VII
# Ti validity ranfe (keV)
#   DT    : 0.2-100
#   D3He  : 0.5-190
#   DDn3He: 0.2-100
#   DDpt  : 0.2-100

coeff_dt     = [1.17302e-9 , 1.51361e-2, 7.51886e-2, 4.60643e-3,  1.35000e-2, -1.06750e-4, 1.36600e-5]
coeff_ddpt   = [5.43360e-12, 5.85778e-3, 7.68222e-3,          0, -2.96400e-6,           0,          0]
coeff_ddn3he = [5.65718e-12, 3.41267e-3, 1.99167e-3,          0,  1.05060e-5,           0,          0]
coeff_d3he   = [5.51036e-10, 6.41918e-3, -202896e-3, -1.9108e-5,  1.35776e-4,           0,          0]

mDc2_keV = 1e3*con.mDc2
mTc2_keV = 1e3*con.mTc2
mHe3c2_keV = 1e3*con.mHe3c2

def react(T, m1_c2=mDc2_keV, m2_c2=mDc2_keV, Z1=con.Z_D, Z2=con.Z_D, coeff=coeff_ddn3he):
    '''Input:
        Ti: reactant temperature (scalar or 1d array)
    '''
# Reduced mass
 
    mu_c2 = (m1_c2 * m2_c2)/(m1_c2 + m2_c2)

    Bg = np.pi * con.alpha * Z1 * Z2 * np.sqrt(2 * mu_c2)
    theta = T/(  1 - \
        (T * (coeff[1] + T * (coeff[3] + T * coeff[5]))) / \
        (1 + T * (coeff[2] + T * (coeff[4] + T * coeff[4])))  \
    )
    csi = (Bg**2/(4*theta))**0.333
    react = coeff[0] * theta * np.sqrt(csi/(mu_c2 * T**3)) * np.exp(-3*csi)

    return react


if __name__ == '__main__':

    import matplotlib.pylab as plt

    Ti_keV = np.linspace(1., 60., 60)
    ddn3he = react(Ti_keV, m1_c2=mDc2_keV, m2_c2=mDc2_keV  , coeff=coeff_ddn3he)
    ddpt   = react(Ti_keV, m1_c2=mDc2_keV, m2_c2=mDc2_keV  , coeff=coeff_ddpt)
    dt     = react(Ti_keV, m1_c2=mDc2_keV, m2_c2=mTc2_keV  , coeff=coeff_dt)
    dhe3   = react(Ti_keV, m1_c2=mDc2_keV, m2_c2=mHe3c2_keV, coeff=coeff_d3he)
    plt.semilogy(Ti_keV, ddn3he, 'b-', label='DDn3He')
    plt.semilogy(Ti_keV, ddpt  , 'r-', label='DDpt')
    plt.semilogy(Ti_keV, dt    , 'g-', label='DT')
    plt.semilogy(Ti_keV, dhe3  , 'k-', label='DHe3')
    plt.xlim([0, 60])
    plt.xlabel(r'T$_i$ [keV]')
    plt.ylabel(r'Reactivity [cm$^3$/s]')
    plt.grid()
    plt.legend(loc=2)
    plt.show()
