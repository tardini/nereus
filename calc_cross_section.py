import numpy as np
from scipy.interpolate import interp1d
from constants import epsilon0, echarge
import cross_section_tables as cs_tab


def coulomb_sigma_diff(E_in_MeV, mu_in, q1, q2):
    '''q1, q2 are the electric charge numbers of the scattering particles'''

    q1 *= echarge
    q2 *= echarge
    E_in_J = 1e6*echarge*E_in_MeV
    if mu_in == 1 or E_in_MeV == 0:
        return 0
    k = 1./(4*np.pi*epsilon0)
    coul = q1*q2*k
    s = coul/(E_in_J*(1 - mu_in))
    cross_sec = 0.25* s**2 * 10**31 # 10**31: m**2 -> mbarn
    return cross_sec


def tabulated_sigma_diff(E_in_MeV, mu_in, reac='dp'):

    from scipy.interpolate import interp2d

    if reac == 'dp':
        cs = cs_tab.DP
    elif reac == 'd3he':
        cs = cs_tab.D3He
    f = interp2d(cs.En, cs.mu, cs.sigma_diff, kind='linear')
    return f(E_in_MeV, mu_in)


def legendre_sigma_diff(E_in_MeV, mu_in, reac='dt'):

    from scipy.special import legendre, eval_legendre

    if reac == 'dt':
        cs = cs_tab.DT
    elif reac == 'dd':
        cs = cs_tab.DDn3He
    n_leg = cs.leg_coeff.shape[1]
    data = interp1d(cs.En, cs.leg_coeff, axis=0)
    cs_leg = eval_legendre(range(n_leg), mu_in)
    cs_sum = np.sum(cs_leg*data(E_in_MeV))

    return legendre_sigma_tot(E_in_MeV, reac=reac)*cs_sum/(4*np.pi*data(E_in_MeV)[0])


def legendre_sigma_tot(E_in_MeV, Emin_MeV=5.e-4, reac='dt'):

    if reac == 'dt':
        cs = cs_tab.DT
    elif reac == 'dd':
        cs = cs_tab.DDn3He

    if E_in_MeV < Emin_MeV:
        return 0
    if E_in_MeV < cs.BoschLiskBound:
        E_keV = 1e3*E_in_MeV
        if E_in_MeV <= cs.loHiBound:
            num  =  cs.A[0] + E_keV*(cs.A[1] + E_keV*(cs.A[2] + E_keV*(cs.A[3] + E_keV*cs.A[4])))
            den1 =       1. + E_keV*(cs.B[0] + E_keV*(cs.B[1] + E_keV*(cs.B[2] + E_keV*cs.B[3])))
        else:
            num  =  cs.Ahi[0] + E_keV*(cs.Ahi[1] + E_keV*(cs.Ahi[2] + E_keV*(cs.Ahi[3] + E_keV*cs.Ahi[4])))
            den1 =         1. + E_keV*(cs.Bhi[0] + E_keV*(cs.Bhi[1] + E_keV*(cs.Bhi[2] + E_keV*cs.Bhi[3])))
        den2 = E_keV * np.exp(cs.Bg/np.sqrt(E_keV))
        return num/(den1*den2) # millibarn
    else:
        return np.interp(E_in_MeV, cs.Etot_MeV, cs.y)


if __name__ == '__main__':

    import matplotlib.pylab as plt

# Cross-sections

# Total - correct, page 9 of ControlRoom manual!
    E = 0.5
    dttot = legendre_sigma_tot(E, reac='dt')
    print('DT tot cross-section for E=%8.4f MeV:' %E)
    print('%12.4e millibarn' %dttot)

# Differential - correct, page 9 of ControlRoom manual!
    E = 1
    mu = -1
    print('DT cross-section for E=%8.4f MeV, mu=%6.3f:' %(E, mu))
    dt_diff = legendre_sigma_diff(E, mu, reac='dt')
    print('%12.4e millibarn' %dt_diff)

# Differential tab-sigma_diff, DP:
    E = 1
    mu = -1
    print('DP cross-section for E=%8.4f MeV, mu=%6.3f:' %(E, mu))
    dp_diff = tabulated_sigma_diff(E, mu, reac='dp')
    print('%12.4e millibarn' %dp_diff)

# Differential Coulomnb alpha scattering
    E = 3.5
    mu = -1
    print('Coul cross-section for E=%8.4f MeV, mu=%6.3f:' %(E, mu))
    coul_diff = coulomb_sigma_diff(E, mu, 2, 2)
    print('%12.4e millibarn' %coul_diff)

    
    theta = np.linspace(0, np.pi, 61)
    mu_grid = np.cos(theta)
    dd_theta = []
    dt_theta = []
    d3he_theta = []
    coul_alpha = []

    E = 0.45
    for mu in mu_grid:
        dd_theta.append(legendre_sigma_diff(E, mu, reac='dd'))
        dt_theta.append(legendre_sigma_diff(E, mu, reac='dt'))
        d3he_theta.append(tabulated_sigma_diff(E, mu, reac='d3he'))
        coul_alpha.append(coulomb_sigma_diff(E, mu, 2, 2))
    plt.figure('Cross-sections', (19, 6))

    plt.subplot(1, 4, 1)
    plt.title('DD cross-section(theta)')
    plt.plot(np.degrees(theta), dd_theta)
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.subplot(1, 4, 2)
    plt.title('DT cross-section(theta)')
    plt.plot(np.degrees(theta), dt_theta)
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.subplot(1, 4, 3)
    plt.title('Coulomb alpha-alpha cross-section(theta)')
    plt.semilogy(np.degrees(theta), coul_alpha)
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.subplot(1, 4, 4)
    plt.title('D3He cross-section(theta)')
    plt.plot(np.degrees(theta), d3he_theta)
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.show()
