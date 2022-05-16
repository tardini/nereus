import logging
import numpy as np
from scipy.interpolate import interp1d
from constants import epsilon0, echarge
import cross_section_tables as cs_tab

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('cross-sect')
logger.addHandler(hnd)
logger.setLevel(logging.INFO)

def sigma_diff(E_in_MeV, mu_in, Z1=None, Z2=None, reac='dt'):
    '''General method redirecting to the relevant reaction method'''

    if reac in ('dp', 'd3he', 'alphad', 'alphat', 'd3healphap'):
        return tabulated_sigma_diff(E_in_MeV, mu_in, reac=reac)
    elif reac in ('dt', 'dd'):
        return legendre_sigma_diff(E_in_MeV, mu_in, reac=reac)
    elif Z1 is not None:
        return coulomb_sigma_diff(E_in_MeV, mu_in, Z1, Z2)


def coulomb_sigma_diff(E_in_MeV, mu_in, Z1, Z2):
    '''q1, q2 are the electric charge numbers of the scattering particles'''

    q1 = Z1*echarge
    q2 = Z2*echarge
    E_in_J = 1e6*echarge*E_in_MeV
    if mu_in == 1 or E_in_MeV == 0:
        return 0
    k = 1./(4*np.pi*epsilon0)
    coul = q1*q2*k
    s = coul/(E_in_J*(1 - mu_in))
    cross_sec = 0.25* s**2 * 10**31 # 10**31: m**2 -> mbarn
    return cross_sec


def tabulated_sigma_diff(E_in_MeV, mu_in, reac='dp'):
    '''Performing interp2d on full cross-sectoin table(E, mu)'''

    from scipy.interpolate import interp2d

    if reac == 'dp':
        cs = cs_tab.DP
    elif reac == 'd3he':
        cs = cs_tab.D3He
    elif reac == 'alphad':
        cs = cs_tab.alphad
    elif reac == 'alphat':
        cs = cs_tab.alphat
    elif reac == 'd3healphap':
        cs = cs_tab.D3HeAlphaP
    f = interp2d(cs.En, cs.mu, cs.sigma_diff, kind='linear')
    res = f(E_in_MeV, mu_in).T
    return np.squeeze(res)


def legendre_sigma_diff(E_in_MeV, mu_in, reac='dt'):

    from scipy.special import eval_legendre

    E_in_MeV = np.atleast_1d(E_in_MeV)
    mu_in    = np.atleast_1d(mu_in)

    if reac == 'dt':
        cs = cs_tab.DT
    elif reac == 'dd':
        cs = cs_tab.DDn3He
    n_leg = cs.leg_coeff.shape[1]
    data = interp1d(cs.En, cs.leg_coeff, axis=0)
    data_E = data(E_in_MeV)
    cs_leg = np.zeros((len(mu_in), n_leg))
    for jleg in range(n_leg):
        cs_leg[:, jleg] = eval_legendre(jleg, mu_in)
    cs_sum = np.sum(cs_leg*data_E, axis=1)

    leg_tot = legendre_sigma_tot1d(E_in_MeV, reac=reac)
#    leg_tot = np.zeros_like(cs_sum)
#    for jE, E in enumerate(E_in_MeV):
#        leg_tot[jE] = legendre_sigma_tot(E, reac=reac)
    return np.squeeze(leg_tot*cs_sum/(4*np.pi*data_E[:, 0]))


def legendre_sigma_tot1d(E_in_MeV, Emin_MeV=5.e-4, reac='dt'):

    if reac == 'dt':
        cs = cs_tab.DT
    elif reac == 'dd':
        cs = cs_tab.DDn3He

    E_in_MeV = np.atleast_1d(E_in_MeV)
    E_keV = 1e3*E_in_MeV
    leg_out = np.zeros_like(E_in_MeV)

    (ind_bosch1, ) = np.where((E_in_MeV >= Emin_MeV) & (E_in_MeV < cs.BoschLiskBound) & (E_in_MeV <= cs.loHiBound))
    (ind_bosch2, ) = np.where((E_in_MeV >= Emin_MeV) & (E_in_MeV < cs.BoschLiskBound) & (E_in_MeV > cs.loHiBound))
    (ind_interp, ) = np.where((E_in_MeV >= Emin_MeV) & (E_in_MeV >= cs.BoschLiskBound))
    leg_out[ind_interp] = np.interp(E_in_MeV[ind_interp], cs.Etot_MeV, cs.y)

    num_a =  cs.A[0] + E_keV*(cs.A[1] + E_keV*(cs.A[2] + E_keV*(cs.A[3] + E_keV*cs.A[4])))
    den1_a =       1. + E_keV*(cs.B[0] + E_keV*(cs.B[1] + E_keV*(cs.B[2] + E_keV*cs.B[3])))
    num_ahi = cs.Ahi[0] + E_keV*(cs.Ahi[1] + E_keV*(cs.Ahi[2] + E_keV*(cs.Ahi[3] + E_keV*cs.Ahi[4])))
    den1_ahi =         1. + E_keV*(cs.Bhi[0] + E_keV*(cs.Bhi[1] + E_keV*(cs.Bhi[2] + E_keV*cs.Bhi[3])))
    den2 = E_keV * np.exp(cs.Bg/np.sqrt(E_keV))

    leg_out[ind_bosch1] = num_a[  ind_bosch1]/(den1_a[  ind_bosch1]*den2[ind_bosch1])
    leg_out[ind_bosch2] = num_ahi[ind_bosch2]/(den1_ahi[ind_bosch2]*den2[ind_bosch2])
    
    return leg_out


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
    dp_diff = sigma_diff(E, mu, reac='dp')
    print('%12.4e millibarn' %dp_diff)

# Differential Coulomnb alpha scattering
    E = 3.5
    mu = -1
    print('Coul cross-section for E=%8.4f MeV, mu=%6.3f:' %(E, mu))
    coul_diff = coulomb_sigma_diff(E, mu, 2, 2)
    print('%12.4e millibarn' %coul_diff)

    E = 0.85
    
    theta = np.linspace(0, np.pi, 61)
    mu_grid = np.cos(theta)
    dd_theta = []
    dt_theta = []
    unsort = np.argsort(np.argsort(mu_grid))
    d3he_theta = sigma_diff([E, 2*E], mu_grid, reac='d3he')
    coul_alpha = []
#    coul_alpha = coulomb_sigma_diff(E, mu_grid, 2, 2)

    for mu in mu_grid:
        dd_theta.append(sigma_diff(E, mu, reac='dd'))
        dt_theta.append(sigma_diff(E, mu, reac='dt'))
        coul_alpha.append(sigma_diff(E, mu, 2, 2))

# Plots

    plt.figure('Cross-sections', (19, 6))

    plt.subplot(1, 4, 1)
    plt.title('DD')
    plt.plot(np.degrees(theta), dd_theta)
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.subplot(1, 4, 2)
    plt.title('DT')
    plt.plot(np.degrees(theta), dt_theta)
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.subplot(1, 4, 3)
    plt.title(r'Coulomb $\alpha-\alpha$')
    plt.semilogy(np.degrees(theta), coul_alpha)
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.subplot(1, 4, 4)
    plt.title('D3He')
    plt.plot(np.degrees(theta), d3he_theta[0, unsort])
    plt.plot(np.degrees(theta), d3he_theta[1, unsort])
    plt.xlim([0, 180])
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    plt.show()
