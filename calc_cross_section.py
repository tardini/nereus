import numpy as np
from scipy.interpolate import interp1d
from constants import epsilon0, echarge, reac_lbls
import cross_section_tables as cs_tab


def sigma_diff(E_in_MeV, mu_in, reac, Z1=None, Z2=None, paired=False):
    '''General method redirecting to the relevant reaction method'''

    if reac in ('dp', 'd3he', 'alphad', 'alphat', 'd3healphap'):
        return tabulated_sigma_diff(E_in_MeV, mu_in, reac, paired=paired)
    elif reac in ('dt', 'ddn3he'):
        return legendre_sigma_diff(E_in_MeV, mu_in, reac, paired=paired)
    elif Z1 is not None:
        return coulomb_sigma_diff(E_in_MeV, mu_in, Z1, Z2)


def coulomb_sigma_diff(E_in_MeV, mu_in, Z1, Z2):
    '''q1, q2 are the electric charge numbers of the scattering particles'''

    q1 = Z1*echarge
    q2 = Z2*echarge
    E_in_MeV = np.atleast_1d(E_in_MeV)
    mu_in    = np.atleast_1d(mu_in)
    E_in_J = 1e6*echarge*E_in_MeV
    k = 1./(4*np.pi*epsilon0)
    coul = q1*q2*k
    s = coul/np.outer(1 - mu_in, E_in_J) # denominator=0 yields nan, as wished
    cross_sec = 0.25* s**2 * 10**31 # 10**31: m**2 -> mbarn
    return np.squeeze(cross_sec)


def tabulated_sigma_diff(E_in_MeV, mu_in, reac, paired=False):
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
    E_in_MeV = np.atleast_1d(E_in_MeV)
    mu_in    = np.atleast_1d(mu_in)
    if paired:
        res = f(E_in_MeV, mu_in)
    else:
        res = f(E_in_MeV, mu_in).T
        if len(mu_in > 1):
            unsorted = np.argsort(np.argsort(mu_in))
            res = res[..., unsorted]
    return np.squeeze(res)


def legendre_sigma_diff(E_in_MeV, mu_in, reac, paired=False):

    from scipy.special import eval_legendre

    E_in_MeV = np.atleast_1d(E_in_MeV)
    mu_in    = np.atleast_1d(mu_in)

    if reac == 'dt':
        cs = cs_tab.DT
    elif reac == 'ddn3he':
        cs = cs_tab.DDn3He
    n_leg = cs.leg_coeff.shape[1]
    data = interp1d(cs.En, cs.leg_coeff, axis=0)
    data_E = data(E_in_MeV)
    cs_leg = np.zeros((len(mu_in), n_leg))
    for jleg in range(n_leg):
        cs_leg[:, jleg] = eval_legendre(jleg, mu_in)
    if paired:
        print(cs_leg.shape, data_E.shape)
        cs_sum = np.sum(cs_leg*data_E, axis=1)
    else:
        cs_sum = np.tensordot(cs_leg, data_E, axes=([1, 1]))

    leg_tot = legendre_sigma_tot(E_in_MeV, reac)

    return np.squeeze(leg_tot*cs_sum/(4*np.pi*data_E[:, 0]))


def legendre_sigma_tot(E_in_MeV, reac, Emin_MeV=5.e-4):

    if reac == 'dt':
        cs = cs_tab.DT
    elif reac == 'ddn3he':
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
    
    return np.squeeze(leg_out)


if __name__ == '__main__':

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
    dt_diff = sigma_diff(E, mu, reac='dt')
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
    coul_diff = sigma_diff(E, mu, 'coul', Z1=2, Z2=2)
    print('%12.4e millibarn' %coul_diff)
