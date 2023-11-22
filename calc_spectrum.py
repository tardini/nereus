import logging
import numpy as np
import numba as nb
import constants as const
from reactions import reaction
import calc_kinematics as ck
import calc_cross_section as cs

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('calc_spectrum')
logger.addHandler(hnd)
logger.setLevel(logging.DEBUG)

@nb.njit
def convolve_dists(v1_all, v2_all, versor_out, reac):

    if reac == 'D(T,n)4He':
        m_in1   = const.mDc2
        m_in2   = const.mTc2
        m_prod1 = const.mnc2
        m_prod2 = const.mHe4c2
    elif reac == 'D(D,n)3He':
        m_in1   = const.mDc2
        m_in2   = const.mDc2
        m_prod1 = const.mnc2
        m_prod2 = const.mHe3c2

    E_out = [0.] # numba needs it, then discard
    Ekin_out = [0.]
    cos_out = [0.]
    vrel_sq = [0.]
    m_prod_12 = m_prod1**2 - m_prod2**2
    m_12 = 1./(m_in1 + m_in2)

    for j in range(len(v1_all)):
        v1 = v1_all[j]
        gamma_in1 = 1./np.sqrt(1 - np.sum(v1**2))
        E_in1 = m_in1*gamma_in1
        p_in1 = E_in1*v1
        for i in range(len(v2_all)):
            v2 = v2_all[j]
            vcm = (m_in1*v1 + m_in2*v2) * m_12
            v1_cm_sq = 0.
            v2_cm_sq = 0.
            for j in range(3):
                v1_cm_sq += (v1[j] - vcm[j])**2
                v2_cm_sq += (v2[j] - vcm[j])**2
            Ekin = 0.5*m_in1*v1_cm_sq + \
                   0.5*m_in2*v2_cm_sq
            gamma_in2 = 1./np.sqrt(1 - np.sum(v2**2))
            E_in2 = m_in2*gamma_in2
            p_in2 = E_in2*v2
            E_tot = E_in1 + E_in2
            p_tot = p_in1 + p_in2
            A = 0.5*(E_tot**2 - np.sum(p_tot**2) + m_prod_12)/E_tot
            B = 0.
            for j in range(3):
                B += p_tot[j]*versor_out[j]
            B /= E_tot   # Well < 1
            Bsq1 = 1 - B**2
            disc = A**2 - m_prod1**2 * Bsq1
            if disc < 0:
                continue
            elif disc == 0:
                Eprod = [A/Bsq1]
            else:
                b_sq_disc = B*np.sqrt(disc)
                E1 = (A + b_sq_disc)/Bsq1
                E2 = (A - b_sq_disc)/Bsq1
                Eprod = [E1, E2]
            E_out += Eprod
            gamma_out1 = np.array(Eprod)/m_prod1
            v1_forw = np.sqrt(1 - 1/gamma_out1[0]**2)*versor_out
            v1_forw_norm = 0.
            cos_forw = 0.
            for j in range(3):
                v1_forw_norm += (v1_forw[j] - vcm[j])**2
                cos_forw += (v1[j] - vcm[j])*(v1_forw[j] - vcm[j])
            cos_forw /= np.sqrt(v1_cm_sq *v1_forw_norm)
            Ekin_out.append(Ekin)
            cos_out.append(cos_forw)
            if len(Eprod) == 2:
                v1_back = np.sqrt(1 - 1/gamma_out1[1]**2)*versor_out
                v1_back_norm = 0.
                cos_back = 0.
                for j in range(3):
                    v1_back_norm += (v1_back[j] - vcm[j])**2
                    cos_back += (v1[j] - vcm[j])*(v1_back[j] - vcm[j])
                cos_back /= np.sqrt(v1_cm_sq *v1_back_norm)
                Ekin_out.append(Ekin)
                cos_out.append(cos_back)

    E_out = np.array(E_out[1:]) - m_prod1
    return E_out, np.array(Ekin_out[1:]), np.array(cos_out[1:])


def mono_iso(E1, E2, versor_out, reac, n_sample=1e5):

    print(E1, E2, versor_out, reac, n_sample)
    m_in1 = reaction[reac].in1.m
    m_in2 = reaction[reac].in2.m

    versor_out = np.array(versor_out, dtype=np.float32)
    versor_out /= np.linalg.norm(versor_out)
    vx1, vy1, vz1 = ck.uniform_sample_versor(n_sample=n_sample)

    ver1 = np.vstack((vx1, vy1, vz1)).T
    ver2 = np.vstack((vz1, vx1, vy1)).T
    v1 = ck.E_m2vmod(E1 + m_in1, m_in1) * ver1
    v2 = ck.E_m2vmod(E2 + m_in2, m_in2) * ver2 # some randomness would be good here

    logger.info('Convolving distributions')
    Earr, Ekin, cos_the = convolve_dists(v1, v2, versor_out, reac)
    logger.info('Getting cross-sections')
    weight = cs.sigma_diff(Ekin, cos_the, reac, paired=True)

    return Earr, weight


def calc_spectrum(dens, Earr, weight, n_Ebins=40):

    Ecount, Eedges = np.histogram(Earr, bins=n_Ebins, weights=weight, density=False)
    Egrid = 0.5*(Eedges[1:] + Eedges[:-1])

    Espec = dens*dens*1e-31*Ecount/np.sum(weight) # 1e-31 because of mbarn-> m**2
    neut_tot = np.sum(Espec)*(Eedges[-1] - Eedges[0])
    logger.info('Tot neut.: %12.4e 1/(m**3 s)', neut_tot)

    return Egrid, Espec
