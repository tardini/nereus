import logging
import numpy as np
import constants as con
import calc_kinematics as ck
import calc_cross_section as cs
import numba as nb

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('calc_spectrum')
logger.addHandler(hnd)
logger.setLevel(logging.INFO)

@nb.njit
def convolve_dists(v1_all, v2_all, versor_out, reac):

    if reac == 'dt':
        m_in1   = con.mDc2
        m_in2   = con.mTc2
        m_prod1 = con.mnc2
        m_prod2 = con.mac2
    elif reac == 'dd':
        m_in1   = con.mDc2
        m_in2   = con.mDc2
        m_prod1 = con.mnc2
        m_prod2 = con.mHe3c2

    E_out = [0.]
    Ekin_out = [0.]
    cos_out = [0.]
    m_prod_12 = m_prod1**2 - m_prod2**2
    m_12 = 1./(m_in1 + m_in2)
    for jv, v1 in enumerate(v1_all):
#        logger.debug('%d %10.2e %10.2e %10.2e', jv, *v1)
        gamma_in1 = 1./np.sqrt(1 - np.sum(v1**2))
        E_in1 = m_in1*gamma_in1
        p_in1 = E_in1*v1
        for v2 in v2_all:
            vcm = (m_in1*v1 + m_in2*v2) * m_12
            v1_cm = v1 - vcm
            Ekin = 0.5*m_in1*np.sum((v1_cm)**2) + \
                   0.5*m_in2*np.sum((v2 - vcm)**2)
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
            v1_cm_norm = 0.
            v1_forw_norm = 0.
            cos_forw = 0.
            for j in range(3):
                v1_cm_norm += v1_cm[j]**2
                v1_forw_norm += (v1_forw[j] - vcm[j])**2
                cos_forw += v1_cm[j]*(v1_forw[j] - vcm[j])
            cos_forw /= np.sqrt(v1_cm_norm *v1_forw_norm)
            Ekin_out.append(Ekin)
            cos_out.append(cos_forw)
            if len(Eprod) == 2:
                v1_back = np.sqrt(1 - 1/gamma_out1[1]**2)*versor_out
                v1_back_norm = 0.
                cos_back = 0.
                for j in range(3):
                    v1_back_norm += (v1_back[j] - vcm[j])**2
                    cos_back += v1_cm[j]*(v1_back[j] - vcm[j])
                cos_back /= np.sqrt(v1_cm_norm *v1_back_norm)
                Ekin_out.append(Ekin)
                cos_out.append(cos_back)

    E_out = np.array(E_out[1:]) - m_prod1
    return E_out, np.array(Ekin_out[1:]), np.array(cos_out[1:])


def convolve_dists2(v1, v2, versor_out, reac):

    E_out = []
    sigma_out = []
    for jv, v_1 in enumerate(v1):
        logger.debug('%d %10.2e %10.2e %10.2e', jv, *v_1)
        for v_2 in v2:
            rea = ck.calc_reac(v_1, v_2, versor_out, reac)
            E_out.append(rea.Ekin_prod1a)
            sigma_out.append(rea.sigma_diff_a)
            if hasattr(rea, 'Ekin_prod1b'):
                E_out.append(rea.Ekin_prod1b)
                sigma_out.append(rea.sigma_diff_b)
    return np.array(E_out), np.array(sigma_out)


def mono_iso(E1, E2, n1, n2, versor_out, reac, n_sample=1e5):

    if reac == 'dt':
        m_in1   = con.mDc2
        m_in2   = con.mTc2
    elif reac == 'dd':
        m_in1   = con.mDc2
        m_in2   = con.mDc2

    versor_out = np.array(versor_out, dtype=np.float32)
    versor_out /= np.linalg.norm(versor_out)
    vx1, vy1, vz1 = ck.uniform_sample_versor(n_sample=n_sample)

    ver = np.vstack((vx1, vy1, vz1)).T
    v1 = ck.E_m2vmod(E1+m_in1, m_in1) * ver
    v2 = ck.E_m2vmod(E2+m_in2, m_in2) * ver # some randomness would be good here

    print(con.mDc2, con.mnc2, con.mHe3c2)
    print(2.*con.mDc2 - con.mnc2 - con.mHe3c2)
    logger.info('Convolving distributions')
    Earr, Ekin, cos_the = convolve_dists(v1, v2, versor_out, reac)
    logger.info('Getting cross-sections')
    sigma_diff = cs.legendre_sigma_diff(Ekin, cos_the)

    return Earr, sigma_diff


if __name__ == '__main__':

    import matplotlib.pylab as plt

    dens = 4.e19
    Earr, sigma = mono_iso(0.003, 0.01, dens, dens, [0, 0, 1], 'dd', n_sample=2000)

    logger.info('Creating spectrum histogram')
    n_Ebins = 40
    nx = len(Earr)
    Ecount, Eedges = np.histogram(Earr, bins=n_Ebins, weights=sigma)
    Egrid = 0.5*(Eedges[1:] + Eedges[:-1])

    Espec = dens*dens*1e-31*Ecount/float(nx) # 1e-31 because of mbarn-> m**2
    neut_tot = np.sum(Espec)*(Eedges[-1] - Eedges[0])
    logger.info('Tot neut.: %12.4e 1/(m**3 s)', neut_tot)
    fig = plt.figure('Spectrum', (13, 8))
    plt.plot(Egrid, Espec)
    plt.show()
