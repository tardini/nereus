import logging
import numpy as np
import constants as con
import calc_kinematics as ck
import numba as nb

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('calc_spectrum')
logger.addHandler(hnd)
logger.setLevel(logging.INFO)


def mono_iso(E1, E2, n1, n2, versor_out, reac='dt', n_sample=1e5):

    if reac == 'dt':
        m_in1   = con.mDc2
        m_in2   = con.mTc2
    elif reac == 'dd':
        m_in1   = con.mDc2
        m_in2   = con.mDc2

    versor_out = np.array(versor_out, dtype=np.float32)
    versor_out /= np.linalg.norm(versor_out)
    vx1, vy1, vz1 = ck.uniform_sample_versor(n_sample=100)

    ver = np.vstack((vx1, vy1, vz1)).T
    v1 = ck.E_m2vmod(E1+m_in1, m_in1) * ver
    v2 = ck.E_m2vmod(E2+m_in2, m_in2) * ver

    print(con.mDc2, con.mnc2, con.mHe3c2)
    print(2.*con.mDc2 - con.mnc2 - con.mHe3c2)
    n_part = len(v1)
    E_out = []
    sigma_out = []
    for jv, v_1 in enumerate(v1):
        logger.info('%d %10.2e %10.2e %10.2e', jv, *v_1)
        for v_2 in v2:
            rea = ck.calc_reac(v_1, v_2, versor_out, reac=reac)
            E_out.append(rea.Ekin_prod1a)
            sigma_out.append(rea.sigma_diff_a)
    return np.array(E_out), np.array(sigma_out)


if __name__ == '__main__':

    import matplotlib.pylab as plt

    dens = 4.e19
    Earr, sigma = mono_iso(0.03, 0.01, dens, dens, [1, 0, 0], reac='dt')

    n_Ebins = 40
    nx = len(Earr)
    Espec, Eedges = np.histogram(Earr, bins=n_Ebins)
    Egrid = 0.5*(Eedges[1:] + Eedges[:-1])
    sigma_avg = np.nanmean(sigma)
    print(sigma_avg)
    plt.plot(Egrid, 16e7*sigma_avg*Espec/float(nx))
    plt.show()
