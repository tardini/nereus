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

# BUGS: missing 1 MeV in dd->nHe3 reactoin!
# How to adapt sigma to each event, instead of using sigma_avg?

# Need NumBa here

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
    v2 = ck.E_m2vmod(E2+m_in2, m_in2) * ver # some randomness would be good here

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
            if hasattr(rea, 'Ekin_prod1b'):
                E_out.append(rea.Ekin_prod1b)
                sigma_out.append(rea.sigma_diff_b)
    return np.array(E_out), np.array(sigma_out)


if __name__ == '__main__':

    import matplotlib.pylab as plt

    dens = 4.e19
    Earr, sigma = mono_iso(0.003, 0.01, dens, dens, [1, 0, 0], reac='dt')

    n_Ebins = 40
    nx = len(Earr)
    Ecount, Eedges = np.histogram(Earr, bins=n_Ebins)
    Egrid = 0.5*(Eedges[1:] + Eedges[:-1])
    sigma_avg = np.nanmean(sigma)

    Espec = dens*dens*1e-31*sigma_avg*Ecount
    print('Tot neut.: %12.4e 1/(m**3 s)' %(np.sum(Espec)*n_Ebins))
    fig = plt.figure('Spectrum', (13, 8))
    plt.subplot(1, 2, 1)
    plt.plot(Egrid, Espec/float(nx))
    
    plt.subplot(1, 2, 2)
    plt.plot(Earr, sigma)
    plt.show()
