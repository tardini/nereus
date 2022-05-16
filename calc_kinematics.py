import sys, logging
import numpy as np
import constants as con
import calc_cross_section as cs
import numba as nb

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('kinema')
logger.addHandler(hnd)
logger.setLevel(logging.INFO)

# v in c units

def gamma(v):
    '''Compute relativistic gamma for given velocity as 1/sqrt(1 - |v|**2)'''

    v = np.array(v)
    return 1./np.sqrt(1 - np.sum(v**2))


def gamma2v(gamma):

    return np.sqrt(1 - 1/gamma**2)


def mv2qmom(m, v):
    '''m is the rest mass, v a 3-array of the particle velocity
Output: relativistic quadri-momentum'''

    v = np.array(v)
    E = m*gamma(v)
    p = E*v
    return np.append(E, p)


def v2Ekin(m, v):

    return m*gamma(v) - m


def E_m2vmod(E_in, m_in):

    gamma = E_in/m_in
    return gamma2v(gamma)


def E_m2pmod(E_in, m_in):

    gamma = E_in/m_in
    vmod = gamma2v(gamma)
    return m_in*gamma*vmod


def cos_the(v1, v2):

    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_prod == 0:
        return np.nan
    else:
        return np.dot(v1, v2)/(norm_prod)


@nb.njit
def E_prod1(m_prod1, A, B):
    '''Eq. 4 in J. Eriksson Comp. Phys. Comm 2016
https://www.sciencedirect.com/science/article/pii/S0010465515003902'''

    Bsq1 = 1 - B**2
    disc = A**2 - m_prod1**2 * Bsq1
    if disc < 0:
        return [np.nan]
    elif disc == 0:
        return [A/Bsq1]
    else:
        b_sq_disc = B*np.sqrt(disc)
        E1 = (A + b_sq_disc)/Bsq1
        E2 = (A - b_sq_disc)/Bsq1
        if E1 > E2:
            return [E2, E1]
        else:
            return [E1, E2]


@nb.njit
def uniform_sample_versor(n_sample=1e6):
    '''Regular theta grid, phi-grid size proportional to cos(theta)'''

    n_grid = int(np.sqrt(2*n_sample))
    theta_grid = np.linspace(-0.5*np.pi, 0.5*np.pi, n_grid+1)
    theta_grid = theta_grid[:-1] + 0.5*(theta_grid[1] - theta_grid[0])

    x = []
    y = []
    z = []
    for theta in theta_grid:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        n_phi = int(n_grid*cos_theta + 0.5)
        for jphi in range(n_phi):
            phi = 2*jphi*np.pi/float(n_phi)
            x.append(cos_theta*np.cos(phi))
            y.append(cos_theta*np.sin(phi))
            z.append(sin_theta)
    return np.array(x), np.array(y), np.array(z)

@nb.njit
def uniform_sample_versor2(n_sample=1e6):
    '''Cook-Von Neumann found at
https://mathworld.wolfram.com/SpherePointPicking.html'''

    n_grid = int(np.sqrt(np.sqrt(float(np.pi*n_sample))))
    xgrid = np.linspace(-1, 1, n_grid+1)
    xgrid = xgrid[:-1] + 0.5*(xgrid[1] - xgrid[0])

    x = []
    y = []
    z = []
    for x0 in xgrid:
        for x1 in xgrid:
            sq1 = x1**2 + x0**2
            if sq1 < 1:
                for x2 in xgrid:
                    sq2 = sq1 + x2**2
                    if sq2 < 1:
                        for x3 in xgrid:
                            sq3 = sq2 + x3**2
                            if sq3 < 1:
                                xloc = 2*(x1*x3 + x0*x2)
                                yloc = 2*(x2*x3 - x0*x1)
                                zloc = (x0**2 + x3**2 - x1**2 - x2**2)
                                norm = np.sqrt(xloc**2 + yloc**2 + zloc**2)
                                x.append(xloc/norm)
                                y.append(yloc/norm)
                                z.append(zloc/norm)

    return np.array(x), np.array(y), np.array(z)


class calc_reac:


    def __init__(self, v1, v2, versor_out, reac):


        versor_out = np.array(versor_out, dtype=np.float32)
        self.versor_out = versor_out/np.linalg.norm(versor_out)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        if reac == 'dt':
            self.m_in1   = con.mDc2
            self.m_in2   = con.mTc2
            self.m_prod1 = con.mnc2
            self.m_prod2 = con.mac2
        elif reac == 'dd':
            self.m_in1   = con.mDc2
            self.m_in2   = con.mDc2
            self.m_prod1 = con.mnc2
            self.m_prod2 = con.mHe3c2
        self.qmom_in1 = mv2qmom(self.m_in1, self.v1)
        self.qmom_in2 = mv2qmom(self.m_in2, self.v2)
        self.vcm = (self.m_in1*self.v1 + self.m_in2*self.v2)/(self.m_in1 + self.m_in2)
        self.Ekin = 0.5*self.m_in1*np.sum((self.v1 - self.vcm)**2) + \
                    0.5*self.m_in2*np.sum((self.v2 - self.vcm)**2)
        qmom_tot = self.qmom_in1 + self.qmom_in2
        E_tot = qmom_tot[0]
        p_tot = qmom_tot[1:]
        A = 0.5*(E_tot**2 - np.sum(p_tot**2) + self.m_prod1**2 - self.m_prod2**2)/E_tot
        B = np.dot(p_tot, self.versor_out)/E_tot   # Well < 1

        Eprod1 = E_prod1(self.m_prod1, A, B)
        self.qmom_prod1a = np.zeros(4)
        self.qmom_prod1a[0] = Eprod1[0]
        self.qmom_prod1a[1: ] = E_m2pmod(Eprod1[0], self.m_prod1)*self.versor_out
        self.qmom_prod2a = qmom_tot - self.qmom_prod1a
        self.Ekin_prod1a = self.qmom_prod1a[0] - self.m_prod1
        self.Ekin_prod2a = self.qmom_prod2a[0] - self.m_prod2
        self.v_out1a = self.qmom_prod1a[1: ]/self.qmom_prod1a[0]
        self.v_out2a = self.qmom_prod2a[1: ]/self.qmom_prod2a[0]
        self.cos_theta_a = cos_the(self.v1 - self.vcm, self.v_out1a - self.vcm)
# Ekin, cos_theta are relevant for the differential cross-sections
        self.sigma_diff_a = cs.sigma_diff(self.Ekin, self.cos_theta_a)
        if len(Eprod1) == 2:
            self.qmom_prod1b = np.zeros(4)
            self.qmom_prod1b[0] = Eprod1[1]
            self.qmom_prod1b[1: ] = E_m2pmod(Eprod1[1], self.m_prod1)*self.versor_out
            self.qmom_prod2b = qmom_tot - self.qmom_prod1b
            self.Ekin_prod1b = self.qmom_prod1b[0] - self.m_prod1
            self.Ekin_prod2b = self.qmom_prod2b[0] - self.m_prod2
            self.v_out1b = self.qmom_prod1b[1: ]/self.qmom_prod1b[0]
            self.v_out2b = self.qmom_prod2b[1: ]/self.qmom_prod2b[0]
            self.cos_theta_b = cos_the(self.v1 - self.vcm, self.v_out1b - self.vcm)
            self.sigma_diff_b = cs.sigma_diff(self.Ekin, self.cos_theta_b)
 

class out_versor_scan:

    def __init__(self, v1, v2, reac, n_sample=1000, sample_flag=1):

        '''alpha means just first reaction product, not alpha particle'''

        logger.info('Sampling versor direction')
        if sample_flag == 1:
            self.x, self.y, self.z = uniform_sample_versor(n_sample=n_sample)
        elif sample_flag == 2:
            self.x, self.y, self.z = uniform_sample_versor2(n_sample=n_sample)
        if reac == 'dt':
            self.m_in1   = con.mDc2
            self.m_in2   = con.mTc2
            self.m_prod1 = con.mnc2
            self.m_prod2 = con.mac2
        elif reac == 'dd':
            self.m_in1   = con.mDc2
            self.m_in2   = con.mDc2
            self.m_prod1 = con.mnc2
            self.m_prod2 = con.mHe3c2
        nx = len(self.x)
        print(nx)
        logger.info('Compute Eneut')
        self.qmom_tot = mv2qmom(self.m_in1, v1) + mv2qmom(self.m_in2, v2)
        E_tot = self.qmom_tot[0]
        p_tot = self.qmom_tot[1:]
        A = 0.5*(E_tot**2 - np.sum(p_tot**2) + self.m_prod1**2 - self.m_prod2**2)/E_tot
        B = (p_tot[0]*self.x + p_tot[1]*self.y + p_tot[2]*self.z)/E_tot # array(n_sample)
        self.E_forw = []
        self.E_back = []
        for jv in range(nx):
            Eprod1 = E_prod1(self.m_prod1, A, B[jv])
            if len(Eprod1) == 2: # different alpha recoil; not same probability
                self.E_forw.append(Eprod1[0] - self.m_prod1)
                self.E_back.append(Eprod1[1] - self.m_prod1)
            elif(len(Eprod1) == 1):
                print(Eprod1)


if __name__ == '__main__':

    import matplotlib.pylab as plt

    nsamp1 = 1e6
    nsamp2 = 1e6 #5e6
    v0 = 1e-2
    v1 = [v0, v0, v0]
    v2 = [-0.5*v0, 0, 0]
    scan1 = out_versor_scan(v1, v2, 'dt', n_sample=nsamp1, sample_flag=2)
    scan2 = out_versor_scan(v1, v2, 'dt', n_sample=nsamp2, sample_flag=1)
    logger.info('Plotting Eneut')
    n_Ebins = 80
    count1b, Eedges = np.histogram(scan1.E_back, bins=n_Ebins)
    count1f, Eedges = np.histogram(scan1.E_forw, bins=n_Ebins)
    count2b, Eedges = np.histogram(scan2.E_back, bins=n_Ebins)
    count2f, Eedges = np.histogram(scan2.E_forw, bins=n_Ebins)
    Egrid = 0.5*(Eedges[1:] + Eedges[:-1])

    versor_out = [1, 1, 0]
    kin = calc_reac(v1, v2, versor_out, 'dt')
    print('Cross section', kin.sigma_diff_a)
    print('Cross section root2', kin.sigma_diff_b)

#------
# Plots
#------

    nx1 = len(scan1.x)
    nx2 = len(scan2.x)

    fig1 = plt.figure('E_neut', (12,9))
    plt.plot(Egrid, count1b/nx1, label='Back, #=1e6')
    plt.plot(Egrid, count1f/nx1, label='Forw, #=1e6')
    plt.plot(Egrid, count2f/nx2, label='Forw, th-ph, #=1e6')
    plt.plot(Egrid, count2b/nx2, label='Back, th-ph, #=1e6')
    plt.xlim([12, 16])
    plt.legend()

# Check versor uniformity on unit sphere

#    step = 50
#
#    fig_samp1 = plt.figure('Uniform sampling1', (13, 10))
#    ax_samp1 = fig_samp1.add_subplot(projection='3d')
#    ax_samp1.scatter(scan1.x[::step], scan1.y[::step], scan1.z[::step], marker='o')
#
#    fig3 = plt.figure('Uniform sampling sph', (13, 10))
#    ax3 = fig3.add_subplot(projection='3d')
#    ax3.scatter(scan3.x[::step], scan3.y[::step], scan3.z[::step], marker='o')
    fig_sol = plt.figure('Solutions', (13, 10))
    ax_sol = fig_sol.add_subplot(projection='3d')
    print(kin.qmom_in1[1:])
    print(kin.qmom_in2[1:])
    print(kin.qmom_prod1a[1:])
    print(kin.qmom_prod2a[1:])
    print(kin.qmom_prod1b[1:])
    print(kin.qmom_prod2b[1:])
    plt.plot([0, kin.qmom_in1[1]], [0, kin.qmom_in1[2]], [0, kin.qmom_in1[3]], 'k-')
    plt.plot([0, kin.qmom_in2[1]], [0, kin.qmom_in2[2]], [0, kin.qmom_in2[3]], 'k--')
    if hasattr(kin, 'qmom_prod1b'):
        plt.plot([0, kin.qmom_prod1b[1]], [0, kin.qmom_prod1b[2]], [0, kin.qmom_prod1b[3]], 'b-')
        plt.plot([0, kin.qmom_prod2b[1]], [0, kin.qmom_prod2b[2]], [0, kin.qmom_prod2b[3]], 'b--')
    plt.plot([0, kin.qmom_prod1a[1]], [0, kin.qmom_prod1a[2]], [0, kin.qmom_prod1a[3]], 'r-')
    plt.plot([0, kin.qmom_prod2a[1]], [0, kin.qmom_prod2a[2]], [0, kin.qmom_prod2a[3]], 'r--')
    plt.xlim([-200, 200])
    plt.ylim([-200, 200])

    plt.show()
