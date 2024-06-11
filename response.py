import os, sys, datetime, logging
import numpy as np
import rw_for
from scipy.io import netcdf_file

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('resp')
logger.setLevel(level=logging.DEBUG)
hnd  = logging.StreamHandler()
hnd.setLevel(level=logging.INFO)
hnd.setFormatter(fmt)
logger.addHandler(hnd)
logger.propagate = False

nereusDir = os.path.dirname(os.path.realpath(__file__))
responseDir = '%s/responses' %nereusDir
out_dir = '%s/output' %nereusDir
os.system('mkdir -p %s' %out_dir)


def gauss_kernel(Emid, sigma, Ec_sim):

    dE = Ec_sim/2.5066

#  Gaussian convolution kernel

    gauss_exp = (Emid[:, None] - Emid[None, :])/sigma[None, :]
    gau_kernel =  dE/sigma[None, :]*np.exp(-0.5*gauss_exp**2)

    return gau_kernel


class RESP:


    def __init__(self):

        pass


    def from_nresp(self, f_spc='%s/SPECT_MPI.DAT' %responseDir):

        self.spc_d = {}
        self.tof_d = {}

        logger.info('Reading file %s' %f_spc)

        f = open(f_spc,'r')
        lines = f.readlines()
        f.close()

        sEbin = lines[0]
        self.Ebin_MeVee = float(sEbin)

        En     = []
        En_wid = []
        self.spc_d  = []
        self.count  = []
        sEn_list = []

        jEn = -1
        for lin in lines[1:]:
            slin = lin.strip()
            if (slin == ''):
                continue
            sarr = slin.split()
            try:
                tmp = float(sarr[0])
                for snum in sarr:
                    self.spc_d[jEn][lbl].append(float(snum))
            except:
                lbl, sEn, sEn_wid, snr = slin.split()
                if sEn not in sEn_list:
                    jEn += 1
                    En.append(float(sEn))
                    En_wid.append(float(sEn_wid))
                    self.spc_d.append({})
                    self.count.append({})
                    sEn_list.append(sEn)
                self.spc_d[jEn][lbl] = []
                if lbl[:5] == 'PP3AS':
                    self.count[jEn][lbl] = 0
                else:
                    self.count[jEn][lbl] = int(snr)

        self.En_MeV     = 1e-3*np.array(En    , dtype=np.float32)
        self.En_wid_MeV = 1e-3*np.array(En_wid, dtype=np.float32)
        nEn = len(self.En_MeV)
        self.phs_max = 0
        for j in range(nEn):
            for lbl, spec in self.spc_d[j].items():
                self.phs_max = max(self.phs_max, len(spec))

        self.EphsB_MeVee = self.Ebin_MeVee*np.arange(self.phs_max + 1)
        self.Ephs_MeVee = 0.5*(self.EphsB_MeVee[1:] + self.EphsB_MeVee[:-1])
        self.RespMat = np.zeros((nEn, self.phs_max))

        for jEn in range(nEn):
            for lbl, arr in self.spc_d[jEn].items():
                if lbl[:5] != 'PP3AS':
                    myarr = np.zeros(self.phs_max)
                    nloc = len(arr)
                    myarr[:nloc] = arr[:nloc]
                    self.RespMat[jEn, :] += myarr


    def from_cdf(self, f_cdf):

        logger.info('Reading file %s' %f_cdf)

        cv = netcdf_file(f_cdf, 'r', mmap=False).variables

        self.Ebin_MeVee  = cv['Ebin'][:]
        self.RespMat     = cv['ResponseMatrix'].data
        self.En_MeV      = cv['E_NEUT'][:]
        self.En_wid_MeV  = cv['En_wid'][:]
        self.EphsB_MeVee = cv['E_light_B'][:]
        self.Ephs_MeVee  = cv['E_light'][:]


    def from_hepro(self, f_hep='%s/simresp.rsp' %responseDir):

        logger.info('Reading file %s' %f_hep)

        f = open(f_hep, 'r')

        data = []
        for line in f.readlines():
            data += line.split()
        f.close()

        dsim = np.array(data, dtype = np.float32)
        n_dsim = len(dsim)
        logger.debug('N_DSIM %d', n_dsim)
        ndim = []
        dE = []
        en1 = []
        en2 = []
        spc = []
        self.Ebin_MeVee = dsim[0]
        j = 1
        while j < n_dsim:
            dE.append(dsim[j])
            nsize = int(dsim[j+1])
            ndim.append(nsize)
            en1.append(dsim[j+2])
            en2.append(dsim[j+3])
            j += 4
            logger.debug('nsize %d %d', nsize, j)
            spc.append(dsim[j:j+nsize])
            j += nsize
        self.En_MeV = np.array(dE, dtype=np.float32)
        en1  = np.array(en1, dtype=np.float32)
        en2  = np.array(en2, dtype=np.float32)
        self.En_wid_Mev = en2 - en1
        ndims = np.array(ndim, dtype=np.int32)
        nEn = len(self.En_MeV)
        self.phs_max = np.max(ndims)
        self.RespMat = np.zeros((nEn, self.phs_max), dtype=np.float32)
        for jEn, phs in enumerate(spc):
            nphs = len(phs)
            self.RespMat[jEn, :nphs] = phs[:nphs]
        self.EphsB_MeVee = self.Ebin_MeVee*np.arange(self.phs_max + 1)
        self.Ephs_MeVee = 0.5*(self.EphsB_MeVee[1:] + self.EphsB_MeVee[:-1])


    def broaden(self, f_par='%s/neut_fit.txt' %responseDir):
        '''Gaussian broadening of the response function'''

# Broadening parameters

        logger.info('Gaussian convolution')
        f = open(f_par, 'r')
        lin = f.readlines()[13].split()
        a = float(lin[3])
        b = float(lin[4])
        c = float(lin[5])
        z = float(lin[6])
        Ec_fac = float(lin[7])
        logger.debug('%8.4f %8.4f %8.4f %8.4 %8.4f', a, b, c, z, Ec_fac)

        Emid = self.Ephs_MeVee
        sigma = np.sqrt(a**2 * Emid**2 + b**2 * Emid + c**2)/235.48

        logger.info('Gaussian kernel from file %s', f_par)
        gau_ker = gauss_kernel(self.Ephs_MeVee, sigma, self.Ebin_MeVee)
        self.RespMat_gb = np.einsum('ij,jk->ik', self.RespMat, gau_ker)


    def to_hepro(self, fout='%s/ddnpar.asc' %responseDir):

        f = open(fout, 'w')
        f.write('  %13.7e\n' %self.Ebin_MeVee)
        n_En, n_spc = self.RespMat.shape
        for jEn in range(n_En):
            f.write('  %11.5f       %5d  %11.5f  %11.5f\n' %(self.En_MeV[jEn], n_spc, self.EphsB_MeVee[0], self.EphsB_MeVee[-1]))
            spc_str = rw_for.wr_for(self.RespMat[jEn, :], fmt=' %13.6e', n_lin=6)
            f.write(spc_str)
        f.close()
        logger.info('Written %s' %fout)


    def to_cdf(self, f_cdf=None):

        if f_cdf is None:
            f_cdf = '%s/rm.cdf' %out_dir
        fcdf = f_cdf
        jcdf = 0
        while os.path.isfile(fcdf):
            jcdf += 1
            fcdf = '%s%s' %(f_cdf, jcdf)

        len_pos = []
        nEn = len(self.En_MeV)
        for jEn in range(nEn):
            phs = self.RespMat[jEn, :]
            if phs.any() > 0:
                (ind_pos, ) = np.where(phs > 0.)
                len_pos.append(ind_pos[-1])
            else:
                len_pos.append(0)
        jEn_max = np.argmax(len_pos)
        nEp = min( len_pos[jEn_max] + 1, len(self.RespMat[jEn_max, :]) ) # can get down by one

# NetCDF output

        f = netcdf_file(fcdf, 'w', mmap=False)

        f.history = "Created " + datetime.datetime.today().strftime("%d/%m/%y")

        f.createDimension('E_NEUT'   , nEn)
        f.createDimension('E_light'  , nEp)
        f.createDimension('E_light_B', nEp + 1)
        f.createDimension('Ebin_dim' , 1)

        En = f.createVariable('E_NEUT', np.float32, ('E_NEUT', ))
        En[:] = self.En_MeV
        En.units = 'MeV'
        En.long_name = 'Neutron energy'

        Ewid = f.createVariable('En_wid', np.float32, ('E_NEUT', ))
        Ewid[:] = self.En_wid_MeV
        Ewid.units = 'MeV'
        Ewid.long_name = 'NRESP-energy width for each En'

        Ep = f.createVariable('E_light', np.float32, ('E_light', ))
        Ep[:] = self.Ephs_MeVee[:nEp]
        Ep.units = 'MeVee'
        Ep.long_name = 'Equivalent photon energy grid'

        EpB = f.createVariable('E_light_B', np.float32, ('E_light_B', ))
        EpB[:] = self.EphsB_MeVee[:nEp+1]
        EpB.units = 'MeVee'
        EpB.long_name = 'Equivalent photon energy bins'

        Ebin = f.createVariable('Ebin', np.float32, ('Ebin_dim', ))
        Ebin[:] = self.Ebin_MeVee
        Ebin.units = 'MeVee'
        Ebin.long_name = 'Step for PHS bins'

        rm = f.createVariable('ResponseMatrix', np.float32, ('E_NEUT', 'E_light'))
        rm.units = '1/(s MeVee)'
        rm.long_name = 'Response functions for several neutron energies'
        rm[:] = self.RespMat[:, :nEp]

        f.close()
        logger.info('Stored %s' %fcdf)


if __name__ == "__main__":

    import matplotlib.pylab as plt

    f_nresp = '%s/SPECT_MPI.DAT' %responseDir
#    f_nresp = 'nresp.dat'
    f_cdf = 'rm2.cdf'

# Store NetCDF
    rsp = RESP()
    rsp.from_nresp(f_spc=f_nresp)
    rsp.to_cdf(f_cdf)
# Store HEPRO output for TOFANA
    rsp.to_hepro()

# Gaussian broadening, storing as cdf
    rsp.from_cdf(f_cdf)
    rsp.broaden()
    rsp.RespMat = rsp.RespMat_gb

# Store gaussian-broadened into NetCDF
    fcdf_out = 'rm_bg.cdf'
    rsp.to_cdf(fcdf_out)

# Plot a few functions

    cv1 = netcdf_file('rm2.cdf'  , 'r', mmap=False).variables
    cv2 = netcdf_file('rm_bg.cdf', 'r', mmap=False).variables

    Emid = cv1['E_light'][:]
    En = cv1['E_NEUT'][:]
    jEn = 0
    plt.plot(Emid, cv1['ResponseMatrix'][jEn] , 'r-', label='En = %5.3f MeV'     %En[jEn])
    plt.plot(Emid, cv2['ResponseMatrix'][jEn] , 'g-', label='En = %5.3f MeV, GB' %En[jEn])
    jEn = -10
    plt.plot(Emid, cv1['ResponseMatrix'][jEn], 'm-', label='En = %5.3f MeV'     %En[jEn])
    plt.plot(Emid, cv2['ResponseMatrix'][jEn], 'b-', label='En = %5.3f MeV, GB' %En[jEn])

    plt.xlim([0, 5])
    plt.ylim([0, 1])
    plt.legend()

    plt.show()
