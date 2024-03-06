import os, json, logging
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
import matplotlib.pylab as plt
from nresp.settings import nrespDir

logger = logging.getLogger('nresp.cs')
logger.setLevel(level=logging.DEBUG)

crossDir='%s/cross-sections' %nrespDir
flt_typ = np.float64

class crossSections:

    
    def __init__(self, json=True):
        
        if json:
            self.fromJSON()


    def fromJSON(self):
        '''Read cross-sections from JSON files'''

        logger.info('Reading JSON cross-sections from dir "%s"', crossDir)
        f_json = '%s/crossSections.json' %crossDir
        with open(f_json, 'r') as fjson:
            crSec = json.load(fjson)
        self.EgridTot  = crSec['EgridTot']
        self.Egrid = 0.02*np.arange(1001)
        f_json = '%s/alphas3.json' %crossDir
        with open(f_json, 'r') as fjson:
            self.alphas3 = json.load(fjson)
        CSalphas3 = np.array(self.alphas3['crossSec'], dtype=flt_typ).T

        self.int_alphas3 = interp1d(self.alphas3['Egrid'], CSalphas3, axis=1, assume_sorted=True, kind='linear', fill_value='extrapolate')
        self.max_level = CSalphas3.shape[0]

        self.crSec_d  = {}
        csDiff_d = {}
        self.reacTot  = []
        self.reacDiff = []
        for reac in crSec['reactions']:
            f_json = '%s/%s.json' %(crossDir, reac)
            if os.path.isfile(f_json):
                with open(f_json, 'r') as fjson:
                    self.crSec_d[reac] = json.load(fjson)
                    if 'crossTot' in self.crSec_d[reac].keys():
                        self.reacTot.append(reac)
                        if 'EgridTot' not in self.crSec_d[reac].keys():
                            nE = len(self.crSec_d[reac]['crossTot'])
                            self.crSec_d[reac]['EgridTot'] = self.EgridTot[-nE:]
                    if 'crossDiff' in self.crSec_d[reac].keys():
                        self.reacDiff.append(reac)
                        crSecArray = 1.e-4*np.array(self.crSec_d[reac]['crossDiff'], dtype=flt_typ)
                        n_the, nE = crSecArray.shape
                        if 'n_theta' not in globals():
                            n_theta = n_the + 2
                            theta_grid = np.linspace(0, 1, n_theta, endpoint=True)
                        csDiff_d[reac] = np.vstack((np.zeros(nE), crSecArray, np.pi + np.zeros(nE))).T

        self.reacTotUse = self.reacTot[:10] # exclude CarTot, AlTot, HE1, HE2

        self.csd_d = {}
        for reac in self.reacDiff:
            self.csd_d[reac] = RectBivariateSpline(self.crSec_d[reac]['EgridDiff'], theta_grid, csDiff_d[reac], kx=2, ky=2)

        self.cst1d = {}
        for reac in self.reacTot:
            Interp = interp1d(self.crSec_d[reac]['EgridTot'], self.crSec_d[reac]['crossTot'], kind='linear', assume_sorted=True, fill_value='extrapolate')
            self.cst1d[reac] = Interp(self.Egrid)


    def cosInterpReac2d(self, reac, En_in, randomAngle):

        if reac in self.reacDiff:
            return np.cos(self.csd_d[reac](En_in, randomAngle, grid=False))
        else:
            logger.error('No differential cross-section for label "%s"', reac)
            return None


    def plot(self):

        plt.figure('Cross-sections', (14, 5))
        cs_c  = np.zeros_like(np.float32(self.EgridTot))
        cs_al = np.zeros_like(np.float32(self.EgridTot))
        for reac in self.reacTot:
            if reac[:6] == "12C(N,":
                plt.subplot(1, 3, 2)
                En = self.crSec_d[reac]['EgridTot']
                cs = self.crSec_d[reac]['crossTot']
                cs_tmp = np.interp(self.EgridTot, En, cs)
                cs_c += cs_tmp
                plt.plot(En, cs, label=reac)
            elif reac[:8] == "27AL(N,N":
                plt.subplot(1, 3, 3)
                En = self.crSec_d[reac]['EgridTot']
                cs = self.crSec_d[reac]['crossTot']
                cs_tmp = np.interp(self.EgridTot, En, cs)
                cs_al += cs_tmp
                plt.plot(En, cs, label=reac)

        plt.subplot(1, 3, 1)
        reac = "H(N,N)H"
        plt.plot(self.crSec_d[reac]['EgridTot'], self.crSec_d[reac]['crossTot'], label=reac)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.ylim([0, 5000])
        reac = 'CarTot'
        plt.plot(self.crSec_d[reac]['EgridTot'], self.crSec_d[reac]['crossTot'], label=reac)
        plt.plot(self.EgridTot, cs_c, label='Sum')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.ylim([0, 6000])
        reac = 'AlTot'
        plt.plot(self.crSec_d[reac]['EgridTot'], self.crSec_d[reac]['crossTot'], label=reac)
        plt.plot(self.EgridTot, cs_al, label='Sum')
        plt.legend()
    
        plt.show()

#  LocalWords:  interp
