#!/usr/bin/env python

import os, logging
import numpy as np
import matplotlib.pylab as plt
from multiprocessing import Pool, cpu_count
from nresp.en2light import En2light
from nresp import crossSections
import rw_for

nrespDir = os.path.dirname(os.path.realpath(__file__))
os.system('mkdir -p %s/output' %nrespDir)

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('nresp')
logger.setLevel(level=logging.DEBUG)
hnd  = logging.StreamHandler()
flog = '%s/output/nresp.log' %nrespDir
fhnd = logging.FileHandler(flog, mode='w') # mode='a' for appending
hnd.setLevel(level=logging.INFO)
fhnd.setLevel(level=logging.DEBUG)
hnd.setFormatter(fmt)
fhnd.setFormatter(fmt)
logger.addHandler(hnd)
logger.addHandler(fhnd)
logger.propagate = False

CS = crossSections.crossSections()
#CS.plot()
flt_typ = np.float64
int_typ = np.int32


class NRESP:


    def __init__(self, nresp_set, parallel=True):

        self.reac_names = [x for x in CS.reacTotUse]
        self.reac_names.append('light-guide')
        self.En_MeV = np.atleast_1d(eval(nresp_set['Energy array']))
        self.nresp_set = nresp_set
        self.nEn = len(self.En_MeV)
        jmax = np.argmax(self.En_MeV)
        self.n_react = len(self.reac_names)
        self.En_wid_MeV = self.nresp_set['En_wid_frac']*self.En_MeV
        En_wid = self.En_wid_MeV[jmax]
        if self.nresp_set['distr'] == 'mono':
            self.phs_max = int(self.En_MeV[jmax]/self.nresp_set['Ebin_MeVee'])
        elif self.nresp_set['distr'] == 'gauss':
            self.phs_max = int((self.En_MeV[jmax] + 5*En_wid)/self.nresp_set['Ebin_MeVee'])
        self.EphsB_MeVee = self.nresp_set['Ebin_MeVee']*np.arange(self.phs_max + 1)
        self.Ephs_MeVee = 0.5*(self.EphsB_MeVee[1:] + self.EphsB_MeVee[:-1])

        if parallel:
            self.run_multi()
        else:
            self.run_serial()


    def run_multi(self):

        timeout_pool = int(self.nresp_set['nmc']/1e3)
        pool = Pool(cpu_count())
        out = pool.map_async(En2light, [(EMeV, self.phs_max, self.nresp_set) for EMeV in self.En_MeV]).get(timeout_pool)
        pool.close()
        pool.join()

        logger.info('END light output calculation, nMC=%d, nEn=%d', self.nresp_set['nmc'], len(self.En_MeV))
        self.count_reac   = np.zeros((self.nEn, self.n_react), dtype=int_typ)
        self.phs_dim_rea  = np.zeros((self.nEn, self.n_react), dtype=int_typ)
        self.count_pp3as  = np.zeros((self.nEn, CS.max_level), dtype=int_typ)
        self.phs_dim_pp3  = np.zeros((self.nEn, CS.max_level), dtype=int_typ)
        self.pp3as_output = np.zeros((self.nEn, CS.max_level, self.phs_max), dtype=flt_typ)
        self.light_output = np.zeros((self.nEn, self.n_react, self.phs_max), dtype=flt_typ)
        for jE, x in enumerate(out):
            self.count_reac  [jE] = x[0]
            self.count_pp3as [jE] = x[1]
            self.phs_dim_rea [jE] = x[2]
            self.phs_dim_pp3 [jE] = x[3]
            self.light_output[jE] = x[4]
            self.pp3as_output[jE] = x[5]
        self.phs_dim_rea += 1
        self.phs_dim_pp3 += 1
        self.RespMat = np.sum(self.light_output, axis=1)


    def run_serial(self):

        self.count_reac   = np.zeros((self.nEn, self.n_react), dtype=int_typ)
        self.phs_dim_rea  = np.zeros((self.nEn, self.n_react), dtype=int_typ)
        self.count_pp3as  = np.zeros((self.nEn, CS.max_level), dtype=int_typ)
        self.phs_dim_pp3  = np.zeros((self.nEn, CS.max_level), dtype=int_typ)
        self.pp3as_output = np.zeros((self.nEn, CS.max_level, self.phs_max), dtype=flt_typ)
        self.light_output = np.zeros((self.nEn, self.n_react, self.phs_max), dtype=flt_typ)
        for jE, EMeV in enumerate(self.En_MeV):
            self.count_reac[jE], self.count_pp3as[jE], self.phs_dim_rea[jE], self.phs_dim_pp3[jE], \
                self.light_output[jE], self.pp3as_output[jE] = En2light((EMeV, self.phs_max, self.nresp_set))
        self.phs_dim_rea += 1
        self.phs_dim_pp3 += 1
        self.RespMat = np.sum(self.light_output, axis=1)


    def to_nresp(self, fout='%s/output/spect.dat' %nrespDir):

        f = open(fout, 'w')
        f.write('%15.6e\n' %self.nresp_set['Ebin_MeVee'])
        for jEn, En in enumerate(self.En_MeV):
            for jreac, reac in enumerate(self.reac_names):
                count = self.count_reac[jEn, jreac]
                if count > 0:
                    phs_dim = self.phs_dim_rea[jEn, jreac]
                    f.write('%-30s %8.2f %8.2f %13d\n' %(reac, 1e3*En, 1e3*self.En_wid_MeV[jEn], count))
                    spc_str = rw_for.wr_for(self.light_output[jEn, jreac, :phs_dim], fmt=' %13.6e', n_lin=5)
                    f.write(spc_str)

            for jlevel in range(CS.max_level):
                count = self.count_pp3as[jEn, jlevel]
                if count > 0:
                    phs_dim = self.phs_dim_pp3[jEn, jlevel]
                    lbl = 'PP3AS%d' %(jlevel+1)
                    f.write('%-30s %8.2f %8.2f %13d\n' %(lbl, En, self.En_wid_MeV[jEn], count))
                    spc_str = rw_for.wr_for(self.pp3as_output[jEn, jlevel, :phs_dim], fmt=' %13.6e', n_lin=5)
                    f.write(spc_str)
        f.close()
        logger.info('Written %s' %fout)


    def plotResponse(self, E_MeV=2.):

        jEn = np.argmin(np.abs(self.En_MeV - E_MeV))
        n_react = self.light_output.shape[1]

        fig = plt.figure('NRESP reactions', (8.8, 5.9))
        fig.clf()

        fig.text(0.5, 0.95, r'E$_n$=%6.3f MeV' %self.En_MeV[jEn], ha='center')
        ax1 = fig.add_subplot(1, 2, 1)
        for jreact in range(n_react):
            ax1.plot(self.Ephs_MeVee, self.light_output[jEn, jreact], label=self.reac_names[jreact])
        ax1.set_ylim([0., 20./E_MeV])
        ax1.set_xlabel('Pulse Height [MeVee]')
        ax1.set_ylabel('Pulse Height Spectrum')
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        resp = np.sum(self.light_output, axis=1)
        ax2.plot(self.Ephs_MeVee, resp[jEn])
        ax2.set_ylim([0., 100./E_MeV])
        ax2.set_xlabel('Pulse Height [MeVee]')
        ax2.set_ylabel('Pulse Height Spectrum')
        return fig
