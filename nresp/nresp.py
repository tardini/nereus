#!/usr/bin/env python

import os, logging
import numpy as np
import matplotlib.pylab as plt
from multiprocessing import Pool, cpu_count
from en2light import En2light
import crossSections, settings, rw_for

os.system('mkdir -p %s/output' %settings.nrespDir)

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('nresp')
logger.setLevel(level=logging.DEBUG)
hnd  = logging.StreamHandler()
flog = '%s/output/nresp.log' %settings.nrespDir
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
flt_typ = settings.flt_typ
int_typ = settings.int_typ


class NRESP:


    def __init__(self, En_in_MeV):

        self.reac_names = CS.reacTotUse
        self.reac_names.append('light-guide')
        self.En_MeV = En_in_MeV
        self.nEn = len(self.En_MeV)
        jmax = np.argmax(self.En_MeV)
        self.n_react = len(self.reac_names)
        self.En_wid_MeV = settings.En_wid_frac*self.En_MeV
        En_wid = self.En_wid_MeV[jmax]
        if settings.distr == 'mono':
            self.phs_max = int(self.En_MeV[jmax]/settings.Ebin_MeVee)
        elif settings.distr == 'gauss':
            self.phs_max = int((self.En_MeV[jmax] + 5*En_wid)/settings.Ebin_MeVee)
        self.EphsB_MeVee = settings.Ebin_MeVee*np.arange(self.phs_max + 1)
        self.Ephs_MeVee = 0.5*(self.EphsB_MeVee[1:] + self.EphsB_MeVee[:-1])

        self.run()


    def run(self):

        timeout_pool = int(settings.nmc/1e3)
        pool = Pool(cpu_count())
        out = pool.map_async(En2light, [(EMeV, self.phs_max) for EMeV in self.En_MeV]).get(timeout_pool)
        pool.close()
        pool.join()

        logger.info('END light output calculation, nMC=%d, nEn=%d', settings.nmc, len(self.En_MeV))
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


    def to_nresp(self, fout='%s/output/spect.dat' %settings.nrespDir):

        f = open(fout, 'w')
        f.write('%15.6e\n' %settings.Ebin_MeVee)
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

        plt.figure('Reactions', (14, 8))
        plt.subplot(1, 2, 1)
        for jreact in range(n_react):
            plt.plot(self.Ephs_MeVee, self.light_output[jEn, jreact], label=self.reac_names[jreact])
        plt.ylim([0., 0.004])
        plt.legend()

        plt.subplot(1, 2, 2)
        resp = np.sum(self.light_output, axis=1)
        plt.plot(self.Ephs_MeVee, resp[jEn])
        plt.ylim([0., 0.02])


if __name__ == '__main__':

    import response

    nEn = 9
    En_in_MeV = np.linspace(2, 18, nEn)
    nrsp = NRESP(En_in_MeV)
    nrsp.to_nresp(fout='%s/output/nresp.dat' %settings.nrespDir)
    nrsp.plotResponse(E_MeV=16.)
    rsp = response.RESP()
    rsp.from_nresp(f_spc='%s/output/nresp.dat' %settings.nrespDir)
    logger.info('Log stored in %s' %flog)
    plt.show()
