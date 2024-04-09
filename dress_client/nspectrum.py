import os, logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from multiprocessing import Pool, cpu_count
import dress
import response
from dress_client import fi_codes

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('DRESS LoS')
logger.addHandler(hnd)
logger.setLevel(logging.DEBUG)

flt = np.float64


def calcvols(tuple_in):

    vols, dist1, dist2, scalc, En_bins = tuple_in
    nes = dress.utils.calc_vols(vols, dist1, dist2, scalc, En_bins, integrate=True, quiet=True)

    return nes


class nSpectrum:


    def __init__(self, f_in1, f_in2, f_los=None, src='transp', samples_per_volume_element=1e4):

        self.samples_per_volume_element = samples_per_volume_element
        self.code = src

        if src == 'transp':
            self.codeClass = fi_codes.TRANSP2DRESS(f_in1, f_in2)
        elif src == 'ascot':
            self.codeClass = fi_codes.ASCOT2DRESS(f_in1)

        if f_los is None:
            self.dressInput = self.codeClass.code_d
            self.dressInput['solidAngle'] = 4*np.pi*np.ones_like(self.dressInput['dV'])
        else:
            x_m, y_m, z_m, omega, V_m3 = np.loadtxt(f_los, unpack=True)
            R_m = np.hypot(x_m, y_m)
            self.los_dressInput(R_m, z_m)
            self.dressInput['R']  = R_m[self.inside]
            self.dressInput['Z']  = z_m[self.inside]
            self.dressInput['dV'] = V_m3[self.inside]
            self.dressInput['solidAngle'] = omega[self.inside]


    def los_dressInput(self, R_m, z_m):
        '''Mapping quantities from original volumes to LoS volumes. Removing LoS volumes outside a {R, z} domain (separatrix or 2D cartesian grid)'''

        n_los = len(R_m)

# LoS volumes inside R, z domain

        RZ_points = np.hstack((R_m, z_m)).reshape(2, n_los).T
        sepPolygon = np.hstack((self.codeClass.Rbnd, self.codeClass.Zbnd)).reshape(2, len(self.codeClass.Rbnd)).T
        self.sepPath = Path(sepPolygon)
        self.inside = self.sepPath.contains_points(RZ_points)
        logger.debug('Volumes inside Sep %d out of %d', np.sum(self.inside), n_los)

        los2fbm = np.zeros(n_los, dtype=np.int32)
        for jlos in range(n_los):
            if self.inside[jlos]:
                d2 = (self.codeClass.code_d['R'] - R_m[jlos])**2 + (self.codeClass.code_d['Z'] - z_m[jlos])**2
                los2fbm[jlos] = np.argmin(d2)
        los_sep = los2fbm[self.inside]

        self.dressInput = {}
        for key, val in self.codeClass.code_d.items():
            if key in ('E', 'pitch'):
                 self.dressInput[key] = val
            else: # map variables onto LoS volumes
                self.dressInput[key] = val[los_sep]


    def run(self, parallel=True):
        '''Execute DRESS calculation, computing Neutron Emission Spectra: beam-target, thermonuclear, beam-beam'''

        logger.info('Running DRESS')

        Ncells = len(self.dressInput['rho'])

        Bdir = np.atleast_2d([0, -1, 0])
        B_dir = np.repeat(Bdir, Ncells, axis=0)   # B-dir for each spatial location

        if 'v_rot' not in self.dressInput.keys():
            self.dressInput['v_rot'] = np.zeros((Ncells, 3), dtype=flt)
            if 'ang_freq' in self.dressInput.keys():
                self.dressInput['v_rot'][:, 1] = self.dressInput['ang_freq']*self.dressInput['R']

        dd    = dress.reactions.DDNHe3Reaction()
        scalc = dress.SpectrumCalculator(dd, n_samples=self.samples_per_volume_element)

# Neutron energy bins [keV]
        bin_keV = 10.
        En_bins = np.arange(1500, 3500, bin_keV)    # bin edges
        self.En = 0.5*(En_bins[1:] + En_bins[:-1])  # bin centers

# Compute spectra components

        vols = dress.utils.make_vols(self.dressInput['dV'], self.dressInput['solidAngle'], pos=(self.dressInput['R'], self.dressInput['Z']))
        fast_dist = dress.utils.make_dist('energy-pitch', 'd', Ncells, self.dressInput['density'],
            energy_axis=self.dressInput['E'], pitch_axis=self.dressInput['pitch'], distvals=self.dressInput['F'], ref_dir=B_dir)
        bulk_dist = dress.utils.make_dist('maxwellian', 'd', Ncells, self.dressInput['nd'], temperature=self.dressInput['Ti'], v_collective=self.dressInput['v_rot'])
        logger.debug('T #nan: %d, #T<=0: %d', np.sum(np.isnan(bulk_dist.T)), np.sum(bulk_dist.T <= 0))
        logger.debug('nd #nan: %d, #nd<=0: %d', np.sum(np.isnan(bulk_dist.density)), np.sum(bulk_dist.density <= 0))

# Split arrays

        self.dressSplit = {}
        if hasattr(self, 'inside'):
            n_split = 20
        else:
            n_split = 10
        for key in ('dV', 'solidAngle', 'R', 'Z', 'density', 'F', 'nd', 'Ti', 'v_rot'):
            self.dressSplit[key] = np.split(self.dressInput[key], n_split)
        vols_spl = {}
        fast_spl = {}
        bulk_spl = {}
        for j in range(n_split):
            Nvols = len(self.dressSplit['dV'][j])
            B_dir = np.repeat(Bdir, Nvols, axis=0)   # B-dir for each spatial location
            vols_spl[j] = dress.utils.make_vols(self.dressSplit['dV'][j], self.dressSplit['solidAngle'][j], pos=(self.dressSplit['R'][j], self.dressSplit['Z'][j]))
            fast_spl[j] = dress.utils.make_dist('energy-pitch', 'd', Nvols, self.dressSplit['density'][j],
                energy_axis=self.dressInput['E'], pitch_axis=self.dressInput['pitch'], distvals=self.dressSplit['F'][j], ref_dir=B_dir)
            bulk_spl[j] = dress.utils.make_dist('maxwellian', 'd', Nvols, self.dressSplit['nd'][j], temperature=self.dressSplit['Ti'][j], v_collective=self.dressSplit['v_rot'][j])
            

        if parallel:
            timeout_pool = 2000
            pool = Pool(cpu_count())
            logger.info('Computing beam-target')
            bt = pool.map_async( calcvols, [(vols_spl[j], fast_spl[j], bulk_spl[j], scalc, En_bins) for j in range(n_split)]).get(timeout_pool)
            logger.info('Computing thermonuclear')
            th = pool.map_async( calcvols, [(vols_spl[j], bulk_spl[j], bulk_spl[j], scalc, En_bins) for j in range(n_split)]).get(timeout_pool)
            logger.info('Computing beam-beam')
            bb = pool.map_async( calcvols, [(vols_spl[j], fast_spl[j], fast_spl[j], scalc, En_bins) for j in range(n_split)]).get(timeout_pool)
            bt = np.array(bt)
            th = np.array(th)
            bb = np.array(bb)
            self.bt = np.sum(bt, axis=0)
            self.th = np.sum(th, axis=0)
            self.bb = np.sum(bb, axis=0)
        else:
            logger.info('Computing beam-target')
            self.bt = dress.utils.calc_vols(vols, fast_dist, bulk_dist, scalc, En_bins, integrate=True, quiet=False)
            logger.info('Computing thermonuclear')
            self.th = dress.utils.calc_vols(vols, bulk_dist, bulk_dist, scalc, En_bins, integrate=True, quiet=False)
            logger.info('Computing beam-beam')
            self.bb = dress.utils.calc_vols(vols, fast_dist, fast_dist, scalc, En_bins, integrate=True, quiet=False)     

        for spec in self.bt, self.bb, self.th:
            spec /= bin_keV
        self.bb *= 0.5
        self.th *= 0.5

        rate_bt = bin_keV*np.sum(self.bt)
        rate_th = bin_keV*np.sum(self.th)
        rate_bb = bin_keV*np.sum(self.bb)
        if hasattr(self, 'inside'): # LoS
            logger.info('b-t rate at detector %12.4e N/s', rate_bt)
            logger.info('th  rate at detector %12.4e N/s', rate_th)
            logger.info('b-b rate at detector %12.4e N/s', rate_bb)
            logger.info('Neutron rate at detector %12.4e N/s', rate_bt + rate_th + rate_bb)
        else: # Total
            logger.info('Total b-t neutrons %12.4e N/s', rate_bt)
            logger.info('Total th  neutrons %12.4e N/s', rate_th)
            logger.info('Total b-b neutrons %12.4e N/s', rate_bb)
            logger.info('Total neutron rate %12.4e N/s', rate_bt + rate_th + rate_bb)


    def nes2phs(self, f_resp='responses/rm_bg.cdf'):

        resp = response.RESP()
        fname, ext = os.path.splitext(f_resp)
        if ext.lower() == '.cdf':
            resp.from_cdf(f_resp)
        else:
            resp.from_hepro(f_resp)

        En_MeV = 1e-3*self.En
        nbins = len(resp.Ephs_MeVee)
        self.phs = {}
        self.phs['Elight_MeVee'] = resp.Ephs_MeVee

        for reac in ('bt', 'bb', 'th'):
            self.phs[reac] = np.zeros(nbins, dtype=flt)
            for jEn, En in enumerate(En_MeV):
                dist = (resp.En_MeV - En)**2
                jclose = np.argmin(dist)
                self.phs[reac] += self.__dict__[reac][jEn]*resp.RespMat[jclose, :]


    def storeSpectra(self, f_out='dress_client/output/Spectrum.dat'):
        '''Store ASCII output for DRESS Neutron Emission Spectra'''

        header = 'Eneut      Thermonucl. Beam-target Beam-beam'
        np.savetxt(f_out, np.c_[self.En, self.th, self.bt, self.bb], fmt='%11.4e', header=header)
        logger.info('Stored file %s', f_out)


    def fromFile(self, f_spec):
        '''Read (e.g. for plotting) DRESS ASCII output files, instead of computing Neutron Emission Spectra'''

        self.En, self.th, self.bt, self.bb = np.loadtxt(f_spec, unpack=True, skiprows=1)


    def plotInput(self, jcell=100):
        '''Plot some DRESS input quantities'''

        logger.info('Plotting input')
        plt.figure()
        plt.plot(self.dressInput['rho'], self.dressInput['nd'], 'k+')
        plt.title('Bulk D density')
        plt.xlabel('rho')
        plt.ylabel('density (particles/m^3)')

        plt.figure()
        plt.plot(self.dressInput['rho'], self.dressInput['Ti'], 'go')
        plt.title('Ion temperature')
        plt.xlabel('rho')
        plt.ylabel('temperature (keV)')

        plt.figure()
        plt.plot(self.dressInput['rho'], self.dressInput['v_rot'], 'ro')
        plt.title('Toroidal velocity')
        plt.xlabel('rho')
        plt.ylabel('Toroidal velocity (m/s)')

# Plot fast D density
        plt.figure()
        plt.tripcolor(self.dressInput['R'], self.dressInput['Z'], self.dressInput['density'])
        if hasattr(self, 'sepPath'):
            patch = patches.PathPatch(self.sepPath, facecolor='None', lw=2)
            plt.gca().add_patch(patch)
        plt.title('Fast D density')
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.axis('scaled')

# Plot the (E, pitch) distribution at a given MC cell
        plt.figure()
        R = self.dressInput['R'][jcell]
        Z = self.dressInput['Z'][jcell]
        F = self.dressInput['F'][jcell].T

        plt.pcolor(self.dressInput['E'], self.dressInput['pitch'], F)
        plt.title(f'D distribution at (R, Z) = ({round(R,2)}, {round(Z,2)}) m')
        plt.xlabel('E (keV)')
        plt.ylabel('pitch')


    def plotSpectra(self):
        '''Plot DRESS output Neutron Emission Spectra'''

        logger.info('Plotting LoS spectrum')
        fig = plt.figure(figsize=(8.8, 5.9))

        ax_nes = fig.add_subplot(1, 2, 1)
        plt.plot(self.En, self.bt, label='BT')
        plt.plot(self.En, self.th, label='TH')
        plt.plot(self.En, self.bb, label='BB')
        ymax = ax_nes.get_ylim()[1]
        plt.plot([2452, 2452], [0, 1.5*ymax], 'k-') # Reference d-d neutron energy
        plt.ylim([0, ymax])
        plt.xlabel('Neutron energy (keV)')
        plt.ylabel('Energy spectrum (neuts/keV/s)')
        if hasattr(self, 'inside'):
            plt.title('Line-of-sight Neutron Emission Spectrum')
        else:
            plt.title('Total Neutron Emission Spectrum')
        plt.legend()
        plt.xlim(2000, 3000)

        ax_phs = fig.add_subplot(1, 2, 2)
        plt.plot(self.phs['Elight_MeVee'], self.phs['bt'], label='BT')
        plt.plot(self.phs['Elight_MeVee'], self.phs['th'], label='TH')
        plt.plot(self.phs['Elight_MeVee'], self.phs['bb'], label='BB')
        plt.xlabel('Light equivalent energy (MeVee)')
        plt.xlim(0, 2)
        ax_phs.set_ylim(bottom=0)

        if hasattr(self, 'inside'):
            plt.title('Line-of-sight Pulse Height Spectrum')
        else:
            plt.title('Total Pulse Height')

        return fig
        

if __name__ == '__main__':

    f_los   = 'input/aug_BC501A.los' # Detector LoS file
# TRANSP: plasma, distribution
    f_pl_tr = 'input/36557D05.CDF'
    f_f_tr  = 'input/36557D05_fi_1.cdf'
#ASCOT output
    f_as    = 'input/29795_3.0s_ascot.h5'

# Total spectra
#    spec = nSpectrum(f_pl_tr, f_f_tr, src='transp', samples_per_volume_element=1e3)
    spec = nSpectrum(f_as, f_as, src='ascot', samples_per_volume_element=1e3)

# LoS spectra
#    spec = nSpectrum(f_pl_tr, f_f_tr, f_los=f_los, src='transp', samples_per_volume_element=1e1)
#    spec = nSpectrum(f_as, f_as, f_los=f_los, src='ascot', samples_per_volume_element=1e3)

    spec.run()

#    spec.fromFile('output/tot_tr_mc1e5.dat')
#    spec.fromFile('output/tot_as_mc1e5.dat')
#    spec.fromFile('output/los_tr_mc1e1.dat')
#    spec.fromFile('output/los_tr_mc1e2.dat')
#    spec.fromFile('output/los_tr_mc1e3.dat')
#    spec.fromFile('output/los_as_mc1e1.dat')
#    spec.fromFile('output/los_as_mc1e2.dat')
#    spec.fromFile('output/los_as_mc1e3.dat')

    spec.plotInput()
    spec.plotSpectra()
    spec.storeSpectra()

    plt.show()
