import logging
import numpy as np
import h5py
from scipy.io import netcdf_file
from scipy.interpolate import RectBivariateSpline

logger = logging.getLogger('DRESS LoS.fi')
logger.setLevel(logging.DEBUG)

flt = np.float64


class TRANSP2DRESS:
    '''Reads TRANSP CDF and NUBEAM cdf output files, converting units into DRESS'''

    def __init__(self, f_plasma, f_fast):

        logger.info('Reading TRANSP distribution')

        cv = netcdf_file(f_plasma, 'r', mmap=False).variables
        fv = netcdf_file(f_fast, 'r', mmap=False).variables

        spc_lbl = b''.join(fv['SPECIES_1'].data).decode()
        tfbm = fv['TIME' ].data
        tim  = cv['TIME3'].data
        jt = np.argmin(np.abs(tim - tfbm))

# Separatrix
        self.Rbnd = 1e-2*fv['RSURF'].data[-1, :]
        self.Zbnd = 1e-2*fv['ZSURF'].data[-1, :]

        rhot_cdf = cv['X'].data[jt, :]
        rhot_fbm = fv['X2D'].data

        self.code_d = {}
        self.code_d['E']        = 1.e-3*fv['E_%s' %spc_lbl].data # eV -> keV
        self.code_d['pitch']    = fv['A_%s' %spc_lbl].data
        self.code_d['R']        = 1e-2*fv['R2D'].data
        self.code_d['Z']        = 1e-2*fv['Z2D'].data
        self.code_d['rho']      = rhot_fbm
        self.code_d['density']  = fv['bdens2'].data
        self.code_d['Ti']       = 1e-3*np.interp(rhot_fbm, rhot_cdf, cv['TI'].data[jt, :]) # eV -> keV
        self.code_d['nd']       =  1e6*np.interp(rhot_fbm, rhot_cdf, cv['ND'].data[jt, :]) # 1/cm**3 -> 1/m**3
        self.code_d['ang_freq'] =      np.interp(rhot_fbm, rhot_cdf, cv['OMEGA'].data[jt, :])
        self.code_d['dV']       = 1e-6*fv['BMVOL'].data # cm**3 -> m**3
        self.code_d['F']        = np.transpose(fv['F_%s' %spc_lbl].data, axes=(0, 2, 1)) # -> cell, E, pitch; no need to normalise


class ASCOT2DRESS:
    '''Reads ASCOT HDF5 output file, converting units into DRESS'''


    def __init__(self, f_fast):

        logger.info('Reading ASCOT distribution')

# Reading HDF5 file

        unit_d = {}
        f = h5py.File(f_fast, 'r')

        pl1d       = f['plasma/1d']
        bfield     = f['bfield']
        dist_grids = f['distributions/rzPitchEdist/abscissae']
        fbm_h5     = f['distributions/rzPitchEdist/ordinate'][:] # species, time, energy, pitch, z, R, ?

        rhop_pl  = pl1d['rho'][:]
        ti = 1.e-3*pl1d['ti' ][:]    # eV -> keV
        nD =       pl1d['ni' ][:, 0] # m**-3
        vt =       pl1d['vtor'][:]   # m/s
        for lbl in ('ti', 'ni', 'vtor'):
            unit_d[lbl] = pl1d[lbl].attrs['unit'].decode()

        (Raxis, ) = bfield['raxis']
        (zaxis, ) = bfield['zaxis']
        R_psi = bfield['r'][:]
        z_psi = bfield['z'][:]
        psi2d = bfield['2d/psi'][:].T # R, z

        grid = {}
        grid_b = {}
        for var in dist_grids.values():
            lbl  = var.attrs['name'].decode()
            unit = var.attrs['unit'].decode()
            unit_d[lbl] = unit
            grid_b[lbl] = var[:]
            grid[lbl] = 0.5*(var[1:] + var[:-1])

        f.close()

# R, z rectangular domain
        Rmin = grid_b['R'][0]
        Rmax = grid_b['R'][-1]
        Zmin = grid_b['z'][0]
        Zmax = grid_b['z'][-1]
        self.Rbnd = [Rmin, Rmax, Rmax, Rmin]
        self.Zbnd = [Zmin, Zmin, Zmax, Zmax]

# Fast ion distribution and grids
        fbm = fbm_h5[0, 0, :, ::-1, :, :, 0] # energy, pitch, z, R; no need to normalise
        nE, nmu, nz, nR = fbm.shape
        nRz = nR*nz
        dR  = (Rmax - Rmin)/float(nR)
        dz  = (Zmax - Zmin)/float(nz)
        dE  = (grid_b['energy'][-1] - grid_b['energy'][0])/float(nE)
        dmu = (grid_b['pitch' ][-1] - grid_b['pitch' ][0])/float(nmu)

        self.code_d = {}
        Rmesh, Zmesh = np.meshgrid(grid['R'], grid['z'], indexing='ij') # 2D grid
        self.code_d['R'] = Rmesh.ravel() # Keep everything 1d, 1st index is R
        self.code_d['Z'] = Zmesh.ravel()
        self.code_d['E'] = grid['energy']*6.242e+15 # J -> keV
        self.code_d['pitch'] = grid['pitch']
        self.code_d['dV'] = 2.*np.pi*self.code_d['R']*dR*dz
        self.code_d['F'] = np.transpose(fbm, axes=(3, 2, 0, 1)).reshape(nRz, nE, nmu)  # Rz, energy, pitch

# 2D grid for rho_pol
        psi_sep = 0
        f_psi = RectBivariateSpline(R_psi, z_psi, psi2d)
        psi_axis = f_psi(Raxis, zaxis)[0][0]
        psi_red  = f_psi(self.code_d['R'], self.code_d['Z'], grid=False)
        psi_norm = (psi_red - psi_axis)/(psi_sep - psi_axis)
        self.code_d['rho'] = np.sqrt(psi_norm)

# Integrate distribution over E, pitch
        self.code_d['density'] = np.sum(self.code_d['F'], axis=(1, 2))*dE*dmu

# Interpolate Ti, nd, v_rot to (unrolled) 2D grid
        self.code_d['Ti']    = np.zeros(nRz, dtype=flt)
        self.code_d['nd']    = np.zeros(nRz, dtype=flt)
        self.code_d['v_rot'] = np.zeros((nRz, 3), dtype=flt)
        nrho_pl = len(rhop_pl)
        for jR in range(nR):
            for jz in range(nz):
                xrho = self.code_d['rho'][jR + jz*nR]
                if np.isnan(xrho) or xrho > 1.: # outside separatrix
                    self.code_d['Ti'][jR + jz*nR] = np.nan
                    self.code_d['nd'][jR + jz*nR] = np.nan
                else:
                    jrho = np.argmin(np.abs(rhop_pl - xrho))
                    if jrho == 0:
                        ind = [jrho, jrho + 1]
                    elif jrho == nrho_pl-1:
                        ind = [jrho-1, jrho]
                    else:
                        ind  = [jrho-1, jrho, jrho+1]
                    self.code_d['Ti'   ][jR + jz*nR] = np.interp(xrho, rhop_pl[ind], ti[ind])
                    self.code_d['nd'   ][jR + jz*nR] = np.interp(xrho, rhop_pl[ind], nD[ind])
#                    self.code_d['v_rot'][jR, jz, 1] = np.interp(xrho, rhop_pl[ind], vt[ind]) # Too high in repo case
