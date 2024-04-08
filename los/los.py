import logging
import numpy as np
import matplotlib.pylab as plt

logger = logging.getLogger('neutReac.los')
logger.setLevel(logging.DEBUG)


class CELL:
    pass


class DETECTOR:
    pass


class PHILOS:

    def __init__(self, par_d, npoints=5000):

        theta  = par_d['theta']
        phi_in = par_d['phi'] 
        x0     = par_d['x0']
        y0     = par_d['y0']
        z0     = par_d['z0']
        xend   = par_d['xend']
        phi0 = np.arctan2(x0, y0)
        phi  = phi_in + phi0
        dx   = xend - x0
        dr   = dx/np.sin(phi)
        self.dr = np.abs(dr)
        yend = y0 + dr*np.cos(phi)
        zend = z0 + self.dr*np.tan(theta)
        self.xline = np.linspace(x0, xend, npoints)
        self.yline = np.linspace(y0, yend, npoints)
        self.zline = np.linspace(z0, zend, npoints)
        self.rline = np.hypot(self.xline, self.yline)


class XYLOS:

    def __init__(self, par_d, npoints=5000):

        theta  = par_d['theta']
        x0 = par_d['x0']
        y0 = par_d['y0']
        z0 = par_d['z0']
        xend = par_d['xend']
        yend = par_d['yend']
        dx = xend - x0
        dy = yend - y0
        self.dr = np.hypot(dx, dy)
        zend = z0 + self.dr*np.tan(theta)
        self.xline = np.linspace(x0, xend, npoints)
        self.yline = np.linspace(y0, yend, npoints)
        self.zline = np.linspace(z0, zend, npoints)
        self.rline = np.hypot(self.xline, self.yline)


class DETECTOR_LOS:

    def __init__(self, geo):
        self.geo = geo
        self.run()

    def run(self):

        los_d = {}
        los_d['x0']    = -self.geo['y_det']
        los_d['y0']    = 0.001
        los_d['z0']    = self.geo['z_det']
        los_d['xend']  = 0 # 1 plasma crossing
#        los_d['xend']  = self.geo['y_det'] # 2 plasma crossings
        los_d['theta'] = np.radians(self.geo['tilt'])
        los_d['phi']   = -np.arctan2(self.geo['tan_radius'], self.geo['y_det'])

        ctilt = np.cos(los_d['theta'])
        dy = self.geo['disk_thick']*ctilt
        ndisks = int(-2*self.geo['y_det']/dy)

        self.det = DETECTOR()
        self.det.los = PHILOS(los_d, npoints=ndisks)

# Get separatrix {R,z}

        rmin = self.geo['Rmaj']  - self.geo['r_chamb'] 
        rmax = self.geo['Rmaj']  + self.geo['r_chamb'] 

# Restrict to LOS inside the plasma

        dl = np.hypot(self.det.los.rline[1] - self.det.los.rline[0], self.det.los.zline[1] - self.det.los.zline[0])
        ind = (self.det.los.rline > rmin - dl) & (self.det.los.rline < rmax + dl)
        r_in  = self.det.los.rline[ind]
        self.det.los.z = self.det.los.zline[ind]
        self.det.los.y = self.det.los.xline[ind]
        y_in  = self.det.los.yline[ind]

# Write ASCII file for LOS, used in the spectrum evaluation

        self.det.pos = (self.geo['tan_radius'], self.geo['y_det'], self.geo['z_det'])
        det_dist = np.hypot(self.det.los.y - self.det.pos[1], self.det.los.z - self.det.pos[2])
        ctilt = np.cos(np.radians(self.geo['tilt']))
        stilt = np.sin(np.radians(self.geo['tilt']))
        dy = self.geo['disk_thick']*ctilt
        coll_rad = 0.5*self.geo['coll_diam']
        dist_corr = self.geo['d_det_coll']/(1. + self.geo['det_radius']/coll_rad) # Shifting the cone vertex to the point where lines (det-left - coll-right, det-right - coll-left) cross
        self.det.tan_cone_aper = coll_rad/dist_corr
#        logger.debug(coll_rad, dist_corr, tan_cone_aper)
        offset = self.geo['d_det_coll'] - dist_corr # scalar, r2
        disk_radius = (det_dist - offset)*self.det.tan_cone_aper # array on discretised LoS
        n_disks = len(self.det.los.y)

# Each disk is divided in n_circles circular sectors,
# n_circles depends on the disk radius (which is almost constant)
# Every circular sector is divided in n_sectors sectors,
# equidistant poloidally; n_sectors is proportional to the radius
# of the circular sector

        self.cell = CELL()
        self.cell.x     = []
        self.cell.y     = []
        self.cell.z     = []
        self.cell.omega = []
        self.cell.vol   = []

        for jdisk in range(n_disks):
            n_circles = int(0.5 + disk_radius[jdisk]/self.geo['cell_radius'])
            delta_radius = disk_radius[jdisk]/float(n_circles)

# radius, alpha in the 'middle' of the sector
# The central circle has only one sector (cell)
            cvol = np.pi * delta_radius**2 * dy
            self.cell.vol = np.append(self.cell.vol, np.repeat(cvol, n_circles**2))

            radius = (0.5 + np.arange(n_circles))*delta_radius
            radius[0] = 0.
            omega_fac = np.pi * self.geo['det_radius']**2/det_dist[jdisk]**3
            cell_detDist = np.hypot(det_dist[jdisk], radius)
            omegaCircle = omega_fac * cell_detDist
            for j_circle in range(n_circles):
                n_sectors = 2*j_circle + 1
                self.cell.omega += n_sectors*[omegaCircle[j_circle]]
# Poloidal sectors (cells) in a circle
                alpha = np.linspace(0, 2.*np.pi, n_sectors, endpoint=False)
                rcos  = list(radius[j_circle]*np.cos(alpha))
                rssin = list(-self.det.los.y[jdisk] - radius[j_circle]*np.sin(alpha)*stilt)
                rcsin = list( self.det.los.z[jdisk] + radius[j_circle]*np.sin(alpha)*ctilt)
# cell_pos: with respect to torus center
                self.cell.x += rcos
                self.cell.y += rssin
                self.cell.z += rcsin
        self.cell.x = np.array(self.cell.x) + self.det.pos[0]
        self.cell.y = np.array(self.cell.y)
        self.cell.z = np.array(self.cell.z)
        self.cell.omega = np.array(self.cell.omega)
        logger.info('Done LOS cone calculation')


    def writeLOS(self):

        n_cells = len(self.cell.vol)
        n_disks = len(self.det.los.y)
        cone_aper = np.degrees(np.arctan(self.det.tan_cone_aper))
        ctilt = np.cos(np.radians(self.geo['tilt']))
        dy = self.geo['disk_thick']*ctilt

        header = \
           'LOS\n' + \
           '   y1 = %9.4f m\n'  % self.det.los.y[0]  + \
           '   y2 = %9.4f m\n'  % self.det.los.y[-1] + \
           '   z1 = %9.4f m\n'  % self.det.los.z[0]  + \
           '   z2 = %9.4f m\n'  % self.det.los.z[-1] + \
           'Detector:\n' + \
           '   Position [m] x, y, z: %9.4f, %9.4f, %9.4f\n' %self.det.pos + \
           '   Radius = %9.4f m\n' % self.geo['det_radius'] + \
           '   Collimation angle = %9.4f deg\n' % cone_aper + \
           'Disks:\n' + \
           '   Thickness = %9.4f m\n' % dy + \
           '   # disks = %5d\n' % n_disks + \
           'Cells:\n' + \
           '   Radius = %9.4f m\n' % self.geo['cell_radius'] + \
           '   # cells = %d\n' %n_cells + \
"""
   (x,y,x) cell cartensian coordinates [m]
   Omega  Steradians is the volume in the solid angle
   Vol = cell volume [m**3]
x             y             z             Omega         Vol
"""

        los_file = 'los/%s.los' %self.geo['label']

        logger.info('Storing ASCII output, n_cells=%d', n_cells)
        np.savetxt(los_file, np.hstack((self.cell.x, self.cell.y, self.cell.z, self.cell.omega, self.cell.vol)).reshape(5, n_cells).T, header=header, fmt='%13.6E')
        logger.info('Written output file %s' %los_file)


    def plotLOS(self, show=False, figsize=(8.8, 5.9)):

        R_m = np.hypot(self.cell.x, self.cell.y)

        fig = plt.figure(1, figsize=figsize)
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.98, wspace=0.2)

        ax1 = plt.subplot(1, 2, 1, aspect='equal')
        ax1.set_xlim([0.5, 3.5])
        ax1.set_ylim([-1.5, 1.5])
        ax1.set_xlabel('R [m]', labelpad=2)
        ax1.set_ylabel('z [m]', labelpad=2)
        ax1.plot(R_m, self.cell.z, 'ro') 

        ax2 = plt.subplot(1, 2, 2, aspect='equal')
        ax2.set_xlim([-3, 3])

# Plot AUG wall
        try:
            import aug_sfutils as sf
        except:
            logger.error('Missing vessel contour data')
        if 'sf' in locals():
            gc_d  = sf.getgc()
            tor_d = sf.getgc_tor() 
            for gc in gc_d.values():
                ax1.plot(gc.r, gc.z, 'b-')
            for tor in tor_d.values():
                ax2.plot(tor.x, tor.y, 'b-')

        ax2.plot(self.cell.x, self.cell.y, 'ro')
        ax2.set_xlabel('x [m]', labelpad=2)
        ax2.set_ylabel('y [m]', labelpad=2)
        ax2.tick_params(which='major', length=4, width=0.5)

        if show:
            plt.show()
        else:
            return fig
        
