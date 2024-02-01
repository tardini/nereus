#!/usr/bin/env python

__author__  = 'Giovanni Tardini (Tel. +49 89 3299-1898)'
__version__ = '0.0.1'
__date__    = '19.05.2022'

import os, sys, logging, webbrowser, json

try:
    from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGridLayout, QMenu, QAction, QLabel, QPushButton, QLineEdit, QCheckBox, QFileDialog, QRadioButton, QButtonGroup, QTabWidget, QVBoxLayout, QComboBox
    from PyQt5.QtGui import QPixmap, QIcon, QIntValidator, QDoubleValidator
    from PyQt5.QtCore import Qt, QRect, QSize
    qt5 = True
except:
    from PyQt4.QtCore import Qt, QRect, QSize, QIntValidator, QDoubleValidator
    from PyQt4.QtGui import QPixmap, QIcon, QMainWindow, QWidget, QApplication, QGridLayout, QMenu, QAction, QLabel, QPushButton, QLineEdit, QCheckBox, QFileDialog, QRadioButton, QButtonGroup, QTabWidget, QVBoxLayout, QComboBox
    qt5 = False

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from reactivities import *
import calc_cross_section as cs
import calc_kinematics as ck
import calc_spectrum as spc
import plot_rekin
from reactions import reaction

os.environ['BROWSER'] = '/usr/bin/firefox'

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('rekin')
logger.addHandler(hnd)
logger.setLevel(logging.DEBUG)

rekin_dir = os.path.dirname(os.path.realpath(__file__))


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


class REAC_GUI(QMainWindow):


    def __init__(self):

        if sys.version_info[0] == 3:
            super().__init__()
        else:
            super(QMainWindow, self).__init__()

        xwin  = 805
        yhead = 44
        yline = 30
        ybar  = 54
        yicon = 46
        xicon = yicon
        ywin  = 15*yline + yhead + ybar

        qhead  = QWidget(self)
        qbar   = QWidget(self)
        qtabs  = QTabWidget(self)
        qhead.setGeometry(QRect(0,     0, xwin, yhead))
        qbar.setGeometry(QRect(0, yhead, xwin, ybar))
        qtabs.setGeometry(QRect(0, yhead+ybar+10, xwin, ywin-yhead-ybar-10))
        qtabs.setStyleSheet("QTabBar::tab { width: 160 }")
        header_grid = QGridLayout(qhead) 
        tbar_grid   = QGridLayout(qbar) 

#-----
# Tabs
#-----

        qreac = QWidget()
        reac_layout = QGridLayout()
        qreac.setLayout(reac_layout)
        qtabs.addTab(qreac, 'Reactivities')

        qcross = QWidget()
        cross_layout = QGridLayout()
        qcross.setLayout(cross_layout)
        qtabs.addTab(qcross, 'Cross-sections')

        qkin = QWidget()
        kin_layout = QGridLayout()
        qkin.setLayout(kin_layout)
        qtabs.addTab(qkin, 'Scattering kinematics')

        qspec = QWidget()
        spec_layout = QGridLayout()
        qspec.setLayout(spec_layout)
        qtabs.addTab(qspec, 'Neutron spectra')

        qlos = QWidget()
        los_layout = QGridLayout()
        qlos.setLayout(los_layout)
        qtabs.addTab(qlos, 'Detector LoS')

#--------
# Menubar
#--------

        menubar = self.menuBar()
        fileMenu = QMenu('&File', self)
        jsonMenu  = QMenu('&Setup', self)
        helpMenu = QMenu('&Help', self)
        menubar.addMenu(fileMenu)
        menubar.addMenu(jsonMenu)
        menubar.addMenu(helpMenu)

        fmap = {'reac': self.reactivity, 'csec': self.cross_section, 'kine': self.scatt_kine, \
                'spec': self.spectra, 'los': self.los, 'exit': sys.exit}
        qlbl = {'reac': '&Reactivity', 'csec': '&Cross-section', 'kine': '&Kinematics', \
                'spec': '&Spectra', 'los': '&Line-of-Sight', 'exit': 'Exit'}
        Action = {}
        for key, lbl in qlbl.items():
            Action = QAction(lbl, fileMenu)
            Action.triggered.connect(fmap[key])
            if key == 'exit':
                fileMenu.addSeparator()
            fileMenu.addAction(Action)

        loadAction  = QAction('&Load Setup', jsonMenu)
        saveAction  = QAction('&Save Setup', jsonMenu)
        jsonMenu.addAction(loadAction)
        jsonMenu.addAction(saveAction)
        loadAction.triggered.connect(self.load_json)
        saveAction.triggered.connect(self.save_json)

        aboutAction = QAction('&Web docu', helpMenu)
        aboutAction.triggered.connect(self.about)
        helpMenu.addAction(aboutAction)

        header_grid.addWidget(menubar, 0, 0, 1, 10)

# Icons
        dum_lbl  = QLabel(200*' ')
        xpos = 0
        ypos = 0
        for jpos, key in enumerate(qlbl.keys()):
            but = QPushButton()
            but.setGeometry(xpos, xicon, ypos, yicon)
            but.setStyleSheet("min-height: %dpx; padding: 0.0em 0.0em 0.0em 0.0em" %yicon)
            fgif = '%s/%s.gif' %(rekin_dir, key)
            if os.path.isfile(fgif):
                but.setIcon(QIcon(fgif))
            else:
                fpng = '%s/%s.png' %(rekin_dir, key)
                but.setIcon(QIcon(fpng))
            but.setIconSize(QSize(ybar, ybar))
            but.clicked.connect(fmap[key])
            tbar_grid.addWidget(but, 0, jpos)
            xpos += yicon + 4

        tbar_grid.addWidget(dum_lbl, 0, jpos+1, 1, 10)

# User options

        f_json = '%s/settings/default.json' %rekin_dir
        with open(f_json, 'r') as fjson:
            self.setup_init = json.load(fjson)
        self.gui = {}
        for node in self.setup_init.keys():
            self.gui[node] = {}
        user = os.getenv('USER')

#------------
# Reactivity
#------------

        cb = ['D(D,n)3He', 'D(D,P)T', 'D(T,n)4He', 'D(3He,P)4He']
        self.fill_layout(reac_layout, 'reac', checkbuts=cb)

#---------------
# Cross-sections
#---------------

        entries = ['E', 'Z1', 'Z2']
        cb = ['log_scale']
        combos = {'reac': reaction.keys()}
        self.fill_layout(cross_layout, 'cross', entries=entries, checkbuts=cb, combos=combos)

#-----------
# Kinematics
#-----------

        entries = ['v1x', 'v1y', 'v1z', 'v2x', 'v2y', 'v2z', 'losx', 'losy', 'losz']
        combos = {'reac': reaction.keys()}
        self.fill_layout(kin_layout, 'kinematics', entries=entries, combos=combos, lbl_wid=40, ent_wid=80)

#--------
# Spectra
#--------

        entries = ['dens', 'E1', 'E2', 'losx', 'losy', 'losz', 'n_sample']
        combos = {'reac': reaction.keys()}
        self.fill_layout(spec_layout, 'spectrum', entries=entries, combos=combos)

#---------
# LOS
#---------

        entries = ['disk_thick', 'cell_radius', 'coll_diam', 'd_det_coll', 'det_radius', 'tilt',
            'tan_radius', 'y_det', 'z_det', 'Rmaj', 'r_chamb', 'label']
        cb = ['writeLOS']
        self.fill_layout(los_layout, 'detector', entries=entries, checkbuts=cb)

        
#-----------
# GUI layout
#-----------

        self.setStyleSheet("QLabel { width: 4 }")
        self.setStyleSheet("QLineEdit { width: 4 }")
        self.setGeometry(10, 10, xwin, ywin)
        self.setWindowTitle('ReKin')
        self.show()


    def about(self):

        webbrowser.open('https://www.aug.ipp.mpg.de/~git/rekin/index.html')


    def fill_layout(self, layout, node, entries=[], checkbuts=[], combos={}, lbl_wid=140, ent_wid=180, col_shift=0):
# Checkbutton

        jrow = 0

        for key in checkbuts:
            self.gui[node][key] = QCheckBox(key)
            layout.addWidget(self.gui[node][key], jrow, col_shift)
            if self.setup_init[node][key]:
                self.gui[node][key].setChecked(True)
            jrow += 1

        for key in entries:
            val = self.setup_init[node][key]
            qlbl = QLabel(key)
            qlbl.setFixedWidth(lbl_wid)
            self.gui[node][key] = QLineEdit(str(val))
            self.gui[node][key].setFixedWidth(ent_wid)
            if isinstance(val, int):
                valid = QIntValidator()
                self.gui[node][key].setValidator(valid)
            elif isinstance(val, float):
                valid = QDoubleValidator()
                self.gui[node][key].setValidator(valid)
            layout.addWidget(qlbl               , jrow, col_shift)
            layout.addWidget(self.gui[node][key], jrow, col_shift+1)
            jrow += 1

        for key, combs in combos.items():
            self.gui[node][key] = QComboBox()
            for comb in combs:
                self.gui[node][key].addItem(comb.strip())
            index = self.gui[node][key].findText(self.setup_init[node][key].strip())
            self.gui[node][key].setCurrentIndex(index)
            layout.addWidget(self.gui[node][key], jrow, col_shift)
            jrow += 1

        layout.setRowStretch(layout.rowCount(), 1)
        layout.setColumnStretch(layout.columnCount(), 1)


    def gui2json(self):
        '''Returns a dic'''
        json_d = {}
        for node in self.gui.keys():
            json_d[node] = self.get_gui_tab(node)
        return json_d


    def get_gui_tab(self, node):

        node_dic = {}
        for key, val in self.gui[node].items():
            node_dic[key] = {}
            if isinstance(val, QLineEdit):
                if isinstance(self.setup_init[node][key], int):
                    node_dic[key] = int(val.text())
                elif isinstance(self.setup_init[node][key], float):
                    node_dic[key] = float(val.text())
                else:
                    node_dic[key] = val.text()
            elif isinstance(val, QCheckBox):
                node_dic[key] = val.isChecked()
            elif isinstance(val, QComboBox):
                node_dic[key] = val.currentText()
            elif isinstance(val, QButtonGroup):
                bid = val.checkedId()
                node_dic[key] = self.rblists[key][bid]

        return node_dic


    def set_gui(self, json_d):

        for node, val1 in json_d.items():
            for key, vald in val1.items():
                if key not in self.gui[node].keys():
                    continue
                widget = self.gui[node][key]
                if isinstance(widget, QCheckBox):
                    widget.setChecked(vald)
                elif isinstance(widget, QButtonGroup):
                    for but in widget.buttons():
                        if but.text() == vald:
                            but.setChecked(True)
                elif isinstance(widget, QLineEdit):
                    if vald:
                        widget.setText(str(vald))
                elif isinstance(widget, QComboBox):
                    for index in range(widget.count()):
                        if widget.itemText(index).strip() == vald.strip():
                            widget.setCurrentIndex(index)
                            break


    def load_json(self):

        ftmp = QFileDialog.getOpenFileName(self, 'Open file', \
            '%s/settings' %rekin_dir, "json files (*.json)")
        if qt5:
            f_json = ftmp[0]
        else:
            f_json = str(ftmp)

        with open(f_json) as fjson:
            setup_d = json.load(fjson)
        self.set_gui(setup_d)


    def save_json(self):

        out_dic = self.gui2json()
        ftmp = QFileDialog.getSaveFileName(self, 'Save file', \
            '%s/settings' %rekin_dir, "json files (*.json)")
        if qt5:
            f_json = ftmp[0]
        else:
            f_json = str(ftmp)
        with open(f_json, 'w') as fjson:
            fjson.write(json.dumps(out_dic))


    def reactivity(self):

        reac_dic = self.get_gui_tab('reac')

        Ti_keV = np.linspace(1., 1000., 1000)
        
        logger.info('Reactivities')

        reac_d = {}

        for key, reac in reaction.items():
            if hasattr(reac, 'coeff_reac') and reac_dic[key]:
                reac_d[key] = react(Ti_keV, key)

        if not hasattr(self, 'wid'):
            self.wid = plot_rekin.plotWindow()
        fig_reac =  plot_rekin.fig_reactivity(reac_d, Ti_keV)
        self.wid.addPlot('Reactivities', fig_reac)
        self.wid.show()


    def cross_section(self):

        cross_dic = self.get_gui_tab('cross')

        Egrid = np.array([float(x) for x in eval(cross_dic['E'])], dtype=np.float32)
        logger.info('Cross-section')

        theta = np.linspace(0, np.pi, 61)
        mu_grid = np.cos(theta)
        reac_lbl = cross_dic['reac']
        sigma = cs.sigma_diff(Egrid, mu_grid, reac_lbl, 2, 2)

        if not hasattr(self, 'wid'):
            self.wid = plot_rekin.plotWindow()
        fig_cross = plot_rekin.fig_cross(sigma, theta, Egrid, log_scale=cross_dic['log_scale'])
        self.wid.addPlot('Cross-sections', fig_cross)
        self.wid.show()


    def scatt_kine(self):

        scatt_dic = self.get_gui_tab('kinematics')
        logger.info('Scattering kinematics')
        reac_lbl = scatt_dic['reac']
        v1 = [scatt_dic['v1x'], scatt_dic['v1y'], scatt_dic['v1z']]
        v2 = [scatt_dic['v2x'], scatt_dic['v2y'], scatt_dic['v2z']]
        versor_out = [scatt_dic['losx'], scatt_dic['losy'], scatt_dic['losz']]
        print(v1)
        print(versor_out)

        kin = ck.calc_reac(v1, v2, versor_out, reac_lbl)

        if not hasattr(self, 'wid'):
            self.wid = plot_rekin.plotWindow()
        fig_kine = plt.figure('Kinematics', (8.8, 5.9), dpi=100)
        self.wid.addPlot('Scattering kinematics', fig_kine)
# Need to define axes after canvas creation, in order to rotate the figure interactively
        plot_rekin.ax_scatt(fig_kine, kin)
        self.wid.show()


    def spectra(self):

        rekin_dic = self.get_gui_tab('spectrum')

        logger.info('Spectra')
        dens = rekin_dic['dens']
        reac_lbl = rekin_dic['reac']
        n_sample = rekin_dic['n_sample']
        E1 = rekin_dic['E1']
        E2 = rekin_dic['E2']
        losx = rekin_dic['losx']
        losy = rekin_dic['losy']
        losz = rekin_dic['losz']
        Earr, weight = spc.mono_iso(E1, E2, [losx, losy, losz], reac_lbl, n_sample=n_sample)
        Egrid, Espec = spc.calc_spectrum(dens, Earr, weight)

        if not hasattr(self, 'wid'):
            self.wid = plot_rekin.plotWindow()
        fig_spec = plot_rekin.fig_spec(Egrid, Espec)
        self.wid.addPlot('Neutron Spectrum', fig_spec)
        self.wid.show()


    def los(self):

        logger.info('Starting LOS cone calculation')
        geo = self.get_gui_tab('detector')

        los_d = {}
        los_d['x0']    = -geo['y_det']
        los_d['y0']    = 0.001
        los_d['z0']    = geo['z_det']
        los_d['xend']  = 0 # 1 plasma crossing
#        los_d['xend']  = geo['y_det'] # 2 plasma crossings
        los_d['theta'] = np.radians(geo['tilt'])
        los_d['phi']   = -np.arctan2(geo['tan_radius'], geo['y_det'])

        ctilt = np.cos(los_d['theta'])
        dy = geo['disk_thick']*ctilt
        ndisks = int(-2*geo['y_det']/dy)

        self.det = DETECTOR()
        self.det.los = PHILOS(los_d, npoints=ndisks)

# Get separatrix {R,z}

        rmin = geo['Rmaj']  - geo['r_chamb'] 
        rmax = geo['Rmaj']  + geo['r_chamb'] 

# Restrict to LOS inside the plasma

        dl = np.hypot(self.det.los.rline[1] - self.det.los.rline[0], self.det.los.zline[1] - self.det.los.zline[0])
        ind = (self.det.los.rline > rmin - dl) & (self.det.los.rline < rmax + dl)
        r_in  = self.det.los.rline[ind]
        self.det.los.z = self.det.los.zline[ind]
        self.det.los.y = self.det.los.xline[ind]
        y_in  = self.det.los.yline[ind]

# Write ASCII file for LOS, used in the spectrum evaluation

        self.det.pos = (geo['tan_radius'], geo['y_det'], geo['z_det'])
        det_dist = np.hypot(self.det.los.y - self.det.pos[1], self.det.los.z - self.det.pos[2])
        ctilt = np.cos(np.radians(geo['tilt']))
        stilt = np.sin(np.radians(geo['tilt']))
        dy = geo['disk_thick']*ctilt
        coll_rad = 0.5*geo['coll_diam']
        dist_corr = geo['d_det_coll']/(1. + geo['det_radius']/coll_rad) # Shifting the cone vertex to the point where lines (det-left - coll-right, det-right - coll-left) cross
        self.det.tan_cone_aper = coll_rad/dist_corr
#        logger.debug(coll_rad, dist_corr, tan_cone_aper)
        offset = geo['d_det_coll'] - dist_corr # scalar, r2
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
            n_circles = int(0.5 + disk_radius[jdisk]/geo['cell_radius'])
            delta_radius = disk_radius[jdisk]/float(n_circles)

# radius, alpha in the 'middle' of the sector
# The central circle has only one sector (cell)
            cvol = np.pi * delta_radius**2 * dy
            self.cell.vol = np.append(self.cell.vol, np.repeat(cvol, n_circles**2))

            radius = (0.5 + np.arange(n_circles))*delta_radius
            radius[0] = 0.
            omega_fac = np.pi * geo['det_radius']**2/det_dist[jdisk]**3
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
# Plot
        if not hasattr(self, 'wid'):
            self.wid = plot_rekin.plotWindow()
        fig_los = plot_rekin.fig_los(self.cell)
        self.wid.addPlot('Detector LoS', fig_los)
        self.wid.show()
# Write output
        if geo['writeLOS']:
            self.writeLOS()


    def writeLOS(self):

        n_cells = len(self.cell.vol)
        n_disks = len(self.det.los.y)
        cone_aper = np.degrees(np.arctan(self.det.tan_cone_aper))
        geo = self.get_gui_tab('detector')
        ctilt = np.cos(np.radians(geo['tilt']))
        dy = geo['disk_thick']*ctilt

        header = \
           'LOS\n' + \
           '   y1 = %9.4f m\n'  % self.det.los.y[0]  + \
           '   y2 = %9.4f m\n'  % self.det.los.y[-1] + \
           '   z1 = %9.4f m\n'  % self.det.los.z[0]  + \
           '   z2 = %9.4f m\n'  % self.det.los.z[-1] + \
           'Detector:\n' + \
           '   Position [m] x, y, z: %9.4f, %9.4f, %9.4f\n' %self.det.pos + \
           '   Radius = %9.4f m\n' % geo['det_radius'] + \
           '   Collimation angle = %9.4f deg\n' % cone_aper + \
           'Disks:\n' + \
           '   Thickness = %9.4f m\n' % dy + \
           '   # disks = %5d\n' % n_disks + \
           'Cells:\n' + \
           '   Radius = %9.4f m\n' % geo['cell_radius'] + \
           '   # cells = %d\n' %n_cells + \
"""
   (x,y,x) cell cartensian coordinates [m]
   Omega  Steradians is the volume in the solid angle
   Vol = cell volume [m**3]
x             y             z             Omega         Vol
"""

        los_file = '%s.los' %geo['label']

        logger.info('Storing ASCII output, n_cells=%d', n_cells)
        np.savetxt(los_file, np.hstack((self.cell.x, self.cell.y, self.cell.z, self.cell.omega, self.cell.vol)).reshape(5, n_cells).T, header=header, fmt='%13.6E')
        logger.info('Written output file %s' %los_file)


if __name__ == '__main__':


    app = QApplication(sys.argv)
    main = REAC_GUI()
    app.exec_()
