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
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except:
    from PyQt4.QtCore import Qt, QRect, QSize, QIntValidator, QDoubleValidator
    from PyQt4.QtGui import QPixmap, QIcon, QMainWindow, QWidget, QApplication, QGridLayout, QMenu, QAction, QLabel, QPushButton, QLineEdit, QCheckBox, QFileDialog, QRadioButton, QButtonGroup, QTabWidget, QVBoxLayout, QComboBox
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    qt5 = False

import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from reactivities import *
import calc_cross_section as cs
import calc_kinematics as ck
import calc_spectrum as spc
import los
import plots
from reactions import reaction
from nresp import nresp

os.environ['BROWSER'] = '/usr/bin/firefox'

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('neutReac')
logger.addHandler(hnd)
logger.setLevel(logging.DEBUG)

neutReacDir = os.path.dirname(os.path.realpath(__file__))


class REAC_GUI(QMainWindow):


    def __init__(self):

        if sys.version_info[0] == 3:
            super().__init__()
        else:
            super(QMainWindow, self).__init__()

        xwin  = 1000
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

        qnresp = QWidget()
        nresp_layout = QGridLayout()
        qnresp.setLayout(nresp_layout)
        qtabs.addTab(qnresp, 'NRESP')

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
                'spec': self.spectra, 'los': self.los, 'nresp': self.nresp, 'exit': sys.exit}
        qlbl = {'reac': '&Reactivity', 'csec': '&Cross-section', 'kine': '&Kinematics', \
                'spec': '&Spectra', 'los': '&Line-of-Sight', 'nresp': '&NRESP', 'exit': 'Exit'}
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
        loadAction.triggered.connect(self.loadJSON)
        saveAction.triggered.connect(self.saveJSON)

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
            fgif = '%s/%s.gif' %(neutReacDir, key)
            if os.path.isfile(fgif):
                but.setIcon(QIcon(fgif))
            else:
                fpng = '%s/%s.png' %(neutReacDir, key)
                but.setIcon(QIcon(fpng))
            but.setIconSize(QSize(ybar, ybar))
            but.clicked.connect(fmap[key])
            tbar_grid.addWidget(but, 0, jpos)
            xpos += yicon + 4

        tbar_grid.addWidget(dum_lbl, 0, jpos+1, 1, 10)

# User options

        f_json = '%s/settings/default.json' %neutReacDir
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
        cb = ['Write LOS']
        self.fill_layout(los_layout, 'detector', entries=entries, checkbuts=cb)

#---------
# NRESP
#---------

        entries = ['Energy array', 'f_detector', 'f_in_light', 'nmc', 'En_wid_frac', 'Ebin_MeVee', 'Energy for PHS plot']
        combos = {'distr': ['gauss', 'mono']}
        cb = ['Write nresp']
        self.fill_layout(nresp_layout, 'nresp', entries=entries, combos=combos, checkbuts=cb)

#-----------
# GUI layout
#-----------

        self.setStyleSheet("QLabel { width: 4 }")
        self.setStyleSheet("QLineEdit { width: 4 }")
        self.setGeometry(10, 10, xwin, ywin)
        self.setWindowTitle('NeutReac')
        self.show()


    def about(self):

        webbrowser.open('https://github.com/tardini/neut_reac/wiki')


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
            qlbl = QLabel(key)
            qlbl.setFixedWidth(lbl_wid)
            self.gui[node][key] = QComboBox()
            for comb in combs:
                self.gui[node][key].addItem(comb.strip())
            index = self.gui[node][key].findText(self.setup_init[node][key].strip())
            self.gui[node][key].setCurrentIndex(index)
            layout.addWidget(qlbl               , jrow, col_shift)
            layout.addWidget(self.gui[node][key], jrow, col_shift+1)
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


    def loadJSON(self):

        ftmp = QFileDialog.getOpenFileName(self, 'Open file', \
            '%s/settings' %neutReacDir, "json files (*.json)")
        if qt5:
            f_json = ftmp[0]
        else:
            f_json = str(ftmp)

        with open(f_json) as fjson:
            setup_d = json.load(fjson)
        self.set_gui(setup_d)


    def saveJSON(self):

        out_dic = self.gui2json()
        ftmp = QFileDialog.getSaveFileName(self, 'Save file', \
            '%s/settings' %neutReacDir, "json files (*.json)")
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
            self.wid = plots.plotWindow()
        fig_reac =  plots.fig_reactivity(reac_d, Ti_keV)
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
            self.wid = plots.plotWindow()
        fig_cross = plots.fig_cross(sigma, theta, Egrid, log_scale=cross_dic['log_scale'])
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
            self.wid = plots.plotWindow()
        fig_kine = plt.figure('Kinematics', (8.8, 5.9), dpi=100)
        self.wid.addPlot('Scattering kinematics', fig_kine)
# Need to define axes after canvas creation, in order to rotate the figure interactively
        plots.ax_scatt(fig_kine, kin)
        self.wid.show()


    def spectra(self):

        neutReac_dic = self.get_gui_tab('spectrum')

        logger.info('Spectra')
        dens = neutReac_dic['dens']
        reac_lbl = neutReac_dic['reac']
        n_sample = neutReac_dic['n_sample']
        E1 = neutReac_dic['E1']
        E2 = neutReac_dic['E2']
        losx = neutReac_dic['losx']
        losy = neutReac_dic['losy']
        losz = neutReac_dic['losz']
        Earr, weight = spc.mono_iso(E1, E2, [losx, losy, losz], reac_lbl, n_sample=n_sample)
        Egrid, Espec = spc.calc_spectrum(dens, Earr, weight)

        if not hasattr(self, 'wid'):
            self.wid = plots.plotWindow()
        fig_spec = plots.fig_spec(Egrid, Espec)
        self.wid.addPlot('Neutron Spectrum', fig_spec)
        self.wid.show()


    def los(self):

        logger.info('Starting LOS cone calculation')
        geo = self.get_gui_tab('detector')

        dlos = los.DETECTOR_LOS(geo)

# Plot
        if not hasattr(self, 'wid'):
            self.wid = plots.plotWindow()
        fig_los = dlos.plotLOS()
        self.wid.addPlot('Detector LoS', fig_los)
        self.wid.show()

# Write output
        if geo['Write LOS']:
            dlos.writeLOS()


    def nresp(self):

        nresp_set = self.get_gui_tab('nresp')
        nEn = 17
        reComp = False
        if not hasattr(self, 'nresp_set'):
            reComp = True
        else:
            for key, val in self.nresp_set.items():
                if key not in ('Energy for PHS plot', 'Write nresp'):
                    if nresp_set[key] != val:
                        reComp = True
                        break
        if reComp:
            nrsp = nresp.NRESP(nresp_set)
        else:
            nrsp = self.nrsp
        fig_nr = nrsp.plotResponse(E_MeV=nresp_set['Energy for PHS plot'])
        if not hasattr(self, 'wid'):
            self.wid = plots.plotWindow()
        self.wid.addPlot('NRESP', fig_nr)
        self.wid.show()
        self.nresp_set = {key: val for key, val in nresp_set.items()}
        self.nrsp = nrsp

# Write output
        if nresp_set['Write nresp']:
            nrsp.to_nresp()


if __name__ == '__main__':


    app = QApplication(sys.argv)
    main = REAC_GUI()
    app.exec_()
