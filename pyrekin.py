#!/usr/bin/env python

__author__  = 'Giovanni Tardini (Tel. 1898)'
__version__ = '0.0.1'
__date__    = '29.03.2022'

import os, sys, logging

try:
    from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGridLayout, QMenu, QAction, QLabel, QPushButton, QLineEdit, QCheckBox, QFileDialog, QRadioButton, QButtonGroup, QTabWidget, QVBoxLayout, QComboBox
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtCore import Qt, QRect, QSize
    qt5 = True
except:
    from PyQt4.QtCore import Qt, QRect, QSize
    from PyQt4.QtGui import QPixmap, QIcon, QMainWindow, QWidget, QApplication, QGridLayout, QMenu, QAction, QLabel, QPushButton, QLineEdit, QCheckBox, QFileDialog, QRadioButton, QButtonGroup, QTabWidget, QVBoxLayout, QComboBox
    qt5 = False

import numpy as np
import matplotlib.pylab as plt
import dixm
from reactivities import *
import constants as con
import calc_cross_section as cs


xml = dixm.DIX()

fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s: %(message)s', '%H:%M:%S')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logger = logging.getLogger('rekin')
logger.addHandler(hnd)
logger.setLevel(logging.DEBUG)

rekin_dir = os.path.dirname(os.path.realpath(__file__))


class REKIN(QMainWindow):


    def __init__(self):

        if sys.version_info[0] == 3:
            super().__init__()
        else:
            super(QMainWindow, self).__init__()

        xwin  = 643
        yhead = 44
        yline = 30
        ybar  = 48
        ywin  = xwin + yhead + ybar

        qhead  = QWidget(self)
        qbar   = QWidget(self)
        qtabs  = QTabWidget(self)
        qhead.setGeometry(QRect(0,     0, xwin, yhead))
        qbar.setGeometry(QRect(0, yhead, xwin, ybar))
        qtabs.setGeometry(QRect(0, yhead+ybar, xwin, ywin-yhead-ybar))
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

#--------
# Menubar
#--------

        menubar = self.menuBar()
        fileMenu = QMenu('&File', self)
        xmlMenu  = QMenu('&Setup', self)
        helpMenu = QMenu('&Help', self)
        menubar.addMenu(fileMenu)
        menubar.addMenu(xmlMenu)
        menubar.addMenu(helpMenu)

        fmap = {'reac': self.reactivity, 'csec': self.cross_section, 'kine': self.scatt_kine, \
            'spec': self.spectra, 'exit': sys.exit}
        qlbl = {'reac': '&Reactivity', 'csec': '&Cross-section', 'kine': '&Kinematics', \
            'spec': '&Spectra', 'exit': 'Exit'}
        Action = {}
        for key, lbl in qlbl.items():
            Action = QAction(lbl, fileMenu)
            Action.triggered.connect(fmap[key])
            if key == 'exit':
                fileMenu.addSeparator()
            fileMenu.addAction(Action)

        loadAction  = QAction('&Load Setup', xmlMenu)
        saveAction  = QAction('&Save Setup', xmlMenu)
        xmlMenu.addAction(loadAction)
        xmlMenu.addAction(saveAction)
        loadAction.triggered.connect(self.load_xml)
        saveAction.triggered.connect(self.save_xml)

        aboutAction = QAction('&About', helpMenu)
        aboutAction.triggered.connect(self.about)
        helpMenu.addAction(aboutAction)

        header_grid.addWidget(menubar, 0, 0, 1, 10)

# Icons
        dum_lbl  = QLabel(200*' ')
        for jpos, key in enumerate(qlbl.keys()):
            but = QPushButton()
            but.setIcon(QIcon('%s/%s.gif' %(rekin_dir, key)))
            but.setIconSize(QSize(ybar, ybar))
            but.clicked.connect(fmap[key])
            tbar_grid.addWidget(but, 0, jpos)

        tbar_grid.addWidget(dum_lbl, 0, jpos+1, 1, 10)

# User options

        self.setup_init = xml.xml2dict('%s/xml/default.xml' %rekin_dir)
        self.gui = {}
        user = os.getenv('USER')

#------------
# Reactivity
#------------

        cb = ['ddn3he', 'ddpt', 'dt', 'd3he']
        self.new_tab(reac_layout, checkbuts=cb)

#---------------
# Cross-sections
#---------------

        entries = ['E_cs', 'reac_Z1', 'reac_Z2']
        cb = ['log_scale']
        combos = {'reac_cs': con.reac_lbls}
        self.new_tab(cross_layout, entries=entries, checkbuts=cb, combos=combos)

#-----------
# GUI layout
#-----------

        self.setStyleSheet("QLabel { width: 4 }")
        self.setStyleSheet("QLineEdit { width: 4 }")
        self.setGeometry(10, 10, xwin, ywin)
        self.setWindowTitle('ReKin')
        self.show()


    def about(self):

        mytext = 'Documentation at <a href="http://www.aug.ipp.mpg.de/~git/tot/index.html">TOT/TTH diagnostic homepage</a>'
        h = tkhyper.HyperlinkMessageBox("Help", mytext, "500x60")


    def new_tab(self, layout, entries=[], checkbuts=[], combos={}):
# Checkbutton

        jrow = 0
        for key in checkbuts:
            if key in con.reac_lbls.keys():
                lbl = con.reac_lbls[key]
            else:
                lbl = key
            self.gui[key] = QCheckBox(lbl)
            layout.addWidget(self.gui[key], jrow, 0, 1, 2)
            if self.setup_init[key].lower().strip() == 'true':
                self.gui[key].setChecked(True)
            jrow += 1

        for key in entries:
            val = self.setup_init[key]
            if key in con.reac_lbls.keys():
                lbl = con.reac_lbls[key]
            else:
                lbl = key
            qlbl = QLabel(lbl)
            qlbl.setFixedWidth(200)
            self.gui[key] = QLineEdit(val)
            self.gui[key].setFixedWidth(180)
            layout.addWidget(qlbl         , jrow, 0)
            layout.addWidget(self.gui[key], jrow, 1)
            jrow += 1

        for key, combo_d in combos.items():
            self.gui[key] = QComboBox()
            for comb in combo_d.keys():
                self.gui[key].addItem(comb.strip())
            index = self.gui[key].findText(self.setup_init[key].strip())
            self.gui[key].setCurrentIndex(index)
            layout.addWidget(self.gui[key], jrow, 0, 1, 2)
            jrow += 1

        layout.setRowStretch(layout.rowCount(), 1)
        layout.setColumnStretch(layout.columnCount(), 1)


    def get_gui(self):

        rekin_dic = {}
        for key, val in self.gui.items():
            if isinstance(val, QLineEdit):
                rekin_dic[key] = val.text()
            elif isinstance(val, QCheckBox):
                rekin_dic[key] = val.isChecked()
            elif isinstance(val, QButtonGroup): # for radiobuttons
                bid = val.checkedId()
                rekin_dic[key] = self.rblists[key][bid]
            elif isinstance(val, QComboBox):
                rekin_dic[key] = val.itemText(val.currentIndex())

        return rekin_dic


    def set_gui(self, xml_d):

        for key, val in xml_d.items():
            val = val.strip()
            val_low = val.lower()
            widget = self.gui[key]
            if isinstance(widget, QCheckBox):
                if val_low == 'false':
                    widget.setChecked(False)
                elif val_low == 'true':
                    widget.setChecked(True)
            elif isinstance(widget, QButtonGroup):
                for but in widget.buttons():
                    if but.text().lower() == val_low:
                        but.setChecked(True)
            elif isinstance(widget, QLineEdit):
                if val_low == '':
                    widget.setText(' ')
                else:
                    widget.setText(val)
            elif isinstance(widget, QComboBox):
                for index in range(widget.count()):
                    if widget.itemText(index).strip() == val.strip():
                        widget.setCurrentIndex(index)
                        break


    def load_xml(self):

        ftmp = QFileDialog.getOpenFileName(self, 'Open file', \
            '%s/xml' %rekin_dir, "xml files (*.xml)")
        if qt5:
            fxml = ftmp[0]
        else:
            fxml = str(ftmp)
        setup_d = xml.xml2dict(fxml)
        self.set_gui(setup_d)


    def save_xml(self):

        rekin_dic = self.get_gui()
        ftmp = QFileDialog.getSaveFileName(self, 'Save file', \
            '%s/xml' %rekin_dir, "xml files (*.xml)")
        if qt5:
            fxml = ftmp[0]
        else:
            fxml = str(ftmp)
        xml.dict2xml(rekin_dic, fxml)


    def reactivity(self):

        rekin_dic = self.get_gui()

        Ti_keV = np.linspace(1., 60., 60)
        
        logger.info('Reactivities')

        plt.figure('Reac', (10,8))
        for key, coeff in coeff_reac.items():
            if rekin_dic[key]:
                reac = react(Ti_keV, m1_c2=con.m[key].in1, m2_c2=con.m[key].in2, coeff=coeff)
                plt.semilogy(Ti_keV, reac, label=con.reac_lbls[key])
        plt.xlim([0, 60])
        plt.xlabel(r'T$_i$ [keV]')
        plt.ylabel(r'Reactivity [cm$^3$/s]')
        plt.grid()
        plt.legend(loc=2)
        plt.show()
        

    def cross_section(self):

        rekin_dic = self.get_gui()

        Egrid = np.array([float(x) for x in eval(rekin_dic['E_cs'])], dtype=np.float32)
        print(rekin_dic['log_scale'])
        logger.info('Cross-section')
        theta = np.linspace(0, np.pi, 61)
        mu_grid = np.cos(theta)

        plt.figure('Cross-sections', (10, 8))

        key = rekin_dic['reac_cs'].lower().strip()
        reac = cs.sigma_diff(Egrid, mu_grid, key, 2, 2)
        plt.title(con.reac_lbls[key])
        if rekin_dic['log_scale']:
            plt.semilogy(np.degrees(theta), reac)
        else:
            plt.plot(np.degrees(theta), reac)
        plt.xlim([0, 180])
        plt.xlabel('Angle [deg]')
        plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

        plt.show()


    def scatt_kine(self):

        logger.info('Scattering kinematics')


    def spectra(self):

        logger.info('Spectra')


if __name__ == '__main__':


    app = QApplication(sys.argv)
    main = REKIN()
    app.exec_()
