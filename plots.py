import sys
import matplotlib
import numpy as np

try:
    from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout
    from PyQt5.QtCore import QRect
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
except:
    from PyQt4.QtGui import QWidget, QTabWidget, QVBoxLayout
    from PyQt4.QtCore import QRect
    matplotlib.use('Qt4Agg')
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

fsize   = 8
titsize = 10
lblsize = 10
fig_size = (8.8, 5.9)
lwid = 2.5


class plotWindow(QWidget):


    def __init__(self, geom=(80, 30, 900, 680)):

        if sys.version_info[0] == 3:
            super().__init__()
        else:
            super(QWidget, self).__init__()
        self.setGeometry(QRect(*geom))
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("QTabBar::tab { width: 165 }")
        self.setWindowTitle('Nuclear reactions properties')


    def addPlot(self, title, figure):

        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)
        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        for jtab in range(self.tabs.count()):
            if self.tabs.tabText(jtab) == title:
                self.tabs.removeTab(jtab)
        self.tabs.addTab(new_tab, title)
        self.tabs.setCurrentIndex(self.tabs.count()-1)


def fig_reactivity(reac_d, Ti_keV, color='#d0d0d0', figSize=None):

    if figSize is None:
        figSize = fig_size
    fig_reac = plt.figure('Reac', figSize, dpi=100, facecolor=color)
    plt.cla()
    for lbl, reac in reac_d.items():
#        if lbl[:3] != 'D(D':
            plt.loglog(Ti_keV, reac, label=lbl, linewidth=lwid)
    dd = reac_d['D(D,n)3He'] + reac_d['D(D,P)T']
#    plt.loglog(Ti_keV, dd, label='DD', linewidth=lwid)
    plt.xlim([Ti_keV[0], Ti_keV[-1]])
    plt.ylim([1e-19, 1e-14])
    plt.xlabel(r'T$_i$ [keV]')
    plt.ylabel(r'Reactivity [cm$^3$/s]')
    plt.grid()
    plt.legend(loc=4)

    return fig_reac


def fig_cross(reac, theta, Egrid, log_scale=False, color='#d0d0d0'):

    fig_cs = plt.figure('Cross-sections', fig_size, dpi=100, facecolor=color)
    plt.cla()
    the_deg = np.degrees(theta)
    for j, E in enumerate(Egrid):
        if log_scale:
            plt.semilogy(the_deg, reac[:, j], linewidth=lwid, label='E=%5.3f MeV' %E)
        else:
            plt.plot(the_deg, reac[:, j], linewidth=lwid, label='E=%5.3f MeV' %E)
    plt.xlim([the_deg[0], the_deg[-1]])
    plt.legend()
    plt.xlabel('Scattering angle [deg]')
    plt.ylabel(r'$\frac{d\sigma}{d\Omega}$ [mbarn]')

    return fig_cs


def fig_resp(resp, En_MeV=2.5, color='#d0d0d0'):

    d2 = (resp.En_MeV - En_MeV)**2
    jEn = np.argmin(d2)
    fig_resp = plt.figure('Response', (8.8, 5.9), dpi=100)
    plt.cla()
    plt.plot(resp.Ephs_MeVee, resp.RespMat[jEn], 'r-', label='En=%5.3f MeV' %resp.En_MeV[jEn])
    if hasattr(resp, 'RespMat_gb'):
        plt.plot(resp.Ephs_MeVee, resp.RespMat_gb[jEn], 'b-', label='En=%5.3f MeV, GB' %resp.En_MeV[jEn])
    plt.legend()
    plt.xlim([0, 3])
    plt.ylim([0, 0.04])

    return fig_resp


def fig_spec(Egrid, Espec, color='#d0d0d0'):

    fig_spc = plt.figure('Spectrum', fig_size, dpi=100, facecolor=color)
    plt.cla()
#    plt.semilogy(Egrid, Espec)
    plt.plot(Egrid, Espec, linewidth=lwid)
    plt.xlabel('En [MeV]')
    plt.ylabel('dN/dEn')

    return fig_spc
