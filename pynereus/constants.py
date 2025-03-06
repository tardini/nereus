import scipy.constants as sc

con = sc.physical_constants
amu = con['atomic mass constant energy equivalent in MeV'][0]
mpc2 = con['proton mass energy equivalent in MeV'][0]
mDc2 = con['deuteron mass energy equivalent in MeV'][0]
mTc2 = con['triton mass energy equivalent in MeV'][0]
mnc2 = con['neutron mass energy equivalent in MeV'][0]
mHe3c2 = 3.014932150*amu # MeV
mac2 = con['alpha particle mass energy equivalent in MeV'][0]
mHe4c2 = con['alpha particle mass energy equivalent in MeV'][0]
alpha = sc.alpha # Fine structure constant
epsilon0 = sc.epsilon_0
echarge = sc.elementary_charge
c = sc.c
