import scipy.constants as sc

con = sc.physical_constants
amu = con['atomic mass constant energy equivalent in MeV'][0]
mpc2 = con['proton mass energy equivalent in MeV'][0]
mDc2 = con['deuteron mass energy equivalent in MeV'][0]
mTc2 = con['triton mass energy equivalent in MeV'][0]
mnc2 = con['neutron mass energy equivalent in MeV'][0]
mHe3c2 = 3.014932150*amu # MeV
mac2 = con['alpha particle mass energy equivalent in MeV'][0]
alpha = sc.alpha
epsilon0 = sc.epsilon_0
echarge = sc.elementary_charge
c = sc.c
Z_D = 1

class mass:
    pass

m = {}
for reac in ('dp', 'd3he', 'alphad', 'alphat', 'd3healphap', 'dt', 'dd'):
    m[reac] = mass()

m['dp'].in1 = mpc2
m['dp'].in2 = mDc2

m['dd'].in1   = mDc2
m['dd'].in2   = mDc2
m['dd'].prod1 = mnc2
m['dd'].prod2 = mHe3c2

m['dt'].in1   = mDc2
m['dt'].in2   = mTc2
m['dt'].prod1 = mnc2
m['dt'].prod2 = mac2
