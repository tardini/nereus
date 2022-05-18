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
for reac in ('dp', 'd3he', 'alphad', 'alphat', 'd3healphap', 'dt', 'ddpt', 'ddn3he'):
    m[reac] = mass()

m['dp'].in1 = mpc2
m['dp'].in2 = mDc2

m['ddn3he'].in1   = mDc2
m['ddn3he'].in2   = mDc2
m['ddn3he'].prod1 = mnc2
m['ddn3he'].prod2 = mHe3c2

m['ddpt'].in1   = mDc2
m['ddpt'].in2   = mDc2
m['ddpt'].prod1 = mpc2
m['ddpt'].prod2 = mTc2

m['d3he'].in1   = mDc2
m['d3he'].in2   = mHe3c2
m['d3he'].prod1 = mpc2
m['d3he'].prod2 = mac2

m['dt'].in1   = mDc2
m['dt'].in2   = mTc2
m['dt'].prod1 = mnc2
m['dt'].prod2 = mac2

reac_lbls = {'ddn3he': 'DD->n3He', 'ddpt': 'DD->pT', 'dt': 'DT->n4He', 'd3he': 'D3He->p4He', 'coul': 'Coulomb'}
