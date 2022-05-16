import scipy.constants as sc

con = sc.physical_constants
amu = con['atomic mass constant energy equivalent in MeV'][0]
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

if __name__ == '__main__':
    print('neut. mass in Mev: %12.4e' %mnc2)
    print('mass_D in Mev: %12.4e' %mDc2)
    print('mass_T in Mev: %12.4e' %mTc2)
    print('mass_He3 in Mev: %12.4e' %mHe3c2)
    print(mDc2 + mDc2 - mnc2 - mHe3c2)
