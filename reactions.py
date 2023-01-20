import constants as con

class reac_class:
    pass


class proton:

    m = con.mpc2
    Z = 1


class neutron:

    m = con.mnc2
    Z = 0


class deuteron:

    m = con.mDc2
    Z = 1


class triton:

    m = con.mTc2
    Z = 1


class he3:

    m = con.mHe3c2
    Z = 2


class he4:

    m = con.mHe4c2
    Z = 2

# Reaction classes

reaction = {}

reaction['dp'] = reac_class()
reaction['dp'].in1 = proton
reaction['dp'].in2 = deuteron

reaction['ddn3he'] = reac_class()
reaction['ddn3he'].in1 = deuteron
reaction['ddn3he'].in2 = deuteron
reaction['ddn3he'].prod1 = neutron
reaction['ddn3he'].prod2 = he3
reaction['ddn3he'].coeff_reac = [5.65718e-12, 3.41267e-3, 1.99167e-3, 0, 1.05060e-5, 0, 0]

reaction['ddpt'] = reac_class()
reaction['ddpt'].in1 = deuteron
reaction['ddpt'].in2 = deuteron
reaction['ddpt'].prod1 = proton
reaction['ddpt'].prod2 = triton
reaction['ddpt'].coeff_reac = [5.43360e-12, 5.85778e-3, 7.68222e-3, 0, -2.96400e-6, 0, 0]

reaction['d3he'] = reac_class()
reaction['d3he'].in1 = deuteron
reaction['d3he'].in2 = he3
reaction['d3he'].prod1 = proton
reaction['d3he'].prod2 = he4
reaction['d3he'].coeff_reac = [5.51036e-10, 6.41918e-3, -202896e-3, -1.9108e-5,  1.35776e-4, 0, 0]

reaction['dt'] = reac_class()
reaction['dt'].in1 = deuteron
reaction['dt'].in2 = triton
reaction['dt'].prod1 = neutron
reaction['dt'].prod2 = he4
reaction['dt'].coeff_reac = [1.17302e-9 , 1.51361e-2, 7.51886e-2, 4.60643e-3,  1.35000e-2, -1.06750e-4, 1.36600e-5]
