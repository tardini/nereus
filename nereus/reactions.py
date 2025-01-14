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
# neutrons/paper/bosch_nf92.pdf page 625 table VII

reaction = {}

reaction['DP'] = reac_class()
reaction['DP'].in1 = proton
reaction['DP'].in2 = deuteron

reaction['D(D,n)3He'] = reac_class()
reaction['D(D,n)3He'].in1 = deuteron
reaction['D(D,n)3He'].in2 = deuteron
reaction['D(D,n)3He'].prod1 = neutron
reaction['D(D,n)3He'].prod2 = he3
reaction['D(D,n)3He'].coeff_reac = [5.4336e-12, 5.85778e-3, 7.68222-3, 0, -2.964e-6, 0, 0]

reaction['D(D,P)T'] = reac_class()
reaction['D(D,P)T'].in1 = deuteron
reaction['D(D,P)T'].in2 = deuteron
reaction['D(D,P)T'].prod1 = proton
reaction['D(D,P)T'].prod2 = triton
reaction['D(D,P)T'].coeff_reac = [5.65718e-12, 3.41267e-3, 1.99167e-3, 0, 1.05060e-5, 0, 0]

reaction['D(3He,P)4He'] = reac_class()
reaction['D(3He,P)4He'].in1 = he3
reaction['D(3He,P)4He'].in2 = deuteron
reaction['D(3He,P)4He'].prod1 = proton
reaction['D(3He,P)4He'].prod2 = he4
reaction['D(3He,P)4He'].coeff_reac = [5.51036e-10, 6.41918e-3, -2.02896e-3, -1.9108e-5,  1.35776e-4, 0, 0]

reaction['D(T,n)4He'] = reac_class()
reaction['D(T,n)4He'].in1 = deuteron
reaction['D(T,n)4He'].in2 = triton
reaction['D(T,n)4He'].prod1 = neutron
reaction['D(T,n)4He'].prod2 = he4
reaction['D(T,n)4He'].coeff_reac = [1.17302e-9 , 1.51361e-2, 7.51886e-2, 4.60643e-3,  1.35000e-2, -1.06750e-4, 1.36600e-5]
