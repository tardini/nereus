import os, time, logging, json
import numpy as np
from scipy.linalg import norm
from scipy.interpolate import interp1d
import numba as nb

import settings, crossSections

logger = logging.getLogger('nresp.en2light')
logger.setLevel(level=logging.DEBUG)

# Measure CPU time for several methods

time_cyl  = 0.
count_cyl = 0

PI2 = 2.*np.pi

CS = crossSections.crossSections()

flt_typ = settings.flt_typ
int_typ = settings.int_typ

mediaCross = np.array([3, 2, 2, 0, 2, 1], dtype=int_typ)

Egrid, det_light = np.loadtxt(settings.f_in_light, skiprows=1, unpack=True)
light_int = interp1d(Egrid, det_light, assume_sorted=True, fill_value='extrapolate')

f_mass = '%s/inc/nucleiMassMeV.json' %settings.nrespDir
with open(f_mass, 'r') as fjson:
    massMeV = json.load(fjson)

with open(settings.f_detector, 'r') as fjson:
    detector = json.load(fjson)

with open(settings.f_poly, 'r') as fjson:
    poly = json.load(fjson)


@nb.njit
def scatteringDirection(cx_in, ctheta, PHI):

    '''Calculating flight direction of scattered particles in the lab frame
ctheta: cos of scattering angle w.r.t. incident direction
PHI     : azimutal scattering angle 
cx_in : versor of incident particle
cx_out: versor of scattered particle'''

    stheta = np.sqrt(1. - ctheta**2)
    X3 = [stheta*np.cos(PHI), stheta*np.sin(PHI), ctheta]

    if cx_in[2] > 0.999999:
        cx_out = np.array(X3, dtype=flt_typ)
    elif cx_in[2] < -0.999999:
        cx_out = -np.array(X3, dtype=flt_typ)
    else:
        cx_out = cx_in*X3[2]
        S1 = np.sqrt(1. - cx_in[2]**2)
        cx_s = cx_in[:2]/S1
        cx_out[0] +=  cx_s[1]*X3[0] + cx_s[0]*cx_in[2]*X3[1]
        cx_out[1] += -cx_s[0]*X3[0] + cx_s[1]*cx_in[2]*X3[1]
        cx_out[2] += -S1*X3[1]

    return cx_out


def reactionType(ZUU, jEne, reac_list):
    '''Throwing dices for the reaction occurring in a given material'''

    for reac in reac_list:
        ZUU -= CS.cst1d[reac][jEne]
        if ZUU < 0.:
            return reac
    return None


def reactionHC(MediumID, En_in, SH, SC, rnd):
    '''Throwing dices for the reaction in a C+H material'''

    if MediumID == 0:
        alpha_sh = detector['alpha_sc']*SH #XNH/XNC
    else:
        alpha_sh = detector['alpha_lg']*SH #XNHL/XNCL
    ZUU = rnd*(alpha_sh + SC) - alpha_sh
    if ZUU < 0.:
        return 'H(N,N)H'
    jEne = min(int(En_in*50.), len(CS.Egrid)-1)

    return reactionType(ZUU, jEne, ["12C(N,N)12C", "12C(N,N')12C", "12C(N,A)9BE",
        "12C(N,A)9BE'->N+3A", "12C(N,N')3A", "12C(N,P)12B", "12C(N,D)11B"])


def photo_out(elementID, En_in, zr_dl):
    '''Light yield for an arbitrary element'''

    if zr_dl < 0:
        return 0

    if elementID == 1: # H
        if En_in >= Egrid[-1]:
            photo = poly['DLT0'] + poly['DLT1']*En_in
        else:
            photo = light_int(En_in)
    elif elementID == 2: # D
        En = 0.5*En_in
        if En >= Egrid[-1]:
            photo = poly['DLT0'] + poly['DLT1']*En
        else:
            photo = light_int(En)
        photo *= 2.
    elif elementID == 3: # He
        if En_in >= poly['ENAL1']:
            photo = poly['GLT0'] + (poly['GLT1'] + poly['GLT2']*En_in)*En_in
        else:
            photo = poly['FLT1'] * En_in**poly['FLT2']
    elif elementID == 4: # Be
        photo = poly['RLT1']*En_in
    elif elementID == 5: # B, C
        photo = poly['SLT1']*En_in

    return photo


def photo_B8to2alpha(EA1, En_in, CX1, CXS, dEnucl, rnd):
    '''Light yield of B->2alpha reactions'''

    CTCM = 2.*rnd[0] - 1.
    ctheta, cthetar, enr_loc, ENE = kinema(massMeV['B8'], 0., massMeV['He'], dEnucl, CTCM, En_in)
    PHI2 = PI2*rnd[1]
    PHI3 = PHI2 + np.pi
    CX2 = scatteringDirection(CXS, ctheta , PHI2)
    CX3 = scatteringDirection(CXS, cthetar, PHI3)
    CA12 = np.dot(CX1, CX2)
    CA13 = np.dot(CX1, CX3)
    CA23 = np.dot(CX2, CX3)
    elementIndex = [3, 3, 3]
    energy = [EA1, ENE, enr_loc]
    CA0 = 0.999999

    if CA12 >= CA0:
        if energy[0] >= energy[1]:
            elementIndex[1] = 5
        else:
            elementIndex[0] = 5
    if CA13 >= CA0:
        if energy[0] >= energy[2]:
            elementIndex[2] = 5
        else:
            elementIndex[0] = 5
    if CA23 >= CA0:
        if energy[1] >= energy[2]:
            elementIndex[2] = 5
        else:
            elementIndex[1] = 5

    phot_B8to2alpha = 0.
    for j in range(3):
        phot_B8to2alpha += photo_out(elementIndex[j], energy[j], 1.)

    return phot_B8to2alpha


@nb.njit
def cylinder_crossing(Radius, H, H0, X0, CX):
    '''Calculating intersections of a straight line crossing a cylinder
z-axis = symmetry axis of the cylinder
If the source point is at the cylinder, this is not counted
as intersection point'''

    H1 = H + H0
    SQ = Radius**2 - X0[0]**2 - X0[1]**2
    C1 = CX[0]**2 + CX[1]**2

    if SQ >= 0 and X0[2] <= H1 and X0[2] >= H0: # Source point inside cylinder, 1 ray intersection
        W1 = 0.
        if C1 < 1E-12: # horizontal
            if CX[2] < 0: W2 = (H0 - X0[2])/CX[2]
            if CX[2] > 0: W2 = (H1 - X0[2])/CX[2]
        else:
            S1 = (CX[0]*X0[0] + CX[1]*X0[1])/C1
            W2 = -S1 + np.sqrt(S1**2 + SQ/C1)
            Z = X0[2] + W2*CX[2]
            if CX[2] < 0:
                if Z < H0: W2 = (H0 - X0[2])/CX[2]
            else:
                if Z > H1: W2 = (H1 - X0[2])/CX[2]
    else:  # Source point outside cylinder
        if C1 <= 0: # horizontal, CX[2] = 1 or -1
            if SQ < 0.:
                return 0., 0.
            W1 = (H0 - X0[2])/CX[2]
            if W1 < 0.:
                return 0., 0.
            W2 = (H1 - X0[2])/CX[2]
            if W2 <= W1:
                W1, W2 = W2, W1
        else:
            S1 = (CX[0]*X0[0] + CX[1]*X0[1])/C1
            SQ2 = S1**2 + SQ/C1
            if SQ2 <= 0.:
                return 0., 0.
            SQ3 = np.sqrt(SQ2)
            W10 = -S1 - SQ3
            W20 = -S1 + SQ3
            Z1 = X0[2] + W10*CX[2]
            Z2 = X0[2] + W20*CX[2]
    
            if CX[2] == 0:
                return 0., 0. # Prevent division by zero
            if Z1 <= H0 and Z2 <= H0:
                return 0., 0.
            if Z1 >= H1 and Z2 >= H1:
                return 0., 0.
# Two ray interesections
            if Z1 > H0 and Z1 < H1:
                Z2 = max(Z2, H0)
                Z2 = min(Z2, H1)
                W1 = W10
                W2 = (Z2 - X0[2])/CX[2]
            elif Z2 > H0 and Z2 < H1:
                Z1 = max(Z1, H0)
                Z1 = min(Z1, H1)
                W2 = W20
                W1 = (Z1 - X0[2])/CX[2]
            else:
                W1 = (H0 - X0[2])/CX[2]
                W2 = (H1 - X0[2])/CX[2]
                if W2 <= W1:
                    W1, W2 = W2, W1
        if W1 == W2 or W2 < 0.:
            return 0., 0.

    return W1, W2


def geom(D, RG, DSZ, RSZ, DL, RL, X0, CX):
    '''Calculating the flight path's crossing points through the three cylinders
MediaSequence    material id:  0 scintillator, 1 light pipe, 2 Al, 3 vacuum (MAT-1)
CrossPathLen     path length to a crossing point(WEG)'''

    global time_cyl, count_cyl

    tcyl1 = time.time()
    W1, W2 = cylinder_crossing(RG , D  , 0., X0, CX) # Outer cylinder
    count_cyl += 1
    tcyl2 = time.time()
    time_cyl += tcyl2 - tcyl1
    if W2 == 0.: # No intersections at all
        return None, None
    tcyl3 = time.time()
    W3, W4 = cylinder_crossing(RSZ, DSZ, DL, X0, CX) # Scintillator
    W5, W6 = cylinder_crossing(RL , DL , 0., X0, CX) # Light guide
    count_cyl += 2
    tcyl4 = time.time()
    time_cyl += tcyl4 - tcyl3

# Mapping paths and medium
    pathl = np.array([W1, W2, W3, W4, W5, W6], dtype=flt_typ)
    IndexPath = [j+2 for j, x in enumerate(pathl[2:]) if x] 
    if W1 > 0.:
        IndexPath.insert(0, 0)

# Swap last 2 pairs in case (light guide crossed before scintillator)
    crosspath = pathl[IndexPath]
    if len(crosspath) > 3:
        if crosspath[-2] <= crosspath[-4]:
            IndexPath[-4:] = IndexPath[-2:] + IndexPath[-4:-2]

# Last material is always "3" (vacuum)
    IndexPath.append(1)

# Delete 3rd-last element if its path is longer than the previous
    crosspath = pathl[IndexPath]
    if len(crosspath) > 3:
        if crosspath[-3] <= crosspath[-4]:
            del IndexPath[-3]

    if len(crosspath) > 1:
        if crosspath[-1] <= crosspath[-2]:
            del IndexPath[-1]

    return mediaCross[IndexPath], pathl[IndexPath]


@nb.njit
def kinema(M1, M2, M3, dEnucl, CTCM, T1LAB):
    '''Scattering kinematics (relativistic)'''

    M4 = M1 + M2 - M3 - dEnucl
    m3_sq = M3**2
    E1LAB = T1LAB + M1
    ARG = max(0., E1LAB**2 - M1**2)
    P1LAB = np.sqrt(ARG)
    BETA = P1LAB/(E1LAB + M2)
    GAMMA = 1./np.sqrt(1. - BETA**2)
    ECM = np.sqrt((M1 + M2)**2 + 2.*M2*T1LAB)
    E3CM = (ECM**2 + m3_sq - M4**2)/(2.*ECM)
    ARG = max(0., E3CM**2 - m3_sq)
    P3CM = np.sqrt(ARG)
    E3LAB = GAMMA*(E3CM + BETA*P3CM*CTCM)
    T3LAB = max(0., E3LAB - M3)
    T4LAB = max(0., dEnucl + T1LAB - T3LAB)
    ARG = E3LAB**2 - m3_sq
    if ARG > 0.:
        P3LAB = np.sqrt(ARG)
        ctheta = GAMMA*(P3CM*CTCM + BETA*E3CM)/P3LAB
    else:
        ctheta = 0.
    ARG = T4LAB**2 + 2.*T4LAB*M4 # E4LAB - M4**2
    if ARG > 0.:
        E4CM = ECM - E3CM
        P4CM = P3CM
        P4LAB = np.sqrt(ARG)
        cthetar = GAMMA*(-P4CM*CTCM + BETA*E4CM)/P4LAB
    else:
        cthetar = 0.

    return ctheta, cthetar, T4LAB, T3LAB


def En2light(E_phsdim):

    En_in_MeV, phs_max = E_phsdim
    nmc = int(settings.nmc)
    En_wid = settings.En_wid_frac*En_in_MeV

    time_reac1 = 0.
    time_reac2 = 0.
    time_geom = 0.
    tim = np.zeros(7)
    time_slow = np.zeros(len(tim) - 1)

    GWT_EXP = np.zeros(6, dtype=flt_typ)
    SIGM = np.zeros(4, dtype=flt_typ)
    X00  = np.zeros(3, dtype=flt_typ)
    X0   = np.zeros_like(X00)
    CX   = np.zeros_like(X00)

# Derived quantities

    rsz_sq = detector['RSZ']**2
    rg_sq  = detector['RG' ]**2
    detector['D']  = detector['DG'] + detector['DL']
    detector['RL'] = detector['RSZ']
    XNC  = detector['dens_sc']*6.023*massMeV['amu']/(detector['alpha_sc']*massMeV['H'] + massMeV['C12'])
    XNH  = detector['alpha_sc']*XNC
    XNCL = detector['dens_lg']*6.023*massMeV['amu']/(detector['alpha_lg']*massMeV['H'] + massMeV['C12'])
    XNHL = detector['alpha_lg']*XNCL
    XNAL = 0.60316

    the = np.radians(detector['theta'])
    cos_the = np.cos(the)
    sin_the = np.sin(the)
    if sin_the == 0.:
        cotan_the = 0. #0.5*np.pi
    else:
        cotan_the = cos_the/sin_the

# Geometry

    if detector['theta'] <= 0. :
        CTMAX = 0.
        distance = detector['dist'] - (detector['DG'] - 0.5*detector['DSZ'])
        if distance > 0.:
            CTMAX = 1./np.sqrt(1. + (detector['RG']/distance)**2)
    elif detector['theta'] < 90. :
        if sin_the < 0.999:
            R0 = detector['RG'] + detector['D']*sin_the/cos_the
    else:
        RR  = detector['RG']*np.sqrt(1. - (detector['RG']/detector['dist'])**2)
        ASS = detector['RG']/RR*(detector['dist'] - np.sqrt(rg_sq - RR**2))

    X00[0] = detector['dist']*sin_the
    X00[1] = 0.
    X00[2] = detector['DL'] + 0.5*detector['DSZ'] + detector['dist']*cos_the

    logger.info('START - Eneut: %8.4f MeV', En_in_MeV)

# Intialise

    n_react = len(CS.reacTotUse) + 1
    count_reac   = np.zeros(n_react     , dtype=int_typ)
    count_pp3as  = np.zeros(CS.max_level, dtype=int_typ)
    phs_dim_rea  = np.zeros(n_react     , dtype=flt_typ)
    phs_dim_pp3  = np.zeros(CS.max_level, dtype=flt_typ)
    light_output = np.zeros((n_react     , phs_max), dtype=flt_typ)
    pp3as_output = np.zeros((CS.max_level, phs_max), dtype=flt_typ)

    n_rand   = min(300000, nmc*10)
    n_rand1  = int(0.99*n_rand)
    ng_rand  = min(300000, nmc)
    ng_rand1 = int(0.99*ng_rand)
    jrand  = n_rand
    jg_rnd = ng_rand

# MC loop
    np.random.seed(0)
    for j_mc in range(nmc):
        if jrand > n_rand1:
            jrand = 0
            rand = np.random.random_sample(n_rand)
            logger.debug('Re-random flat %d %d', j_mc, nmc)
        weight = 1.

        if detector['theta'] == 0.: # Source position at 0 deg
            if CTMAX > 0.9999:
                RR0 = detector['RG']*np.sqrt(rand[jrand])
                FI0 = PI2*rand[jrand+1]
                jrand += 2
                X0[0] = RR0*np.cos(FI0)
                X0[1] = RR0*np.sin(FI0)
                X0[2] = detector['D']
                CX[0] = 0.
                CX[1] = 0.
                CX[2] = -1.
            else:
                X0[0] = 1e6
                while np.abs(X0[0]) >= detector['RG']:
                    CX[2] = -1. + (1. - CTMAX)*rand[jrand]
                    jrand += 1
                    if CX[2] <= -1.:
                        CX[2] = -1. + 1E-10
                    CX[0] = np.sqrt(1. - CX[2]**2)
                    X0[0] = -distance*CX[0]/CX[2]
                CX[1] = 0.
                X0[1] = 0.
                X0[2] = detector['D']
        elif detector['theta'] == 90.: # Source position at 90 deg, versor perp to the scintillator axis
            X0[2] = 0.
            while (X0[2] == detector['D'] or X0[2] == 0.):
                X0[1] = RR*(2*rand[jrand] - 1.)
                X0[2] = detector['D']*rand[jrand+1]
                jrand += 2
            X0[0] = np.sqrt(rg_sq - X0[1]**2)
            H1 = norm(X00 - X0)
            CX = (X0 - X00)/H1
        else:  # (only for distance > 500 cm, assuming parallel neutron beam)
            CX[0] = -sin_the
            CX[1] = 0.
            CX[2] = -cos_the
            X0[1] = detector['RG']*(2.*rand[jrand] - 1.)
            H1 = np.sqrt(rg_sq - X0[1]**2)
            X0[0] = -detector['RG'] + (R0 + detector['RG'])*rand[jrand+1]
            jrand += 2
            while (-X0[0] >= H1 or X0[0] >= H1 + R0 - detector['RG']):
                X0[0] = -detector['RG'] + (R0 + detector['RG'])*rand[jrand]
                jrand += 1
            X0[2] = detector['D']
            if X0[0] > H1:
                X0[2] = detector['D'] - (X0[0] - H1)*cotan_the
                X0[0] = H1

        n_scat = 0
        LightYieldChain = 0.
        LEVEL0 = 0
        ENE = En_in_MeV # MeV

        if settings.distr == 'gauss':
            if jg_rnd > ng_rand1:
                logger.debug('Re-random gauss')
                jg_rnd = 0
                gauss_rnd = np.random.normal(loc=En_in_MeV, scale=En_wid, size=ng_rand)
            ENE = gauss_rnd[jg_rnd]
            jg_rnd += 1


# Chain of reactions
        while(True):

# Flight path
# Input: D, RG, DSZ, RSZ, DL, RL, X0[2], CX[2]
            tg1 = time.time()
            MediaSequence, CrossPathLen = geom(detector['D'], detector['RG'], detector['DSZ'], detector['RSZ'], detector['DL'], detector['RL'], X0, CX)
            tg2 = time.time()
            time_geom += tg2 - tg1

            if MediaSequence is None:
                break

            n_cross_cyl = len(MediaSequence)
            tim[0] = time.time()
            jEne = min(int(ENE*50.), len(CS.Egrid)-1)
            SH  = CS.cst1d['H(N,N)H'][jEne] # No log, different from fortran
            SC  = CS.cst1d['CarTot'][jEne]
            SAL = CS.cst1d['AlTot' ][jEne]
            SIGM[0] = XNH*SH  + XNC*SC
            SIGM[1] = XNHL*SH + XNCL*SC
            SIGM[2] = XNAL*SAL

            tim[1] = time.time()

            SIG = 1e-4*SIGM[MediaSequence]
            RHO = rand[jrand]
            jrand += 1

            tim[2] = time.time()

            GWT_EXP[0] = CrossPathLen[0]*SIG[0]
            for I in range(1, n_cross_cyl):
                GWT_EXP[I] = GWT_EXP[I-1] + (CrossPathLen[I] - CrossPathLen[I-1])*SIG[I]

            RGWT = 1. - np.exp(-GWT_EXP)
            tim[3] = time.time()

            if n_scat <= 0:
                weight = RGWT[n_cross_cyl-1]
                RHO *= weight

            log_RHO = np.log(1. - RHO)
            MediumID = 3
            PathInMedium = 0.
            tim[4] = time.time()
            tim[5] = time.time()
            if SIG[0] > 0. and RGWT[0] >= RHO:
                PathInMedium = -log_RHO/SIG[0]
                MediumID = MediaSequence[0]
            else:
                for I in range(1, n_cross_cyl):
                    if RGWT[I] >= RHO:
                        PathInMedium = CrossPathLen[I-1] - (GWT_EXP[I-1] + log_RHO)/SIG[I]
                        MediumID = MediaSequence[I]
                        break

            if PathInMedium > CrossPathLen[n_cross_cyl-1]:
                MediumID = 3
            tim[6] = time.time()
            time_slow += np.diff(tim)

            XR = X0 + PathInMedium*CX
            zr_dl = XR[2] - detector['DL']

            if weight < 2.E-5 or MediumID == 3:
                break # Reac. chain
            n_scat += 1

# Random scattering angle

            PHI = PI2*rand[jrand]
            Frnd = rand[jrand+1]
            jrand += 2

            if MediumID in (0, 1):

#---------------------
# Random reaction type
#---------------------

                reac_type = None
                tre1 = time.time()
                while reac_type is None:
                    reac_type = reactionHC(MediumID, ENE, SH, SC, rand[jrand])
                    jrand += 1
                tre2 = time.time()
                time_reac1 += tre2 - tre1

                if n_scat == 1: # Label first neutron reaction
                    if MediumID == 0:
                        first_reac_type = CS.reacTotUse.index(reac_type)
                    elif MediumID == 1:
                        first_reac_type = n_react - 1 # 1st reaction (no matter which) in light guide

#-----------
# Kinematics
#-----------

                if reac_type == 'H(N,N)H':
                    CTCM = 2.*Frnd - 1.
                    if ENE > 2.:   # Angular distribution
                        AAA = CS.cst1d['HE1'][jEne]
                        BBB = CS.cst1d['HE2'][jEne]
                        CTCM1 = (CTCM + AAA)/(1. - BBB + AAA*CTCM  + BBB*CTCM **2)
                        CTCM  = (CTCM + AAA)/(1. - BBB + AAA*CTCM1 + BBB*CTCM1**2)
                    dEnucl = CS.crSec_d[reac_type]['dEnucl']
                    ctheta, cthetar, ENR, ENE = kinema(massMeV['neutron'], massMeV['H'], massMeV['neutron'], dEnucl, CTCM, ENE)
                    if zr_dl >= 0.:
                        if ENR <= 0.2: 
                            BR = 1.507E-3*ENR
                        else:
                            BR = 2.0457E-3*(ENR + 0.15045)**1.8194
                        LightYieldChain += photo_out(1, ENR, zr_dl)
                        rsz_sq_xr = rsz_sq - XR[0]**2 - XR[1]**2
                        if zr_dl <= BR or detector['DSZ'] - zr_dl <= BR or rsz_sq_xr <= 2.*detector['RSZ']*BR:
                            PHIR = PHI - np.pi
                            CXR = scatteringDirection(CX, cthetar, PHIR)
                            if CXR[2] < 0:
                                WMZ = -zr_dl/CXR[2]
                            elif CXR[2] > 0:
                                WMZ = (detector['DSZ'] - zr_dl)/CXR[2]
                            if np.abs(CXR[2]) > 0.9999:
                                PATHM = WMZ
                            else:
                                cxr_fac = 1./(1. - CXR[2]**2)
                                WR = (XR[0]*CXR[0] + XR[1]*CXR[1])  * cxr_fac
                                WM = rsz_sq_xr * cxr_fac
                                if WM <= 0.:
                                    WMR = -WR + np.abs(WR)
                                else:
                                    WMR = -WR + np.sqrt(WM + WR**2)
                                if WMR < WMZ or CXR[2] == 0.:
                                    PATHM = WMR
                                else:
                                    PATHM = WMZ
                            PATH = BR - PATHM
                            if PATH > 0.:
                                if PATH > 3.104E-4:
                                    ENT = -0.150 + (PATH*488.83)**0.5496
                                else:
                                    ENT = 663.57*PATH
                                LightYieldChain -= photo_out(1, ENT, zr_dl)

                elif reac_type in ('12C(N,N)12C', "12C(N,N')12C"):
                    CTCM = CS.cosInterpReac2d(reac_type, ENE, Frnd) # elastic
                    dEnucl = CS.crSec_d[reac_type]['dEnucl']
                    ctheta, cthetar, ENR, ENE = kinema(massMeV['neutron'], massMeV['C12'], massMeV['neutron'], dEnucl, CTCM, ENE)
                    LightYieldChain += photo_out(5, ENR, zr_dl)

                elif reac_type == '12C(N,A)9BE':
                    if zr_dl > 0.:
                        CTCM = CS.cosInterpReac2d(reac_type, ENE, Frnd)
                        dEnucl = CS.crSec_d[reac_type]['dEnucl']
                        ctheta, cthetar, ENR, ENE = kinema(massMeV['neutron'], massMeV['C12'], massMeV['He'], dEnucl, CTCM, ENE)
                        LightYieldChain += photo_out(3, ENE, zr_dl) + photo_out(4, ENR, zr_dl)
                    break # reac. chain

                elif reac_type =="12C(N,A)9BE'->N+3A":
                    CTCM = 2.*Frnd - 1.
                    dEnucl = CS.crSec_d[reac_type]['dEnucl']
                    ctheta, cthetar, ENR, EA1 = kinema(massMeV['neutron'], massMeV['C12'], massMeV['He'], dEnucl, CTCM, ENE)
                    CX1 = scatteringDirection(CX, ctheta, PHI)
                    PHI += np.pi
                    CX = scatteringDirection(CX, cthetar, PHI)
                    CTCM = 2.*rand[jrand] - 1.
                    jrand += 1
                    dEnucl = 0.761
                    ctheta, cthetar, ENR, ENE = kinema(massMeV['B9'], 0., massMeV['neutron'], dEnucl, CTCM, ENR)
                    if zr_dl >= 0.:
                        PHI = PI2*rand[jrand] + np.pi
                        jrand += 1
                        CXS = scatteringDirection(CX, cthetar, PHI)
                        dEnucl = 0.095
                        if n_scat == 1:
                            LEVEL0 = 10
                        LightYieldChain += photo_B8to2alpha(EA1, ENR, CX1, CXS, dEnucl, rand[jrand:jrand+2])
                        jrand += 2

                elif reac_type == "12C(N,N')3A":
                    LEVEL = 0
                    if ENE >= 10.:
                        NRA = rand[jrand]
                        jrand += 1
                        NL = 0.
                        tmp = CS.int_alphas3(ENE)
                        for LEVEL in range(CS.max_level):
                            NL += tmp[LEVEL]
                            if NL >= NRA:
                                break
                    dEnucl = CS.alphas3['q3a'][LEVEL]
                    if LEVEL == 1:
                        CTCM = CS.cosInterpReac2d("12C(N,N')12C*", ENE, Frnd)
                    else:
                        CTCM = 2.*Frnd - 1.
                    ctheta, cthetar, ENR, ENE = kinema(massMeV['neutron'], massMeV['C12'], massMeV['neutron'], dEnucl, CTCM, ENE)
                    if zr_dl >= 0.:
                        PHI += np.pi
                        CXR = scatteringDirection(CX, cthetar, PHI)
                        CTCM = 2.*rand[jrand] - 1.
                        PHI1 = PI2*rand[jrand+1]
                        jrand += 2
                        if n_scat == 1:
                            LEVEL0 = LEVEL
                        dEnucl = -dEnucl - 7.369
                        if LEVEL > 1:
                            LEX = rand[jrand]
                            jrand += 1
                            if LEX <= CS.alphas3['3MeV'][LEVEL]:
                                dEnucl -= 3.
                        ctheta1, cthetar, ENR, EA1 = kinema(massMeV['C12'], 0., massMeV['He'], dEnucl, CTCM, ENR)
                        CX1 = scatteringDirection(CXR, ctheta1, PHI1)
                        PHIR = PHI1 + np.pi
                        CXS = scatteringDirection(CXR, cthetar, PHIR)
                        dEnucl = 0.095
                        if LEVEL > 1 and LEX <= CS.alphas3['3MeV'][LEVEL]:
                            dEnucl += 3.
                        LightYieldChain += photo_B8to2alpha(EA1, ENR, CX1, CXS, dEnucl, rand[jrand: jrand+2])
                        jrand += 2

                elif reac_type == '12C(N,P)12B':
                    if zr_dl > 0.:
                        CTCM = 2.*Frnd - 1.
                        dEnucl = CS.crSec_d[reac_type]['dEnucl']
                        ctheta, cthetar, ENR, ENE = kinema(massMeV['neutron'], massMeV['C12'], massMeV['H'], dEnucl, CTCM, ENE)
                        LightYieldChain += photo_out(1, ENE, zr_dl) + photo_out(5, ENR, zr_dl)
                    break # reac_chain

                elif reac_type == '12C(N,D)11B':
                    if zr_dl > 0.:
                        CTCM = 2.*Frnd - 1.
                        dEnucl = CS.crSec_d[reac_type]['dEnucl']
                        ctheta, cthetar, ENR, ENE = kinema(massMeV['neutron'], massMeV['C12'], massMeV['D'], dEnucl, CTCM, ENE)
                        LightYieldChain += photo_out(2, ENE, zr_dl) + photo_out(5, ENR, zr_dl)
                    break # reac_chain

# Reaction in aluminium cage
            elif MediumID == 2:
                tre3 = time.time()
                ZUU = rand[jrand]*SAL
                jrand += 1
                reac_type = reactionType(ZUU, jEne, ["27AL(N,N)27AL", "27AL(N,N')27AL'"])
                tre4 = time.time()
                time_reac2 += tre4 - tre3
                if reac_type is None:
                    break #reac_chain
                if n_scat <= 1:
                    first_reac_type = CS.reacTotUse.index(reac_type)

                if reac_type == '27AL(N,N)27AL':
                    CTCM = CS.cosInterpReac2d(reac_type, ENE, Frnd)
                    dEnucl = CS.crSec_d[reac_type]['dEnucl']
                elif reac_type == "27Al(N, N')27Al":
                    CTCM = 2.*Frnd - 1.
                    dEnucl = -rand[jrand]*(ENE - 0.5) - 0.5
                    jrand += 1
                ctheta, cthetar, ENR, ENE = kinema(massMeV['neutron'], massMeV['Al'], massMeV['neutron'], dEnucl, CTCM, ENE)

            if ENE <= 0.01:
                break #reac_chain
            CX = scatteringDirection(CX, ctheta, PHI)
            X0 = XR

# End reaction chain

        if weight >= 2E-5 and LightYieldChain > 0.:
            phsBin = int(LightYieldChain/settings.Ebin_MeVee)
            light_output[first_reac_type, phsBin] += weight
            count_reac  [first_reac_type] += 1 # count
            phs_dim_rea [first_reac_type] = max(phsBin, phs_dim_rea[first_reac_type])
            if first_reac_type == 5:
                count_pp3as [LEVEL0] += 1
                phs_dim_pp3 [LEVEL0] = max(phsBin, phs_dim_pp3[LEVEL0])
                pp3as_output[LEVEL0, phsBin] += weight

# End Energy MC loop (incoming neutron)

# Unnormalise arrays w.r.t. NMC and viewing solid angle

    norm_mc_F0 = (np.pi*rg_sq*cos_the + 2.*detector['D']*detector['RG']*sin_the)/float(nmc)
    light_output *= norm_mc_F0
    pp3as_output *= norm_mc_F0

    logger.debug('Time analysis:')
    logger.debug('geom %8.4f', time_geom)
    logger.debug('cyl_cross %8.4f %d', time_cyl , count_cyl)
    logger.debug('reac1 %f8.4', time_reac1)
    logger.debug('reac2 %f8.4', time_reac2)
    logger.debug('Bottle necks in energy chain, En=%8.4f MeV' %En_in_MeV)
    for ts in time_slow:
        logger.debug(ts)
    logger.debug(np.sum(time_slow))

    return count_reac, count_pp3as, phs_dim_rea, phs_dim_pp3, light_output, pp3as_output
