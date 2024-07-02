from numba import njit, types, prange
from numba.typed import Dict

import numpy as np
import numpy.ma as ma

from sharptab import interp
from sharptab import thermo
from sharptab import profile
from sharptab.params import DefineParcel, DefineParcel_2, cape1, cape
from calc import derived

from .constants import *

#########################################################################################
# Ahead-of-time compilation of the parcelx function. Need to ensure CC environment 
# variable is the same as the one used to compile the numpy.disutils package. 
# Running this with python -m sharptab.params will produce a file called parcelx_module
# within the sharptab folder which will be read in during NSE execution.
from numba.pycc import CC
cc = CC('aot_module')
#########################################################################################

@cc.export('parcelx', 'DictType(unicode_type, f8)(f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],i4)')
def parcelx(pres, tmpc, dwpc, wspd, wdir, hght, flag):
    """
    !! Re-written July 2024 to take in vectors as opposed to a profile object to allow for 
    ahead-of-time compilation with numba !!

    Lifts the specified parcel, calculates various levels and parameters from the
    profile object. B+/B- are calculated based on the specified layer. Such parameters
    include CAPE, CIN, LCL height, LFC height, buoyancy minimum. EL height, MPL height.

    !! All calculations use the virtual temperature correction unless noted. !!

    Parameters
    ----------
    prof : profile object
        Profile Object
    flag : number (optional; default = 5)
        Flag to determine what kind of parcel to create; See DefineParcel for
        flag values

    Returns
    -------
        Parcel Object
    """
    dp = -1
    pcl = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    pcl['pres'] = -99  # Parcel beginning pressure (mb)
    pcl['tmpc'] = -99.0  # Parcel beginning temperature (C)
    pcl['dwpc'] = -99.0  # Parcel beginning dewpoint (C)
    pcl['ptrace'] = -99.0  # Parcel trace pressure (mb)
    pcl['ttrace'] = -99.0  # Parcel trace temperature (C)
    pcl['blayer'] = (
        -99.0
    )  # Pressure of the bottom of the layer the parcel is lifted (mb)
    pcl['tlayer'] = (
        -99.0
    )  # Pressure of the top of the layer the parcel is lifted (mb)
    pcl['entrain'] = 0.0  # A parcel entrainment setting (not yet implemented)
    pcl['lclpres'] = -99.0  # Parcel LCL (lifted condensation level) pressure (mb)
    pcl['lclhght'] = -99.0  # Parcel LCL height (m AGL)
    pcl['lfcpres'] = -99.0  # Parcel LFC (level of free convection) pressure (mb)
    pcl['lfchght'] = -99.0  # Parcel LFC height (m AGL)
    pcl['elpres'] = -99.0  # Parcel EL (equilibrium level) pressure (mb)
    pcl['elhght'] = -99.0  # Parcel EL height (m AGL)
    pcl['mplpres'] = -99.0  # Maximum Parcel Level (mb)
    pcl['mplhght'] = -99.0  # Maximum Parcel Level (m AGL)
    pcl['bplus'] = -99.0  # Parcel CAPE (J/kg)
    pcl['bminus'] = -99.0  # Parcel CIN (J/kg)
    pcl['pbot'] = -99.0
    pcl['ptop'] = -99.0
    pcl['lplhght'] = -99.0  # Lifted Parcel Height (m AGL)
    pcl['b3km'] = -99.0 # Parcel CAPE up to 3 km (J/kg)

    prof = profile.create_profile(pres, tmpc, dwpc, wspd, wdir, hght)

    LPLVALS = DefineParcel(prof, flag)

    # Variables
    pres = LPLVALS.pres
    tmpc = LPLVALS.tmpc
    dwpc = LPLVALS.dwpc
    # pcl.lplhght = interp.to_agl(prof, interp.hght(prof, pres))
    pcl['lplhght'] = interp.hght(prof, pres)
    pcl['pres'] = pres
    pcl['tmpc'] = tmpc
    pcl['dwpc'] = dwpc

    # cap_strength = -9999.
    # cap_strengthpres = -9999.
    # li_max = -9999.
    # li_maxpres = -9999.
    totp = 0.0
    totn = 0.0
    tote = 0.0
    cinh_old = 0.0

    # See if default layer is specified

    pbot = prof.pres[prof.sfc]
    pcl['blayer'] = pbot
    pcl['pbot'] = pbot

    ptop = prof.pres[prof.pres.shape[0] - 1]
    pcl['tlayer'] = ptop
    pcl['ptop'] = ptop

    # Make sure this is a valid layer
    if pbot > pres:
        pbot = pres
        pcl['blayer'] = pbot

    # if type(interp.vtmp(prof, pbot)) == type(ma.masked) or type(interp.vtmp(prof, ptop)) == type(ma.masked):
    #    return pcl

    # Begin with the Mixing Layer
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    tp1 = thermo.virtemp(pres, tmpc, dwpc)
    ttrace = [tp1]
    ptrace = [pe1]

    # Lift parcel and return LCL pres (hPa) and LCL temp (C)
    pe2, tp2 = thermo.drylift(pres, tmpc, dwpc)

    # CALCS SAME TO HERE...

    # if type(pe2) == type(ma.masked) or np.isnan(pe2):
    #    return pcl
    blupper = pe2
    h2 = interp.hght(prof, pe2)
    te2 = interp.vtmp(prof, pe2)

    pcl['lclpres'] = min(pe2, prof.pres[prof.sfc])  # Make sure the LCL pressure is
    # never below the surface
    pcl['lclhght'] = interp.to_agl(prof, h2)

    ptrace.append(pe2)
    ttrace.append(thermo.virtemp(pe2, tp2, tp2))

    # Calculate lifted parcel theta for use in iterative CINH loop below
    # RECALL: lifted parcel theta is CONSTANT from LPL to LCL
    theta_parcel = thermo.theta(pe2, tp2, 1000.0)

    # Environmental theta and mixing ratio at LPL
    bltheta = thermo.theta(pres, interp.temp(prof, pres), 1000.0)
    blmr = thermo.mixratio(pres, dwpc)

    # ACCUMULATED CINH IN THE MIXING LAYER BELOW THE LCL
    # This will be done in 'dp' increments and will use the virtual
    # temperature correction where possible
    pp = np.arange(pbot, blupper + dp, dp, dtype=type(pbot))
    hh = interp.hght(prof, pp)
    tmp_env_theta = thermo.theta(pp, interp.temp(prof, pp), 1000.0)
    tmp_env_dwpt = interp.dwpt(prof, pp)
    tv_env = thermo.virtemp(pp, tmp_env_theta, tmp_env_dwpt)
    tmp1 = thermo.virtemp(pp, theta_parcel, thermo.temp_at_mixrat(blmr, pp))
    tdef = (tmp1 - tv_env) / thermo.ctok(tv_env)
    # tdef = np.divide((tmp1 - tv_env), thermo.ctok(tv_env))

    tidx1 = np.arange(0, len(tdef) - 1, 1)
    tidx2 = np.arange(1, len(tdef), 1)
    lyre = G * (tdef[tidx1] + tdef[tidx2]) / 2 * (hh[tidx2] - hh[tidx1])
    # lyre = np.divide(G * (tdef[tidx1]+tdef[tidx2]),  2 * (hh[tidx2]-hh[tidx1]))
    totn = lyre[lyre < 0].sum()
    if not totn:
        totn = 0.0

    # Move the bottom layer to the top of the boundary layer
    if pbot > pe2:
        pbot = pe2
        pcl['blayer'] = pbot

    # Calculate height of various temperature levels
    #p0c = temp_lvl(prof, 0.)

    #pm10c = temp_lvl(prof, -10.)
    #pm20c = temp_lvl(prof, -20.)
    #pm30c = temp_lvl(prof, -30.)
    #hgt0c = interp.hght(prof, p0c)

    #hgtm10c = interp.hght(prof, pm10c)
    #hgtm20c = interp.hght(prof, pm20c)
    #hgtm30c = interp.hght(prof, pm30c)
    #pcl.p0c = p0c
    #pcl.pm10c = pm10c
    #pcl.pm20c = pm20c
    #pcl.pm30c = pm30c
    #pcl.hght0c = hgt0c

    #pcl.hghtm10c = hgtm10c
    #pcl.hghtm20c = hgtm20c
    #pcl.hghtm30c = hgtm30c

    if pbot < prof.pres[-1]:
        # Check for the case where the LCL is above the
        # upper boundary of the data (e.g. a dropsonde)
        return pcl

    # Find lowest observation in layer
    lptr = np.where(pbot >= prof.pres)[0].min()
    uptr = np.where(ptop <= prof.pres)[0].max()

    # START WITH INTERPOLATED BOTTOM LAYER
    # Begin moist ascent from lifted parcel LCL (pe2, tp2)
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    te1 = interp.vtmp(prof, pe1)

    tp1 = thermo.wetlift3(pe2, tp2, pe1)

    lyre = 0
    lyrlast = 0

    iter_ranges = np.arange(lptr, prof.pres.shape[0])
    ttraces = np.zeros(len(iter_ranges))
    ptraces = np.zeros(len(iter_ranges))

    for i in iter_ranges:
        pe2 = prof.pres[i]
        h2 = prof.hght[i]
        te2 = prof.vtmp[i]
        tp2 = thermo.wetlift3(pe1, tp1, pe2)
        tdef1 = (thermo.virtemp(pe1, tp1, tp1) - te1) / thermo.ctok(te1)
        # tdef1 = np.divide((thermo.virtemp(pe1, tp1, tp1) - te1), thermo.ctok(te1))
        tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2)
        # tdef2 = np.divide((thermo.virtemp(pe2, tp2, tp2) - te2), thermo.ctok(te2))

        ptraces[i - iter_ranges[0]] = pe2
        ttraces[i - iter_ranges[0]] = thermo.virtemp(pe2, tp2, tp2)
        lyrlast = lyre
        lyre = G * (tdef1 + tdef2) / 2.0 * (h2 - h1)
        # lyre = np.divide(G * (tdef1 + tdef2), 2. * (h2 - h1))

        # print(pe1, pe2, te1, te2, tp1, tp2, lyre, totp, totn)

        # Add layer energy to total positive if lyre > 0
        if lyre > 0:
            totp += lyre
        # Add layer energy to total negative if lyre < 0, only up to EL
        else:
            if pe2 > 500.0:
                totn += lyre

        # Check for Max LI
        # mli = thermo.virtemp(pe2, tp2, tp2) - te2
        # if  mli > li_max:
        #    li_max = mli
        #    li_maxpres = pe2

        # Check for Max Cap Strength
        # mcap = te2 - mli
        # if mcap > cap_strength:
        #    cap_strength = mcap
        #    cap_strengthpres = pe2

        tote += lyre
        pelast = pe1
        pe1 = pe2
        te1 = te2
        tp1 = tp2

        # Is this the top of the specified layer
        if i >= uptr:
            pe3 = pe1
            h3 = h2
            te3 = te1
            tp3 = tp1
            lyrf = lyre
            if lyrf > 0:
                pcl['bplus'] = totp - lyrf
                pcl['bminus'] = totn
            else:
                pcl['bplus'] = totp
                if pe2 > 500.0:
                    pcl['bminus'] = totn + lyrf
                else:
                    pcl['bminus'] = totn
            pe2 = ptop
            h2 = interp.hght(prof, pe2)
            te2 = interp.vtmp(prof, pe2)
            tp2 = thermo.wetlift3(pe3, tp3, pe2)
            tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / thermo.ctok(te3)
            tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2)
            lyrf = G * (tdef3 + tdef2) / 2.0 * (h2 - h3)

            # tdef3 = np.divide((thermo.virtemp(pe3, tp3, tp3) - te3), thermo.ctok(te3))
            # tdef2 = np.divide((thermo.virtemp(pe2, tp2, tp2) - te2), thermo.ctok(te2))
            # lyrf = np.divide(G * (tdef3 + tdef2), 2. * (h2 - h3))

            if lyrf > 0:
                pcl['bplus'] += lyrf
            else:
                if pe2 > 500.0:
                    pcl['bminus'] += lyrf
            if pcl['bplus'] == 0:
                pcl['bminus'] = 0.0

        # Is this the freezing level
        if te2 < 0.0:
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift3(pe1, tp1, pe3)
            lyrf = lyre
            # if lyrf > 0.: pcl.bfzl = totp - lyrf
            # else: pcl.bfzl = totp
            # if p0c > pe3:
            #    pcl.bfzl = 0

            # elif utils.QC(pe2):
            #    te2 = interp.vtmp(prof, pe2)
            #    tp2 = thermo.wetlift(pe3, tp3, pe2)
            #    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
            #        thermo.ctok(te3)
            #    tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
            #        thermo.ctok(te2)
            #    lyrf = G * (tdef3 + tdef2) / 2. * (hgt0c - h3)
            #    if lyrf > 0: pcl.bfzl += lyrf
        # Is this the -10C level
        if te2 < -10.0:
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift3(pe1, tp1, pe3)
            lyrf = lyre
            # if lyrf > 0.: pcl.wm10c = totp - lyrf
            # else: pcl.wm10c = totp
            # if not utils.QC(pm10c) or pm10c > pcl.lclpres:
            #    pcl.wm10c = 0
            # elif utils.QC(pe2):
            #    te2 = interp.vtmp(prof, pe2)
            #    tp2 = thermo.wetlift(pe3, tp3, pe2)
            #    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
            #        thermo.ctok(te3)
            #    tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
            #        thermo.ctok(te2)
            #    lyrf = G * (tdef3 + tdef2) / 2. * (hgtm10c - h3)
            #    if lyrf > 0: pcl.wm10c += lyrf

        # Is this the -20C level
        if te2 < -20.0:
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift3(pe1, tp1, pe3)
            lyrf = lyre
            # if lyrf > 0.: pcl.wm20c = totp - lyrf
            # else: pcl.wm20c = totp
            # if not utils.QC(pm20c) or pm20c > pcl.lclpres:
            #    pcl.wm20c = 0
            # elif utils.QC(pe2):
            #    te2 = interp.vtmp(prof, pe2)
            #    tp2 = thermo.wetlift(pe3, tp3, pe2)
            #    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
            #        thermo.ctok(te3)
            #    tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
            #        thermo.ctok(te2)
            #    lyrf = G * (tdef3 + tdef2) / 2. * (hgtm20c - h3)
            #    if lyrf > 0: pcl.wm20c += lyrf

        # Is this the -30C level
        if te2 < -30.0:
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift3(pe1, tp1, pe3)
            lyrf = lyre
            # if lyrf > 0.: pcl.wm30c = totp - lyrf
            # else: pcl.wm30c = totp
            # if not utils.QC(pm30c) or pm30c > pcl.lclpres:
            #    pcl.wm30c = 0
            # elif utils.QC(pe2):
            #    te2 = interp.vtmp(prof, pe2)
            #    tp2 = thermo.wetlift(pe3, tp3, pe2)
            #    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
            #        thermo.ctok(te3)
            #    tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
            #        thermo.ctok(te2)
            #    lyrf = G * (tdef3 + tdef2) / 2. * (hgtm30c - h3)
            #    if lyrf > 0: pcl.wm30c += lyrf

        # Is this the 3km level
        if pcl['lclhght'] < 3000.0:
            if interp.to_agl(prof, h1) <= 3000.0 and interp.to_agl(prof, h2) >= 3000.0:
                pe3 = pelast
                h3 = interp.hght(prof, pe3)
                te3 = interp.vtmp(prof, pe3)
                tp3 = thermo.wetlift3(pe1, tp1, pe3)
                lyrf = lyre
                if lyrf > 0:
                    pcl['b3km'] = totp - lyrf
                else: pcl['b3km'] = totp
                h4 = interp.to_msl(prof, 3000.)
                pe4 = interp.pres(prof, h4)
                # if utils.QC(pe2):
                te2 = interp.vtmp(prof, pe4)
                tp2 = thermo.wetlift3(pe3, tp3, pe4)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                         thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe4, tp2, tp2) - te2) / \
                         thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (h4 - h3)
                if lyrf > 0: pcl['b3km'] += lyrf
        else: pcl['b3km'] = 0.
        # Is this the 6km level
        if pcl['lclhght'] < 6000.0:
            if interp.to_agl(prof, h1) <= 6000.0 and interp.to_agl(prof, h2) >= 6000.0:
                pe3 = pelast
                h3 = interp.hght(prof, pe3)
                te3 = interp.vtmp(prof, pe3)
                tp3 = thermo.wetlift3(pe1, tp1, pe3)
                lyrf = lyre
                # if lyrf > 0: pcl.b6km = totp - lyrf
                # else: pcl.b6km = totp
                # h4 = interp.to_msl(prof, 6000.)
                # pe4 = interp.pres(prof, h4)
                # if utils.QC(pe2):
                #    te2 = interp.vtmp(prof, pe4)
                #    tp2 = thermo.wetlift(pe3, tp3, pe4)
                #    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                #        thermo.ctok(te3)
                #    tdef2 = (thermo.virtemp(pe4, tp2, tp2) - te2) / \
                #        thermo.ctok(te2)
                #    lyrf = G * (tdef3 + tdef2) / 2. * (h4 - h3)
                #    if lyrf > 0: pcl.b6km += lyrf
        # else: pcl.b6km = 0.

        h1 = h2

        # LFC Possibility
        if lyre >= 0.0 and lyrlast <= 0.0:
            tp3 = tp1
            # te3 = te1
            pe2 = pe1
            pe3 = pelast
            if interp.vtmp(prof, pe3) < thermo.virtemp(
                pe3, thermo.wetlift3(pe2, tp3, pe3), thermo.wetlift3(pe2, tp3, pe3)
            ):
                # Found an LFC, store height/pres and reset EL/MPL
                pcl['lfcpres'] = pe3
                pcl['lfchght'] = interp.to_agl(prof, interp.hght(prof, pe3))
                pcl['elpres'] = -999.0
                pcl['elhght'] = -999.0
                pcl['mplpres'] = -999.0

            else:

                ###############################################################
                # This minimization block (while loop) was causing
                # ZeroDivisionErrors
                ###############################################################
                # while interp.vtmp(prof, pe3) > thermo.virtemp(pe3, thermo.wetlift3(pe2, tp3, pe3),
                #                                              thermo.wetlift3(pe2, tp3, pe3)) and pe3 > 0:
                while (
                    interp.vtmp(prof, pe3)
                    > thermo.virtemp(
                        pe3,
                        thermo.wetlift3(pe2, tp3, pe3),
                        thermo.wetlift3(pe2, tp3, pe3),
                    )
                    and pe3 > 5
                ):
                    pe3 -= 5
                if pe3 > 0:
                    # Found a LFC, store height/pres and reset EL/MPL
                    pcl['lfcpres'] = pe3
                    pcl['lfchght'] = interp.to_agl(prof, interp.hght(prof, pe3))
                    cinh_old = totn
                    tote = 0.0
                    li_max = -9999.0
                    # if cap_strength < 0.: cap_strength = 0.
                    # pcl.cap = cap_strength
                    # pcl.cappres = cap_strengthpres

                    pcl['elpres'] = -999.0
                    pcl['elhght'] = -999.0
                    pcl['mplpres'] = -999.0

            # Hack to force LFC to be at least at the LCL
            if pcl['lfcpres'] >= pcl['lclpres']:
                pcl['lfcpres'] = pcl['lclpres']
                pcl['lfchght'] = pcl['lclhght']

        # EL Possibility
        if lyre <= 0.0 and lyrlast >= 0.0:
            tp3 = tp1
            # te3 = te1
            pe2 = pe1
            pe3 = pelast
            while interp.vtmp(prof, pe3) < thermo.virtemp(
                pe3, thermo.wetlift3(pe2, tp3, pe3), thermo.wetlift3(pe2, tp3, pe3)
            ):
                pe3 -= 5
            pcl['elpres']= pe3
            pcl['elhght'] = interp.to_agl(prof, interp.hght(prof, pcl['elpres']))
            # pcl.mplpres = ma.masked
            # pcl.limax = -li_max
            # pcl.limaxpres = li_maxpres
        """
        # MPL Possibility
        if tote < 0.:
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift3(pe1, tp1, pe3)
            totx = tote - lyre
            pe2 = pelast
            while totx > 0:
                pe2 -= 1
                te2 = interp.vtmp(prof, pe2)
                tp2 = thermo.wetlift3(pe3, tp3, pe2)
                h2 = interp.hght(prof, pe2)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                    thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
                    thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (h2 - h3)
                totx += lyrf
                tp3 = tp2
                te3 = te2
                pe3 = pe2
            pcl.mplpres = pe2
            pcl.mplhght = interp.to_agl(prof, interp.hght(prof, pe2))

        # 500 hPa Lifted Index
        if prof.pres[i] <= 500. and not utils.QC(pcl.li5):
            a = interp.vtmp(prof, 500.)
            b = thermo.wetlift(pe1, tp1, 500.)
            pcl.li5 = a - thermo.virtemp(500, b, b)

        # 300 hPa Lifted Index
        if prof.pres[i] <= 300. and not utils.QC(pcl.li3):
            a = interp.vtmp(prof, 300.)
            b = thermo.wetlift(pe1, tp1, 300.)
            pcl.li3 = a - thermo.virtemp(300, b, b)

#    pcl.bminus = cinh_old
    """
    pcl['bplus'] = totp
    """
    # Calculate BRN if available
    bulk_rich(prof, pcl)

    # Save params
    if np.floor(pcl.bplus) == 0: pcl.bminus = 0.
    pcl.ptrace = ma.concatenate((ptrace, ptraces))
    pcl.ttrace = ma.concatenate((ttrace, ttraces))

    # Find minimum buoyancy from Trier et al. 2014, Part 1
    idx = np.ma.where(pcl.ptrace >= 500.)[0]
    if len(idx) != 0:
        b = pcl.ttrace[idx] - interp.vtmp(prof, pcl.ptrace[idx])
        idx2 = np.ma.argmin(b)
        pcl.bmin = b[idx2]
        pcl.bminpres = pcl.ptrace[idx][idx2]
    """
    return pcl


@cc.export('effective_inflow_layer', '(f8[:])(f8[:],f8[:],f8[:],f8[:],f8[:],f8[:])')
def effective_inflow_layer(pres, tmpc, dwpc, wspd, wdir, hght):
    """NUMBA DOES NOT SUPPORT RECURSIVE CLASS CALLS!

    Calculate the top and bottom of the effective inflow layer based on research by [3]_.

    Parameters
    ----------
    prof : profile object
        Profile object
    ecape : number (optional; default=100)
        Minimum amount of CAPE in the layer to be considered part of the
        effective inflow layer.
    echine : number (optional; default=250)
        Maximum amount of CINH in the layer to be considered part of the
        effective inflow layer
    mupcl : parcel object
        Most Unstable Layer parcel

    Returns
    -------
    pbot : number
        Pressure at the bottom of the layer (hPa)
    ptop : number
        Pressure at the top of the layer (hPa)
    """
    ecape = 100.0
    ecinh = -250.0

    prof = profile.create_profile(pres, tmpc, dwpc, wspd, wdir, hght)
    mulplvals = DefineParcel_2(prof, 3)
    mupcl = cape1(prof, mulplvals)

    mucape = mupcl.bplus
    mucinh = mupcl.bminus
    pbot = -999
    ptop = -999
    if mucape != 0:
        if mucape >= ecape and mucinh > ecinh:
            # Begin at surface and search upward for effective surface
            for i in range(prof.sfc, prof.top):
                pcl = cape(prof, prof.pres[i], prof.tmpc[i], prof.dwpc[i])

                if pcl.bplus >= ecape and pcl.bminus > ecinh:
                    pbot = prof.pres[i]
                    break
            bptr = i
            # Keep searching upward for the effective top
            for i in range(bptr + 1, prof.top):
                if not prof.dwpc[i] or not prof.tmpc[i]:
                    continue
                pcl = cape(prof, prof.pres[i], prof.tmpc[i], prof.dwpc[i])

                if (
                    pcl.bplus < ecape or pcl.bminus <= ecinh
                ):  # Is this a potential "top"?
                    j = 1
                    if (prof.dwpc[i - j] < -999) and (prof.tmpc[i - j] < -999.0):
                        j += 1
                    ptop = prof.pres[i - j]
                    if ptop > pbot:
                        ptop = pbot
                    break
    ret = np.zeros((2))
    ret[0] = pbot
    ret[1] = ptop
    return ret


float_array = types.float64[:,:] 
@njit(parallel=True)
@cc.export('fast_loop', 'DictType(unicode_type,types.float64[:,:])(f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:],f8[:,:,:])')
def fast_loop(pres, tmpc, dwpc, wspd, wdir, hght, vort, incoming):
    shape = pres.shape
    d = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )

    ####################################################################################
    # In order for this all to work within the ahead-of-time-compiled numba function,
    # need to explicitly and individually declare arrays. Trying to figure out how to
    # pass List(SCALAR_PARAMS.keys() and List(VECTOR_PARAMS.keys()) from config file.
    # Scalars
    d['eff_inflow_base'] = incoming[0]
    d['eff_inflow_top'] = incoming[1]
    d['mlcape'] = incoming[2]
    d['mlcin'] = incoming[3]
    d['mucape'] = incoming[4]
    d['cape3km'] = incoming[5]
    d['mllcl'] = incoming[6]
    d['elhght'] = incoming[7]
    d['esrh'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['srh500'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['srh01km'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['lr03km'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['estp'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['nst'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['deviance'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['snsq'] = np.zeros((shape[1], shape[2]), dtype='float64')

    # Vectors
    d['ebwd_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['ebwd_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr1_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr1_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr3_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr3_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr6_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr6_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr8_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['shr8_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['rm5_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['rm5_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['lm5_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['lm5_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['devtor_u'] = np.zeros((shape[1], shape[2]), dtype='float64')
    d['devtor_v'] = np.zeros((shape[1], shape[2]), dtype='float64')
    
    for j in prange(shape[1]):
        for i in prange(shape[2]):
            prof = profile.create_profile(pres=pres[:,j,i], tmpc=tmpc[:,j,i],
                                          hght=hght[:,j,i], dwpc=dwpc[:,j,i],
                                          wspd=wspd[:,j,i], wdir=wdir[:,j,i])
            
            eff_inflow = d['eff_inflow_base'][j,i], d['eff_inflow_top'][j,i]
            d['esrh'][j,i] = derived.srh(prof, effective_inflow_layer=eff_inflow)
            d['srh500'][j,i] = derived.srh(prof, lower=0, upper=500)
            d['srh01km'][j,i] = derived.srh(prof, lower=0, upper=1000)
            d['lr03km'][j,i] = derived.lapse_rate(prof, lower=0, upper=3000)
            d['snsq'][j,i] = derived.snsq(prof)
            
            d['ebwd_u'][j,i], d['ebwd_v'][j,i] = derived.ebwd_aot(prof, d['elhght'][j,i], 
                                                                  d['eff_inflow_base'][j,i])
            d['shr1_u'][j,i], d['shr1_v'][j,i] = derived.bulk_shear(prof, height=1000)
            d['shr3_u'][j,i], d['shr3_v'][j,i] = derived.bulk_shear(prof, height=3000)
            d['shr6_u'][j,i], d['shr6_v'][j,i] = derived.bulk_shear(prof, height=6000)
            d['shr8_u'][j,i], d['shr8_v'][j,i] = derived.bulk_shear(prof, height=8000)
            d['rm5_u'][j,i], d['rm5_v'][j,i] = derived.rm5(prof)
            d['lm5_u'][j,i], d['lm5_v'][j,i] = derived.lm5(prof)
            devtor = derived.devtor(prof)
            d['devtor_u'][j,i], d['devtor_v'][j,i], d['deviance'][j,i] = devtor

            d['estp'][j,i] = derived.estp_aot(d['mlcape'][j,i], -1*d['mlcin'][j,i], 
                                              d['esrh'][j,i], d['ebwd_u'][j,i], 
                                              d['ebwd_v'][j,i], d['mllcl'][j,i],
                                              d['eff_inflow_base'][j,i], prof)
            d['nst'][j,i] = derived.nst(d['cape3km'][j,i], d['mlcin'][j,i], vort[j,i], prof)

    return d

if __name__ == '__main__':
    cc.compile()