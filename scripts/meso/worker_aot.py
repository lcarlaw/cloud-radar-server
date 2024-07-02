from numba import njit, prange
from numba.typed import Dict
from numba.core import types
import numpy as np 
from numba.pycc import CC

import sharptab.profile as profile
from calc import derived

cc = CC('fast_loop')

float_array = types.float64[:,:] 
@njit(parallel=True)
@cc.export('worker', 'DictType(unicode_type,types.float64[:,:])(f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:,:],f8[:,:],f8[:,:,:])')
def worker(pres, tmpc, dwpc, wspd, wdir, hght, vort, incoming):
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
    
    for j in prange(shape[0]):
        for i in prange(shape[1]):
            
            prof = profile.create_profile(pres=pres[:,j,i], tmpc=tmpc[:,j,i],
                                          hght=hght[:,j,i], dwpc=dwpc[:,j,i],
                                          wspd=wspd[:,j,i], wdir=wdir[:,j,i])
            
            eff_inflow = d['eff_inflow_base'][j,i], d['eff_inflow_top'][j,i]
            d['esrh'][j,i] = derived.srh(prof, effective_inflow_layer=eff_inflow)
            d['srh500'][j,i] = derived.srh(prof, lower=0, upper=500)
            d['lr03km'][j,i] = derived.lapse_rate(prof, lower=0, upper=3000)
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