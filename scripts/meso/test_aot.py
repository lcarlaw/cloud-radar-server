import pickle
from multiprocessing import freeze_support
from multiprocessing import Pool

import sharptab.profile as profile
import sharptab.params as params
from calc import derived
from sharptab.winds import vec2comp
from sharptab.constants import KTS2MS

from sharptab import aot_module

from plotconfigs import SCALAR_PARAMS, VECTOR_PARAMS
from time import time

import numpy as np 
from numba import njit, prange, set_num_threads
from numba import i4, f8
from numba.typed import Dict, List
from numba.core import types

import matplotlib.pyplot as plt 


float_array = types.float64[:,:] 

def process_element(args):
    """
    Perform calculations on variables return from ahead-of-time-compiled functions. 
    Additional derived variables handled in separate function using parallelized numba.
    """
    d = {}
    pres, tmpc, dwpc, wspd, wdir, hght, j, i = args
    pres_value = pres[:,j,i]
    tmpc_value = tmpc[:,j,i]
    dwpc_value = dwpc[:,j,i]
    wspd_value = wspd[:,j,i]
    wdir_value = wdir[:,j,i]
    hght_value = hght[:,j,i]
    
    mlpcl = aot_module.parcelx(pres_value, tmpc_value, dwpc_value, wspd_value, wdir_value, hght_value, 4)
    mupcl = aot_module.parcelx(pres_value, tmpc_value, dwpc_value, wspd_value, wdir_value, hght_value, 3)
    eff_inflow = aot_module.effective_inflow_layer_aot(pres_value, tmpc_value, dwpc_value, wspd_value, wdir_value, hght_value)
    d['eff_inflow_bot'] = eff_inflow[0]
    d['eff_inflow_top'] = eff_inflow[1]
    d['mlcape'] = mlpcl['bplus']
    d['mlcin'] = -1 * mlpcl['bminus']
    d['mucape'] = mupcl['bplus']
    d['cape3km'] = mlpcl['b3km']
    d['mllcl'] = mlpcl['lclhght']
    d['elhght'] = mupcl['elhght']
    return (j, i), d


# Can't parallelize the aot function with numba...
def test_aot_version(pres, tmpc, dwpc, wspd, wdir, hght, vort):
    shape = pres.shape
    
    # About 10 seconds with 4 processes
    with Pool(processes=4) as pool:
        args = [(pres, tmpc, dwpc, wspd, wdir, hght, j, i) for j in range(shape[1]) for i in range(shape[2])]
        results = pool.map(process_element, args)
    
    ####################################
    # 9 seconds
    #num_tasks = shape[1] * shape[2]
    #chunksize = max(1, num_tasks // (2 * 4))
    #args = [(pres, tmpc, dwpc, wspd, wdir, hght, j, i) for j in range(shape[1]) for i in range(shape[2])]
    #with Pool(processes=4) as pool:
    #    results = pool.map(process_element, args, chunksize=chunksize)

    # Put everything back together as a big dictionary of arrays. Ensure the dictionary
    # is understood by numba (which we're going to pass this into) with explicit types.
    #data_dict = Dict.empty(
    #    key_type=types.unicode_type,
    #    value_type=float_array,
    #)
    #for variable in results[0][1].keys():
    #    data = np.zeros((shape[1], shape[2]))
    #    for key, result in results:
    #        data[key] = result[variable]
    #    data_dict[variable] = data
    
    n_vars = len(results[0][1].keys())
    processed_data = np.zeros((n_vars, shape[1], shape[2]))
    for i, variable in enumerate(results[0][1].keys()):
        for key, result in results:
            processed_data[i, key[0], key[1]] = result[variable]
            #data[key] = result[variable]
        #data_dict[variable] = data
    temp = aot_module.worker(pres, tmpc, dwpc, wspd, wdir, hght, vort, processed_data)
    return temp



@njit(parallel=True)
def test_jit_version(pres, tmpc, dwpc, wspd, wdir, hght, vort, SCALARS, VECTORS):
    # Declare 'jit-able' dictionary and fill it with empty arrays
    d = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    for scalar in SCALARS:
        d[scalar] = np.zeros((tmpc.shape[1], tmpc.shape[2]), dtype='float64')
    for vector in VECTORS:
        d[vector+'_u'] = np.zeros((tmpc.shape[1], tmpc.shape[2]), dtype='float64')
        d[vector+'_v'] = np.zeros((tmpc.shape[1], tmpc.shape[2]), dtype='float64')

    for j in prange(tmpc.shape[1]):
        for i in prange(tmpc.shape[2]):
            prof = profile.create_profile(pres=pres[:,j,i], tmpc=tmpc[:,j,i],
                                          hght=hght[:,j,i], dwpc=dwpc[:,j,i],
                                          wspd=wspd[:,j,i], wdir=wdir[:,j,i])

            # Compile the big jitted methods. This will slow down the very 1st iteration.
            mlpcl = params.parcelx_original(prof, flag=4)
            eff_inflow = params.effective_inflow_layer(prof)
            mupcl = params.parcelx_original(prof, flag=3)    
            
            # Scalars
            if ('fzl-lfc-diff' in SCALARS) and ('el-lfc-diff' in SCALARS) and \
                ('mu-el' in SCALARS):
                hail = derived.hail_parms(prof, mupcl) 
                d['mu-el'][j,i], d['el-lfc-diff'][j,i], d['fzl-lfc-diff'][j,i] = hail 
            if 'esrh' in SCALARS:
                d['esrh'][j,i] = derived.srh(prof, effective_inflow_layer=eff_inflow)
            if 'mucape' in SCALARS:
                d['mucape'][j,i] = mupcl.bplus
            if 'mlcin' in SCALARS:
                d['mlcin'][j,i] = mlpcl.bminus * -1
            if 'mlcape' in SCALARS:
                d['mlcape'][j,i] = mlpcl.bplus
            if 'cape3km' in SCALARS:
                d['cape3km'][j,i] = mlpcl.b3km
            if 'srh500' in SCALARS:
                d['srh500'][j,i] = derived.srh(prof, lower=0, upper=500)
            if 'srh01km' in SCALARS:
                d['srh01km'][j,i] = derived.srh(prof, lower=0, upper=1000)
            if 'lr03km' in SCALARS:
                d['lr03km'][j,i] = derived.lapse_rate(prof, lower=0, upper=3000)
            if 'mllcl' in SCALARS:
                d['mllcl'][j,i] = mlpcl.lclhght
            if 'snsq' in SCALARS:
                d['snsq'][j,i] = derived.snsq(prof)

            # Vectors: returned as (u, v) tuples
            if 'ebwd' in VECTORS:
                d['ebwd_u'][j,i], d['ebwd_v'][j,i] = derived.ebwd(prof, mupcl, eff_inflow)
            if 'shr1' in VECTORS:
                d['shr1_u'][j,i], d['shr1_v'][j,i] = derived.bulk_shear(prof, height=1000)
            if 'shr3' in VECTORS:
                d['shr3_u'][j,i], d['shr3_v'][j,i] = derived.bulk_shear(prof, height=3000)
            if 'shr6' in VECTORS:
                d['shr6_u'][j,i], d['shr6_v'][j,i] = derived.bulk_shear(prof, height=6000)
            if 'shr8' in VECTORS:
                d['shr8_u'][j,i], d['shr8_v'][j,i] = derived.bulk_shear(prof, height=8000)
            if 'rm5' in VECTORS:
                d['rm5_u'][j,i], d['rm5_v'][j,i] = derived.rm5(prof)
            if 'lm5' in VECTORS:
                d['lm5_u'][j,i], d['lm5_v'][j,i] = derived.lm5(prof)
            if 'devtor' in VECTORS:
                devtor = derived.devtor(prof)
                d['devtor_u'][j,i], d['devtor_v'][j,i], d['deviance'][j,i] = devtor

            # Special parameters: prohibitive to re-compute all of the inputs...
            if 'estp' in SCALARS: d['estp'][j,i] = derived.estp(d['mlcape'][j,i],
                                                                mlpcl.bminus,
                                                                d['esrh'][j,i],
                                                                d['ebwd_u'][j,i],
                                                                d['ebwd_v'][j,i],
                                                                mlpcl, eff_inflow[0],
                                                                prof)
            if 'nst' in SCALARS: d['nst'][j,i] = derived.nst(d['cape3km'][j,i],
                                                             d['mlcin'][j,i], vort[j,i],
                                                             prof)
    return d


def do():
    set_num_threads(8)
    fname = '../../tests/numba-aot/standard.pickle'
    with open(fname, 'rb') as f: data = pickle.load(f)

    pres = data['pres']
    tmpc = data['tmpc']
    dwpc = data['dwpc']
    wspd = data['wspd']
    wdir = data['wdir']
    hght = data['hght']
    lons = data['lons']
    lats = data['lats']

    # Vorticity calculations for NST parameter. 0th index from hybrid files is ~10m agl.
    u, v = vec2comp(wdir[0,:,:], wspd[0,:,:]*KTS2MS)
    vort = derived.vorticity(u, v, lons, lats)

    #ret = fast_loop.worker(pres, tmpc, dwpc, wspd, wdir, hght, hght)
    #print(ret)
    t1 = time()
    results = test_aot_version(pres, tmpc, dwpc, wspd, wdir, hght, vort)
    t2 = time() 
    print(results['estp'])
    levels = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
    fig = plt.figure()
    plt.contour(results['mllcl'], levels)
    plt.savefig('aot_mllcl.png')
    plt.close(fig)

    levels = [50, 100, 200, 300, 400, 500, 600, 700, 800]
    fig = plt.figure()
    plt.contour(results['esrh'], levels)
    plt.savefig('aot_esrh.png')
    plt.close(fig)

    #t3 = time()
    #results = test_aot_version(pres, tmpc, dwpc, wspd, wdir, hght)
    #t4 = time() 

    t3 = time()
    results = test_jit_version(pres, tmpc, dwpc, wspd, wdir, hght, vort, 
                               List(SCALAR_PARAMS.keys()), List(VECTOR_PARAMS.keys()))
    t4 = time()
    levels = [500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
    fig = plt.figure()
    plt.contour(results['mllcl'], levels)
    plt.savefig('original_mllcl.png')
    plt.close(fig)

    levels = [50, 100, 200, 300, 400, 500, 600, 700, 800]
    fig = plt.figure()
    plt.contour(results['esrh'], levels)
    plt.savefig('original_esrh.png')
    plt.close(fig)

    #t5 = time()
    #results = test_jit_version(pres, tmpc, dwpc, wspd, wdir, hght, 
    #                           List(SCALAR_PARAMS.keys()), List(VECTOR_PARAMS.keys()))
    #t6 = time()
    #print(results)
    
    print(f"AOT parallel: {t2-t1} seconds")
    #print(f"AOT parallel loop 2: {t4-t3} seconds")
    print(f"JIT parallel: {t4-t3} seconds")
    #print(f"JIT parallel 2nd loop: {t6-t5} seconds")


if __name__ == '__main__':
    #cc.compile()
    freeze_support()
    do()
    