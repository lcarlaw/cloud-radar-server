import pickle
from multiprocessing import freeze_support
from multiprocessing import Pool

import sharptab.profile as profile
import sharptab.params as params
from sharptab.parcelx_module import parcelx
from time import time

import numpy as np 
from utils.timing import timeit
from numba import njit, prange, set_num_threads
from numba.typed import Dict
from numba.core import types
import numba as nb
import concurrent.futures

float_array = types.float64[:,:]

def process_element(args):
    pres, tmpc, dwpc, wspd, wdir, hght, j, i = args
    pres_value = pres[:,j,i]
    tmpc_value = tmpc[:,j,i]
    dwpc_value = dwpc[:,j,i]
    wspd_value = wspd[:,j,i]
    wdir_value = wdir[:,j,i]
    hght_value = hght[:,j,i]
    mlpcl = parcelx(pres_value, tmpc_value, dwpc_value, wspd_value, wdir_value, hght_value, 4)
    mupcl = parcelx(pres_value, tmpc_value, dwpc_value, wspd_value, wdir_value, hght_value, 3)
    
    result = mlpcl['bplus']
    return result

# Can't parallelize the aot function with numba...
def test_aot_version(pres, tmpc, dwpc, wspd, wdir, hght):
    shape = pres.shape
    
    # About 10 seconds with 4 processes
    with Pool(processes=4) as pool:
        args = [(pres, tmpc, dwpc, wspd, wdir, hght, j, i) for j in range(shape[1]) for i in range(shape[2])]
        results = pool.map(process_element, args)
    processed_results = []
    for res in results: 
        processed_results.append(res)

    ####################################
    # 9 seconds
    #num_tasks = shape[1] * shape[2]
    #chunksize = max(1, num_tasks // (2 * 4))
    #args = [(pres, tmpc, dwpc, wspd, wdir, hght, j, i) for j in range(shape[1]) for i in range(shape[2])]
    #with Pool(processes=4) as pool:
    #    results = pool.map(process_element, args, chunksize=chunksize)
    #processed_results = []
    #for res in results: 
    #    processed_results.append(res)
    
    return processed_results


@njit(parallel=True)
def test_jit_version(pres, tmpc, dwpc, wspd, wdir, hght):
    result = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )
    result['mlcape'] = np.zeros((tmpc.shape[1], tmpc.shape[2]), dtype='float64')
    for j in prange(tmpc.shape[1]):
        for i in prange(tmpc.shape[2]):
            prof = profile.create_profile(pres=pres[:,j,i], tmpc=tmpc[:,j,i],
                                          hght=hght[:,j,i], dwpc=dwpc[:,j,i],
                                          wspd=wspd[:,j,i], wdir=wdir[:,j,i])
            mlpcl_original = params.parcelx_original(prof, flag=4)
            mupcl_original = params.parcelx_original(prof, flag=3)
            result['mlcape'][j,i] = mlpcl_original.bplus
            #eff_inflow = params.effective_inflow_layer(prof)
    
    return result


def do():
    #set_num_threads(8)
    fname = '../../tests/standard.pickle'
    with open(fname, 'rb') as f: data = pickle.load(f)

    pres = data['pres']
    tmpc = data['tmpc']
    dwpc = data['dwpc']
    wspd = data['wspd']
    wdir = data['wdir']
    hght = data['hght']
    t1 = time()
    results = test_aot_version(pres, tmpc, dwpc, wspd, wdir, hght)
    t2 = time() 
    print(results)
    
    t3 = time()
    results = test_jit_version(pres, tmpc, dwpc, wspd, wdir, hght)
    t4 = time() 
    print(results)
    
    t5 = time()
    results = test_jit_version(pres, tmpc, dwpc, wspd, wdir, hght)
    t6 = time()
    print(results)
    
    print(f"AOT parallel: {t2-t1} seconds")
    print(f"JIT parallel: {t4-t3} seconds")
    print(f"JIT parallel 2nd loop: {t6-t5} seconds")


if __name__ == '__main__':
    freeze_support()
    do()
    