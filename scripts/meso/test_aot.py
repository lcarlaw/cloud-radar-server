import pickle
from multiprocessing import freeze_support
from multiprocessing import Pool

import numpy as np 
from numba import set_num_threads
from numba.core import types

from calc import derived
from sharptab.winds import vec2comp
from sharptab.constants import KTS2MS
from sharptab import aot_module
from time import time

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
    eff_inflow = aot_module.effective_inflow_layer(pres_value, tmpc_value, dwpc_value, wspd_value, wdir_value, hght_value)
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
    
    # Chunking data test
    #num_tasks = shape[1] * shape[2]
    #chunksize = max(1, num_tasks // (2 * 4))
    #args = [(pres, tmpc, dwpc, wspd, wdir, hght, j, i) for j in range(shape[1]) for i in range(shape[2])]
    #with Pool(processes=4) as pool:
    #    results = pool.map(process_element, args, chunksize=chunksize)    
    
    # Gather output. Best to use numpy array instead of dictionary for numba purposes.
    n_vars = len(results[0][1].keys())
    processed_data = np.zeros((n_vars, shape[1], shape[2]))
    for i, variable in enumerate(results[0][1].keys()):
        for key, result in results:
            processed_data[i, key[0], key[1]] = result[variable]

    ret = aot_module.fast_loop(pres, tmpc, dwpc, wspd, wdir, hght, vort, processed_data)

    # Converison back to a 'normal' Python dictionary
    final_output = {}
    for k, v in ret.items(): final_output[k] = v
    return final_output


def do_comparison():
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

    t1 = time()
    results = test_aot_version(pres, tmpc, dwpc, wspd, wdir, hght, vort)
    t2 = time() 

    with open(f"../../tests/numba-aot/aot.pickle", 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    #t3 = time()
    #results = worker(pres, tmpc, dwpc, wspd, wdir, hght, vort, 
    #                 List(SCALAR_PARAMS.keys()), List(VECTOR_PARAMS.keys()))
    #t4 = time()
    
    print(f"AOT parallel: {t2-t1} seconds")
    #print(f"JIT parallel: {t4-t3} seconds")

if __name__ == '__main__':
    freeze_support()
    do_comparison()
    