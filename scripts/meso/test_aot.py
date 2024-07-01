import pickle

import sharptab.profile as profile
import sharptab.params as params
from sharptab.parcelx_module import parcelx
from time import time

fname = '/Users/leecarlaw/scripts/tmp/data/standard.pickle'
with open(fname, 'rb') as f: data = pickle.load(f)

arr = data['tmpc']
t1 = time()
for j in range(arr.shape[1]):
    for i in range(arr.shape[2]):
        pres = data['pres'][:,j,i]
        tmpc = data['tmpc'][:,j,i]
        dwpc = data['dwpc'][:,j,i]
        wspd = data['wspd'][:,j,i]
        wdir = data['wdir'][:,j,i]
        hght = data['hght'][:,j,i]
        mlpcl_aot = parcelx(pres, tmpc, dwpc, wspd, wdir, hght, 4)
        mupcl_aot = parcelx(pres, tmpc, dwpc, wspd, wdir, hght, 3)
        #effective_info_aot = parcelx(pres, tmpc, dwpc, wspd, wdir, hght, 6)
t2 = time()


t3 = time()
for j in range(arr.shape[1]):
    for i in range(arr.shape[2]):
        prof = profile.create_profile(pres=data['pres'][:,j,i], tmpc=data['tmpc'][:,j,i],
                                     hght=data['hght'][:,j,i], dwpc=data['dwpc'][:,j,i],
                                     wspd=data['wspd'][:,j,i], wdir=data['wdir'][:,j,i])
        mlpcl_original = params.parcelx_original(prof, flag=4)
        mupcl_original = params.parcelx_original(prof, flag=3)
        #eff_inflow = params.effective_inflow_layer(prof)
t4 = time()

print(f"Ahead-of-time loop: {t2-t1} seconds")
print(f"Original JIT loop:  {t4-t3} seconds")