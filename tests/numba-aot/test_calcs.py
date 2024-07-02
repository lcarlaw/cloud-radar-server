"""
Verifying the sharppy calculations completed via aot functions match original calculations. 
Did have to move the pickle exports in process.py to above the filtering code. 
"""
import pickle
import numpy as np 

fpath = "/Users/leecarlaw/scripts/cloud-radar-server/tests/numba-aot"
with open(f'{fpath}/aot.pickle', 'rb') as f: aot = pickle.load(f)
with open(f'{fpath}/sharppy.pickle', 'rb') as f: sharppy = pickle.load(f)

for k in sharppy[1]:
    if k in aot:
        sum_diff = np.nansum(aot[k] - sharppy[1][k])
        print(f"Difference sum for {k}: {sum_diff}")
    else: 
        print(f"{k} not in aot dataset")