import pyart
from glob import glob 
from pathlib import Path 
from time import time 

from multiprocessing import Pool, freeze_support
from concurrent.futures import ThreadPoolExecutor

def read_and_dealias(filename):
    radar = pyart.io.read(filename)
    print(radar)

    # create a gate filter which specifies gates to exclude from dealiasing
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_invalid("velocity")
    gatefilter.exclude_invalid("reflectivity")
    gatefilter.exclude_outside("reflectivity", 0, 80)

    # perform dealiasing
    dealias_data = pyart.correct.dealias_region_based(radar, gatefilter=gatefilter)
    radar.add_field("corrected_velocity", dealias_data)

    return 

def execute_multiprocessing():
    # Limited to 8 processes on the AWS instance
    pool = Pool(4)
    pool.map(read_and_dealias, files)
    pool.close()
    pool.terminate()

def execute_multithread():
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(read_and_dealias, files)
        executor.shutdown(wait=True)

def serial():
    for f in files:
        read_and_dealias(f)

if __name__ == '__main__':
    freeze_support()
    files = glob(f"{Path(__file__).parents[0]}/data/*V06")
    print(f"Reading {len(files)} radar files")

    ts = time()
    execute_multiprocessing()
    te = time() 
    multiprocess = te - ts 

    ts = time()
    serial()
    te = time() 
    serial = te - ts 

    ts = time()
    execute_multithread()
    te = time() 
    multithread = te - ts 

    print(f"Total radar files: {len(files)}")
    print(f"Multiprocessing (4 processes) took {round(multiprocess, 1)} seconds")
    print(f"Multithread (4 threads) took {round(multithread, 1)} seconds")
    print(f"Serial executation took {round(serial, 1)} seconds")


    
