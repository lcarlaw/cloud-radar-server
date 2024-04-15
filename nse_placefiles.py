"""
Wrapper script used by radar-server to launch the NSE placefile scripts. The first 
download step is blocking, as the placefiles can't be generated until .grib files have
been downloaded to disk. 

Alert banners are sent to the front end if 
"""

from datetime import datetime
import subprocess 

def make_nse_placefiles(sim_start, sim_end, scripts_path):
    model_alert = False
    start_string = datetime.strftime(sim_start,"%Y-%m-%d/%H")
    end_string = datetime.strftime(sim_end,"%Y-%m-%d/%H")
    time_bounds = f"-s {start_string} -e {end_string}".split()

    # Some way to check and report out percent completion? 
    #script_progress("Downloading RAP Data")
    get_data_status = subprocess.run(
        ["python", f"{scripts_path}/meso/get_data.py"] + time_bounds)
    
    #script_progress("ANOTHER UPDATE")
    # Only proceed if we have a good return status. Either full or partial download
    if get_data_status.returncode in [0, 2]:
        #script_progress('Proceeding to NSE placefile generation.')
        ret = subprocess.run(
            ["python", f"{scripts_path}/meso/process.py"] + time_bounds + ['-meso'],
            check=True
        )
        # Partial download
        #if get_data_status.returncode == 2:
            
    # No data available. Could be dates in the future as well. Issue an alert banner. 
    else:
        model_alert = True
    return model_alert