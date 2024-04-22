"""
All 4 background callbacks are linked to the same input (click of the Run Scripts 
button). Can't figure out how to keep the button set to "disabled=True" until all 
associated scripts have completed. Currently, it will become active ("disabled=False")
as soon aas the shortest-running background callback completes. Same thing with the 
Cancel button. 
    - Possible the Run Script button could just be disabled once it's clicked. Do
      we need to allow for multiple runs? Currently, this is how I've set things.
    - Need to ensure there's no way for a single user to dispatch numerous processes
      by making multiple button clicks.

Background callbacks offer ability to cancel running code -- haven't found this 
available with anything else. 
"""

import time
from uuid import uuid4
import diskcache
from dash import Dash, html, DiskcacheManager, CeleryManager, Input, Output, State, dcc, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta 
import subprocess
import boto3
import botocore
from botocore.client import Config
from pathlib import Path
import os 
import calendar

import layout_components as lc

class RadarSimulator(Config):
    """
    A class representing a radar simulator.

    Attributes:
        start_time (datetime): The start time of the simulation.
        duration (int): The duration of the simulation in seconds.
        radars (list): A list of radar objects.
        product_directory_list (list): A list of directories in the data directory.

    """

    def __init__(self):
        super().__init__()
        self.start_year = 2023
        self.start_month = 6
        self.days_in_month = 30
        self.start_day = 15
        self.start_hour = 18
        self.start_minute = 30
        self.duration = 60
        self.timeshift = None
        self.timestring = None
        self.sim_clock = None
        self.radar = None
        self.lat = None
        self.lon = None
        self.t_radar = 'None'
        self.tlat = None
        self.tlon = None
        self.simulation_running = False
        self.current_dir = Path.cwd()
        self.make_directories()
        self.make_radar_download_folders()
        self.make_times()
        self.make_prefix()
        self.bucket = boto3.resource('s3',
                                    config=Config(signature_version=botocore.UNSIGNED,
                                    user_agent_extra='Resource')).Bucket('noaa-nexrad-level2')

    def make_directories(self):
        self.csv_file = self.current_dir / 'radars.csv'
        self.data_dir = self.current_dir / 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        self.scripts_path = self.current_dir / 'scripts'
        self.obs_script_path = self.current_dir / 'obs_placefile.py'
        self.hodo_script_path = self.scripts_path / 'hodo_plot.py'
        self.nexrad_script_path = self.scripts_path / 'get_nexrad.py'
        self.assets_dir = self.current_dir / 'assets'
        self.hodo_images = self.assets_dir / 'hodographs'
        os.makedirs(self.hodo_images, exist_ok=True)
        self.placefiles_dir = self.assets_dir / 'placefiles'
        os.makedirs(self.placefiles_dir, exist_ok=True)
        return
    
    def make_times(self):
        self.sim_start = datetime(self.start_year,self.start_month,self.start_day,
                                  self.start_hour,self.start_minute,second=0)
        self.sim_clock = self.sim_start
        self.sim_start_str = datetime.strftime(self.sim_start,"%Y-%m-%d %H:%M:%S UTC")
        self.sim_end = self.sim_start + timedelta(minutes=int(self.duration))
        return
   
    def get_days_in_month(self):
        self.days_in_month = calendar.monthrange(self.start_year, self.start_month)[1]
        return

    def make_radar_download_folders(self):
        if self.radar is not None:
            self.radar_site_dir = self.data_dir / 'radar' / self.radar
            os.makedirs(self.radar_site_dir, exist_ok=True)
            self.radar_site_download_dir = self.radar_site_dir / 'downloads'
            os.makedirs(self.radar_site_download_dir, exist_ok=True)
            self.cf_dir = self.radar_site_download_dir / 'cf_radial'
            os.makedirs(self.cf_dir, exist_ok=True)
            return
        return

    def make_prefix(self):
        first_folder = self.sim_start.strftime('%Y/%m/%d/')
        second_folder = self.sim_end.strftime('%Y/%m/%d/')
        self.prefix_day_one = f'{first_folder}{self.radar}'  
        if first_folder != second_folder:
            self.prefix_day_two = f'{second_folder}{self.radar}'
        else:
            self.prefix_day_two = None
        
        return


launch_uid = uuid4()
cache = diskcache.Cache('./cache')

background_callback_manager = DiskcacheManager(
    cache, cache_by=[lambda: launch_uid], expire=60
)

sa = RadarSimulator()
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           background_callback_manager=background_callback_manager)

app.layout = dbc.Container([
    dcc.Interval(id='status_monitor', interval=1000),
    dcc.Store(id='placefile_callback_status'), 
    dcc.Store(id='radar_callback_status'), 
    dcc.Store(id='hodo_callback_status'), 
    dcc.Store(id='nse_callback_status'),
    lc.scripts_button,
    lc.status_section,
    html.Button(id='cancel_button_id', children='Cancel all requests'),
    ])  # end of app.layout


def percent_progress(iteration, total):
    percent = int((iteration / total) * 100)
    return percent

def create_mesowest(set_progress):
    """
    """
    #Mesowest(sa.radar,str(sa.lat),str(sa.lon),sa.sim_start_str,str(sa.duration))
    total = 5
    for i in range(total):
        time.sleep(0.5)
        percent = percent_progress(iteration=i+1, total=total)
        set_progress([int(percent), f"{int(percent)} %"])

def get_radar(set_progress):
    """
    """
    #NexradDownloader(sa.radar, sa.sim_start_str, sa.duration)
    total = 10
    for i in range(total):
        time.sleep(0.5)
        percent = percent_progress(iteration=i+1, total=total)
        set_progress([int(percent), f"{int(percent)} %"])

def create_hodographs(set_progress):
    """
    """
    total = 15
    for i in range(total):
        time.sleep(0.5)
        percent = percent_progress(iteration=i+1, total=total)
        set_progress([int(percent), f"{int(percent)} %"])

def create_nse_placefiles(set_progress):
    """
    """
    proceed = True
    knt = 0
    while proceed:
        time.sleep(0.5)
        try:
            with open(f'{sa.data_dir}/download_status.txt', 'r') as f: 
               status = f.readlines()[0].split(',')
        except FileNotFoundError:
            status = ['0', '999', '0\n']
        except IndexError:
            pass
        percent = percent_progress(iteration=int(status[2]), total=int(status[1]))
        set_progress([int(percent), f"{int(percent)} %"])

        # Temporary hack for testing.
        if percent > 99:
            proceed=False

        knt += 1

# Long callback 1 for mesowest placefiles
@app.callback(
    Output('placefile_callback_status', 'data', allow_duplicate=True),
    Input('run_scripts', 'n_clicks'),
    running=[
        (Output('run_scripts', 'disabled'), True, True),
        (Output('cancel_button_id', 'disabled'), False, False),
    ],
    cancel=[Input('cancel_button_id', 'n_clicks')],
    progress=[
        Output('placefile_status', 'value'),
        #Output("placefile_status", "label"),
    ],
    interval=1000,
    background=True,
    prevent_initial_call=True
)
def mesowest(set_progress, n_clicks):
    if n_clicks > 0:
        create_mesowest(set_progress) 
    return True

# Long callback 2 for radar downloads
@app.callback(
    Output('radar_callback_status', 'data', allow_duplicate=True),
    Input('run_scripts', 'n_clicks'),
    running=[
        (Output('run_scripts', 'disabled'), True, True),
        (Output('cancel_button_id', 'disabled'), False, False),
    ],
    cancel=[Input('cancel_button_id', 'n_clicks')],
    progress=[
        Output('radar_status', 'value'),
        #Output("radar_status", "label"),
    ],
    interval=1000,
    background=True,
    prevent_initial_call=True
)
def radar(set_progress, n_clicks):
    if n_clicks > 0:
        get_radar(set_progress) 
    return True

# Long callback 3 for hodographs
@app.callback(
    Output('hodo_callback_status', 'data', allow_duplicate=True),
    Input('run_scripts', 'n_clicks'),
    running=[
        (Output('run_scripts', 'disabled'), True, True),
        (Output('cancel_button_id', 'disabled'), False, False),
    ],
    cancel=[Input('cancel_button_id', 'n_clicks')],
    progress=[
        Output('hodo_status', 'value'),
        #Output("hodo_status", "label"),
    ],
    interval=1000,
    background=True,
    prevent_initial_call=True
)
def hodographs(set_progress, n_clicks):
    if n_clicks > 0:
        create_hodographs(set_progress) 
    return True

# Long callback 4 for NSE placefiles
@app.callback(
    Output('nse_callback_status', 'data', allow_duplicate=True),
    Input('run_scripts', 'n_clicks'),
    running=[
        (Output('run_scripts', 'disabled'), True, True),
        (Output('cancel_button_id', 'disabled'), False, False),
    ],
    cancel=[Input('cancel_button_id', 'n_clicks')],
    progress=[
        Output('nse_status', 'value'),
        #Output('nse_status', 'label'),
    ],
    interval=1000,
    background=True,
    prevent_initial_call=True
)
def nse_placefiles(set_progress, n_clicks):
    if n_clicks > 0:
        try:
            os.remove(f'{sa.data_dir}/download_status.txt')
        except FileNotFoundError:
            pass 
        #start_string = datetime.strftime(sa.sim_start,"%Y-%m-%d/%H")
        #end_string = datetime.strftime(sa.sim_end,"%Y-%m-%d/%H")
        # Temporary values for testing. 
        start_string = "2024-04-20/01"
        end_string = "2024-04-20/03"
        args = f"-s {start_string} -e {end_string} -statuspath {sa.data_dir}".split()

        # Non-blocking call via Popen to the necessary python scripts
        subprocess.Popen(["python", f"{sa.scripts_path}/meso/get_data.py"] + args)
        create_nse_placefiles(set_progress) 

        # Once create_nse_placefiles returns, the next process to produce the placefiles
        # can be dispatched. Need someway to show we've moved to this next step with the 
        # status bar. Currently, fills to 100% with completion of the step above. 
        args = f"-s {start_string} -e {end_string} -meso".split()
        #subprocess.Popen(["python", f"{sa.scripts_path}/meso/process.py"] + args)
    
    return True

# This callback is initiated by dcc.Interval and monitors the return status of each of 
# the primary background callbacks. Once every process has returned, the run scripts 
# button is returned to disabled=False and cancellation button to disabled=True. Without 
# this callback, the button disabled states are changed as soon as the first/fastest
# background callback returns, and I can't find a way to avoid that. 

# Ultimately, this step might not be necessary--we could just turn the run scripts button
# off entirely after the initial click. 

# Main issue: for some reason, you have to click the Cancel button twice (after a brief
# delay between each click). Not sure what's going on there.
@app.callback(
    Output('run_scripts', 'disabled'),
    Output('cancel_button_id', 'disabled'),
    Output('placefile_callback_status', 'data', allow_duplicate=True),
    Output('radar_callback_status', 'data', allow_duplicate=True),
    Output('hodo_callback_status', 'data', allow_duplicate=True),
    Output('nse_callback_status', 'data', allow_duplicate=True),
    Input('status_monitor', 'n_intervals'),
    Input('run_scripts', 'n_clicks'),
    Input('cancel_button_id', 'n_clicks'),
    State('placefile_callback_status', 'data'),
    State('radar_callback_status', 'data'),
    State('hodo_callback_status', 'data'),
    State('nse_callback_status', 'data'),
    prevent_initial_call=True,
)
def monitor(n, run_click, cancel_click, mesowest, radar, hodo, nse):
    # Probably also want something checking how long each of the processes has been
    # running and to cancel if we're over a certain limit. Create a datetime object 
    # within each callback? 

    # Run scripts button was clicked 
    # Set the run button to disabled and cancel button to enabled
    if ctx.triggered[0]['prop_id'] == 'run_scripts.n_clicks':
        return True, False, False, False, False, False 
    
    # !! For some reason, have to hit cancel button twice?? Can't figure that out.. 
    # Cancel button was clicked, or all background callbacks have completed
    # Set the run button to enabled and cancel button to disabled
    if ctx.triggered[0]['prop_id'] == 'cancel_button_id.n_clicks' or all([mesowest, radar, hodo, nse]):
        return False, True, False, False, False, False 

    raise PreventUpdate 

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1')