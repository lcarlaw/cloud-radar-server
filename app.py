"""_summary_

    Returns:
    Main page for radar-server
        _type_: _description_
"""

import os
import re
from glob import glob
from datetime import datetime, timedelta
from time import sleep
import calendar
from pathlib import Path
import math
import subprocess
from dash import Dash, html, Input, Output, dcc #, ctx, callback
from dash.exceptions import PreventUpdate
#from dash import diskcache, DiskcacheManager, CeleryManager
#from uuid import uuid4
#import diskcache

import numpy as np
from botocore.client import Config

# bootstrap is what helps styling for a better presentation
import dash_bootstrap_components as dbc
from scripts.obs_placefile import Mesowest
from scripts.Nexrad import NexradDownloader
import layout_components as lc

 # Earth radius (km)
R = 6_378_137

# Regular expressions. First one finds lat/lon pairs, second finds the timestamps.
LAT_LON_REGEX = "[0-9]{1,2}.[0-9]{1,100},[ ]{0,1}[|\\s-][0-9]{1,3}.[0-9]{1,100}"
TIME_REGEX = "[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"

# ----------------------------------------
#        Attempt to set up environment
# ----------------------------------------
TOKEN = 'INSERT YOUR MAPBOX TOKEN HERE'

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'
RADAR_DIR = DATA_DIR / 'radar'
CSV_PATH = BASE_DIR / 'radars.csv'
SCRIPTS_DIR = BASE_DIR / 'scripts'
OBS_SCRIPT_PATH = SCRIPTS_DIR / 'obs_placefile.py'
HODO_SCRIPT_PATH = SCRIPTS_DIR / 'hodo_plot.py'
NEXRAD_SCRIPT_PATH = SCRIPTS_DIR / 'Nexrad.py'

################################################################################################
#       Define class RadarSimulator
################################################################################################

class RadarSimulator(Config):
    """
    A class to simulate radar operations, inheriting configurations from a base Config class.

    This simulator is designed to mimic the behavior of a radar system over a specified period,
    starting from a predefined date and time. It allows for the simulation of radar data generation,
    including the handling of time shifts and geographical coordinates.

    Attributes:
        start_year (int): The year when the simulation starts.
        start_month (int): The month when the simulation starts.
        days_in_month (int): The number of days in the starting month.
        start_day (int): The day of the month when the simulation starts.
        start_hour (int): The hour of the day when the simulation starts (24-hour format).
        start_minute (int): The minute of the hour when the simulation starts.
        duration (int): The total duration of the simulation in minutes.
        timeshift (Optional[int]): The time shift in minutes to apply to the simulation clock. Default is None.
        timestring (Optional[str]): A string representation of the current simulation time. Default is None.
        sim_clock (Optional[datetime]): The current simulation time as a datetime object. Default is None.
        radar (Optional[object]): An instance of a radar object used in the simulation. Default is None.
        lat (Optional[float]): The latitude coordinate for the radar. Default is None.
        lon (Optional[float]): The longitude coordinate for the radar. Default is None.
        t_radar (str): A temporary variable for radar type. Default is 'None'.
        tlat (Optional[float]): Temporary storage for latitude coordinate. Default is None.
        tlon (Optional[float]): Temporary storage for longitude coordinate. Default is None.
        simulation_running (bool): Flag to indicate if the simulation is currently running. Default is False.
    """

    def __init__(self):
        super().__init__()
        self.start_year = 2023
        self.start_month = 6
        self.days_in_month = 30
        self.start_day = 7
        self.start_hour = 21
        self.start_minute = 45
        self.duration = 30
        self.timeshift = None
        self.timestring = None
        self.sim_clock = None
        self.number_of_radars = 0
        self.radar_list = []
        self.radar_dict = {}
        self.radar = None
        self.lat = None
        self.lon = None
        self.tradar = 'None'
        self.tlat = None
        self.tlon = None
        self.simulation_running = False
        self.current_dir = Path.cwd()
        self.define_scripts_and_assets_directories()
        self.make_simulation_times()
        # self.get_radar_coordinates()

    def create_radar_dict(self):
        for _i,radar in enumerate(self.radar_list):
            self.lat = lc.df[lc.df['radar'] == radar]['lat'].values[0]
            self.lon = lc.df[lc.df['radar'] == radar]['lon'].values[0]
            asos_one = lc.df[lc.df['radar'] == radar]['asos_one'].values[0]
            asos_two = lc.df[lc.df['radar'] == radar]['asos_two'].values[0]
            self.radar_dict[radar.upper()] = {'lat':self.lat,'lon':self.lon, 'asos_one':asos_one, 'asos_two':asos_two, 'radar':radar.upper(), 'file_list':[]}
        return
    
    def define_scripts_and_assets_directories(self):
        self.csv_file = self.current_dir / 'radars.csv'
        self.data_dir = self.current_dir / 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        self.scripts_path = self.current_dir / 'scripts'
        self.obs_script_path = self.current_dir / 'obs_placefile.py'
        self.hodo_script_path = self.scripts_path / 'hodo_plot.py'
        self.nexrad_script_path = self.scripts_path / 'get_nexrad.py'
        self.munge_dir = self.scripts_path / 'munge'
        self.assets_dir = self.current_dir / 'assets'
        self.hodo_images = self.assets_dir / 'hodographs'
        os.makedirs(self.hodo_images, exist_ok=True)
        self.placefiles_dir = self.assets_dir / 'placefiles'
        os.makedirs(self.placefiles_dir, exist_ok=True)
        return

    def make_simulation_times(self):
        self.sim_start = datetime(self.start_year,self.start_month,self.start_day,
                                  self.start_hour,self.start_minute,second=0)
        self.sim_clock = self.sim_start
        self.sim_start_str = datetime.strftime(self.sim_start,"%Y-%m-%d %H:%M:%S UTC")
        self.timestring = self.sim_start_str
        self.sim_end = self.sim_start + timedelta(minutes=int(self.duration))
        return
   
    def get_days_in_month(self):
        self.days_in_month = calendar.monthrange(self.start_year, self.start_month)[1]
        return

    def get_timestamp(self,file):
        """
        - extracts datetime info from the radar filename
        - converts it to a datetime timestamp (epoch seconds) object
        """
        file_epoch_time = datetime.strptime(file[4:19], '%Y%m%d_%H%M%S').timestamp()
        return file_epoch_time

    def move_point(self,plat,plon):
        """
        Shift placefiles to a different radar site. Maintains the original azimuth and range
        from a specified RDA and applies it to a new radar location. 

        Parameters:
        -----------
        lat: float 
            Original placefile latitude
        lon: float 
            Original palcefile longitude
        """
        def _clamp(n, minimum, maximum):
            """
            Helper function to make sure we're not taking the square root of a negative 
            number during the calculation of `c` below. Same as numpy.clip(). 
            """
            return max(min(maximum, n), minimum)

        # Compute the initial distance from the original radar location
        phi1, phi2 = math.radians(self.lat), math.radians(plat)
        d_phi = math.radians(plat - self.lat)
        d_lambda = math.radians(plon - self.lon)

        a = math.sin(d_phi/2)**2 + (math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2)
        a = _clamp(a, 0, a)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = R * c

        # Compute the bearing
        y = math.sin(d_lambda) * math.cos(phi2)
        x = (math.cos(phi1) * math.sin(phi2)) - (math.sin(phi1) * math.cos(phi2) * \
                                                math.cos(d_lambda))
        theta = math.atan2(y, x)
        bearing = (math.degrees(theta) + 360) % 360

        # Apply this distance and bearing to the new radar location
        phi_new, lambda_new = math.radians(self.tlat), math.radians(self.tlon)
        phi_out = math.asin((math.sin(phi_new) * math.cos(d/R)) + (math.cos(phi_new) * \
                            math.sin(d/R) * math.cos(math.radians(bearing))))
        lambda_out = lambda_new + math.atan2(math.sin(math.radians(bearing)) *    \
                    math.sin(d/R) * math.cos(phi_new), math.cos(d/R) - math.sin(phi_new) * \
                    math.sin(phi_out))
        return math.degrees(phi_out), math.degrees(lambda_out)



    def shift_placefiles(self, filepath):
        filenames = glob(f"{filepath}/*.txt")
        for file_ in filenames:
            print(f"Shifting placefile: {file_}")
            with open(file_, 'r', encoding='utf-8') as f: data = f.readlines()
            outfile = open(outfilename, 'w', encoding='utf-8')
            outfilename = f"{file_[0:file_.index('.txt')]}.shifted"
            for line in data:
                new_line = line

                if self.timeshift is not None and any(x in line for x in ['Valid', 'TimeRange']):
                    new_line = self.shift_time(line)

                # Shift this line in space
                # This regex search finds lines with valid latitude/longitude pairs
                regex = re.findall(LAT_LON_REGEX, line)
                if len(regex) > 0:
                    idx = regex[0].index(',')
                    lat, lon = float(regex[0][0:idx]), float(regex[0][idx+1:])
                    lat_out, lon_out = self.move_point(lat, lon)
                    new_line = line.replace(regex[0], f"{lat_out}, {lon_out}")

                outfile.write(new_line)
            outfile.close()
        return

    def shift_time(self, line):
        new_line = line
        if 'Valid:' in line:
            idx = line.find('Valid:')
            valid_timestring = line[idx+len('Valid:')+1:-1] # Leave off \n character
            dt = datetime.strptime(valid_timestring, '%H:%MZ %a %b %d %Y')
            new_validstring = datetime.strftime(dt + timedelta(minutes=self.timeshift),
                                                '%H:%MZ %a %b %d %Y')
            new_line = line.replace(valid_timestring, new_validstring)

        if 'TimeRange' in line:
            regex = re.findall(TIME_REGEX, line)
            dt = datetime.strptime(regex[0], '%Y-%m-%dT%H:%M:%SZ')
            new_datestring_1 = datetime.strftime(dt + timedelta(minutes=self.timeshift),
                                                '%Y-%m-%dT%H:%M:%SZ')
            dt = datetime.strptime(regex[1], '%Y-%m-%dT%H:%M:%SZ')
            new_datestring_2 = datetime.strftime(dt + timedelta(minutes=self.timeshift),
                                                '%Y-%m-%dT%H:%M:%SZ')
            new_line = line.replace(f"{regex[0]} {regex[1]}",
                                    f"{new_datestring_1} {new_datestring_2}")
        return new_line


scripts_button = html.Div([
        dbc.Row([
            dbc.Col(
                html.Div([
                    dbc.Button('Make Obs Placefile ... Download radar data ... Make hodo plots', size="lg", id='run_scripts', n_clicks=0),
                    ], className="d-grid gap-2"), style={'vertical-align':'middle'}),
                    html.Div(id='show_script_progress',style=lc.feedback)
        ])
            ], style={'padding':'1em', 'vertical-align':'middle'})


################################################################################################
#      Initialize the app
################################################################################################

sa = RadarSimulator()
app = Dash(__name__,external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True)
app.title = "Radar Simulator"

################################################################################################
#     Build the layout
################################################################################################

sim_day_selection =  dbc.Col(html.Div([
                    lc.step_day,
                    dcc.Dropdown(np.arange(1,sa.days_in_month+1),7,id='start_day',clearable=False
                    ) ]))

app.layout = dbc.Container([
    dcc.Store(id='sim_store'),
    lc.top_section, lc.top_banner,
    dbc.Container([
    dbc.Container([
    html.Div([html.Div([ lc.step_select_time_section,lc.spacer,
        dbc.Row([
                lc.sim_year_section,lc.sim_month_section, sim_day_selection,
                lc.sim_hour_section, lc.sim_minute_section, lc.sim_duration_section,
                lc.spacer,lc.step_time_confirm])],style={'padding':'1em'}),
    ],style=lc.section_box)])
    ]),lc.spacer,lc.spacer,
        dbc.Container([
        dbc.Container([html.Div([lc.radar_select_section],style={'background-color': '#333333', 'border': '2.5px gray solid','padding':'1em'})]),
]),
        lc.spacer,
        lc.map_section, lc.transpose_section, lc.spacer_mini,
        scripts_button,
    lc.status_section,
    lc.toggle_simulation_clock,lc.simulation_clock, lc.radar_id, lc.bottom_section
    ])  # end of app.layout


################################################################################################
# ---------------------------------------- Radar Map Callbacks -------------------------------------
################################################################################################

@app.callback(
    Output('tradar', 'value'),
    Input('tradar', 'value'), prevent_initial_call=True)
def transpose_radar(tradar):
    if tradar != 'None':
        sa.tradar = tradar
        return f'{sa.tradar}'
    return 'None'


@app.callback(
    Output('graph-container', 'style'),
    Input('map_btn', 'n_clicks'))
def toggle_map_display(n):
    if n%2 == 0:
        return {'display': 'none'}
    else:
        return {'padding-bottom': '2px', 'padding-left': '2px','height': '80vh', 'width': '100%'}


@app.callback(
    Output('show_radar_selections', 'children'),
    [Input('graph', 'clickData')])
def display_click_data(clickData):
    if clickData is None:
        return 'No radars selected ...'
    else:
        print (clickData)
        the_link=clickData['points'][0]['customdata']
        if the_link is None:
            return 'No Website Available'
        else:
            sa.radar = the_link
            if sa.radar not in sa.radar_list:
                sa.radar_list.append(sa.radar)
            else:
                radar_list = ', '.join(sa.radar_list)
                return f'{sa.radar} already selected'
            if len(sa.radar_list) <= sa.number_of_radars:
                radar_list = ', '.join(sa.radar_list)
                return f'{radar_list}'
            sa.radar_list = sa.radar_list[1:]
            sa.create_radar_dict()
            radar_list = ', '.join(sa.radar_list)
            return f'{radar_list}'

@app.callback(
    Output('transpose_section', 'style'),
    Input('radar_quantity', 'value'))
def toggle_transpose_display(value):
    """
    Resets the radar list when the number of radars is changed
    """
    sa.number_of_radars = value
    #sa.radar_list = []
    while len(sa.radar_list) > sa.number_of_radars:
        sa.radar_list = sa.radar_list[1:]
    if sa.number_of_radars == 1:
        return lc.section_box
    return {'display': 'none'}


# -------------------------------------
# ---  Run Scripts button ---
# -------------------------------------


def run_hodo_script(args):
    subprocess.run(["python", HODO_SCRIPT_PATH] + args)
    return


@app.callback(
    Output('show_script_progress', 'children'),
    [Input('run_scripts', 'n_clicks')],
    prevent_initial_call=True)
def launch_obs_script(n_clicks):
    if n_clicks > 0:
        sa.make_simulation_times()
        print(sa.radar_list)
        try:
            sa.create_radar_dict()
        except Exception as e:
            print("Error creating radar dictionary: ", e)
        for radar, data in sa.radar_dict.items():
            pass
            try:
                asos_one = data['asos_one']
                asos_two = data['asos_two']
                print(asos_one, asos_two, radar)
            except KeyError as e:
                print("Error getting radar metadata: ", e)
            try:
                file_list = NexradDownloader(radar, sa.timestring, str(sa.duration))
                sa.radar_dict[radar]['file_list'] = file_list
                #print("Nexrad script completed ... Now creating hodographs ...")
            except Exception as e:
                print("Error running nexrad script: ", e)
            try:
                print(f'hodo script: {radar}, {BASE_DIR}, {asos_one}, {asos_two}')
                run_hodo_script([radar, BASE_DIR, asos_one, asos_two])
                #print("Hodograph script completed ...")
            except Exception as e:
                print("Error running hodo script: ", e)

                
        try:
            #mean_lat,mean_lon = sa.get_mean_lat_lon_values()
            print("Running obs script...")
            Mesowest(str(sa.lat),str(sa.lon),sa.timestring,str(sa.duration))
            print("Obs script completed")
        except Exception as e:
            print("Error running obs script: ", e)

        return ""


# -------------------------------------
# --- Transpose if transpose radar selected
# -------------------------------------
@app.callback(
    Output('transpose_status', 'value'),
    Input('run_transpose_script', 'n_clicks'))
def run_transpose_script(n_clicks):
    if sa.tradar == 'None':
        return 100
    if n_clicks > 0:
        sa.shift_placefiles(sa.placefiles_dir)
        return 100
    return 0

################################################################################################
# ---------------------------------------- Time Selection Summary and Callbacks ----------------
################################################################################################

@app.callback(
Output('show_time_data', 'children'),
Input('start_year', 'value'),
Input('start_month', 'value'),
Input('start_day', 'value'),
Input('start_hour', 'value'),
Input('start_minute', 'value'),
Input('duration', 'value'),
)
def get_sim(_yr, _mo, _dy, _hr, _mn, _dur):
    sa.make_simulation_times()
    line1 = f'Start: {sa.sim_start_str[:-7]}Z ____ {sa.duration} minutes'
    return line1

@app.callback(Output('start_year', 'value'),Input('start_year', 'value'))
def get_year(start_year):
    sa.start_year = start_year
    return sa.start_year

@app.callback(
    Output('start_day', 'options'),
    [Input('start_year', 'value'), Input('start_month', 'value')])
def update_day_dropdown(selected_year, selected_month):
    _, num_days = calendar.monthrange(selected_year, selected_month)
    day_options = [{'label': str(day), 'value': day} for day in range(1, num_days+1)]
    return day_options


@app.callback(Output('start_month', 'value'),Input('start_month', 'value'))
def get_month(start_month):
    sa.start_month = start_month
    return sa.start_month

@app.callback(Output('start_day', 'value'),Input('start_day', 'value'))
def get_day(start_day):
    sa.start_day = start_day
    return sa.start_day

@app.callback(Output('start_hour', 'value'),Input('start_hour', 'value'))
def get_hour(start_hour):
    sa.start_hour = start_hour
    return sa.start_hour

@app.callback(Output('start_minute', 'value'),Input('start_minute', 'value'))
def get_minute(start_minute):
    sa.start_minute = start_minute
    return sa.start_minute

@app.callback(Output('duration', 'value'),Input('duration', 'value'))
def get_duration(duration):
    sa.duration = duration
    return sa.duration

################################################################################################
# ---------------------------------------- Clock Callbacks ----------------
################################################################################################

@app.callback(
    Output('clock-container', 'style'),
    Input('enable_sim_clock', 'n_clicks'))
def enable_simulation_clock(n):
    if n % 2 == 0:
        return {'display': 'none'}
    else:
        return {'padding-bottom': '2px', 'padding-left': '2px','height': '80vh', 'width': '100%'}


@app.callback(
    Output('clock-output', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(_n):
    sa.sim_clock = sa.sim_clock + timedelta(seconds=15)
    return sa.sim_clock.strftime("%Y-%m-%d %H:%M:%S UTC")

################################################################################################
# ---------------------------------------- Call app ----------------
################################################################################################

if __name__ == '__main__':
    #app.run_server(host="0.0.0.0", port=8050, threaded=True, debug=True, use_reloader=False)
    app.run(debug=True, port=8050, threaded=True)
    
# pathname_params = dict()
# if my_settings.hosting_path is not None:
#     pathname_params["routes_pathname_prefix"] = "/"                                                                                                                                                                                                                              
#     pathname_params["requests_pathname_prefix"] = "/{}/".format(my_settings.hosting_path)

    # def update_dirlist(self):
    #     """
    #     The dir.list file is needed for GR2Analyst to poll in DRT
    #     """
    #     simulation_counter = self.get_timestamp(self.simulation_files[0].parts[-1]) + 360
    #     last_file_timestamp = self.get_timestamp(self.simulation_files[-1].parts[-1])
    #     print(simulation_counter,last_file_timestamp-simulation_counter)
    #     while simulation_counter < last_file_timestamp:
    #         simulation_counter += 60
    #         self.output = ''     
    #         for file in self.simulation_files:
    #             file_timestamp = self.get_timestamp(file.parts[-1])
    #             if file_timestamp < simulation_counter:
    #                 line = f'{file.stat().st_size} {file.parts[-1]}\n'
    #                 self.output = self.output + line
    #                 with open(f'{self.radar_dir}/dir.list', mode='w', encoding='utf-8') as f:
    #                     f.write(self.output)
    #             else:
    #                 pass

    #         sleep(int(60/self.playback_speed))

    #     print("simulation complete!")

    #     return
    # def catalog_files(self):
    #     self.source_directory = Path(self.radar_site_download_dir)
    #     self.source_files = list(self.source_directory.glob('*V06'))
    #     self.uncompressed_files = list(self.source_directory.glob('*V06.uncompressed'))
    #     self.first_file_epoch_time = self.get_timestamp(self.source_files[0].parts[-1])
    #     self.last_file_epoch_time = self.get_timestamp(self.source_files[-1].parts[-1])
    #     return


    # def uncompress_radar_files(self):
    #     """
    #     example command line: python debz.py KBRO20170825_195747_V06 KBRO20170825_195747_V06.uncompressed
    #     """

    #     os.chdir(self.munge_dir)
    #     self.source_files = list(self.source_directory.glob('*V06'))
    #     for original_file in self.source_files:
    #         command_string = f'python debz.py {str(original_file)} {str(original_file)}.uncompressed'
    #         os.system(command_string)
    #     print("uncompress complete!")
    #     return
    
        # def stage_files_to_munge(self):
        # """
        # stages raw files into munge directory where munger script lives
        # """
        # cp_cmd = f'cp {self.radar_site_download_dir}/* {self.munge_dir}'
        # os.system(cp_cmd)
        
        # return
# def handle_script_progress(n_clicks):
#     try:
#         if n_clicks > 0:
#             print("Running script...")
#             sa.make_simulation_times()
#             sa.create_radar_dict()

#             for key in sa.radar_dict.keys():
#                 try:
#                     asos_one = sa.radar_dict[key]['asos_one']
#                     asos_two = sa.radar_dict[key]['asos_two']
#                     radar = key
#                     print(asos_one, asos_two, radar)
#                 except KeyError as e:
#                     print("Error getting radar metadata: ", e)
#                 except Exception as e:
#                     print("Error: ", e)

#                 try:
#                     NexradDownloader(radar, sa.timestring, str(sa.duration))
#                 except Exception as e:
#                     print("Error running nexrad script: ", e)

#                 try:
#                     run_hodo_script([radar, BASE_DIR, asos_one, asos_two])
#                 except Exception as e:
#                     print("Error running hodograph script: ", e)
#     except Exception as e:
#         print("Error: ", e)



# def handle_script_progress(n_clicks):
#     try:
#         if n_clicks > 0:
#             print("Running script...")
#             sa.make_simulation_times()
#             sa.create_radar_dict()

#             for key in sa.radar_dict.keys():
#                 try:
#                     asos_one = sa.radar_dict[key]['asos_one']
#                     asos_two = sa.radar_dict[key]['asos_two']
#                     radar = key
#                     print(asos_one, asos_two, radar)
#                 except KeyError as e:
#                     print("Error getting radar metadata: ", e)
#                 except Exception as e:
#                     print("Error: ", e)

#                 try:
#                     NexradDownloader(radar, sa.timestring, str(sa.duration))
#                 except Exception as e:
#                     print("Error running nexrad script: ", e)

#                 try:
#                     run_hodo_script([radar, BASE_DIR, asos_one, asos_two])
#                 except Exception as e:
#                     print("Error: ", e)