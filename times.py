from datetime import datetime, timedelta 
import pytz

def date_time_string(dt) -> str:
    """
    Converts a datetime object to a string.
    """
    return datetime.strftime(dt, "%Y-%m-%d %H:%M")


def make_simulation_times(event_start_time, event_duration) -> dict:
    """
    playback_end_time: datetime object
        - set to current time then rounded down to nearest 15 min.
    playback_start_time: datetime object
        - the time the simulation starts.
        - playback_end_time minus the event duration
        - This is "recent enough" for GR2Analyst to poll data
    playback_timer: datetime object
        - the "current" displaced realtime during the playback
    event_start_time: datetime object
        - the historical time the actual event started.
        - based on user inputs of the event start time
    simulation_time_shift: timedelta object
        the difference between the playback start time and the event start time
    simulation_seconds_shift: int
        the difference between the playback start time and the event start time in seconds

    Variables ending with "_str" are the string representations of the datetime objects
    """

    now = datetime.now(pytz.utc).replace(second=0, microsecond=0)
    playback_end = now - timedelta(minutes=now.minute % 15)
    playback_end_str = date_time_string(playback_end)

    playback_start = playback_end - timedelta(minutes=event_duration)
    playback_start_str = date_time_string(playback_start)

    # The playback clock is set to 10 minutes after the start of the simulation
    playback_clock = playback_start + timedelta(seconds=600)
    playback_clock_str = date_time_string(playback_clock)

    # a timedelta object is not JSON serializable, so cannot be included in the output
    # dictionary stored in the dcc.Store object. All references to simulation_time_shift
    # will need to use the simulation_seconds_shift reference instead.
    simulation_time_shift = playback_start - event_start_time
    simulation_seconds_shift = round(simulation_time_shift.total_seconds())
    event_start_str = date_time_string(event_start_time)
    increment_list = []
    for t in range(0, int(event_duration/5) + 1, 1):
        new_time = playback_start + timedelta(seconds=t*300)
        new_time_str = date_time_string(new_time)
        increment_list.append(new_time_str)

    playback_dropdown_dict = [
        {'label': increment, 'value': increment} for increment in increment_list]

    sim_times = {
        'event_start_str': event_start_str,
        'simulation_seconds_shift': simulation_seconds_shift,
        'playback_start_str': playback_start_str,
        'playback_start': playback_start,
        'playback_end_str': playback_end_str,
        'playback_end': playback_end,
        'playback_clock_str': playback_clock_str,
        'playback_clock': playback_clock,
        'playback_dropdown_dict': playback_dropdown_dict,
        'event_duration': event_duration
    }

    return sim_times