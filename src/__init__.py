from sys import argv
from gym.envs.registration import register
from typing import Optional, Union, Tuple

register(
    id='SumoEnv-v0', 
    entry_point='src.environment:SumoEnvironment',
    kwargs={
        'sumocfg': str,
        'sumonet': str,
        'out_csv_name': None,
        'use_gui': False, 
        'virtual_display': None,
        'begin_time': 0, 
        'num_seconds': 20000, 
        'max_depart_delay': 100000,
        'time_to_teleport': -1, 
        'delta_time': 5, 
        'yellow_time': 2, 
        'min_green': 5, 
        'max_green': 50, 
        'single_agent': False, 
        'sumo_seed': 'random', 
        'fixed_ts': False,
        'rank': 55,
        'sumo_warnings': False,
        'pred_params': {}, 
    }
)