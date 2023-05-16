import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple

from src.encoder.gnn import GNN
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd

from src.environment.traffic_signal import TrafficSignal
import src.infrastructure.pytorch_utils as ptu
from src.dataloader.dataloader import load_graph
import subprocess


LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


class SumoEnvironment(gym.Env):
    """
        SUMO Environment for Traffic Signal Control

        :param net_file: (str) SUMO .net.xml file
        :param route_file: (str) SUMO .rou.xml file
        :param out_csv_name: (Optional[str]) name of the .csv output with simulation results. If None no output is generated
        :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
        :param virtual_display: (Optional[Tuple[int,int]]) Resolution of a virtual display for rendering
        :param begin_time: (int) The time step (in seconds) the simulation starts
        :param num_seconds: (int) Number of simulated seconds on SUMO. The time in seconds the simulation must end.
        :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
        :param delta_time: (int) Simulation seconds between actions
        :param min_green: (int) Minimum green time in a phase
        :param max_green: (int) Max green time in a phase
        :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
        :sumo_seed: (int/string) Random seed for sumo. If 'random' it uses a randomly chosen seed.
        :fixed_ts: (bool) If true, it will follow the phase configuration in the route_file and ignore the actions.
        :sumo_warnings: (bool) If False, remove SUMO warnings in the terminal
    """
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self, 
        sumocfg: str,
        sumonet: str,
        out_csv_name: Optional[str] = None, 
        use_gui: bool = False, 
        virtual_display: Optional[Tuple[int,int]] = None,
        begin_time: int = 0, 
        num_seconds: int = 20000, 
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1, 
        delta_time: int = 5,
        cycle_time: int = 60,  # 120,
        yellow_time: int = 3,
        min_green: int = 5, 
        max_green: int = 50, 
        single_agent: bool = False, 
        sumo_seed: Union[str,int] = 'random', 
        rank: int = 0,
        seed: int = 0,
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        pred_params: dict = {},
        training: bool = True,
        test_log_folder: Optional[str] = "trash",
        eval_log_folder: Optional[str] = "",
        device: str = "cpu",
        mult_factor: float = 0.7,
    ):
        self._sumocfg = sumocfg
        self._sumonet= sumonet
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.virtual_display = virtual_display

        delta_time = cycle_time
        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.cycle_time = cycle_time
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.sumo_seed = 5 # sumo_seed
        self.rank = rank
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.pred_params = pred_params
        self.training = training
        self.test_log_folder = test_log_folder
        self.eval_log_folder = eval_log_folder
        self.eval = False
        # ptu.device = device
        # self.label = str(SumoEnvironment.CONNECTION_LABEL)
        self.label = str(self.rank)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None
        np.random.seed(self.rank+(8*seed))
        print(f"----- Env {self.rank} created with seed {(self.rank+(8*seed))} and cycle {self.cycle_time}")
        print(f"---------------Init rank {self.rank}---------------")
        if self.training:
            self.mult_factor = mult_factor
            self._gen_random_routes()            
        else:
            self.max_test_count = 100 # before: 10 to have 10 test per mult_factor
            self.mult_factor = mult_factor
            self.test_count = 0
            self._load_test_route()
            self.test_count = 0

        if LIBSUMO:
            traci.start(
                [sumolib.checkBinary('sumo'), '-c', self._sumocfg, "--no-step-log", "true", "--no-warnings", "true"], 
                verbose=False, 
                stdout=open(os.devnull, "w"),
                label='init_connection'+self.label
            )  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start(
                [sumolib.checkBinary('sumo'), '-c', self._sumocfg, "--no-step-log", "true", "--no-warnings", "true"], 
                label='init_connection'+self.label, 
                verbose=False, 
                stdout=open(os.devnull, "w")
            )
            conn = traci.getConnection('init_connection'+self.label)
            print(f"'init_connection'+{self.label}")
        self.ts_ids = list(conn.trafficlight.getIDList())
        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time,
                                                  self.cycle_time,
                                                  self.yellow_time,
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  conn) for ts in self.ts_ids}
        self.tls_mask = {ts: self.traffic_signals[ts].model_phases_idx for ts in self.ts_ids}
        # Coorect way to close the connection
        traci.switch('init_connection'+self.label)
        traci.close()
        # conn.close()
        # self.close()  # no hace nada porque sumo = None
        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = EnvSpec('SUMORL-v0')
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}

        self.predictor = pred_params['loaded_predictor']
        self.g = pred_params['loaded_graph']

        self.pred_buf = np.zeros((9, 8, 12))
        self.pred_buf_dict = {ts: None for ts in self.ts_ids}

    
    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                     '-c', self._sumocfg,
                    '--max-depart-delay', str(self.max_depart_delay),
                     '--waiting-time-memory', str(self.sim_max_time),  # 10000
                     '--time-to-teleport', str(self.time_to_teleport),
                     "--no-step-log", "true", 
                     # "--duration-log.disable", 
                     "--no-warnings", "true",
                     "--num-clients", str(1),
                    ]

        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self.virtual_display is not None:
                sumo_cmd.extend(['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")
        if not self.training:
            sumo_cmd.extend([#"--summary", f"{self.test_log_folder}/summary_{(self.mult_factor - 0.1)}.xml",
                     #"--emission-output", f"{self.test_log_folder}/emission_{(self.mult_factor - 0.1)}.xml",
                     "--duration-log.statistics",
                     "--verbose",
                     "--log", f"{self.test_log_folder}/verbose_{self.mult_factor}_{self.test_count}.xml",])
        elif self.eval:
            sumo_cmd.extend(["--duration-log.statistics", 
                     "--verbose",
                     "--log", f"{self.eval_log_folder}/eval_mf{self.mult_factor}_log.xml",])
        
        traci_port = sumolib.miscutils.getFreeSocketPort()
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(
                sumo_cmd, 
                label=self.label, 
                port=traci_port,
                numRetries=240,
                verbose=False, 
                stdout=open(os.devnull, "w")
            )
            self.sumo = traci.getConnection(self.label)
        
        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def _gen_random_routes(self):
        mult_factor = self.mult_factor #np.random.normal(loc=0.9, scale=0.1)
        seed = "5" #str(np.random.randint(1000000000))
        # return_code = subprocess.check_output(["./src/data/sumo_optimal/optimal_routes_constructor.sh", mult_factor, seed])

        N_HOURS = 1
        DEPART_DURATION = 3600 * N_HOURS
        MAX_N_BIKE = 400 * N_HOURS
        MAX_N_MOTO = 360 * N_HOURS
        MAX_N_VEH = 1000 * N_HOURS
        MAX_N_BUS = 240 * N_HOURS
        MAX_N_TRUCK = 200 * N_HOURS

        vehicle_dict = {'bicycle':
                            {'prefix': 'bike', 'fringe_factor': 2, 'min_dist': 50,
                             'max_dist': 800, 'veh_p': DEPART_DURATION / (MAX_N_BIKE * mult_factor)},
                        'motorcycle':
                            {'prefix': 'moto', 'fringe_factor': 2, 'min_dist': 100, 'max_dist': 800,
                             'veh_p': DEPART_DURATION / (MAX_N_MOTO * mult_factor)},
                        'passenger':
                            {'prefix': 'veh', 'fringe_factor': 5, 'min_dist': 200, 'max_dist': 0,
                             'veh_p': DEPART_DURATION / (MAX_N_VEH * mult_factor)},
                        'bus':
                            {'prefix': 'bus', 'fringe_factor': 5, 'min_dist': 400, 'max_dist': 0,
                             'veh_p': DEPART_DURATION / (MAX_N_BUS * mult_factor)},
                        'truck':
                            {'prefix': 'truck', 'fringe_factor': 5, 'min_dist': 400, 'max_dist': 0,
                             'veh_p': DEPART_DURATION / (MAX_N_TRUCK * mult_factor)}
                        }

        for veh_name in vehicle_dict:
            subprocess_command = [sys.executable, f"{tools}/randomTrips.py",
                                  '-n', 'src/data/sumo_optimal/optimal.net.xml',
                                  '--seed', seed,
                                  '--fringe-factor', f'{vehicle_dict[veh_name]["fringe_factor"]}',
                                  '-p', str(vehicle_dict[veh_name]["veh_p"]),
                                  '-o', f'src/data/sumo_optimal/optimal.{veh_name}_{self.rank}.trips.xml',
                                  '-e', str(DEPART_DURATION),
                                  '-r', f'src/data/sumo_optimal/optimal.{veh_name}_{self.rank}.rou.xml',
                                  '--vehicle-class', f'{veh_name}',
                                  '--vclass', f'{veh_name}',
                                  '--prefix', f'{vehicle_dict[veh_name]["prefix"]}',
                                  #'--min-distance', f'{vehicle_dict[veh_name]["min_dist"]}',
                                  #'--max-distance', f'{vehicle_dict[veh_name]["max_dist"]}',
                                  '--fringe-start-attributes', "departSpeed='max'",
                                  '--trip-attributes', "departLane='best'",
                                  '--validate', '--remove-loops']
            if veh_name == 'passenger':
                subprocess_command.append('--lanes')
            # subprocess.run(subprocess_command)
            print(f"Generating routes: rank {self.rank}, seed: {seed}, mult_factor {mult_factor}, vehicle {veh_name}")
            return_code=subprocess.check_output(subprocess_command, timeout=10)
            # print(f"Routes Generated:   rank {self.rank}, seed: {seed}, mult_factor {mult_factor}, vehicle {veh_name}")
        return

    def _load_test_route(self):
        # mult_factor = self.mult_factor
        # if self.test_count == self.max_test_count:
        #     self.test_count = 1
        #     self.mult_factor = self.mult_factor + 0.1
        # else:
        #     self.test_count += 1
        # seed = str(int(22552255 -1 + self.test_count))
        mult_factor = self.mult_factor # np.random.normal(loc=0.9, scale=0.01) 
        if self.test_count == self.max_test_count:
            self.test_count = 1
        else:
            self.test_count += 1  
        seed = "5" #str(np.random.randint(1000000000))
        # return_code = subprocess.check_output(["./src/data/sumo_optimal/optimal_routes_constructor.sh", mult_factor, seed])

        N_HOURS = 1
        DEPART_DURATION = 3600 * N_HOURS
        MAX_N_BIKE = 400 * N_HOURS
        MAX_N_MOTO = 360 * N_HOURS
        MAX_N_VEH = 1000 * N_HOURS
        MAX_N_BUS = 240 * N_HOURS
        MAX_N_TRUCK = 200 * N_HOURS

        vehicle_dict = {'bicycle':
                            {'prefix': 'bike', 'fringe_factor': 2, 'min_dist': 50,
                             'max_dist': 800, 'veh_p': DEPART_DURATION / (MAX_N_BIKE * mult_factor)},
                        'motorcycle':
                            {'prefix': 'moto', 'fringe_factor': 2, 'min_dist': 100, 'max_dist': 800,
                             'veh_p': DEPART_DURATION / (MAX_N_MOTO * mult_factor)},
                        'passenger':
                            {'prefix': 'veh', 'fringe_factor': 5, 'min_dist': 200, 'max_dist': 0,
                             'veh_p': DEPART_DURATION / (MAX_N_VEH * mult_factor)},
                        'bus':
                            {'prefix': 'bus', 'fringe_factor': 5, 'min_dist': 400, 'max_dist': 0,
                             'veh_p': DEPART_DURATION / (MAX_N_BUS * mult_factor)},
                        'truck':
                            {'prefix': 'truck', 'fringe_factor': 5, 'min_dist': 400, 'max_dist': 0,
                             'veh_p': DEPART_DURATION / (MAX_N_TRUCK * mult_factor)}
                        }

        for veh_name in vehicle_dict:
            subprocess_command = [sys.executable, f"{tools}/randomTrips.py",
                                  '-n', 'src/data/sumo_optimal/optimal.net.xml',
                                  '--seed', seed,
                                  '--fringe-factor', f'{vehicle_dict[veh_name]["fringe_factor"]}',
                                  '-p', str(vehicle_dict[veh_name]["veh_p"]),
                                  '-o', f'src/data/sumo_optimal/optimal.{veh_name}_{self.rank}.trips.xml',
                                  '-e', str(DEPART_DURATION),
                                  '-r', f'src/data/sumo_optimal/optimal.{veh_name}_{self.rank}.rou.xml',
                                  '--vehicle-class', f'{veh_name}',
                                  '--vclass', f'{veh_name}',
                                  '--prefix', f'{vehicle_dict[veh_name]["prefix"]}',
                                  #'--min-distance', f'{vehicle_dict[veh_name]["min_dist"]}',
                                  #'--max-distance', f'{vehicle_dict[veh_name]["max_dist"]}',
                                  '--fringe-start-attributes', "departSpeed='max'",
                                  '--trip-attributes', "departLane='best'",
                                  '--validate', '--remove-loops']
            if veh_name == 'passenger':
                subprocess_command.append('--lanes')
            # subprocess.run(subprocess_command)
            print(f"Generating routes: rank {self.rank}, seed: {seed}, mult_factor {mult_factor}, vehicle {veh_name}")
            return_code=subprocess.check_output(subprocess_command, timeout=10)
            # print(f"Routes Generated:   rank {self.rank}, seed: {seed}, mult_factor {mult_factor}, vehicle {veh_name}")
        return

    def reset(self):
        if self.run != 0:
            self.run = 0
            obs = self._compute_observations()
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        else:
            # print(f"---------------Reset rank {self.rank}---------------")
            self.run = 1
            if self.training:
                #self._gen_random_routes()
                pass
            else:
                #self._load_test_route()
                if self.test_count == self.max_test_count:
                    self.test_count = 1
                else:
                    self.test_count += 1  
            self._start_simulation()
            self.traffic_signals = {ts: TrafficSignal(self, 
                                                    ts, 
                                                    self.delta_time,
                                                    self.cycle_time,
                                                    self.yellow_time, 
                                                    self.min_green, 
                                                    self.max_green, 
                                                    self.begin_time,
                                                    self.sumo) for ts in self.ts_ids}
            self.pred_buf_dict = {ts: None for ts in self.ts_ids}  # reset pred_buf
            self._compute_predictor()
            self.vehicles = dict()
            obs = self._compute_observations()

        if self.single_agent:
            return obs[self.ts_ids[0]]
        else:
            return obs

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    def step(self, action):
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].next_action_time = self.sim_step  # activate time to act
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], dones['__all__'], {}
        else:
            return observations, rewards, dones['__all__'], {}

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            if self.sim_step % 300 == 0: 
                # Compute h
                self._compute_predictor()

            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True
            
            tsa = 'intersection/0002/tls'
            # print(f'Sim Step: {self.sim_step}, Phase 0002: {self.sumo.trafficlight.getPhase(tsa)}, Duration: {self.sumo.trafficlight.getPhaseDuration(tsa)}, Next_switch: {self.sumo.trafficlight.getNextSwitch(tsa)}')
            # print(f'Sim Step: {self.sim_step}, Time to Act: {self.traffic_signals[tsa].time_to_act}')
            # print(f'Sim Step: {self.sim_step}, Rank: {self.rank}')

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_tls_splits(actions)
        else:
            for ts, action in actions.items():
                #print(f'ts: {ts}')
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_tls_splits(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        if (not self.training) or self.eval:
            if (self.sim_step >= self.sim_max_time) or (traci.simulation.getMinExpectedNumber() == 0):
                dones['__all__'] = True
            else:
                dones['__all__'] = False
        else:
            dones['__all__'] = self.sim_step >= self.sim_max_time
            
        return dones
    
    def _compute_info(self):
        info = self._compute_step_info()
        self.metrics.append(info)

    def _compute_observations(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        obs = {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
        return {ts: np.concatenate([obs[ts], self.h[i]], -1) for i, ts in enumerate(obs.keys())}
        # return {
        #     'obs': obs, 
        #     'h': self.h
        # }

    def _compute_predictor(self): 
        # aca computar la gnn con un buffer de observaciones de 5 min 
        # shape salida 
        # (intersections, phases, features)
        # (9, 8, gnn output)
        self._compute_prediction_buffer()
        self.h = self.predictor.pred_step(self.g, self.pred_buf)

    def _compute_prediction_buffer(self): 
        # Calls TLS and then concatenate its own buffers.
        self.pred_buf_dict.update({ts: self.traffic_signals[ts].compute_predicton_buffer() for ts in self.ts_ids})
        self.pred_buf = np.array([self.pred_buf_dict[ts] for ts in self.ts_ids])

    def _compute_rewards(self):
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        # (num_detectors, features)
        shape = (
            self.traffic_signals[self.ts_ids[0]].num_detectors,
            self.traffic_signals[self.ts_ids[0]].num_obs + self.pred_params['hidden_size'] * 2
        )
        return spaces.Box(
            low=np.zeros(shape, dtype=np.float32),
            high=np.ones(shape, dtype=np.float32)
        )
        # return self.traffic_signals[self.ts_ids[0]].observation_space
    
    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': self.traffic_signals[self.ts_ids[0]].last_reward,
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(sum(self.traffic_signals[ts].get_waiting_time_per_detector()) for ts in self.ts_ids)
        }

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        self.sumo = None
        print(f"SUMO {self.rank} closed")

    def __del__(self):
        self.close()
    
    def render(self, mode='human'):
        if self.virtual_display:
            #img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg", 
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            if mode == 'rgb_array':
                return np.array(img)
            return img         
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_conn{}_run{}'.format(self.label, run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)

