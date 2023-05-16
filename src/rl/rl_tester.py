import gym
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
from collections import OrderedDict

from src.infrastructure import pytorch_utils as ptu
from src.infrastructure.logger import Logger
from src.rl.ppo_agent import PPOAgent
from src.rl import utils
from src.environment.vec_env import make_vec_env


class RL_Tester(object): 

    def __init__(self, params) -> None:
        
        ############
        ## Init
        ############

        self.params = params
        self.logger = Logger(self.params['logdir'])

        ############
        ## ENV
        ############

        # Make the gym environment
        self.envs = make_vec_env(
            env_params=params['env_params'],
            num_cpu=1,
        )
        # self.gym_env used to initialize the agent
        self.gym_env = gym.make('SumoEnv-v0', **params['env_params'])

        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.gym_env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.gym_env.observation_space.shape) > 2
        self.params['agent_params']['discrete'] = discrete
        # Observation and action sizes
        self.params['agent_params']['ob_space'] = self.gym_env.observation_space
        self.params['agent_params']['ac_space'] = self.gym_env.action_space
        self.params['agent_params']['ob_dim'] = self.gym_env.observation_space.shape
        self.params['agent_params']['ac_dim'] = self.gym_env.action_space.shape[0]
        self.params['agent_params']['n_agents'] = 9
        self.params['agent_params']['tls_masks'] = self.gym_env.tls_mask

        # max path len
        # ep_len correspond to the seconds to simulate sumo
        self.max_path_len = self.params['ep_len'] // self.gym_env.cycle_time

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.gym_env):
            self.fps = 1/self.gym_env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        else:
            self.fps = 10

        self.gym_env.close()

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params['agent_params'])

    def run_testing(self, agent, n_scenarios):
        """
        param ac: Actor Critc MLP or GNN
        """
        
        # init vars at beginning of testing
        self.start_time = time.time()
        # collect test trajectories, for logging
        print("\nCollecting data for test...")
        rews, lens, act_ts = utils.sample_test_trajectory(
            env=self.envs, 
            agent=agent, 
            ntraj=n_scenarios,   # 5 testing scenarios managed by env.py
            max_path_length=self.max_path_len,
        )

        # log/save
        print('\nBeginning logging procedure...')
        # self.perform_logging(rews, lens)
        self.plot_action_time_hist(act_ts, lens)
        print(f"*********************************\n FINISH rl_tester_l{self.params['agent_params']['n_layers']}_hs{self.params['agent_params']['size']}_gnl{self.params['agent_params']['gnn_n_layers']}_ghs{self.params['agent_params']['gnn_size']}.py\n*********************************")


    def perform_logging(self, rews, lens):         
        # save test metrics
        test_returns = np.stack(rews)
        test_ep_len = np.stack(lens)

        for scenario in range(len(test_returns)):
            logs = OrderedDict()
            logs["Test_Return"] = test_returns[scenario]
            logs["Test_EpLen"] = test_ep_len[scenario]
            # TODO: Agregar variables a loguear

            logs["TimeSinceStart"] = time.time() - self.start_time

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, scenario)
            print('Done logging...\n\n')

            self.logger.flush()   


    def is_outlier(self, points, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return modified_z_score > thresh


    def plot_action_time_hist(self, act_ts, lens):
        act_ts_np = np.array([x for act_t in act_ts for x in act_t], dtype=np.float64)
        np.save(f"{self.params['env_params']['test_log_folder']}/action_time_nl{self.params['agent_params']['n_layers']}_hs{self.params['agent_params']['size']}_gnl{self.params['agent_params']['gnn_n_layers']}_ghs{self.params['agent_params']['gnn_size']}.npy", act_ts_np, allow_pickle=True)
        # load saved act_ts_np
        # act_ts_np = np.load(f"{self.params['env_params']['test_log_folder']}/action_time.npy", allow_pickle=True)
        act_ts_ms = 1000*act_ts_np
        act_ts_ms_filt = act_ts_ms[~self.is_outlier(act_ts_ms, 20)]
        plt.hist(act_ts_ms_filt, bins = int(len(act_ts_ms_filt)/10), range = [act_ts_ms_filt.min(), act_ts_ms_filt.max()]) 
        plt.title("Action time histogram (ms)")
        image_format = 'svg' # e.g .png, .svg, etc.
        image_name = f"{self.params['env_params']['test_log_folder']}/act_time_hist_nl{self.params['agent_params']['n_layers']}_hs{self.params['agent_params']['size']}_gnl{self.params['agent_params']['gnn_n_layers']}_ghs{self.params['agent_params']['gnn_size']}.svg"
        plt.savefig(image_name, format=image_format, dpi=1200)
        # plt.show()

        print("######################## Agent PARAMETERS ##########################")
        with open(f"{self.params['env_params']['test_log_folder']}/model_params_nl{self.params['agent_params']['n_layers']}_hs{self.params['agent_params']['size']}_gnl{self.params['agent_params']['gnn_n_layers']}_ghs{self.params['agent_params']['gnn_size']}.txt", 'w') as f:
            print(f"Total de parámetros = {sum(p.numel() for p in self.agent.ac.pi.parameters())}")
            f.write(f"Total de parámetros = {sum(p.numel() for p in self.agent.ac.pi.parameters())}")



