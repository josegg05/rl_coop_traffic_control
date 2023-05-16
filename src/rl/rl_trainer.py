
import imp
import gym
import numpy as np
import time
import torch
import os
from collections import OrderedDict

from src.infrastructure import pytorch_utils as ptu
from src.infrastructure.logger import Logger
from src.rl.ppo_agent import PPOAgent
from src.rl import utils
from src.environment.vec_env import make_vec_env


class RL_Trainer(object): 

    def __init__(self, params) -> None:
        
        ############
        ## Init
        ############

        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set seeds 
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        ############
        ## ENV
        ############

        # Make the gym environment
        print(f"ENVIROMENT seed = {seed}")
        self.envs = make_vec_env(
            env_params=params['env_params'],
            num_cpu=params['num_cpu'],
            seed=seed
        )
        time.sleep(params['num_cpu']*3)
        # self.env used to initialize the agent
        self.env = gym.make('SumoEnv-v0', **params['env_params'])

        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2
        self.params['agent_params']['discrete'] = discrete
        # Observation and action sizes
        self.params['agent_params']['ob_space'] = self.env.observation_space
        self.params['agent_params']['ac_space'] = self.env.action_space
        self.params['agent_params']['ob_dim'] = self.env.observation_space.shape
        self.params['agent_params']['ac_dim'] = self.env.action_space.shape[0]
        self.params['agent_params']['n_agents'] = 9
        self.params['agent_params']['tls_masks'] = self.env.tls_mask

        # max path len
        # ep_len correspond to the seconds to simulate sumo
        self.max_path_len = self.params['ep_len'] // self.env.cycle_time

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        else:
            self.fps = 10

        self.env.close()

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params['agent_params'])

    def run_training_loop(self, n_iter, agent):
        """
        param n_iter: number of iterations
        param ac: Actor Critc MLP or GNN
        """
        
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        all_logs = []
        self.max_rews = agent.max_rews
        self.min_epoch = agent.itr
        self.min_wt_epoch = agent.itr
        self.min_wt = agent.min_wt

        for itr in range(agent.itr, n_iter): 
            print(f"\n\n********** Iteration {itr} ************")

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                agent, self.params['steps_per_epoch'] 
            )
            # DONE: update paths and envsteps
            # add collected data to replay buffer
            # DONE: implement replay buffer

            # train agent (using sampled data from replay buffer)
            train_logs, ac_model_state_dict, pi_optimizer_state_dict, vf_optimizer_state_dict = self.train_agent(agent)
            all_logs.append(train_logs)

            # log/save
            print('\nBeginning logging procedure...')
            self.perform_logging(itr, agent, all_logs, ac_model_state_dict, pi_optimizer_state_dict, vf_optimizer_state_dict)

            # TODO: save model
        torch.save({'epoch': itr,
                    'model_state_dict': ac_model_state_dict,
                    'pi_optimizer_state_dict': pi_optimizer_state_dict,
                    'vf_optimizer_state_dict': vf_optimizer_state_dict,
                    'reward': self.max_rews,
                    'waiting_time': self.min_wt}, f"{self.params['ac_out_model_folder']}/GNN_RL_last.pt")
        print('*********************************\n FINISH rl_trainer.py\n*********************************')

    def collect_training_trajectories(self, agent, batch_size): 
        """
        :param itr:
        :param actor_critic:  the current actor critic using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        
        print("\nCollecting data to be used for training...")
        ep_rews, ep_lens = utils.sample_trajectories(
            env=self.envs, 
            agent=agent,
            min_timesteps_per_batch=self.params['steps_per_epoch'] // self.params['num_cpu'],
            max_path_length=self.max_path_len
        )
        self.total_envsteps += self.params['steps_per_epoch']
        return [ep_rews, ep_lens]

    def train_agent(self, agent): 
        return agent.train()

    def perform_logging(self, itr, agent, all_logs, ac_model_state_dict, pi_optimizer_state_dict, vf_optimizer_state_dict): 
        
        last_log = all_logs[-1]

        #######################
        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        ntraj = (self.params['eval_ntraj'] // self.params['num_cpu']) + 1
        ep_rews, ep_lens = utils.sample_n_eval_trajectories(
            env=self.envs, 
            agent=agent, 
            ntraj=ntraj,
            max_path_length=self.max_path_len,
        )
        ep_rews_norm = [ep_rews[i] / ep_lens[i] for i in range(len(ep_lens))]

        #######################
        # Load eval pareameters
        teleports = 0
        for filename in os.scandir(self.params['ac_out_model_folder']):
            if f"eval_mf{self.params['mult_factor']}" in filename.name:
                with open(filename) as f:
                    content = f.readlines()
                    ready_to_load = False
                    for line_idx in range(len(content)):
                        if content[line_idx][:24] == "Simulation ended at time":
                            ending_time = int(content[line_idx][26:-4])
                        if content[line_idx][0:8] == "Vehicles":
                            inserted_veh = int(content[line_idx+1][11:15])
                            running_veh = int(content[line_idx+2][10:-1])
                            waiting_veh = int(content[line_idx+3][10:-1])
                        if content[line_idx][0:9] == "Teleports":
                            teleports = int(content[line_idx][11:14])
                        if content[line_idx][0:10] == "Statistics":
                            ready_to_load = True
                        if "WaitingTime" in content[line_idx] and ready_to_load:
                            waiting_time = float(content[line_idx][14:-1])
                            time_loss  = float(content[line_idx+1][11:-1])
                            ready_to_load = False
                        if "DepartDelay" in content[line_idx]:
                            depart_delay = float(content[line_idx][14:-1])
                            break

        #######################
        # save best waiting time model
        if (waiting_time < self.min_wt) and (waiting_veh == 0):
            self.min_wt = waiting_time
            self.min_wt_epoch = itr
            torch.save({'epoch': itr,
                        'model_state_dict': ac_model_state_dict,
                        'pi_optimizer_state_dict': pi_optimizer_state_dict,
                        'vf_optimizer_state_dict': vf_optimizer_state_dict,
                        'reward': np.stack(ep_rews_norm).mean(),
                        'waiting_time': self.min_wt}, f"{self.params['ac_out_model_folder']}/GNN_RL_best_wt.pt")

        #######################
        # save best reward model
        if np.stack(ep_rews_norm).mean() > self.max_rews:
            self.max_rews = np.stack(ep_rews_norm).mean()
            self.min_epoch = itr
            print('Saving best model\n')
            torch.save({'epoch': itr,
                        'model_state_dict': ac_model_state_dict,
                        'pi_optimizer_state_dict': pi_optimizer_state_dict,
                        'vf_optimizer_state_dict': vf_optimizer_state_dict,
                        'reward': self.max_rews,
                        'waiting_time': waiting_time}, f"{self.params['ac_out_model_folder']}/GNN_RL_iter{itr}_rew_{self.max_rews}.pt")
            torch.save({'epoch': itr,
                        'model_state_dict': ac_model_state_dict,
                        'pi_optimizer_state_dict': pi_optimizer_state_dict,
                        'vf_optimizer_state_dict': vf_optimizer_state_dict,
                        'reward': self.max_rews,
                        'waiting_time': waiting_time}, f"{self.params['ac_out_model_folder']}/GNN_RL_best.pt")
        print(f'best_model_epoch = {self.min_epoch}, max_rew = {self.max_rews}')
        print(f'best_wt_model_epoch = {self.min_wt_epoch}, min_wt = {self.min_wt}')

        #######################
        # save last model
        torch.save({'epoch': itr,
                    'model_state_dict': ac_model_state_dict,
                    'pi_optimizer_state_dict': pi_optimizer_state_dict,
                    'vf_optimizer_state_dict': vf_optimizer_state_dict,
                    'reward': np.stack(ep_rews_norm).mean(),
                    'waiting_time': waiting_time}, f"models/opt/ac_last_model/GNN_RL_last.pt")
        

        #######################
        # save eval metrics
        if self.logmetrics:
            # returns, for logging

            eval_returns = np.stack(ep_rews_norm)
            eval_ep_len = np.stack(ep_lens)

            # decide what to log
            logs = OrderedDict()
            logs["Ending Time"] = ending_time
            logs["Inserted_Vehicles"] = inserted_veh
            logs["Running_Vehicles"] = running_veh
            logs["Waiting_Vehicles"] = waiting_veh
            logs["Teleports"] = teleports

            logs["Waiting_Time"] = waiting_time
            logs["Time_Loss"] = time_loss
            logs["Depart_Delay"] = depart_delay

            logs["Eval_AverageReturn"] = eval_returns.mean()
            logs["Eval_StdReturn"] = eval_returns.std()
            logs["Eval_MaxReturn"] = eval_returns.max()
            logs["Eval_MinReturn"] = eval_returns.min()
            logs["Eval_AverageEpLen"] = eval_ep_len.mean()
            logs['Eval_simulations'] = ntraj * self.params['num_cpu']

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["Train_Iterations"] = self.total_envsteps * self.params['agent_params']['train_steps_per_iter']
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()    

