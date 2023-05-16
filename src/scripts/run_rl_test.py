from email.policy import default
import os
import time 
from src.rl.ppo_agent import PPOAgent
from src.rl.rl_tester import RL_Tester
from src.encoder.gnn import GNN
from src.dataloader.dataloader import load_graph
import src.infrastructure.pytorch_utils as ptu 
import torch

class PPO_Tester(object): 
    
    def __init__(self, params) -> None:
        
        self.params = params

        # Init Device
        ptu.init_gpu(
            use_gpu = not self.params['not_gpu'],
            gpu_id = self.params['which_gpu']
        )

        #####################
        ## SET AGENT PARAMS
        #####################
        
        agent_params = dict(
            gamma=params['gamma'],
            gae_lambda=params['gae_lambda'],
            size=params['size'],
            n_layers=params['n_layers'],
            gnn_size=params['gnn_size'],
            gnn_n_layers=params['gnn_n_layers'],
            steps_per_epoch=params['steps_per_epoch'],
            train_steps_per_iter=params['train_steps_per_iter'],
            clip_ratio=params['clip_ratio'],
            pi_lr=params['pi_lr'],
            vf_lr=params['vf_lr'],
            graph_path=params['graph_path'],
            num_cpu=params['num_cpu'],
            ac_in_model_path=params['ac_in_model_path'],
            ac_in_model_type=params['ac_in_model_type'],
            retrain_reset=params['retrain_reset']
        )

        #####################
        ## SET ENV PARAMS
        #####################

        pred_params = dict(
            graph_path=params['graph_path'],
            in_feat=12, 
            out_feat=12, 
            n_layers=2, 
            hidden_size=128, 
            adaptadj=True, 
            num_nodes=72,
        )

        self.predictor = GNN(
            in_feat=pred_params['in_feat'], 
            out_feat=pred_params['out_feat'],
            n_layers=pred_params['n_layers'], 
            hidden_size=pred_params['hidden_size'],
            adaptadj=pred_params['adaptadj'], 
            num_nodes=pred_params['num_nodes']
        )

        # Load Predictor GNN model
        if params['pred_model_path']:
            print(f"Prediction model {params['pred_model_path']} loaded")
            # comment "map_location=torch.device(ptu.device)))" if you are using GPU and pops an error
            self.predictor.load_state_dict(torch.load(params['pred_model_path'], map_location=torch.device(ptu.device)))
        else: 
            print('\nWARNING: PREDICTION NETOWRK IS RANDOMLY INITIALIZED\n')
        self.g = load_graph(pred_params['graph_path'])
        self.predictor = self.predictor.to("cpu")  #ptu.device)
        self.g = self.g.to("cpu")  #ptu.device)

        pred_params['loaded_predictor'] = self.predictor
        pred_params['loaded_graph'] = self.g
        
        env_params = dict(
            sumocfg=params['env_path'],
            sumonet=params['net_path'],
            out_csv_name=None,
            use_gui=params['use_gui'],
            num_seconds=params['ep_len'],
            min_green=5,
            max_green=50,
            fixed_ts=params['fixed_ts'],
            sumo_warnings=True,
            yellow_time=3,
            pred_params=pred_params,
            training=False,
            time_to_teleport = params['time_to_teleport'],
            test_log_folder = params['test_log_folder'],
            device = ptu.device,
            mult_factor=params["mult_factor"],
            cycle_time=params["cycle_time"],
        )

        ################
        ## RL TRAINER
        ################
        
        self.params['agent_class'] = PPOAgent
        self.params['agent_params'] = agent_params
        self.params['env_params'] = env_params

        self.rl_tester = RL_Tester(self.params)

    def run_testing(self): 
        
        self.rl_tester.run_testing(
            n_scenarios=self.params['n_scenarios'],
            agent=self.rl_tester.agent
        )

def main(): 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', type=str, help='sumoconfig file')
    parser.add_argument('--net_path', type=str, help='net file')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='rl')
    parser.add_argument('--n_scenarios', '-n', type=int, default=50)

    parser.add_argument('--gae_lambda', type=float, default=0.97)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--steps_per_epoch', '-b', type=int, default=60)
    parser.add_argument('--eval_ntraj', type=int, default=2) 
    parser.add_argument('--eval_steps', type=int, default=60)
    parser.add_argument('--train_steps_per_iter', type=int, default=80)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--ep_len', type=int, default=3600, help='Seconds to simulate on sumo')
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--gnn_size', '-gs', type=int, default=16)
    parser.add_argument('--gnn_n_layers', '-gl', type=int, default=2)
    parser.add_argument('--graph_path', '-gp', type=str)
    parser.add_argument('--pred_model_path', type=str, default=None)

    # Retrain or Testing
    parser.add_argument('--ac_in_model_folder', type=str, default=None)
    parser.add_argument('--ac_in_model_type', type=str, default="last")
    parser.add_argument('--retrain_reset', action='store_true', default=False)
    parser.add_argument('--test_log_folder', type=str, default="")    

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--mult_factor', type=float, default=0.7)
    parser.add_argument('--cycle_time', type=int, default=60)
    parser.add_argument('--fixed_ts', action='store_true', default=False)
    parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--time_to_teleport', type=int, default=-1)
    parser.add_argument('--not_gpu', action='store_true', default=False)
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--num_cpu', type=int, default=0, help="num of cpu cores used to run parallel envs")

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'runs_rl/'
    params['test_log_folder'] = f"{params['test_log_folder']}/{params['ac_in_model_type']}/{params['model_name']}"

    logdir = logdir_prefix + args.exp_name + '/' 
    exp_name = args.model_name + f'_nl{args.n_layers}' + f'_hs{args.size}' + f'_gnl{args.gnn_n_layers}' + f'_ghs{args.gnn_size}' 
    exp_time = '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

    params['logdir'] = logdir + exp_name + exp_time
    if params['ac_in_model_folder']:
        params['ac_in_model_path'] = f"{params['ac_in_model_folder']}/{params['model_name']}/GNN_RL_{params['ac_in_model_type']}.pt"  # + exp_time
    else:
        params['ac_in_model_path'] = None

    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    
    if not(os.path.exists(params['test_log_folder'])):
        os.makedirs(params['test_log_folder'])

    ###################
    ### RUN TRAINING
    ###################

    tester = PPO_Tester(params)
    tester.run_testing()


if __name__ == '__main__': 
    main()
    print('*********************************\n FINISH run_rl_test.py\n*********************************')
