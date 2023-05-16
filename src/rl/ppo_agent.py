import pickle as pkl
import torch
import torch.nn as nn
import dgl
from src.rl.policies.mlp_actorcritic import MLPActorCritic
from src.rl.policies.gnn_actorcritic import GNNActorCritic
from src.rl.ppo_buffer import PPOBuffer
from src.dataloader.dataloader import load_graph
import src.infrastructure.pytorch_utils as ptu

class BaseAgent(object):
    def __init__(self, **kwargs) -> None:
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class PPOAgent(BaseAgent):
    def __init__(self, agent_params) -> None:
        super(PPOAgent, self).__init__()

        # init vars
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.lam = self.agent_params['gae_lambda']

        # self.ac = MLPActorCritic(
        #     ob_space=self.agent_params['ob_space'],
        #     ac_space=self.agent_params['ac_space'],
        #     size=self.agent_params['size'],
        #     n_layer=self.agent_params['n_layers'],
        #     pi_lr=self.agent_params['pi_lr'],
        #     vf_lr=self.agent_params['vf_lr'],
        #     tls_masks=self.agent_params['tls_masks'],
        #     activation=nn.ReLU()
        # )

        n_graphs_step = agent_params['num_cpu']
        n_graphs_forw = agent_params['num_cpu'] * (agent_params['steps_per_epoch'] // agent_params['num_cpu'])
        g = load_graph(self.agent_params['graph_path'])
        g_step = dgl.batch([g for _ in range(n_graphs_step)])
        g_forw = dgl.batch([g for _ in range(n_graphs_forw)])

        self.ac = GNNActorCritic(
            ob_space=self.agent_params['ob_space'],
            ac_space=self.agent_params['ac_space'],
            g_step=g_step,
            g_forw=g_forw,
            # gnn_n_layers=2,
            # gnn_size=16,  # TODO: Puede ser mayor?
            gnn_n_layers=self.agent_params['gnn_n_layers'],
            gnn_size=self.agent_params['gnn_size'],
            mlp_n_layers=self.agent_params['n_layers'],
            mlp_size=self.agent_params['size'],
            pi_lr=self.agent_params['pi_lr'],
            vf_lr=self.agent_params['vf_lr'],
            tls_masks=self.agent_params['tls_masks'],
            num_cpu=self.agent_params['num_cpu'],
            mlp_activation=nn.ReLU()
        )

        self.max_rews = -1000000
        self.itr = 0
        self.min_wt = 10000
        # Load saved ac model if it exists:
        if agent_params["ac_in_model_path"] is not None:
            print(f'\n*-*-*-*-*-*- Loading AC Network in {agent_params["ac_in_model_path"]} -*-*-*-*-*-*\n')
            checkpoint = torch.load(f'{agent_params["ac_in_model_path"]}',map_location=torch.device(ptu.device))
            if "model_state_dict" in checkpoint.keys():
                self.ac.load_state_dict(checkpoint["model_state_dict"])
                self.ac.pi_optimizer.load_state_dict(checkpoint["pi_optimizer_state_dict"])
                self.ac.vf_optimizer.load_state_dict(checkpoint["vf_optimizer_state_dict"])
                print("Model pi and vf Optimizers loaded for retraining")
                # from torch.optim import Adam
                # self.ac.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.ac.pi_optimizer.param_groups[0]['lr'])
                # self.ac.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.ac.vf_optimizer.param_groups[0]['lr'])
                if not agent_params["retrain_reset"]:
                    self.max_rews = checkpoint["reward"]
                    self.itr = checkpoint["epoch"]
                    self.min_wt = checkpoint["waiting_time"]
                    print("Rew, Iter and Min_wt loaded")
                else:
                    print("Reset retrain!!!")
            else:
                self.ac.load_state_dict(checkpoint)  # old version models

        # print("######################## Agent PARAMETERS ##########################")
        for name, param in self.ac.pi.named_parameters():
            print(name, param.numel())
        print(f"Total de parámetros pi = {sum(p.numel() for p in self.ac.pi.parameters())}")
        print(f"Total de parámetros v = {sum(p.numel() for p in self.ac.v.parameters())}")


        self.ppobuffer = PPOBuffer(
            obs_dim=self.agent_params['ob_dim'],
            act_dim=self.agent_params['ac_dim'],
            n_agents=self.agent_params['n_agents'],
            size=self.agent_params['steps_per_epoch'],
            num_cpu=self.agent_params['num_cpu'],
            gamma=self.gamma,
            lam=self.lam
        )

    def train(self):
        print('\nTraining agent using sampled data from buffer...')
        train_log = []
        self.ac.train()
        data = self.ppobuffer.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)  #, 0)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # TODO: Probar entrenamiento por escenario
        # data_copy = data.copy()
        # data = {}
        # for idx in data_copy:
        #     data_copy[idx] = data_copy[idx].transpose(0,1)
        #     data_copy[idx] = data_copy[idx].flatten(end_dim=1)
        # n_scenarios = int(data["obs"].shape[0] * data["obs"].shape[1] / 30)  # 30 es 3600 seg/scenario / 120 seg/steps = 30 steps/scenario

        # Train policy with multiple steps of gradient descent
        for train_steps in range(self.agent_params['train_steps_per_iter']):
            # TODO: Probar entrenamiento por escenario
            # for scenario in range(n_scenarios):
            #     for idx in data_copy:
            #         data[idx] = data_copy[idx][scenario*30:(scenario+1)*30]
            #         data[idx] = data[idx].unsqueeze(1)

            loss_pi, pi_info = self.compute_loss_pi(data)  #, train_steps)
            # TODO: Early stoping due to reaching max kl
            # print(f"loss_pi = {loss_pi}")
            self.ac.update_pi(loss_pi)

        # Value function learning
        for train_steps in range(self.agent_params['train_steps_per_iter']):
            # TODO: Probar entrenamiento por escenario
            # for scenario in range(n_scenarios):
            #     for idx in data_copy:
            #         data[idx] = data_copy[idx][scenario*30:(scenario+1)*30]
            #         data[idx] = data[idx].unsqueeze(1)
            loss_vf = self.compute_loss_v(data)
            self.ac.update_v(loss_vf)

        # Log changes from update
        train_log = {
            'Loss_pi': loss_pi,
            'Loss_vf': loss_vf,
            'Pi_Entropy': pi_info_old['ent'],
            'Pi_KL': pi_info['kl'],
            'Pi_ClipFraction': pi_info['cf'],
            'Delta_Loss_Pi': loss_pi - pi_l_old,
            'Delta_Loss_Vf': loss_vf - v_l_old,
        }

        # print("######################## Agent PARAMETERS ##########################")
        # for name, param in self.ac.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        return train_log, self.ac.state_dict(), self.ac.pi_optimizer.state_dict(), self.ac.vf_optimizer.state_dict()

    def compute_loss_pi(self, data):  #, train_steps):
        """Set up function for computing PPO policy loss"""
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        clip_ratio = self.agent_params['clip_ratio']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)  #, train_steps)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        #
        # import pickle
        # print(f"act={act}")
        # print(f"pi={pi}")
        # print(f"logp={logp}")
        # print(f"loss_pi={loss_pi}")
        # print(f"pi_info={pi_info}")
        # test_var_dict2 = {"loss_pi":loss_pi, "pi_info":pi_info, "logp":logp, "logp_old":logp_old, "ratio":ratio, "act":act}
        # a_file = open("test_var_dict2.pkl", "wb")
        # pickle.dump(test_var_dict2, a_file)
        # a_file.close()
        # torch.save(self.ac.state_dict(), 'ac.pth')
        #

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        """Set up function for computing value loss"""
        obs, rtg = data['obs'], data['rtg']
        return ((self.ac.v(obs, self.ac.g_forw) - rtg) ** 2).mean()