import numpy as np
import numpy as np
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical

import src.infrastructure.pytorch_utils as ptu
from src.encoder.gnn import GNN, MLP, GNN_layer
from src.rl.policies.mlp_actorcritic import MLPGaussianActor, \
    MLPCritic


class GNNGaussianActor(nn.Module): 

    def __init__(
        self, 
        obs_dim, 
        act_dim, 
        g_forw, 
        gnn_n_layers,
        gnn_size,
        mlp_n_layers,
        mlp_size,
        mlp_activation,
        num_cpu
    ) -> None: 
        super().__init__()

        self.obs_dim = obs_dim
        self.phases, self.features = self.obs_dim
        self.g_forw = g_forw
        self.num_cpu = num_cpu
        
        modules = []
        in_size = self.features
        for _ in range(gnn_n_layers): 
            modules.append(GNN_layer(in_size, gnn_size))
            in_size = 2 * gnn_size
        self.gnns = nn.ModuleList(modules)
        
        self.mlp_actor = MLPGaussianActor(
            obs_dim = in_size * self.phases, 
            act_dim = act_dim,
            hidden_size = mlp_size,
            n_layer = mlp_n_layers, 
            activation = mlp_activation
        )
    
    def _distribution(self, obs, g):  #, train_steps):
        steps = obs.shape[0]
        h = obs.reshape(-1, self.features)
        for gnn_layer in self.gnns: 
            h = gnn_layer(g, h)
        h = h.reshape(steps, self.num_cpu, -1, h.shape[-1] * self.phases)
        return self.mlp_actor._distribution(h)  #, obs, g, self.gnns, steps, train_steps)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):  #, train_steps=0):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under 
        # those distributions.
        
        pi = self._distribution(obs, self.g_forw)  #, train_steps)
        logp_a = None
        if act is not None: 
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
   

class GNNCritic(nn.Module): 

    def __init__(
        self, 
        obs_dim, 
        gnn_n_layers,
        gnn_size,
        mlp_n_layers,
        mlp_size,
        mlp_activation, 
        num_cpu,
    ) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.phases, self.features = self.obs_dim
        self.num_cpu = num_cpu

        modules = []
        in_size = self.features
        for _ in range(gnn_n_layers): 
            modules.append(GNN_layer(in_size, gnn_size))
            in_size = 2 * gnn_size
        self.gnns = nn.ModuleList(modules)

        self.mlp_critic = MLPCritic(
            obs_dim = in_size * self.phases, 
            hidden_size = mlp_size, 
            n_layer = mlp_n_layers, 
            activation = mlp_activation
        )

    def forward(self, obs, g): 
        steps = obs.shape[0]
        h = obs.reshape(-1, self.features)
        for gnn_layer in self.gnns: 
            h = gnn_layer(g, h)
        h = h.reshape(steps, self.num_cpu, -1, h.shape[-1] * self.phases)
        return self.mlp_critic(h)


class GNNActorCritic(nn.Module): 
    def __init__(
        self, 
        ob_space, 
        ac_space, 
        gnn_n_layers, 
        gnn_size, 
        mlp_n_layers, 
        mlp_size, 
        pi_lr, 
        vf_lr, 
        tls_masks, 
        g_step,
        g_forw, 
        num_cpu,
        mlp_activation=nn.Tanh,
    ) -> None:
        super().__init__()

        self.tls_masks = tls_masks
        self.mask = self._order_masks(tls_masks)
        obs_dim = ob_space.shape
        act_dim = ac_space.shape[0]
        self.g_step = g_step.to(ptu.device)
        self.g_forw = g_forw.to(ptu.device)

        self.pi = GNNGaussianActor(
            obs_dim, act_dim, self.g_forw, gnn_n_layers, gnn_size, mlp_n_layers, 
            mlp_size, mlp_activation, num_cpu
        )
        self.pi.to(ptu.device)

        # build value function
        self.v = GNNCritic(
            obs_dim, gnn_n_layers, gnn_size, mlp_n_layers, mlp_size, 
            mlp_activation, num_cpu
        )
        self.v.to(ptu.device)

        # Optimizers
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.v.parameters(), lr=vf_lr)

    def step(self, obs):
        # TODO: predictor
        #   
        """ 
        Function that samples an action from a group of observations and return 
        a dictionary of action for each agent. 
        """ 
        obs = ptu.from_numpy(obs[None])
        with torch.no_grad(): 
            pi = self.pi._distribution(obs, self.g_step)  #, 0)
            a = pi.sample()
            # TODO revisar backprop con softmax, asoft guardado es distinto del a calculado en el forward
            asoft = a
            # asoft = F.softmax(a + self.mask, -1)
            logp_a = self.pi._log_prob_from_distribution(pi, asoft)
            v = self.v(obs, self.g_step)
        return ptu.to_numpy(asoft), \
            ptu.to_numpy(v), ptu.to_numpy(logp_a)

    def update_pi(self, loss): 
        self.pi_optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        #
        # for name, param in self.pi.named_parameters():
        #     print(name, torch.isfinite(param.grad).all())
        #
        self.pi_optimizer.step()
        
    def update_v(self, loss): 
        self.vf_optimizer.zero_grad()
        loss.backward()
        self.vf_optimizer.step()

    def _order_masks(self, tls_mask):
        sortedkeys = sorted(tls_mask.keys(), key=lambda x:x.lower())
        batch = np.stack([tls_mask[key] for key in sortedkeys]).astype(np.bool)
        return ptu.from_numpy((~batch) * -1e9)        