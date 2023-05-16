import numpy as np
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import src.infrastructure.pytorch_utils as ptu


class Actor(nn.Module): 

    def _distribution(self, obs): 
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act): 
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under 
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None: 
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor): 

    def __init__(self, obs_dim, act_dim, hidden_size, n_layer, activation) -> None:
        super().__init__()
        self.logits_net = ptu.build_mlp(
            obs_dim, act_dim, n_layer, hidden_size, activation)
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)


class MLPGaussianActor(Actor): 
    
    def __init__(self, obs_dim, act_dim, hidden_size, n_layer, activation) -> None:
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = ptu.build_mlp(
            obs_dim, act_dim, n_layer, hidden_size, activation)

    def _distribution(self, obs):  #, obs2, g, gnn, steps, train_steps):  # borrar obs2, g, steps, train_steps
        """ 
        Distribution forward prop. 
        
        Detailed explanation on why it is used a MultivariateNormal with a 
        diagonal covariance matrices. 
        1. https://discuss.pytorch.org/t/optimized-multivariatenormal-with-diagonal-covariance-matrix/29803/3
        2. https://pytorch.org/docs/stable/distributions.html#independent
        
        Implementation: UC Berkeley CS285 HW2
        https://github.com/gepizar/CS285_HWs_Fall2021/blob/main/hw2/cs285/policies/MLP_policy.py#L100-L114
        """
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        scale_tril = torch.diag(std)
        batch_dim = mu.shape[:-1]
        batch_scale_tril = scale_tril.repeat(*batch_dim, 1, 1)
        distrib = MultivariateNormal(mu, scale_tril=batch_scale_tril)
        # distrib = Normal(mu, std)  # Otra forma -- hacer ".sum(axis=-1)" en el logp_a=self._log_prob_from_distribution(pi, act).sum(axis=-1)
        #
        # import pickle
        # if steps == 30:
            # print(f"train_steps = {train_steps}")
            # print(f"$$$$$$$$$$$   obs[15,0,0,0,:6] = {obs2[15,0,0,0,:6]}   $$$$$$$$$$$$$")
            # print(f"$$$$$$$$$$$   obs[15,0,0,0,6:] = {obs2[15,0,0,0,6:]}   $$$$$$$$$$$$$")
            # print(f"$$$$$$$$$$$   h[0,0,0,:] = {obs[15,0,0,:]}   $$$$$$$$$$$$$") 
            # print(f"gnn[0].mlp.model[0]._parameters[weight] = {gnn[0].mlp.model[0]._parameters['weight']}" )           
        #
        # test_var_dict = {
        #     "obs":obs, 
        #     "mu":mu, 
        #     "scale_tril":scale_tril, 
        #     "batch_dim":batch_dim, 
        #     "batch_scale_tril":batch_scale_tril, 
        #     "distrib":distrib
        # }
        # a_file = open("test_var_dict.pkl", "wb")
        # pickle.dump(test_var_dict, a_file)
        # a_file.close()
        #

        return distrib

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
         

class MLPCritic(nn.Module): 

    def __init__(self, obs_dim, hidden_size, n_layer, activation) -> None:
        super().__init__()
        self.v_net = ptu.build_mlp(obs_dim, 1, n_layer, hidden_size, activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has the right shape. 


class MLPActorCritic(nn.Module): 
    def __init__(self, ob_space, ac_space, size, n_layer, pi_lr, vf_lr, 
                 tls_masks, activation=nn.Tanh) -> None:
        super().__init__()
        self.tls_masks = tls_masks
        self.mask = self._order_masks(tls_masks)
        obs_dim = ob_space.shape[0]

        # policy builder depends on action space
        if isinstance(ac_space, Box): 
            self.pi = MLPGaussianActor(obs_dim, ac_space.shape[0], size, 
                n_layer, activation)
        elif isinstance(ac_space, Discrete): 
            self.pi = MLPCategoricalActor(obs_dim, ac_space.n, size, n_layer, 
                activation)
        self.pi.to(ptu.device)

        # build value function
        self.v = MLPCritic(obs_dim, size, n_layer, activation)
        self.v.to(ptu.device)

        # Optimizers
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.v.parameters(), lr=vf_lr)

    def step(self, obs): 
        """ 
        Function that samples an action from a group of observations and return 
        a dictionary of action for each agent. 
        """ 
        obs = ptu.from_numpy(obs)
        with torch.no_grad(): 
            pi = self.pi._distribution(obs)
            a = pi.sample()
            # TODO revisar backprop con softmax, asoft guardado es distinto del a calculado en el forward
            asoft = F.softmax(a + self.mask, -1)
            logp_a = self.pi._log_prob_from_distribution(pi, asoft)
            v = self.v(obs)
        return ptu.to_numpy(asoft), ptu.to_numpy(v), ptu.to_numpy(logp_a)

    def update_pi(self, loss): 
        self.pi_optimizer.zero_grad()
        loss.backward()
        self.pi_optimizer.step()
        
    def update_v(self, loss): 
        self.vf_optimizer.zero_grad()
        loss.backward()
        self.vf_optimizer.step()

    def _order_masks(self, tls_mask):
        sortedkeys = sorted(tls_mask.keys(), key=lambda x:x.lower())
        batch = np.stack([tls_mask[key] for key in sortedkeys]).astype(np.bool)
        return ptu.from_numpy((~batch) * -1e9)





    




