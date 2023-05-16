import numpy as np
from src.rl.utils import discount_cumsum, normalize
from src.infrastructure import pytorch_utils as ptu

class PPOBuffer: 
    """
    A buffer for storing trajectories experienced by a PPO agent interacting 
    with the environment, and using Generlized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs. 
    """

    def __init__(self, obs_dim, act_dim, n_agents, size, num_cpu, gamma=0.99, lam=0.95) -> None:
        self.obs_buf = np.zeros((size // num_cpu, num_cpu, n_agents, *obs_dim), dtype=np.float32)    
        self.act_buf = np.zeros((size // num_cpu, num_cpu, n_agents, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros((size // num_cpu, num_cpu, n_agents), dtype=np.float32)
        self.rew_buf = np.zeros((size // num_cpu, num_cpu, n_agents), dtype=np.float32)
        self.rtg_buf = np.zeros((size // num_cpu, num_cpu, n_agents), dtype=np.float32)
        self.val_buf = np.zeros((size // num_cpu, num_cpu, n_agents), dtype=np.float32)
        self.logp_buf = np.zeros((size // num_cpu, num_cpu, n_agents), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size // num_cpu
        self.n_agents = n_agents

        self.size = size
        self.num_cpu = num_cpu
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def store(self, obs, act, rew, val, logp): 
        """
        Append one timestep of agent-environment interaction to the buffer. 
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0): 
        """
        Call this at the end of a trajectory, or when one gets cut off by an 
        epoch ending. This looks back in the buffer to where the trajectory 
        started, and uses rewards and value estimates from the whole trajectory
        to compute advantage estimates with GAE-Lambda, as well as compute 
        rewards-to-go for each state, to use as the targets for the value 
        function.

        The "last_val" argument should be 0 if the trajectory ended because the
        agent reached a terminal state (died), and otherwise should be V(s_T), 
        the value function estimated for the last state. This allow us to 
        bootstrap the reward-to-go calculation to account for timesteps beyond
        the arbitrary episode horizon (or epoch cutoff).  
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.concatenate([self.rew_buf[path_slice], last_val])
        vals = np.concatenate([self.val_buf[path_slice], last_val])

        # The next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # The next line computes rewards-to-go, to be targets for the value function
        self.rtg_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self): 
        """
        Call this at the end of an epoch to get all of the data from the buffer,
        with advantages appropiately normalized (shifted to have mean zero and 
        std one). Also resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx= 0, 0

        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-10)

        data = dict(obs=self.obs_buf, act=self.act_buf, rtg=self.rtg_buf, 
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: ptu.from_numpy(v) for k, v in data.items()}

    def get_eval(self): 
        """
        Call this at the end of an epoch to get all of the data from the buffer,
        with advantages appropiately normalized (shifted to have mean zero and 
        std one). Also resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        path_slice = slice(0, self.ptr)
        self.ptr, self.path_start_idx= 0, 0
        
        return self.rew_buf[path_slice].sum(0)
        
