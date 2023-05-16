from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from src.environment import SumoEnvironment

def make_env1(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env_params = dict(
            sumocfg=f'src/data/sumo_optimal/optimal_{rank}.sumocfg',
            sumonet='src/data/sumo_optimal/optimal.net.xml',
            out_csv_name=None,
            use_gui=False,
            num_seconds=2000,
            min_green=5,
            max_green=50,
            fixed_ts=False,
            rank=rank, 
            sumo_warnings=True,
            cycle_time=120,
            yellow_time=3
        )

        env = SumoEnvironment(**env_params)
        env.seed(rank + (8 * seed))
        return env
    set_random_seed(seed)
    return _init

def make_env(env_params, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        sumocfg = env_params['sumocfg'].split(".sumocfg")[0]
        env_params['sumocfg'] = sumocfg + f'_{rank}.sumocfg'
        env_params['rank'] = rank
        env_params['seed'] = seed
        
        env = SumoEnvironment(**env_params)
        env.seed(rank + (8 * seed))
        return env
    set_random_seed(seed)
    return _init

def make_vec_env(env_params, num_cpu, seed=0):
    num_cpu = num_cpu if num_cpu > 0 else 1 
    return SubprocVecEnv(
        [make_env(env_params, i, seed) for i in range(num_cpu)],
        start_method='spawn'  # 'spawn' for windows, 'forkserver' for linux
    )

if __name__ == "__main__": 
    
    num_cpu = 8

    # envs = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    env_params = dict(
            sumocfg='src/data/sumo_optimal/optimal.sumocfg',
            sumonet='src/data/sumo_optimal/optimal.net.xml',
            out_csv_name=None,
            use_gui=False,
            num_seconds=2000,
            min_green=5,
            max_green=50,
            fixed_ts=False,
            sumo_warnings=True,
            cycle_time=120,
            yellow_time=3
        )
    envs = make_vec_env(env_params, num_cpu)

    # print()

    # import time
    # start_time = time.time()
    # for _ in range(2): 
    #     o = envs.reset()
    #     t = time.time()
    #     print(t - start_time)
    #     start_time = t

    # print()