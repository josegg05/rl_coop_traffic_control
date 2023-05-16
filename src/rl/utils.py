import numpy as np
import scipy.signal
import time
import torch

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def ob_from_multiagent_to_net(obs): 
    # obs shape -> (num_cpu, intersections, phases, feat)
    envs = range(len(obs))
    sortedkeys = sorted(obs[0].keys(), key=lambda x:x.lower())
    obs = np.stack([[obs[env][key] for key in sortedkeys] for env in envs])
    return obs, sortedkeys

def data_to_net(data): 
    """Recieves the multi-env (list) multi-agent (dict) and transformed it to 
    numpy array
    """
    envs = range(len(data))
    obs = [data[env]['obs'] for env in envs]
    h = np.stack([data[env]['h'] for env in envs])
    obs, sk = ob_from_multiagent_to_net(obs)
    o = np.concatenate([obs, h], -1)
    return o, sk
    # h, _ = env_out_from_multiagent_to_net(h, _, sk)

def ac_from_net_to_multiagent(ac, sortedkeys):
    ac = ac.squeeze(0)
    envs = range(ac.shape[0])
    return [{key: ac[env][i] for i, key in enumerate(sortedkeys)} for env in envs]

def env_out_from_multiagent_to_net(obs, rew, sortedkeys): 
    envs = range(len(obs))
    obs = np.stack([[obs[env][key] for key in sortedkeys] for env in envs])
    rew = np.stack([[rew[env][key] for key in sortedkeys] for env in envs])
    return obs, rew


############################################
############################################

def sample_trajectory(env, agent, max_path_length, render=False, render_mode=('rgb_array')):
    # initialize env for the beginning of a new rollout
    # torch.manual_seed(0)  # Uncomment to get exact same scenarios for every sample. Reset the ac actions seed
    o, ep_rew, steps = env.reset(), 0, 0
    o, sk = ob_from_multiagent_to_net(o)

    act_t = []
    while True:
        # Actor critic step
        act_ti = time.perf_counter()
        a, v, logp = agent.ac.step(o)
        act_t.append(time.perf_counter() - act_ti)

        # Env step 
        ac = ac_from_net_to_multiagent(a, sk)
        if not env.get_attr("fixed_ts")[0]:
            next_o, r, d, _ = env.step(ac)
        else:
            next_o, r, d, _ = env.step([{}])
        next_o, r = env_out_from_multiagent_to_net(next_o, r, sk)
        ep_rew += r
        steps += 1
        if steps % 10 == 0:
            print(f'Steps: {steps}, ep_rw: {np.mean(ep_rew)/steps}')

        # save and log 
        agent.ppobuffer.store(o, a, r, v, logp)
        # TODO log vvals

        # Update observation
        o = next_o

        # HINT: rollout can end due to done, or due to max_path_length
        # rollout_done = d['__all__'] or steps >= max_path_length # HINT: this is either 0 or 1

        # ANTES: rollout_done = d[0] or steps >= max_path_length, pero es redundante pues steps >= max_path_length sucede al mismo tiempo que d
        rollout_done = d[0] # d[0] implica que llego al ep_lengh. Se hace un reset() automatico en subproc_vec_env que realiza el env.close().
        epoch_ended = agent.ppobuffer.ptr == agent.ppobuffer.max_size
        if rollout_done or epoch_ended:
            _, v, _ = agent.ac.step(o)
            # env.step([{}] * len(ac)) # (creo que no hace falta) To close the environemnt 
            # print(f"rollout_done = {rollout_done}")
            if not rollout_done:
                env.reset()  # para cerrar la simulacion en caso de que se haya terminado en el medio
            agent.ppobuffer.finish_path(v)
            return ep_rew, steps, act_t
    

def sample_trajectories(env, agent, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.

    """
    ep_rews, ep_lens, act_ts = [], [], []
    while agent.ppobuffer.ptr < min_timesteps_per_batch:
        ep_rew, ep_len, act_t = sample_trajectory(env, agent, max_path_length, render, render_mode)
        ep_rews.append(ep_rew)
        ep_lens.append(ep_len)
        act_ts.append(act_t)
    return ep_rews, ep_lens


def sample_n_eval_trajectories(env, agent, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
    Collect ntraj rollouts. 
    """
    ep_rews, ep_lens, act_ts = [], [], []
    env.set_attr("eval", True)
    agent.ac.eval()
    for _ in range(ntraj): 
        ep_rew, ep_len, act_t = sample_trajectory(env, agent, max_path_length, render, render_mode)
        ep_rews.append(ep_rew)
        ep_lens.append(ep_len)
        act_ts.append(act_t)
        # clean buffer
        _ = agent.ppobuffer.get_eval()
    env.set_attr("eval", False)
    agent.ac.train()
    return ep_rews, ep_lens


def sample_test_trajectory(env, agent, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
    Collect test trajectory. 
    """
    ep_rews, ep_lens, act_ts = [], [], []
    agent.ac.eval()
    for test_n in range(ntraj): 
        ep_rew, ep_len, act_t = sample_trajectory(env, agent, max_path_length, render, render_mode)
        ep_rews.append(ep_rew)
        ep_lens.append(ep_len)
        act_ts.append(act_t)
        # clean buffer
        _ = agent.ppobuffer.get_eval()

        log_file=f"{env.get_attr('test_log_folder')[0]}/verbose_{env.get_attr('mult_factor')[0]}_{env.get_attr('test_count')[0]}.xml"
        teleports = 0
        with open(log_file) as f:
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
                    depart_delay = float(content[-2][14:-1])
                    ready_to_load = False
                    break
        print(f"Test {test_n}: Ending Time: {ending_time}| Inserted Veh: {inserted_veh}| Running Veh: {running_veh}| Waiting Veh: {waiting_veh}| Teleports: {teleports}")
        print(f"        Waiting Time: {waiting_time}| Time Loss: {time_loss}| Depart Delay: {depart_delay}")
    agent.ac.train()
    return ep_rews, ep_lens, act_ts