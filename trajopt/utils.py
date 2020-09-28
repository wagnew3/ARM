"""
Utility functions useful for TrajOpt algos
"""

import numpy as np
import multiprocessing as mp
from trajopt import tensor_utils
from trajopt.envs.utils import get_environment
from trajopt.envs.herb_pushing_env import HerbEnv
import cv2
import trimesh
import copy

max_action=0.1

#@profile
def do_env_rollout(env_name, start_state, act_list, vel_list, top_dir, task, filter_coefs, e):
    """
        1) Construct env with env_name and set it to start_state.
        2) Generate rollouts using act_list.
           act_list is a list with each element having size (H,m).
           Length of act_list is the number of desired rollouts.
    """
    e.reset_model()
    e.real_step = False
    e.set_env_state(start_state, reset=True)
    env_seen_hand_target_dists=len(e.hand_target_dists)
    paths = []
    H = act_list[0].shape[0]
    N = len(act_list)
    for i in range(N):
        e.set_env_state(start_state, reset=False)
        obs = []
        act = []
        rewards = []
        env_infos = []
        states = []

        try:
            for k in range(H):
                obs.append(e._get_obs())
                act.append(act_list[i][k])
                env_infos.append(e.get_env_infos())
                states.append(e.get_env_state())
                s, r, d, ifo = e.step(act[-1])
                e.hand_target_dists=e.hand_target_dists[:env_seen_hand_target_dists]
                e.finger_controls=e.finger_controls[:env_seen_hand_target_dists]
                e.contacting=e.contacting[:env_seen_hand_target_dists]
                rewards.append(r)
            path = dict(observations=np.array(obs),
                actions=np.array(act),
                rewards=np.array(rewards),
                env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                states=states,
                vels=np.array(vel_list[i]))
            paths.append(path)
        
        except:
            print('bad physiscs state!!')

    return paths


def discount_sum(x, gamma, discounted_terminal=0.0):
    """
    discount sum a sequence with terminal value
    """
    y = []
    run_sum = discounted_terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])

def generate_perturbed_actions_old(base_act, filter_coefs):
    """
    Generate perturbed actions around a base action sequence
    """
    sigma, beta_0, beta_1, beta_2 = filter_coefs
    eps = np.random.normal(loc=0, scale=0.1, size=base_act.shape) * sigma
    for i in range(2, eps.shape[0]):
        eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
    return base_act + eps

def generate_perturbed_actions(start_state, base_act, filter_coefs, scale_factor, finger_vel, ctrl_4, ctrl_6, hand_open=1, move=True):
    """
    Generate perturbed actions around a base action sequence
    """
    scale_factor=2*scale_factor
    if not move:
        scale_factor=0
    
    #finger_vel=0.15
    sigma, beta_0, beta_1, beta_2, joint_range, low, high = filter_coefs
    
    ls_scale_factors=[0.0025, 0.00125, 0.0025, 0.005, 0.0025, 0.025, 0.0025]
    
    base_positions=np.concatenate((np.expand_dims(start_state['qp'][:15], axis=0), base_act))
    vels=base_positions[1:]-base_positions[:-1]
    for ind in range(len(ls_scale_factors)):
        vels[:, ind]=np.clip(vels[:, ind], scale_factor*-ls_scale_factors[ind]*joint_range[ind], scale_factor*ls_scale_factors[ind]*joint_range[ind]) 
    vels[:, 7:]=np.clip(vels[:, 7:], finger_vel, finger_vel)

    for pred_ind in range(base_act.shape[0]):
        eps = np.random.normal(loc=0, scale=scale_factor*0.005, size=base_act[0].shape) * sigma
        for ind in range(len(ls_scale_factors)):   
            vels[pred_ind][ind]=np.clip(vels[pred_ind][ind] + eps[ind], scale_factor*-ls_scale_factors[ind]*joint_range[ind], scale_factor*ls_scale_factors[ind]*joint_range[ind])
        vels[pred_ind][7:]=np.clip(vels[pred_ind][7:] + eps[7:], scale_factor*-0.25*joint_range[7:], scale_factor*0.25*joint_range[7:])
        
        if pred_ind==0:
            if hand_open==1:
                vels[pred_ind][8]=-finger_vel
                vels[pred_ind][11]=-finger_vel
                vels[pred_ind][13]=-finger_vel
            elif hand_open==0:
                vels[pred_ind][8]=finger_vel
                vels[pred_ind][11]=finger_vel
                vels[pred_ind][13]=finger_vel
            elif hand_open==-1:
                vels[pred_ind][8]=finger_vel
                vels[pred_ind][11]=finger_vel
                vels[pred_ind][13]=finger_vel
        else:
            vels[pred_ind][8]=vels[0][8]
            vels[pred_ind][11]=vels[0][11]
            vels[pred_ind][13]=vels[0][13]

    base_act[0]=base_positions[0]+vels[0]
    for vel_ind in range(1, vels.shape[0]):
        base_act[vel_ind]=base_act[vel_ind-1]+vels[vel_ind]
        base_act[vel_ind]=np.clip(base_act[vel_ind], low, high)
    base_act[:,4]=ctrl_4
    vels[:,4]=0
    base_act[:,6]=ctrl_6
    vels[:,6]=0
    return base_act, vels

#@profile
def generate_paths(env_name, start_state, N, base_act, filter_coefs, base_seed, top_dir, task, env):
    """
    first generate enough perturbed actions
    then do rollouts with generated actions
    set seed inside this function for multiprocessing
    """
    np.random.seed(base_seed)
    act_list = []#[np.copy(base_act)]
    vel_list=[]
    
    state=env.get_state()
    finger_vel=0.05

    for i in range(0, N):
        if task=='hard_pushing':
            if state==0:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.1875, finger_vel, -2.69, 0.24, hand_open=1)
                act_list.append(act)
                vel_list.append(vel)
            elif state==1:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.1875, finger_vel, -2.69, 0.24, hand_open=-1)
                act_list.append(act)
                vel_list.append(vel)
                break
            elif state==2:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.5, finger_vel, -2.69, 0.24, hand_open=0)
                act_list.append(act)
                vel_list.append(vel)
        elif task=='grasping':
            if state==0:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.1875, finger_vel, -2.69, 0.24, hand_open=1)
                act_list.append(act)
                vel_list.append(vel)
            elif state==1:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.1, finger_vel, -2.69, 0.24, hand_open=-1)
                act_list.append(act)
                vel_list.append(vel)
                break
            elif state==2 or state==3:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.25, finger_vel, -2.69, 0.24, hand_open=0)
                act_list.append(act)
                vel_list.append(vel)
#             elif state==3:
#                 act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 1.5, 0.05, -1.13, 1.59, hand_open=0, move=False)
#                 act_list.append(act)
#                 vel_list.append(vel)
        elif task=='easy_pushing':
            if state==0:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.1875, finger_vel, -2.46, 0, hand_open=1)
                act_list.append(act)
                vel_list.append(vel)
            elif state==1:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.1875, finger_vel, -2.46, 0, hand_open=-1)
                act_list.append(act)
                vel_list.append(vel)
                break
            elif state==2:
                act, vel = generate_perturbed_actions(start_state, np.copy(base_act), filter_coefs, 0.5, finger_vel, -2.46, 0, hand_open=0)
                act_list.append(act)
                vel_list.append(vel)

    paths = do_env_rollout(env_name, start_state, act_list, vel_list, top_dir, task, filter_coefs, env)
    return paths


def generate_paths_star(args_list):
    return generate_paths(*args_list)


def gather_paths_parallel(env_name, start_state, base_act, filter_coefs, base_seed, paths_per_cpu, top_dir, task, env_pool, num_cpu=None):
    num_cpu = mp.cpu_count() if num_cpu is None else num_cpu
    args_list = []
    for i in range(num_cpu):
        cpu_seed = base_seed + i*paths_per_cpu
        args_list_cpu = [env_name, start_state, paths_per_cpu, base_act, filter_coefs, cpu_seed, top_dir, task, env_pool[i]]
        args_list.append(args_list_cpu)

    # do multiprocessing
    results = _try_multiprocess(args_list, num_cpu, max_process_time=300, max_timeouts=4)
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths


def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts):
    # Base case
    if max_timeouts == 0:
        return None

    if num_cpu == 1:
        results = [generate_paths_star(args_list[0])]  # dont invoke multiprocessing unnecessarily

    else:
        pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
        parallel_runs = [pool.apply_async(generate_paths_star,
                                         args=(args_list[i],)) for i in range(num_cpu)]
        try:
            results = [p.get(timeout=max_process_time) for p in parallel_runs]
        except Exception as e:
            print(str(e))
            print("Timeout Error raised... Trying again")
            pool.close()
            pool.terminate()
            pool.join()
            return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts - 1)

        pool.close()
        pool.terminate()
        pool.join()

    return results

