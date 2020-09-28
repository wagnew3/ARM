"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""

import numpy as np
from trajopt.algos.trajopt_base import Trajectory
from trajopt.utils import gather_paths_parallel
from trajopt.envs.herb_pushing_env import HerbEnv
import trimesh
import cv2
import os
import open3d as o3d

class MPPI(Trajectory):
    def __init__(self, run_num, env, top_dir, task, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 seed=123
                 ):
        self.top_dir=top_dir
        self.task=task
        self.env, self.seed = env, seed
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []
        self.act_sequences=[]

        self.env.reset_model(seed=self.seed)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.repeat(np.expand_dims(self.sol_state[-1]['qp'][:15], 0), self.H, axis=0)
        #self.act_sequence = np.ones((self.H, self.m)) * self.mean
        self.env_pool=[HerbEnv(top_dir, None, run_num, obs=False, task=self.task) for i in range(self.num_cpu)]


    def update(self, paths):
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        vel= np.array([paths[i]["vels"] for i in range(num_traj)])
        R = self.score_trajectory(paths)
        S = np.exp(self.kappa*(R-np.max(R)))
        
        best_reward=-float('inf')
        for i in range(len(paths)):
            if paths[i]["rewards"][-1]>best_reward:
                best_reward=paths[i]["rewards"][-1]
        

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        
        weighted_vels = S*vel.T
        weighted_vels = np.sum(weighted_vels.T, axis=0)/(np.sum(S) + 1e-6)
        
        best_seq_ind=np.argmax(R)
        print('best_reward', R[best_seq_ind])
        
        self.vel_sequence=np.copy(weighted_vels)
        self.act_sequence = np.copy(act_sequence)
        self.sim_act_sequence=np.copy(act_sequence)

#     @profile
    def advance_time(self, act_sequence=None, gt_env=None):
        state_now = self.sol_state[-1].copy()
        act_sequence=np.zeros(self.vel_sequence.shape)
        act_sequence[0]=self.sol_state[-1]['qp'][:15]+self.vel_sequence[0]
        for vel_ind in range(1, self.vel_sequence.shape[0]):
            act_sequence[vel_ind]=act_sequence[vel_ind-1]+self.vel_sequence[vel_ind]
        act_sequence[:, 0:7]=self.act_sequence[:, 0:7]
        self.sol_act.append(act_sequence[0])
        
        self.env.set_env_state(state_now)
        _, r, _, _ = self.env.step(act_sequence[0])
        if gt_env!=None:
            gt_env.set_env_state(state_now)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env._get_obs())
        self.sol_reward.append(r)

        # get updated action sequence
        self.act_sequences.append(np.copy(self.act_sequence))
        self.act_sequence[:-1] = act_sequence[1:]
        if self.default_act == 'repeat':
            self.act_sequence[-1] = self.act_sequence[-2]
        else:
            self.act_sequence[-1] = self.mean.copy()

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        for i in range(len(paths)):
            scores[i] = 0.0
            for t in range(paths[i]["rewards"].shape[0]):
                scores[i] += (self.gamma**t)*paths[i]["rewards"][t]
        return scores

    def do_rollouts(self, seed, env_pool, temp_sol_state):
        base_act=np.repeat(np.expand_dims(temp_sol_state['qp'][:15], 0), self.H, axis=0)
        paths = gather_paths_parallel(self.env.env_name,
                                      temp_sol_state,
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.top_dir,
                                      self.task,
                                      env_pool,
                                      self.num_cpu)
        return paths

    
    def train_step(self, obs_env_path, save_id, top_dir, target_object_name, palm_mesh_vertices, gt_env=None, niter=1, gt=False, vis=True, use_last_state=False, last_state=None, last_env=None, cur_env=None):
        #compute next action
        if gt:
            if last_env==None:
                last_env=HerbEnv(obs_env_path, palm_mesh_vertices, save_id, task=self.task, obs=False, push_mesh_vertices=trimesh.load(os.path.join(top_dir, target_object_name)), target_mesh_vertices=trimesh.load(os.path.join(top_dir, target_object_name)).vertices)
                last_env.set_env_state(cur_env.get_env_state().copy())
            env_pool=[last_env]
            temp_sol_state=env_pool[0].get_env_state().copy()
        else:
            if not use_last_state:
                env_pool=[HerbEnv(obs_env_path, palm_mesh_vertices, save_id, task=self.task, obs=True, push_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, f'herb_reconf/temp_{save_id}/target_mesh.stl')), target_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, f'herb_reconf/temp_{save_id}/target_mesh.stl')).vertices)]
                temp_sol_state=env_pool[0].get_env_state().copy()
            else:
                if last_env==None:
                    last_env=HerbEnv(obs_env_path, palm_mesh_vertices, save_id, task=self.task, obs=True, push_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, f'herb_reconf/temp_{save_id}/target_mesh.stl')), target_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, f'herb_reconf/temp_{save_id}/target_mesh.stl')).vertices)
                env_pool=[last_env]
                temp_sol_state={i: np.copy(last_state[i]) for i in last_state}
            
        if vis:
            env_pool[0].reset_model()
            env_pool[0].real_step = False
            env_pool[0].set_env_state(temp_sol_state)
            trgb=env_pool[0].model.render(height=480, width=640, camera_id=0, depth=False)
            cv2.imshow('trbg', trgb)
            rgb=env_pool[0].model.render(height=480, width=640, camera_id=1, depth=False)
            cv2.imshow('obs', rgb)
            rgb=self.env.model.render(height=480, width=640, camera_id=1, depth=False)
            cv2.imshow('conv decomp', rgb)
            cv2.waitKey(20)
            
        t = len(self.sol_state) - 1
        for _ in range(niter):
            paths = self.do_rollouts(self.seed+t, env_pool, temp_sol_state)
            self.update(paths)
        env_pool[0].set_env_state(temp_sol_state)
        env_pool[0].step(self.sim_act_sequence[0], compute_tpd=True)
        
        #step gt env
        self.advance_time(gt_env=gt_env)
        print('env state', env_pool[0].get_state(), env_pool[0].hand_target_dists[-1])
        return env_pool[0]
