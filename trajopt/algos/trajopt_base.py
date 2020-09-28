"""
Base trajectory class
"""
import numpy as np
import cv2
import numpy as np
import os, sys
sys.path.append('/home/willie/workspace/SSC')
from trajopt.envs.herb_pushing_env import HerbEnv
import trimesh

def save_images_as_video(name, frames, top_dir):
    image_folder = os.path.join(top_dir, 'results/videos/')
    video_name = name+'.avi'
    
    images=frames
    height, width, layers = images[0].shape
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 20
    out = cv2.VideoWriter(image_folder+video_name, fourcc, fps, (width, height))
    
    for image in images:
        image=np.uint8(image)
        out.write(image)
    
    cv2.destroyAllWindows()
    out.release()
    print('saved video')

class Trajectory:
    def __init__(self, env, H=32, seed=123):
        self.env, self.seed = env, seed
        self.n, self.m, self.H = env.observation_dim, env.action_dim, H

        # following need to be populated by the trajectory optimization algorithm
        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset_model(seed=self.seed)
        self.sol_state.append(self.env.get_env_state())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.zeros((self.H, self.m))

    def update(self, paths):
        """
        This function should accept a set of trajectories
        and must update the solution trajectory
        """
        raise NotImplementedError

    def animate_rollout(self, t, act):
        """
        This function starts from time t in the solution trajectory
        and animates a given action sequence
        """
        self.env.set_env_state(self.sol_state[t])
        self.env.mujoco_render_frames = True
        for k in range(act.shape[0]):
            # self.env.mj_render()
            self.env.set_env_state(self.sol_state[t+k])
            self.env.step(act[k])
            print(self.env.env_timestep)
            print(self.env.real_step)
        self.env.mujoco_render_frames = False

    def animate_result(self, env_name, top_dir, save_dir ,task, save_id, use_gt, obs_states):
        pixels=[]
        pose_errors=[]
        if not use_gt:
            obs_env_path=os.path.join(top_dir, f'herb_reconf/temp_scene_{save_id}_0.xml')
            obs_env=HerbEnv(obs_env_path, None, save_id, task=self.task, obs=True, push_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, f'herb_reconf/temp_{save_id}/target_mesh.stl')), target_mesh_vertices=trimesh.primitives.Box(extents=np.array([0.11,0.11,0.11])))
        
        for k in range(1, len(self.sol_act)):#:
            state=self.sol_state[k]         
            if not use_gt:
                obs_env.set_env_state(obs_states[k], reset=False)
            
            self.env.set_env_state(state, reset=False)
            rgb=self.env.model.render(height=480, width=640, camera_id=1, depth=False)
            if not use_gt:
                obs_rgb=obs_env.model.render(height=480, width=640, camera_id=1, depth=False)
            if use_gt:
                pixels+=[rgb]
            else:
                pixels+=[np.concatenate((rgb, obs_rgb), axis=1)]
        print(pose_errors)
        self.env.mujoco_render_frames = False
        save_images_as_video(env_name, np.array(pixels), save_dir)
        
        
        
        
        
        
        
        