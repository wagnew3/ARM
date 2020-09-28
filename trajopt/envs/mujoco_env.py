"""
This is a class structure for mujoco environments.
Base functions inherited from gym.
Additional functions needed for trajectory optimization algorithms are included.
"""

import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import time as timer
import dm_control.mujoco as mujoco
from dm_control.rl import control
import trajopt.envs.tasks as tasks
from pose_model_estimator import pose_model_estimator

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    #@profile
    def __init__(self, model_path, frame_skip, has_robot=True):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco.Physics.from_xml_path(model_path)
        task=tasks.TouchTable()
        self.sim = control.Environment(self.model, task, time_limit=1000, control_timestep=0.01)
        self.data = self.sim.physics._data

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        if has_robot:
            self.init_qpos = np.concatenate((self.init_qpos, self.data.qpos.ravel().copy()[self.init_qpos.shape[0]:]))#self.data.qpos.ravel().copy()
        else:
            self.init_qpos=self.data.qpos.ravel().copy()
        self.init_qvel = np.zeros(self.data.qvel.ravel().copy().shape)
        
        if has_robot:
            self.robot_reset()
        observation=self.model.position()
        self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        bounds = self.model.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def mj_viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function.
        """
        pass

    def get_env_state(self):
        return np.concatenate((self.data.qpos.ravel(), self.data.qvel.ravel()))

    def set_env_state(self, state):
        self.sim.physics.set_state(state)

    # -----------------------------

    def _reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    #@profile
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.model.nq,) and qvel.shape == (self.model.model.nv,)
        self.sim.physics.set_state(np.concatenate((qpos, qvel)))
        self.sim.physics.forward()

    @property
    def dt(self):
        return self.model.model.opt.timestep * self.frame_skip

    #@profile
    def do_simulation(self, ctrl, n_frames):
        ctrl[7]=0
        ctrl[10]=0
        ctrl[9]=0
        ctrl[12]=0
        ctrl[14]=0
        ctrl=np.concatenate((ctrl[:7], np.zeros(7), ctrl[7:], np.zeros(8)))
        for _ in range(n_frames):
            self.sim.step(ctrl)

    def mj_render(self):
        u=0
        
    def _get_viewer(self):
        return None

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([
            state.qpos.flat, state.qvel.flat])

    # -----------------------------

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        self.mujoco_render_frames = True
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
        self.mujoco_render_frames = False

    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   frame_size=(640,480),
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   filename='newvid',
                                   camera_name=None):
        import skvideo.io
        for ep in range(num_episodes):
            print("Episode %d: rendering offline " % ep, end='', flush=True)
            o = self.reset()
            d = False
            t = 0
            arrs = []
            t0 = timer.time()
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
                                             mode='offscreen', camera_name=camera_name, device_id=0)
                arrs.append(curr_frame[::-1,:,:])
                print(t, end=', ', flush=True)
            file_name = save_loc + filename + str(ep) + ".mp4"
            skvideo.io.vwrite( file_name, np.asarray(arrs))
            print("saved", file_name)
            t1 = timer.time()
            print("time taken = %f"% (t1-t0))