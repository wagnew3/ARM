import numpy as np
from gym import utils
from trajopt.envs import mujoco_env
import os
import dm_control.mujoco as mujoco
import fileinput
import trimesh
import scipy

def prep_XML(xml_path, replacement_path):
    """
    Prepares MJCF XML code. Replaces mesh and textures directory for compilter
    with that specified in our configuration file.
    """
    compileToReplace = '<compiler coordinate="local" angle="radian" fusestatic="false" meshdir="{}" texturedir="{}"/>'
    compileToReplace = compileToReplace.format(replacement_path, replacement_path)
    for line in fileinput.input(xml_path, inplace=True): 
        if "compiler " in line:
            print(compileToReplace)
        else:
            print(line.rstrip())

class Randomized_Pushing_Env(mujoco_env.MujocoEnv, utils.EzPickle):
    #@profile
    def __init__(self, path='/home/willie/workspace/SSC/herb_reconf/scene.xml', task='easy', obs=False, push_mesh_vertices=np.zeros((1,3)), target_mesh_vertices=np.zeros((1,3)), shapenet_path='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/ShapeNetCore.v2/'):

        # trajopt specific attributes
        self.obs=obs
        self.task=task
        self.env_name = 'herb_pushing_easy'
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0

        # placeholder
        self.hand_sid = -2
        self.target_sid = -1
        
        #prep_XML(path, '/home/willie/workspace/herbpushing/herb_reconf')
        
        self.model = mujoco.Physics.from_xml_path(path)
        self.model.forward()
        
        a=self.model.named.data.xpos.axes.row.names
        
        self.hand_sid = "herb/wam_1/bhand//unnamed_geom_0"
        
        if self.obs:
            self.block_sid_1 = "gen_body_0"
        else:
            self.block_sid_1 = "push_obj"
            
        self.target_sid_1 = "//unnamed_geom_15"
        self.block_sid_2 = "//unnamed_geom_9"
        self.target_sid_2 = "//unnamed_geom_16"
        self.block_sid_3 = "//unnamed_geom_10"
        self.target_sid_3 = "//unnamed_geom_17"
        
        self.push_mesh_vertices=push_mesh_vertices
        self.target_mesh_vertices=target_mesh_vertices
        
        
        
        self.last_block_pos=None
        
        
        self.init_qpos= np.array([-1.48, -1.07, -1.48, 0.899, 0, 1.12,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0
                                ])#, 0.13900801576105609, -0.42142641007555215, 0.3549998,0,0,0,0
        #self.init_qpos[:]=0.0

        mujoco_env.MujocoEnv.__init__(self, path, 1)
        utils.EzPickle.__init__(self)
        self.observation_dim = 66
        self.action_dim = 15

        
        
        self.robot_reset()
        
    #@profile
    def _step(self, a):
        s_ob = self._get_obs()
#         if a.shape[0]==15:
#             zero_vel_a=np.zeros(2*a.shape[0])
#             zero_vel_a[0:15]=a
#             #zero_vel_a[14:22]=a[7:15]
#             a=zero_vel_a
        self.do_simulation(a, self.frame_skip)
        hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
        
        
        
        #target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        #mesh=self.model.named.model.geom_dataid[self.block_sid_1]
        
        block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
        block_orientation_1=np.reshape(self.model.named.data.xmat[self.block_sid_1], (3,3))
        trans_push_mesh_vertices=np.matmul(block_orientation_1, self.push_mesh_vertices.T).T
        trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
        
        target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        trans_target_mesh_vertices=np.matmul(target_orientation_1, self.target_mesh_vertices.T).T
        trans_target_mesh_vertices=trans_target_mesh_vertices+target_pos_1
        
        
        #target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        
#         block_pos_2 = self.model.named.data.geom_xpos[self.block_sid_2]
#         block_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.block_sid_2], (3,3))
#         target_pos_2 = self.model.named.data.geom_xpos[self.target_sid_2]
#         target_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.target_sid_2], (3,3))
#         
#         block_pos_3 = self.model.named.data.geom_xpos[self.block_sid_3]
#         block_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.block_sid_3], (3,3))
#         target_pos_3 = self.model.named.data.geom_xpos[self.target_sid_3]
#         target_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.target_sid_3], (3,3))

        if self.task=='easy':
            #cube_target_ADDS=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            target_loss=np.max(scipy.spatial.distance.cdist(trans_push_mesh_vertices, trans_target_mesh_vertices, 'euclidean'))#scipy.spatial.distance.directed_hausdorff(trans_push_mesh_vertices, trans_target_mesh_vertices)[0]
            robot_block_reward = -np.linalg.norm(hand_pos - block_pos_1)
            
            #print('hand_pos', hand_pos, 'block_pos_1', block_pos_1, 'target_pos_1', target_pos_1)
            
            vel_penalty=0.0
            if not self.last_block_pos is None:
                velocity=np.linalg.norm(self.last_block_pos-block_pos_1)
                if velocity>0.01:
                    vel_penalty=-100*velocity
            self.last_block_pos=np.copy(block_pos_1)
            #a=np.linalg.norm(self.data.qvel)
            reward = 1+0.1 * robot_block_reward+-target_loss#-0.001*np.linalg.norm(self.data.qvel)#+
        elif self.task=='three_blocks':
            cube_target_ADDS_1=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            robot_block_reward_1 = -np.sum(np.abs(hand_pos - block_pos_1))
            
            cube_target_ADDS_2=self.get_cubes_ADDS(block_pos_2, target_pos_2, block_orientation_2, target_orientation_2, 0.055)
            robot_block_reward_2 = -np.sum(np.abs(hand_pos - block_pos_2))
            
            cube_target_ADDS_3=self.get_cubes_ADDS(block_pos_3, target_pos_3, block_orientation_3, target_orientation_3, 0.055)
            robot_block_reward_3 = -np.sum(np.abs(hand_pos - block_pos_3))
            
            if cube_target_ADDS_1>0.05:
                reward = 0.01*robot_block_reward_1+-cube_target_ADDS_1
            elif cube_target_ADDS_2>0.05:
                reward = 0.01*robot_block_reward_2+-cube_target_ADDS_2
            elif cube_target_ADDS_3>0.05:
                reward = 0.01*robot_block_reward_3+-cube_target_ADDS_3
            
            
        ob = self.model.position()
        # keep track of env timestep (needed for continual envs)
        self.env_timestep += 1
        return ob, reward, False, self.get_env_infos()
    
    def get_dist(self):   
        hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
        
        
        
        #target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        #mesh=self.model.named.model.geom_dataid[self.block_sid_1]
        
        block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
        block_orientation_1=np.reshape(self.model.named.data.xmat[self.block_sid_1], (3,3))
        trans_push_mesh_vertices=np.matmul(block_orientation_1, self.push_mesh_vertices.T).T
        trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
        
        target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        trans_target_mesh_vertices=np.matmul(target_orientation_1, self.target_mesh_vertices.T).T
        trans_target_mesh_vertices=trans_target_mesh_vertices+target_pos_1
        
        
        #target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        
#         block_pos_2 = self.model.named.data.geom_xpos[self.block_sid_2]
#         block_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.block_sid_2], (3,3))
#         target_pos_2 = self.model.named.data.geom_xpos[self.target_sid_2]
#         target_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.target_sid_2], (3,3))
#         
#         block_pos_3 = self.model.named.data.geom_xpos[self.block_sid_3]
#         block_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.block_sid_3], (3,3))
#         target_pos_3 = self.model.named.data.geom_xpos[self.target_sid_3]
#         target_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.target_sid_3], (3,3))

        if self.task=='easy':
            #cube_target_ADDS=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            target_loss=np.max(scipy.spatial.distance.cdist(trans_push_mesh_vertices, trans_target_mesh_vertices, 'euclidean'))#scipy.spatial.distance.directed_hausdorff(trans_push_mesh_vertices, trans_target_mesh_vertices)[0]
            robot_block_reward = -np.linalg.norm(hand_pos - block_pos_1)
            
            #print('hand_pos', hand_pos, 'block_pos_1', block_pos_1, 'target_pos_1', target_pos_1)
            
            vel_penalty=0.0
            if not self.last_block_pos is None:
                velocity=np.linalg.norm(self.last_block_pos-block_pos_1)
                if velocity>0.01:
                    vel_penalty=-100*velocity
            self.last_block_pos=np.copy(block_pos_1)
            #a=np.linalg.norm(self.data.qvel)
            reward = 1+0.1 * robot_block_reward+-target_loss#-0.001*np.linalg.norm(self.data.qvel)#+

        return target_loss
    #@profile
    def get_cubes_ADDS(self, cube_1_position, cube_2_position, cube_1_orientation, cube_2_orientation, side_length):
        
        cube_points=np.array([
            [0,0,0],
            [side_length,0,0],
            [0,side_length,0],
            [side_length,side_length,0],
            [0,0,side_length],
            [side_length,0,side_length],
            [0,side_length,side_length],
            [side_length,side_length,side_length]
            ])
        
        cube_1_points=np.zeros(cube_points.shape)
        cube_2_points=np.zeros(cube_points.shape)
        for point_ind in range(8):
            cube_1_points[point_ind]=np.matmul(cube_1_orientation, cube_points[point_ind])
            cube_2_points[point_ind]=np.matmul(cube_2_orientation, cube_points[point_ind])
        
        total_distance=0.0
        for point_1_ind in range(8):
            best_distance=float('inf')
            distances=np.linalg.norm(cube_1_points[point_1_ind]-cube_2_points, axis=1)
            best_distance=np.amin(distances)
            total_distance+=best_distance
        
        total_distance+=8*np.linalg.norm(cube_1_position-cube_2_position)
        
        return total_distance/8.0
        

    def step(self, a):
        # overloading to preserve backwards compatibility
        return self._step(a)

    def _get_obs(self):
        if self.task=='easy':
            return np.concatenate([
                self.model.position(),
                self.model.velocity(),
                self.model.named.data.geom_xpos[self.hand_sid],
                self.model.named.data.xpos[self.block_sid_1],
                self.model.named.data.geom_xpos[self.target_sid_1],
            ])
        elif self.task=='three_blocks':
            return np.concatenate([
                self.model.position(),
                self.model.velocity(),
                self.model.named.data.geom_xpos[self.hand_sid],
                self.model.named.data.geom_xpos[self.block_sid_1],
                self.model.named.data.geom_xpos[self.target_sid_1],
                self.model.named.data.geom_xpos[self.block_sid_2],
                self.model.named.data.geom_xpos[self.target_sid_2],
                self.model.named.data.geom_xpos[self.block_sid_3],
                self.model.named.data.geom_xpos[self.target_sid_3],
            ])

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        target_pos = np.array([0.1, 0.1, 0.1])
        if self.seeding is True:
            target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
            target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
            target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        #self.model.named.data.geom_xpos[self.target_sid] = target_pos
        self.sim.physics.forward()

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        return self._get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.named.data.geom_xpos[self.target_sid_1].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.set_state(qp, qv)
        self.model.named.data.geom_xpos[self.target_sid_1] = target_pos
        self.env_timestep = state['timestep']
        self.last_block_pos=None

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
       u=0
import numpy as np
from gym import utils
from trajopt.envs import mujoco_env
import os
import dm_control.mujoco as mujoco
import fileinput
import trimesh
import scipy

def prep_XML(xml_path, replacement_path):
    """
    Prepares MJCF XML code. Replaces mesh and textures directory for compilter
    with that specified in our configuration file.
    """
    compileToReplace = '<compiler coordinate="local" angle="radian" fusestatic="false" meshdir="{}" texturedir="{}"/>'
    compileToReplace = compileToReplace.format(replacement_path, replacement_path)
    for line in fileinput.input(xml_path, inplace=True): 
        if "compiler " in line:
            print(compileToReplace)
        else:
            print(line.rstrip())

class HerbEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    #@profile
    def __init__(self, path='/home/willie/workspace/SSC/herb_reconf/scene.xml', task='easy', obs=False, push_mesh_vertices=np.zeros((1,3)), target_mesh_vertices=np.zeros((1,3))):

        # trajopt specific attributes
        self.obs=obs
        self.task=task
        self.env_name = 'herb_pushing_easy'
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0

        # placeholder
        self.hand_sid = -2
        self.target_sid = -1
        
        #prep_XML(path, '/home/willie/workspace/herbpushing/herb_reconf')
        
        self.model = mujoco.Physics.from_xml_path(path)
        self.model.forward()
        
        a=self.model.named.data.xpos.axes.row.names
        
        self.hand_sid = "herb/wam_1/bhand//unnamed_geom_0"
        
        if self.obs:
            self.block_sid_1 = "gen_body_0"
        else:
            self.block_sid_1 = "push_obj"
            
        self.target_sid_1 = "//unnamed_geom_15"
        self.block_sid_2 = "//unnamed_geom_9"
        self.target_sid_2 = "//unnamed_geom_16"
        self.block_sid_3 = "//unnamed_geom_10"
        self.target_sid_3 = "//unnamed_geom_17"
        
        self.push_mesh_vertices=push_mesh_vertices
        self.target_mesh_vertices=target_mesh_vertices
        
        
        
        self.last_block_pos=None
        
        
        self.init_qpos= np.array([-1.48, -1.07, -1.48, 0.899, 0, 1.12,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0
                                ])#, 0.13900801576105609, -0.42142641007555215, 0.3549998,0,0,0,0
        #self.init_qpos[:]=0.0

        mujoco_env.MujocoEnv.__init__(self, path, 1)
        utils.EzPickle.__init__(self)
        self.observation_dim = 66
        self.action_dim = 15

        
        
        self.robot_reset()
        
    #@profile
    def _step(self, a):
        s_ob = self._get_obs()
#         if a.shape[0]==15:
#             zero_vel_a=np.zeros(2*a.shape[0])
#             zero_vel_a[0:15]=a
#             #zero_vel_a[14:22]=a[7:15]
#             a=zero_vel_a
        self.do_simulation(a, self.frame_skip)
        hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
        
        
        
        #target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        #mesh=self.model.named.model.geom_dataid[self.block_sid_1]
        
        block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
        block_orientation_1=np.reshape(self.model.named.data.xmat[self.block_sid_1], (3,3))
        trans_push_mesh_vertices=np.matmul(block_orientation_1, self.push_mesh_vertices.T).T
        trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
        
        target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        trans_target_mesh_vertices=np.matmul(target_orientation_1, self.target_mesh_vertices.T).T
        trans_target_mesh_vertices=trans_target_mesh_vertices+target_pos_1
        
        
        #target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        
#         block_pos_2 = self.model.named.data.geom_xpos[self.block_sid_2]
#         block_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.block_sid_2], (3,3))
#         target_pos_2 = self.model.named.data.geom_xpos[self.target_sid_2]
#         target_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.target_sid_2], (3,3))
#         
#         block_pos_3 = self.model.named.data.geom_xpos[self.block_sid_3]
#         block_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.block_sid_3], (3,3))
#         target_pos_3 = self.model.named.data.geom_xpos[self.target_sid_3]
#         target_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.target_sid_3], (3,3))

        if self.task=='easy':
            #cube_target_ADDS=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            target_loss=np.max(scipy.spatial.distance.cdist(trans_push_mesh_vertices, trans_target_mesh_vertices, 'euclidean'))#scipy.spatial.distance.directed_hausdorff(trans_push_mesh_vertices, trans_target_mesh_vertices)[0]
            robot_block_reward = -np.linalg.norm(hand_pos - block_pos_1)
            
            #print('hand_pos', hand_pos, 'block_pos_1', block_pos_1, 'target_pos_1', target_pos_1)
            
            vel_penalty=0.0
            if not self.last_block_pos is None:
                velocity=np.linalg.norm(self.last_block_pos-block_pos_1)
                if velocity>0.01:
                    vel_penalty=-100*velocity
            self.last_block_pos=np.copy(block_pos_1)
            #a=np.linalg.norm(self.data.qvel)
            reward = 1+0.1 * robot_block_reward+-target_loss#-0.001*np.linalg.norm(self.data.qvel)#+
        elif self.task=='three_blocks':
            cube_target_ADDS_1=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            robot_block_reward_1 = -np.sum(np.abs(hand_pos - block_pos_1))
            
            cube_target_ADDS_2=self.get_cubes_ADDS(block_pos_2, target_pos_2, block_orientation_2, target_orientation_2, 0.055)
            robot_block_reward_2 = -np.sum(np.abs(hand_pos - block_pos_2))
            
            cube_target_ADDS_3=self.get_cubes_ADDS(block_pos_3, target_pos_3, block_orientation_3, target_orientation_3, 0.055)
            robot_block_reward_3 = -np.sum(np.abs(hand_pos - block_pos_3))
            
            if cube_target_ADDS_1>0.05:
                reward = 0.01*robot_block_reward_1+-cube_target_ADDS_1
            elif cube_target_ADDS_2>0.05:
                reward = 0.01*robot_block_reward_2+-cube_target_ADDS_2
            elif cube_target_ADDS_3>0.05:
                reward = 0.01*robot_block_reward_3+-cube_target_ADDS_3
            
            
        ob = self.model.position()
        # keep track of env timestep (needed for continual envs)
        self.env_timestep += 1
        return ob, reward, False, self.get_env_infos()
    
    def get_dist(self):   
        hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
        
        
        
        #target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        #mesh=self.model.named.model.geom_dataid[self.block_sid_1]
        
        block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
        block_orientation_1=np.reshape(self.model.named.data.xmat[self.block_sid_1], (3,3))
        trans_push_mesh_vertices=np.matmul(block_orientation_1, self.push_mesh_vertices.T).T
        trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
        
        target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        trans_target_mesh_vertices=np.matmul(target_orientation_1, self.target_mesh_vertices.T).T
        trans_target_mesh_vertices=trans_target_mesh_vertices+target_pos_1
        
        
        #target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        
#         block_pos_2 = self.model.named.data.geom_xpos[self.block_sid_2]
#         block_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.block_sid_2], (3,3))
#         target_pos_2 = self.model.named.data.geom_xpos[self.target_sid_2]
#         target_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.target_sid_2], (3,3))
#         
#         block_pos_3 = self.model.named.data.geom_xpos[self.block_sid_3]
#         block_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.block_sid_3], (3,3))
#         target_pos_3 = self.model.named.data.geom_xpos[self.target_sid_3]
#         target_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.target_sid_3], (3,3))

        if self.task=='easy':
            #cube_target_ADDS=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            target_loss=np.max(scipy.spatial.distance.cdist(trans_push_mesh_vertices, trans_target_mesh_vertices, 'euclidean'))#scipy.spatial.distance.directed_hausdorff(trans_push_mesh_vertices, trans_target_mesh_vertices)[0]
            robot_block_reward = -np.linalg.norm(hand_pos - block_pos_1)
            
            #print('hand_pos', hand_pos, 'block_pos_1', block_pos_1, 'target_pos_1', target_pos_1)
            
            vel_penalty=0.0
            if not self.last_block_pos is None:
                velocity=np.linalg.norm(self.last_block_pos-block_pos_1)
                if velocity>0.01:
                    vel_penalty=-100*velocity
            self.last_block_pos=np.copy(block_pos_1)
            #a=np.linalg.norm(self.data.qvel)
            reward = 1+0.1 * robot_block_reward+-target_loss#-0.001*np.linalg.norm(self.data.qvel)#+

        return target_loss
    #@profile
    def get_cubes_ADDS(self, cube_1_position, cube_2_position, cube_1_orientation, cube_2_orientation, side_length):
        
        cube_points=np.array([
            [0,0,0],
            [side_length,0,0],
            [0,side_length,0],
            [side_length,side_length,0],
            [0,0,side_length],
            [side_length,0,side_length],
            [0,side_length,side_length],
            [side_length,side_length,side_length]
            ])
        
        cube_1_points=np.zeros(cube_points.shape)
        cube_2_points=np.zeros(cube_points.shape)
        for point_ind in range(8):
            cube_1_points[point_ind]=np.matmul(cube_1_orientation, cube_points[point_ind])
            cube_2_points[point_ind]=np.matmul(cube_2_orientation, cube_points[point_ind])
        
        total_distance=0.0
        for point_1_ind in range(8):
            best_distance=float('inf')
            distances=np.linalg.norm(cube_1_points[point_1_ind]-cube_2_points, axis=1)
            best_distance=np.amin(distances)
            total_distance+=best_distance
        
        total_distance+=8*np.linalg.norm(cube_1_position-cube_2_position)
        
        return total_distance/8.0
        

    def step(self, a):
        # overloading to preserve backwards compatibility
        return self._step(a)

    def _get_obs(self):
        if self.task=='easy':
            return np.concatenate([
                self.model.position(),
                self.model.velocity(),
                self.model.named.data.geom_xpos[self.hand_sid],
                self.model.named.data.xpos[self.block_sid_1],
                self.model.named.data.geom_xpos[self.target_sid_1],
            ])
        elif self.task=='three_blocks':
            return np.concatenate([
                self.model.position(),
                self.model.velocity(),
                self.model.named.data.geom_xpos[self.hand_sid],
                self.model.named.data.geom_xpos[self.block_sid_1],
                self.model.named.data.geom_xpos[self.target_sid_1],
                self.model.named.data.geom_xpos[self.block_sid_2],
                self.model.named.data.geom_xpos[self.target_sid_2],
                self.model.named.data.geom_xpos[self.block_sid_3],
                self.model.named.data.geom_xpos[self.target_sid_3],
            ])

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        target_pos = np.array([0.1, 0.1, 0.1])
        if self.seeding is True:
            target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
            target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
            target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        #self.model.named.data.geom_xpos[self.target_sid] = target_pos
        self.sim.physics.forward()

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        return self._get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.named.data.geom_xpos[self.target_sid_1].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.set_state(qp, qv)
        self.model.named.data.geom_xpos[self.target_sid_1] = target_pos
        self.env_timestep = state['timestep']
        self.last_block_pos=None

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
       u=0
import numpy as np
from gym import utils
from trajopt.envs import mujoco_env
import os
import dm_control.mujoco as mujoco
import fileinput
import trimesh
import scipy

def prep_XML(xml_path, replacement_path):
    """
    Prepares MJCF XML code. Replaces mesh and textures directory for compilter
    with that specified in our configuration file.
    """
    compileToReplace = '<compiler coordinate="local" angle="radian" fusestatic="false" meshdir="{}" texturedir="{}"/>'
    compileToReplace = compileToReplace.format(replacement_path, replacement_path)
    for line in fileinput.input(xml_path, inplace=True): 
        if "compiler " in line:
            print(compileToReplace)
        else:
            print(line.rstrip())

class HerbEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    #@profile
    def __init__(self, path='/home/willie/workspace/SSC/herb_reconf/scene.xml', task='easy', obs=False, push_mesh_vertices=np.zeros((1,3)), target_mesh_vertices=np.zeros((1,3))):

        # trajopt specific attributes
        self.obs=obs
        self.task=task
        self.env_name = 'herb_pushing_easy'
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0

        # placeholder
        self.hand_sid = -2
        self.target_sid = -1
        
        #prep_XML(path, '/home/willie/workspace/herbpushing/herb_reconf')
        
        self.model = mujoco.Physics.from_xml_path(path)
        self.model.forward()
        
        a=self.model.named.data.xpos.axes.row.names
        
        self.hand_sid = "herb/wam_1/bhand//unnamed_geom_0"
        
        if self.obs:
            self.block_sid_1 = "gen_body_0"
        else:
            self.block_sid_1 = "push_obj"
            
        self.target_sid_1 = "//unnamed_geom_15"
        self.block_sid_2 = "//unnamed_geom_9"
        self.target_sid_2 = "//unnamed_geom_16"
        self.block_sid_3 = "//unnamed_geom_10"
        self.target_sid_3 = "//unnamed_geom_17"
        
        self.push_mesh_vertices=push_mesh_vertices
        self.target_mesh_vertices=target_mesh_vertices
        
        
        
        self.last_block_pos=None
        
        
        self.init_qpos= np.array([-1.48, -1.07, -1.48, 0.899, 0, 1.12,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0
                                ])#, 0.13900801576105609, -0.42142641007555215, 0.3549998,0,0,0,0
        #self.init_qpos[:]=0.0

        mujoco_env.MujocoEnv.__init__(self, path, 1)
        utils.EzPickle.__init__(self)
        self.observation_dim = 66
        self.action_dim = 15

        
        
        self.robot_reset()
        
    #@profile
    def _step(self, a):
        s_ob = self._get_obs()
#         if a.shape[0]==15:
#             zero_vel_a=np.zeros(2*a.shape[0])
#             zero_vel_a[0:15]=a
#             #zero_vel_a[14:22]=a[7:15]
#             a=zero_vel_a
        self.do_simulation(a, self.frame_skip)
        hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
        
        
        
        #target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        #mesh=self.model.named.model.geom_dataid[self.block_sid_1]
        
        block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
        block_orientation_1=np.reshape(self.model.named.data.xmat[self.block_sid_1], (3,3))
        trans_push_mesh_vertices=np.matmul(block_orientation_1, self.push_mesh_vertices.T).T
        trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
        
        target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        trans_target_mesh_vertices=np.matmul(target_orientation_1, self.target_mesh_vertices.T).T
        trans_target_mesh_vertices=trans_target_mesh_vertices+target_pos_1
        
        
        #target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        
#         block_pos_2 = self.model.named.data.geom_xpos[self.block_sid_2]
#         block_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.block_sid_2], (3,3))
#         target_pos_2 = self.model.named.data.geom_xpos[self.target_sid_2]
#         target_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.target_sid_2], (3,3))
#         
#         block_pos_3 = self.model.named.data.geom_xpos[self.block_sid_3]
#         block_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.block_sid_3], (3,3))
#         target_pos_3 = self.model.named.data.geom_xpos[self.target_sid_3]
#         target_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.target_sid_3], (3,3))

        if self.task=='easy':
            #cube_target_ADDS=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            target_loss=np.max(scipy.spatial.distance.cdist(trans_push_mesh_vertices, trans_target_mesh_vertices, 'euclidean'))#scipy.spatial.distance.directed_hausdorff(trans_push_mesh_vertices, trans_target_mesh_vertices)[0]
            robot_block_reward = -np.linalg.norm(hand_pos - block_pos_1)
            
            #print('hand_pos', hand_pos, 'block_pos_1', block_pos_1, 'target_pos_1', target_pos_1)
            
            vel_penalty=0.0
            if not self.last_block_pos is None:
                velocity=np.linalg.norm(self.last_block_pos-block_pos_1)
                if velocity>0.01:
                    vel_penalty=-100*velocity
            self.last_block_pos=np.copy(block_pos_1)
            #a=np.linalg.norm(self.data.qvel)
            reward = 1+0.1 * robot_block_reward+-target_loss#-0.001*np.linalg.norm(self.data.qvel)#+
        elif self.task=='three_blocks':
            cube_target_ADDS_1=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            robot_block_reward_1 = -np.sum(np.abs(hand_pos - block_pos_1))
            
            cube_target_ADDS_2=self.get_cubes_ADDS(block_pos_2, target_pos_2, block_orientation_2, target_orientation_2, 0.055)
            robot_block_reward_2 = -np.sum(np.abs(hand_pos - block_pos_2))
            
            cube_target_ADDS_3=self.get_cubes_ADDS(block_pos_3, target_pos_3, block_orientation_3, target_orientation_3, 0.055)
            robot_block_reward_3 = -np.sum(np.abs(hand_pos - block_pos_3))
            
            if cube_target_ADDS_1>0.05:
                reward = 0.01*robot_block_reward_1+-cube_target_ADDS_1
            elif cube_target_ADDS_2>0.05:
                reward = 0.01*robot_block_reward_2+-cube_target_ADDS_2
            elif cube_target_ADDS_3>0.05:
                reward = 0.01*robot_block_reward_3+-cube_target_ADDS_3
            
            
        ob = self.model.position()
        # keep track of env timestep (needed for continual envs)
        self.env_timestep += 1
        return ob, reward, False, self.get_env_infos()
    
    def get_dist(self):   
        hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
        
        
        
        #target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        #mesh=self.model.named.model.geom_dataid[self.block_sid_1]
        
        block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
        block_orientation_1=np.reshape(self.model.named.data.xmat[self.block_sid_1], (3,3))
        trans_push_mesh_vertices=np.matmul(block_orientation_1, self.push_mesh_vertices.T).T
        trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
        
        target_pos_1 = self.model.named.data.geom_xpos[self.target_sid_1]
        target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        trans_target_mesh_vertices=np.matmul(target_orientation_1, self.target_mesh_vertices.T).T
        trans_target_mesh_vertices=trans_target_mesh_vertices+target_pos_1
        
        
        #target_orientation_1=np.reshape(self.model.named.data.geom_xmat[self.target_sid_1], (3,3))
        
#         block_pos_2 = self.model.named.data.geom_xpos[self.block_sid_2]
#         block_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.block_sid_2], (3,3))
#         target_pos_2 = self.model.named.data.geom_xpos[self.target_sid_2]
#         target_orientation_2=np.reshape(self.model.named.data.geom_xmat[self.target_sid_2], (3,3))
#         
#         block_pos_3 = self.model.named.data.geom_xpos[self.block_sid_3]
#         block_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.block_sid_3], (3,3))
#         target_pos_3 = self.model.named.data.geom_xpos[self.target_sid_3]
#         target_orientation_3=np.reshape(self.model.named.data.geom_xmat[self.target_sid_3], (3,3))

        if self.task=='easy':
            #cube_target_ADDS=self.get_cubes_ADDS(block_pos_1, target_pos_1, block_orientation_1, target_orientation_1, 0.055)
            target_loss=np.max(scipy.spatial.distance.cdist(trans_push_mesh_vertices, trans_target_mesh_vertices, 'euclidean'))#scipy.spatial.distance.directed_hausdorff(trans_push_mesh_vertices, trans_target_mesh_vertices)[0]
            robot_block_reward = -np.linalg.norm(hand_pos - block_pos_1)
            
            #print('hand_pos', hand_pos, 'block_pos_1', block_pos_1, 'target_pos_1', target_pos_1)
            
            vel_penalty=0.0
            if not self.last_block_pos is None:
                velocity=np.linalg.norm(self.last_block_pos-block_pos_1)
                if velocity>0.01:
                    vel_penalty=-100*velocity
            self.last_block_pos=np.copy(block_pos_1)
            #a=np.linalg.norm(self.data.qvel)
            reward = 1+0.1 * robot_block_reward+-target_loss#-0.001*np.linalg.norm(self.data.qvel)#+

        return target_loss
    #@profile
    def get_cubes_ADDS(self, cube_1_position, cube_2_position, cube_1_orientation, cube_2_orientation, side_length):
        
        cube_points=np.array([
            [0,0,0],
            [side_length,0,0],
            [0,side_length,0],
            [side_length,side_length,0],
            [0,0,side_length],
            [side_length,0,side_length],
            [0,side_length,side_length],
            [side_length,side_length,side_length]
            ])
        
        cube_1_points=np.zeros(cube_points.shape)
        cube_2_points=np.zeros(cube_points.shape)
        for point_ind in range(8):
            cube_1_points[point_ind]=np.matmul(cube_1_orientation, cube_points[point_ind])
            cube_2_points[point_ind]=np.matmul(cube_2_orientation, cube_points[point_ind])
        
        total_distance=0.0
        for point_1_ind in range(8):
            best_distance=float('inf')
            distances=np.linalg.norm(cube_1_points[point_1_ind]-cube_2_points, axis=1)
            best_distance=np.amin(distances)
            total_distance+=best_distance
        
        total_distance+=8*np.linalg.norm(cube_1_position-cube_2_position)
        
        return total_distance/8.0
        

    def step(self, a):
        # overloading to preserve backwards compatibility
        return self._step(a)

    def _get_obs(self):
        if self.task=='easy':
            return np.concatenate([
                self.model.position(),
                self.model.velocity(),
                self.model.named.data.geom_xpos[self.hand_sid],
                self.model.named.data.xpos[self.block_sid_1],
                self.model.named.data.geom_xpos[self.target_sid_1],
            ])
        elif self.task=='three_blocks':
            return np.concatenate([
                self.model.position(),
                self.model.velocity(),
                self.model.named.data.geom_xpos[self.hand_sid],
                self.model.named.data.geom_xpos[self.block_sid_1],
                self.model.named.data.geom_xpos[self.target_sid_1],
                self.model.named.data.geom_xpos[self.block_sid_2],
                self.model.named.data.geom_xpos[self.target_sid_2],
                self.model.named.data.geom_xpos[self.block_sid_3],
                self.model.named.data.geom_xpos[self.target_sid_3],
            ])

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos, self.init_qvel)

    def target_reset(self):
        target_pos = np.array([0.1, 0.1, 0.1])
        if self.seeding is True:
            target_pos[0] = self.np_random.uniform(low=-0.3, high=0.3)
            target_pos[1] = self.np_random.uniform(low=-0.2, high=0.2)
            target_pos[2] = self.np_random.uniform(low=-0.25, high=0.25)
        #self.model.named.data.geom_xpos[self.target_sid] = target_pos
        self.sim.physics.forward()

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        self.target_reset()
        return self._get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.named.data.geom_xpos[self.target_sid_1].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.set_state(qp, qv)
        self.model.named.data.geom_xpos[self.target_sid_1] = target_pos
        self.env_timestep = state['timestep']
        self.last_block_pos=None

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
       u=0
