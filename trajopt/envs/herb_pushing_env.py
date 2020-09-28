import numpy as np
from gym import utils
from trajopt.envs import mujoco_env
import fileinput
import scipy
from pose_model_estimator import hollow_dense_pointcloud
from genre.voxelization import voxel

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

class HerbEnv(mujoco_env.MujocoEnv):
    #@profile
    def __init__(self, path, palm_mesh_vertices, run_num, task='easy', obs=False, push_mesh_vertices=None, target_mesh_vertices=np.zeros((1,3)), state_arm_pos=None, skip=2):

        # trajopt specific attributes
        self.obs=obs
        self.task=task
        self.env_name = 'herb_pushing_easy'
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0
        self.run_num=run_num

        # placeholder
        self.hand_sid = -2
        self.target_sid = -1

        self.hand_sid = "herb/wam_1/bhand//unnamed_geom_0"
        
        if self.obs:
            self.block_sid_1 = "gen_body_0"
            self.target_sid_1 = "gen_body_1"
        else:
            self.block_sid_1 = "push_obj"
            self.target_sid_1 = "target_obj"
            self.block_sid_1 = "gen_body_0"
            self.target_sid_1 = "gen_body_1"

        self.target_sid_2 = "//unnamed_geom_16"
        self.block_sid_3 = "//unnamed_geom_10"
        self.target_sid_3 = "//unnamed_geom_17"
        
        self.palm_name="herb/wam_1/bhand//unnamed_geom_0"
        
        self.push_mesh_vertices=push_mesh_vertices
        self.target_mesh_vertices=target_mesh_vertices

        self.last_block_pos=None
        self.hand_target_dists=[]
        self.finger_controls=[]
        self.contacting=[]
        self.times_closed=0
        
        if task=='data_gen':
            self.init_qpos= np.array([-0.546, -1.47, -0.868, 1.73, -1.13, 0.0,
                                    1.59, 0, 0, 0, 0, 0,
                                    0, 0, 0
                                    ])
            self.init_qpos[:state_arm_pos.shape[0]]=state_arm_pos
        elif task=='hard_pushing' or task=='grasping':
            self.init_qpos= np.array([-0.858, -1.23, -1.12, 1.38, -2.69, -0.725,
                                    0.24, 0, 0, 0, 0, 0,
                                    0, 0, 0
                                    ])
        elif task=='easy_pushing':
            self.init_qpos= np.array([-1.33, -0.992, -1.65, 1.08, -2.69, -1.19,
                                    0.0, 0, 0, 0, 0, 0,
                                    0, 0, 0
                                    ])

        mujoco_env.MujocoEnv.__init__(self, path, skip)
        utils.EzPickle.__init__(self)
        self.observation_dim = 66
        self.action_dim = 15
        
        self.set_palm_verts(palm_mesh_vertices)

        self.dist_improvement_threshold=0.0025
        self.dist_improvement_window=2
        self.last_state=0
        
        
        self.robot_reset()
        self.target_palm_rot=np.array([[0,0,-1], [0,1,0], [1,0,0]])
        
        if push_mesh_vertices!=None:
            scale=2*np.amax(np.abs(self.push_mesh_vertices.bounds))
            scale_mat=np.eye(4)
            scale_mat=scale_mat/scale
            scale_mat[3,3]=1.0
            s_trans_push_mesh=self.push_mesh_vertices.copy().apply_transform(scale_mat)
            self.target_mesh_dense_vertices=voxel.voxelize_model_binvox(s_trans_push_mesh, 32, self.run_num, binvox_add_param='-bb -.5 -.5 -.5 .5 .5 .5', use_cuda_vox=False)
            self.target_mesh_dense_vertices=(np.argwhere(hollow_dense_pointcloud(self.target_mesh_dense_vertices))/32.0-0.5)*scale
            
            trans_push_mesh=self.push_mesh_vertices.copy()
            transform=np.eye(4)
            transform[0:3,0:3]=np.reshape(self.model.named.data.xmat[self.block_sid_1],(3,3))
            trans_push_mesh.apply_transform(transform)
            transform=np.eye(4)
            transform[0:3,3]=self.model.named.data.xpos[self.block_sid_1]
            trans_push_mesh.apply_transform(transform)
            self.target_mesh_vertices=trans_push_mesh.vertices
            
            if self.task=='hard_pushing':
                target_pos_1 = np.array([0.2,-0.2,0.0])
                self.target_mesh_vertices=self.target_mesh_vertices+target_pos_1
            elif self.task=='easy_pushing':
                target_pos_1 = np.array([0.05,0.3,0.0])
                self.target_mesh_vertices=self.target_mesh_vertices+target_pos_1
            elif self.task=='grasping':
                target_pos_1 = np.array([0.0,0.0,0.2])
                self.target_mesh_vertices=self.target_mesh_vertices+target_pos_1
    
    def set_palm_verts(self, palm_mesh_vertices):
        if palm_mesh_vertices is not None:
            self.palm_mesh_dense_vertices=palm_mesh_vertices
            self.palm_mesh_vertices=palm_mesh_vertices
    
    def get_state(self):
        if self.last_state==0 and (len(self.hand_target_dists)<10 or (self.hand_target_dists[-1]>0.0025 and np.min(np.array(self.hand_target_dists))>=np.min(np.array(self.hand_target_dists)[-min(10, len(self.hand_target_dists)):]))):#self.last_state==0 and (len(self.hand_target_dists)<10 or (self.hand_target_dists[-1]>0.005 and np.min(np.array(self.hand_target_dists))>=np.min(np.array(self.hand_target_dists)[-min(20, len(self.hand_target_dists)):]))):# )):
            state=0 #hasn't gotten near object yet
        elif len(self.finger_controls)==0 or np.sum(np.array(self.finger_controls)>0)<75:
            state=1 #hasn't closed hand yet
            self.last_state=1
        else:
            state=2 #has closed hand around object
            self.last_state=2
        return state
    
    def step(self, a, compute_tpd=False, compute_stats=True):
        state=self.get_state()
        
        self.do_simulation(a, self.frame_skip)
        
        if not compute_stats:
            return None
        
        if self.task!='data_gen':
            
            #compute transforms of target mesh and hand
            trans_push_mesh=self.push_mesh_vertices.copy()
            transform=np.eye(4)
            transform[0:3,0:3]=np.reshape(self.model.named.data.xmat[self.block_sid_1],(3,3))
            transform[0:3,3]=self.model.named.data.xpos[self.block_sid_1]
            trans_push_mesh.apply_transform(transform)
            trans_push_mesh_vertices=trans_push_mesh.vertices
            
            trans_push_mesh_dense_vertices=np.copy(self.target_mesh_dense_vertices)
            trans_push_mesh_dense_vertices=np.matmul(transform[0:3,0:3], trans_push_mesh_dense_vertices.T).T
            trans_push_mesh_dense_vertices=trans_push_mesh_dense_vertices+self.model.named.data.xpos[self.block_sid_1]
            
            if state<2 and np.argwhere(trans_push_mesh_vertices[:,2]<0.42).shape[0]>0:
                block_mean_pos=np.mean(trans_push_mesh_vertices[np.argwhere(trans_push_mesh_vertices[:,2]<0.42)][:,0,:], axis=0)
            else:
                block_mean_pos=np.mean(trans_push_mesh_vertices, axis=0)
            
            palm_rot=np.reshape(self.model.named.data.geom_xmat[self.palm_name],(3,3))
            
            transformed_known_mesh=np.copy(self.palm_mesh_vertices)
            transformed_known_mesh=np.matmul(np.reshape(self.model.named.data.geom_xmat[self.palm_name],(3,3)), transformed_known_mesh.T).T+self.model.named.data.geom_xpos[self.palm_name]
            trans_palm_mesh_vertices=transformed_known_mesh#.vertices
        
            transformed_palm_mesh_dense=np.copy(self.palm_mesh_dense_vertices)
            transformed_palm_mesh_dense=np.matmul(np.reshape(self.model.named.data.geom_xmat[self.palm_name],(3,3)), transformed_palm_mesh_dense.T).T+self.model.named.data.geom_xpos[self.palm_name]

            #keep claw parallel to table
            z_upr=np.array([[1],[0],[0]])
            z_up=np.array([[0],[0],[1]])
            rot_z=np.matmul(np.transpose(palm_rot), z_up)
            angle_diff=np.arccos(np.clip(np.dot(z_upr[:,0], rot_z[:,0]), -1.0, 1.0))
            rotation_penalty=-0.5*angle_diff
            
            #compute reward for task
            if self.task=='hard_pushing':
                hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
                block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
                #hand-target object distance
                target_palm_dist=np.mean(np.linalg.norm(block_mean_pos-transformed_palm_mesh_dense, axis=1))
                robot_block_reward = -target_palm_dist
                self.last_block_pos=np.copy(block_pos_1)
                
                #keep hand near table
                hand_penalty=0.0
                if hand_pos[2]>0.38:
                    hand_penalty=-5*(hand_pos[2]-0.38)
                reward = 2+5*robot_block_reward
                if state<2:
                    reward=reward+hand_penalty+rotation_penalty
                if state==2:
                    target_loss=np.linalg.norm(np.mean(trans_push_mesh_vertices, axis=0)-np.mean(self.target_mesh_vertices, axis=0))
                    reward=reward-target_loss
            elif self.task=='easy_pushing':
                hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
                block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
                #hand-target object distance
                target_palm_dist=target_palm_dist=np.mean(np.linalg.norm(block_mean_pos-transformed_palm_mesh_dense, axis=1))
                robot_block_reward = -target_palm_dist
                self.last_block_pos=np.copy(block_pos_1)
                
                #keep hand near table
                hand_penalty=0.0
                if hand_pos[2]>0.38:
                    hand_penalty=-5*(hand_pos[2]-0.38)
                reward = 2+5*robot_block_reward
                if state<2:
                    reward=reward+hand_penalty+rotation_penalty
                if state==2:
                    target_loss=np.linalg.norm(np.mean(trans_push_mesh_vertices, axis=0)-np.mean(self.target_mesh_vertices, axis=0))
                    reward=reward-target_loss
            elif self.task=='grasping':
                hand_pos = self.model.named.data.geom_xpos[self.hand_sid]
                block_pos_1 = self.model.named.data.xpos[self.block_sid_1]
                #hand-target object distance
                target_palm_dist=np.mean(np.linalg.norm(block_mean_pos-transformed_palm_mesh_dense, axis=1))
                robot_block_reward = -target_palm_dist
                self.last_block_pos=np.copy(block_pos_1)
                
                #keep hand near table
                hand_penalty=0.0
                if hand_pos[2]>0.38:
                    hand_penalty=-5*(hand_pos[2]-0.38)
                
                if block_pos_1[2]<0.75:
                    height_reward=block_pos_1[2]
                else:
                    height_reward=0.75-(block_pos_1[2]-0.75)
                
                reward = 2+5*robot_block_reward
                if state<2:
                    reward=reward+hand_penalty+rotation_penalty
                if state==2 or state==3:
                    target_loss=np.linalg.norm(np.mean(trans_push_mesh_vertices, axis=0)-np.mean(self.target_mesh_vertices, axis=0))
                    reward=reward-target_loss
            if compute_tpd:
                target_palm_dist=np.amin(scipy.spatial.distance.cdist(trans_push_mesh_dense_vertices, transformed_palm_mesh_dense, 'euclidean').T)
                self.hand_target_dists.append(target_palm_dist)
        else:
            reward=0
        
        ob = self.model.position()
        self.env_timestep += 1   
        self.finger_controls.append(a[8])
        return ob, reward, False, self.get_env_infos()
    
    def _get_obs(self):
        return np.concatenate([
            self.model.position(),
            self.model.velocity(),
            self.model.named.data.geom_xpos[self.hand_sid],
            self.model.named.data.xpos[self.block_sid_1]
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
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(), timestep=self.env_timestep)

    #@profile
    def set_env_state(self, state, reset=True):
        if reset:
            self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
        self.env_timestep = state['timestep']
        self.last_block_pos=None

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        u=0
