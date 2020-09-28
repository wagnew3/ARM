import sys
sys.path.append('/home/wagnew3/workspace/SSC/')
import os
from optparse import OptionParser
import dill
import scipy
import time

dir_name=os.path.dirname(__file__)
parser = OptionParser()
parser.add_option("--top_dir", dest="top_dir", default='/home/willie/workspace/SSC')
parser.add_option("--task", dest="task", default="hard_pushing")
parser.add_option("--H", type="int", dest="H", default=5)
parser.add_option("--paths_per_cpu", dest="paths_per_cpu", type="int", default=50)
parser.add_option("--steps", dest="steps", type="int", default=120, help="display game frames")
parser.add_option("--num_id", dest="num_id", type="int", default=117) #100: mesh quality studies
parser.add_option("--use_gt", dest="use_gt", type="int", default=1)
parser.add_option("--replay", dest="replay", type="int", default=0)
parser.add_option("--gpu_num", dest="gpu_num", type="int", default=1)
parser.add_option("--reconstruction_quality_level", dest="reconstruction_quality_level", type="float", default=0.01)
parser.add_option("--quality_type", dest="quality_type", default='none') #interp, chamfer, none
parser.add_option("--vis_while_running", dest="vis_while_running", type="int", default=1)
parser.add_option("--restart", dest="restart", type="int", default=0)
parser.add_option("--stability_loss", dest="stability_loss", type="int", default=1)
parser.add_option("--target_object_name", dest="target_object_name", default="036_wood_block")
parser.add_option("--num_cpu", dest="num_cpu", type="int", default=1)
parser.add_option("--save_dir", dest="save_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/sss_saved')
parser.add_option("--use_cuda_vox", dest="use_cuda_vox", type="int", default=1)
parser.add_option("--run_num", dest="run_num", type="int", default=0)
parser.add_option("--total_runs", dest="total_runs", type="int", default=1)
parser.add_option("--save_trajectories", dest="save_trajectories", type="int", default=0)
parser.add_option("--recon_net", dest="recon_net", default='/home/willie/workspace/SSC/genre/logs/0/0013.pt')
parser.add_option("--recon_net_type", dest="recon_net_type", default='genre_given_depth')#_4_channel_predict_other_stability
parser.add_option("--shape_lib", dest="shape_lib", default='downloads')
(options, args) = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(options.gpu_num)
os.environ["OMP_NUM_THREADS"]="1"

import torch
import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Empty

from importlib import reload

import shutil
import traceback

from genre import models
from genre.options import options_test
from genre.loggers import loggers
from genre.voxelization import voxel
import genre.util.util_loadlib as loadlib

import data_augmentation
import data_loader
import segmentation; segmentation = reload(segmentation)

print(options)

from trajopt.algos.mppi import MPPI
from trajopt.envs.utils import get_environment
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
from trajopt.envs.herb_pushing_env import HerbEnv
import cv2
from pose_model_estimator import pose_model_estimator
from pose_model_estimator import compute_mujoco_int_transform, remove_objects_from_mujoco, add_object_to_mujoco
import trimesh
from pose_model_estimator import chamfer_distance

class MyPickler(pickle._Pickler):
    def save(self, obj):
        print('pickling object  {0} of type {1}'.format(obj, type(obj)))
        pickle._Pickler.save(self, obj)

def create_scene_with_mesh(mesh_filename, save_id, top_dir, scene_xml_file, task):
    mujoco_center, _=compute_mujoco_int_transform(mesh_filename, save_id)
    mic=mujoco_center[2]
    mesh=trimesh.load(mesh_filename)
    lower_z=-mic
    z_offset=0.3-mesh.bounds[0,2]
    
    temp_scene_xml_file=os.path.join(top_dir, f'herb_reconf/temp_scene_{save_id}.xml')
    shutil.copyfile(scene_xml_file, temp_scene_xml_file)
    remove_objects_from_mujoco(temp_scene_xml_file, [9])#,10
    if task=='hard_pushing' or task=="grasping":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([0,0,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 0, 0, joint=True, add_mesh_name='init', include_collisions=True) #
    elif task=='easy_pushing':
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([-0.05,-0.35,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 0, 0, joint=True, add_mesh_name='init', include_collisions=True)
    
    if task=="hard_pushing":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([-0.05,-0.35,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 1, 0, joint=False, geom_args=[['contype', '0'], ['conaffinity', '0'], ['group', '1'], ['rgba', '0 0 0 0.0']])
    elif task=="grasping":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([0,0,z_offset+0.2]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 1, 0, joint=False, geom_args=[['contype', '0'], ['conaffinity', '0'], ['group', '1'], ['rgba', '0 0 0 0.0']])
    elif task=="easy_pushing":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([0,0,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 1, 0, joint=False, geom_args=[['contype', '0'], ['conaffinity', '0'], ['group', '1'], ['rgba', '0 0 0 0.1666']])
    return temp_scene_xml_file

ycb_objects=["002_master_chef_can", "003_cracker_box", "005_tomato_soup_can","006_mustard_bottle","013_apple",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "025_mug",
    "036_wood_block",
    "053_mini_soccer_ball",
    "055_baseball",
    "077_rubiks_cube"]

downloaded_objects=[]

#@profile
def run_MPPI(seg_input_queue, seg_output_queue, recon_input_queue, recon_output_queue, task, steps, num_id, use_gt, reconstruction_quality_level, quality_type, target_object_name, top_dir, H, paths_per_cpu, vis_while_running, replay, restart, save_dir, use_cuda_vox, stability_loss, four_channel, seg_model=None, point_completion_model=None):
    
    if target_object_name in ycb_objects:
        mesh_file=f'herb_reconf/assets/ycb_objects/{target_object_name}/google_16k/nontextured.stl'
    else:
        mesh_file=f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{target_object_name}/scene.stl'

    if options.target_object_name is not None:
        scene_name=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/{task}_{target_object_name}_scene.xml')
        #scene_name=create_scene_with_mesh(os.path.join(top_dir, mesh_file), num_id, top_dir, os.path.join(top_dir, f'herb_reconf/{task}_scene.xml'), task)
    else:
        scene_name=f'herb_reconf/{task}_scene.xml'
    
    print('starting', num_id)
    ENV_NAME = f'herb-{task}_40_50_{steps}_{num_id}_{use_gt}_{reconstruction_quality_level}_{quality_type}_{target_object_name}_{stability_loss}'
    PICKLE_FILE = os.path.join(save_dir+'/results/saved_mppi', ENV_NAME + '_mppi.pickle')
    STATS_FILE=os.path.join(save_dir+'/results/mppi_run_stats', ENV_NAME + '_stats.pickle')
#     if os.path.exists(STATS_FILE):
#         return
    
    SEED = num_id
    H_total = 1000
    N_ITER = 1
    
    a=os.path.join(top_dir, 'herb_reconf/assets/bhand_palm_fine-dff1351ac0d212f9757a6bc04b85287b4f13403a.stl')
    palm_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, 'herb_reconf/assets/bhand_palm_fine-dff1351ac0d212f9757a6bc04b85287b4f13403a.stl')).vertices
    
    # =======================================
    e =HerbEnv(os.path.join(top_dir, scene_name), palm_mesh_vertices, push_mesh_vertices=trimesh.load(os.path.join(top_dir, mesh_file)), target_mesh_vertices=trimesh.load(os.path.join(top_dir, mesh_file)).vertices, task=task, obs=False)
    e.reset_model(seed=SEED)
    mean = np.zeros(e.action_dim)
    sigma = 1.0*np.ones(e.action_dim)
    sigma[0:7]=sigma[0:7]*(e.action_space.high[0:7]-e.action_space.low[0:7])
    sigma[0]=1*sigma[0]
    sigma[1]=0.5*sigma[1]
    sigma[2]=1*sigma[2]
    sigma[3]=2*sigma[3]
    sigma[7:]=sigma[7:]*(e.action_space.high[14:22]-e.action_space.low[14:22])
    sigma[7:]=50*sigma[7:]
    sigma[5]=5*sigma[5]
    sigma[-2:]=20*sigma[-2:]
    filter_coefs = [sigma, 0.25, 0.8, 0.0, np.concatenate((e.action_space.high[0:7]-e.action_space.low[0:7], e.action_space.high[14:22]-e.action_space.low[14:22])), np.concatenate((e.action_space.low[0:7], e.action_space.low[14:22])), np.concatenate((e.action_space.high[0:7], e.action_space.high[14:22]))]
     
    agent = MPPI(e, os.path.join(top_dir, scene_name), task, H=H, paths_per_cpu=paths_per_cpu, num_cpu=1,
                 kappa=1000.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
                 default_act='repeat', seed=SEED)
    # 
    # 
    # # replay_start_state_num=10
    obs_states=[]
#     restart=True
#     print('fixed state to 2!!')
    if restart:
        agent, obs_states=pickle.load(open(PICKLE_FILE, 'rb'))
        agent.env=e
        reset_num=90
        agent.filter_coefs=filter_coefs
        agent.sol_state=agent.sol_state[:reset_num]
        obs_states=obs_states[:reset_num]
        agent.act_sequence=agent.act_sequences[reset_num-2]
        agent.act_sequence=agent.act_sequence[:H]
        agent.kappa=1000.0
        agent.env.set_env_state(agent.sol_state[-1])
        rgb=agent.env.model.render(height=480, width=640, camera_id=1, depth=False)
#         cv2.imshow('rbg', rgb)
#         cv2.waitKey(50)
    # # agent.act_sequence=agent.sol_act[replay_start_state_num]
    
    # pickle.dump(agent, open(PICKLE_FILE, 'wb'))#
    if replay:
    #     num_id=1000000000
        replay_start_state_num=0
        agent=pickle.load(open(PICKLE_FILE, 'rb'))
        agent.env=e
        # #agent.filter_coefs=filter_coefs
        #agent.sol_state=agent.sol_state[:63]
        agent.env.set_env_state(agent.sol_state[replay_start_state_num])
        # # # agent.kappa=100.0
        # # # # 
        # # # 
        # agent.act_sequence=np.concatenate((agent.act_sequence, np.repeat(np.expand_dims(agent.act_sequence[-1], axis=0), 60-agent.H, axis=0)))
        # agent.H=60
        # #agent.act_sequence=agent.act_sequence[:H]
        # u=0
        # 
        #agent.paths_per_cpu=50
    
    pose_estimator=pose_model_estimator(seg_input_queue, seg_output_queue, recon_input_queue, recon_output_queue, top_dir, os.path.join(top_dir, scene_name), num_id, use_cuda_vox, 0, model=agent.env.model, simulate_model_quality=(quality_type!='none'), model_quality=reconstruction_quality_level, quality_type=quality_type, four_channel=four_channel)
    pose_estimator.tabletop_segmentor=seg_model
    pose_estimator.point_completion_model=point_completion_model
    
    palm_mesh_ind=pose_estimator.geom_names.index("herb/wam_1/bhand//unnamed_geom_0")
    palm_mesh_vertices=pose_estimator.included_meshes[palm_mesh_ind]
    e.palm_mesh_vertices=palm_mesh_vertices
    
    agent.num_cpu=1
    
    # agent.paths_per_cpu=100
    # agent.gamma=1.0
    reuse_sim_env=True
    losses=[]

    ts = timer.time()
    stepped_sim_env=None
    pred_object_positions=None
    pred_rotationss=None
    body_name=None
    geom_names=None
    #steps=200
    s_time=time.time()
    depth=agent.env.model.render(height=480, width=640, camera_id=1, depth=True)
    rgb=agent.env.model.render(height=480, width=640, camera_id=1, depth=False)
    
    segs=agent.env.model.render(height=480, width=640, camera_id=1, depth=False, segmentation=True)
    can=segs[256, 343]
    table=segs[200, 200]
    back=segs[50, 50]
    arm=segs[284, 515]

#     cv2.imshow('segs', segs)
#     cv2.waitKey(20)
    t=0
    if vis_while_running:
        cv2.imshow('rbg', rgb)
        cv2.waitKey(50)
        
    if not use_gt:     
        cam_pos=agent.env.model.data.cam_xpos[1]
        cam_mat=np.reshape(agent.env.model.data.cam_xmat[1], (3, 3))
             
        if stepped_sim_env is not None:
            pred_object_positions=[stepped_sim_env.model.named.data.xpos[body_name]]
            pred_rotationss=[stepped_sim_env.model.named.data.geom_xmat[geom_name] for geom_name in geom_names]
          
        if replay:
            pred_object_positions=[]
        
        if not reuse_sim_env or t==0:
            gt_target_mesh=trimesh.load(os.path.join(top_dir, mesh_file))#trimesh.primitives.Box(extents=np.array([0.11,0.11,0.11]))##
             
            transform=np.eye(4)
            transform[:3,:3]=np.reshape(agent.env.model.named.data.xmat["gen_body_0"], (3,3))
            gt_target_mesh.apply_transform(transform)
             
            transform=np.eye(4)
            transform[:3,3]=agent.env.model.named.data.xpos["gen_body_0"]
            gt_target_mesh.apply_transform(transform) 
             
            #gt_target_mesh.show()
    #         if t>=1:
    #             rgb[:,:,:]=0
    #             depth[:,:]=0
            _,color_seg_masks,obs_xml_path,body_name,geom_names=pose_estimator.estiamte_poses(rgb, depth, agent.env.model, cam_pos, cam_mat, pred_object_positions, pred_rotationss, segs, task, stability_loss, step=t, gt_mesh=gt_target_mesh)
#                     exit()
            print(f'{target_object_name} chamfer loss', pose_estimator.obs_cds[-1]['chamfer'])
#             if (quality_type=='chamfer') and abs(pose_estimator.cd-reconstruction_quality_level)>0.05:
#                 break
        if vis_while_running:             
            cv2.imshow('color_seg_masks', color_seg_masks)
    if vis_while_running:
        cv2.imshow('rbg', rgb)
        cv2.waitKey(20)
        
    if use_gt:
        step_env=agent.env
    else:
        step_env=HerbEnv(obs_xml_path, palm_mesh_vertices, task=task, obs=True)
    
    step_env.set_env_state(step_env.get_env_state().copy())
    step_env.push_mesh_vertices=trimesh.load(os.path.join(top_dir, mesh_file))
    base_act=step_env.get_env_state().copy()['qp'][:15]
    timesteps=[0, 1, 4, 10, 50]
    for timestep in range(timesteps[-1]+1):
        if timestep in timesteps:
            front=step_env.model.render(height=480, width=640, camera_id=1, depth=False)
            cv2.imwrite(f'/home/willie/workspace/SSC/plots/raw_images/front_{target_object_name}_{timestep}_{use_gt}_{four_channel}.jpg', front) 
            back=step_env.model.render(height=480, width=640, camera_id=0, depth=False)
            cv2.imwrite(f'/home/willie/workspace/SSC/plots/raw_images/back_{target_object_name}_{timestep}_{use_gt}_{four_channel}.jpg', back)
        step_env.step(base_act)
   

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    #set up neural networks
    #set up segmentation
    data_loader = reload(data_loader)
    data_augmentation = reload(data_augmentation)
    dsn_params = {
        
        # Sizes
        'feature_dim' : 64, # 32 would be normal
        
        # algorithm parameters
        'lr' : 1e-2, # learning rate
        'iter_collect' : 20, # Collect results every _ iterations
        'max_iters' : 100000,
        
        # architecture parameters
        'use_coordconv' : False,
    
        # Loss function stuff
        'lambda_fg' : 1,
        'lambda_direction' : 1.,
    
        # Hough Voting parameters
        'skip_pixels' : 5, 
        'inlier_threshold' : 0.9, 
        'angle_discretization' : 100,
        'inlier_distance' : 20,
        'percentage_threshold' : 0.5, # this depends on skip_pixels, angle_discretization, inlier_distance. just gotta try it to see if it works
        'object_center_kernel_radius' : 10,
    
    }
    rrn_params = {
        
        # Sizes
        'feature_dim' : 64, # 32 would be normal
        
        # algorithm parameters
        'lr' : 1e-2, # learning rate
        'iter_collect' : 20, # Collect results every _ iterations
        'max_iters' : 100000,
        
        # architecture parameters
        'use_coordconv' : False,
        
    }
    tts_params = {
        
        # Padding for RGB Refinement Network
        'padding_percentage' : 0.25,
        
        # Open/Close Morphology for IMP (Initial Mask Processing) module
        'use_open_close_morphology' : True,
        'open_close_morphology_ksize' : 9,
        
        # Closest Connected Component for IMP module
        'use_closest_connected_component' : True,
        
        # RANSAC to estimate table plane
        'table_RANSAC' : False,
        'RANSAC_min_samples' : 5,
        'RANSAC_residual_threshold' : 0.01, # measured in meters
        'RANSAC_max_trials' : 100,
        'false_positive_table_percentage_threshold' : 0.7,
        
    }
    
    #20: 1361.1076367082503
    #30: 895.6921938165307
    #37.5: 707.017201090989
    #45: 579.4112549695428
    camera_params={'fx':579.4112549695428, 'fy':579.4112549695428, 'img_width':640, 'img_height': 480}#{'fx':346.6040174, 'fy':462.1286899, 'img_width':640, 'img_height': 480} #f=(D/2)/tan(FOV/2)
    root_dir=options.top_dir
    checkpoint_dir = root_dir+'/checkpoints/'
    dsn_filename = os.path.join(checkpoint_dir, 'DepthSeedingNetwork_iter100000_TableTop_v5_64c_checkpoint.pth.tar')
    rrn_filename = os.path.join(checkpoint_dir, 'RRN_iter100000_TR_AC_MT_E_TableTop_v5_64c_checkpoint.pth.tar')
    tts_params['final_close_morphology'] = 'TableTop_v5' in rrn_filename
    tabletop_segmentor = segmentation.TableTopSegmentor(tts_params, 
                                                        dsn_filename,
                                                        dsn_params,
                                                        rrn_filename,
                                                        rrn_params)
    
    #set up genre no stability
    opt = options_test.parse()
    opt.gpu=str(options.gpu_num)
    opt.full_logdir = None
    opt.net_file=options.recon_net
    opt.out_channels=1
#     if options.recon_net_type=='genre_given_depth':
#         opt.net_file=os.path.join(options.recon_net, "bst.pt")
#         opt.out_channels=1
#     elif options.recon_net_type=='genre_given_depth_4_channel':
#         opt.net_file=os.path.join(options.recon_net+'4_channel', "bst.pt")
#         opt.out_channels=1
#     elif options.recon_net_type=='genre_given_depth_4_channel_predict_other':
#         opt.net_file=os.path.join(options.recon_net+'_4_channel_predict_other', "bst.pt")
#         opt.out_channels=1
#     elif options.recon_net_type=='genre_given_depth_4_channel_predict_other_stability':
#         opt.net_file=os.path.join(options.recon_net+'_4_channel_predict_other_stability', "bst.pt")
#         opt.out_channels=1
    opt.output_dir=''
    #print(opt)
    opt.batch_size=1
    if opt.gpu == '-1':
        device = torch.device('cpu')
    else:
        loadlib.set_gpu(opt.gpu, check=False)
        device = torch.device('cuda')
    if opt.manual_seed is not None:
        loadlib.set_manual_seed(opt.manual_seed)
    output_dir = opt.output_dir
    output_dir += ('_' + opt.suffix.format(**vars(opt))) \
        if opt.suffix != '' else ''
    opt.output_dir = output_dir
    logger_list = [
    loggers.TerminateOnNaN(),
    ]
    logger = loggers.ComposeLogger(logger_list)
    
    if options.recon_net_type=='genre_given_depth':
        Model = models.get_model(options.recon_net_type, test=True)
    elif options.recon_net_type=='genre_given_depth_4_channel':
        Model = models.get_model(options.recon_net_type, test=True)
    elif options.recon_net_type in ['genre_given_depth_4_channel_predict_other', 'genre_given_depth_4_channel_predict_other_stability']:
        Model = models.get_model('genre_given_depth_4_channel', test=True)
    
    point_completion_model = Model(opt, logger)
    point_completion_model.to(device)

    tasks=['grasping']
    target_objects=["vase_2"]
    
    args_list=[]
    
    m = mp.Manager()
    seg_input_queue=m.Queue()
    seg_output_queues={}
    recon_input_queue=m.Queue()
    recon_output_queues={}
    
    num_id=0
    for task in tasks:
        for target_object in target_objects:
#             seg_output_queues[num_id]=m.Queue()
#             recon_output_queues[num_id]=m.Queue()
#             args_list.append([seg_input_queue, seg_output_queues[num_id], recon_input_queue, recon_output_queues[num_id], task, options.steps, num_id, 1, 0, None, target_object, options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.replay, options.restart, options.save_dir, options.use_cuda_vox, options.stability_loss, 0])
#             num_id+=1
            seg_output_queues[num_id]=m.Queue()
            recon_output_queues[num_id]=m.Queue()
            args_list.append([seg_input_queue, seg_output_queues[num_id], recon_input_queue, recon_output_queues[num_id], task, options.steps, num_id, 0, 0, 'none', target_object, options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.replay, options.restart, options.save_dir, options.use_cuda_vox, 1, options.recon_net_type in ['genre_given_depth_4_channel', 'genre_given_depth_4_channel_predict_other', 'genre_given_depth_4_channel_predict_other_stability'], 0 ])
            num_id+=1
#             seg_output_queues[num_id]=m.Queue()
#             recon_output_queues[num_id]=m.Queue()
#             args_list.append([seg_input_queue, seg_output_queues[num_id], recon_input_queue, recon_output_queues[num_id], task, options.steps, num_id, 0, 0, 'none', target_object, options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.replay, options.restart, options.save_dir, options.use_cuda_vox, 0])
#             num_id+=1

#             for chamfer_dist in chamfer_dists:
#                 seg_output_queues[num_id]=m.Queue()
#                 recon_output_queues[num_id]=m.Queue()
#                 args_list.append([seg_input_queue, seg_output_queues[num_id], recon_input_queue, recon_output_queues[num_id], task, options.steps, num_id, 0, chamfer_dist, 'chamfer-add', target_object, options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.replay, options.restart, options.save_dir, options.use_cuda_vox, options.stability_loss])
#                 num_id+=1
#                 seg_output_queues[num_id]=m.Queue()
#                 recon_output_queues[num_id]=m.Queue()
#                 args_list.append([seg_input_queue, seg_output_queues[num_id], recon_input_queue, recon_output_queues[num_id], task, options.steps, num_id, 0, chamfer_dist, 'chamfer-sub', target_object, options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.replay, options.restart, options.save_dir, options.use_cuda_vox, options.stability_loss])
#                 num_id+=1
#                     
#     run_MPPI(args_list[0][0], args_list[0][1], args_list[0][2], args_list[0][3], args_list[0][4], args_list[0][5], args_list[0][6], args_list[0][7],
#               args_list[0][8], args_list[0][9], args_list[0][10], args_list[0][11], args_list[0][12], args_list[0][13], args_list[0][14],
#                args_list[0][15], args_list[0][16], args_list[0][17], args_list[0][18], args_list[0][19], seg_model=tabletop_segmentor, point_completion_model=point_completion_model)
    args_list=[]
    args_list.append([seg_input_queue, seg_output_queues[0], recon_input_queue, recon_output_queues[0], 'grasping', options.steps, 0, 0, 0, 'none', "vase_3", options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.replay, options.restart, options.save_dir, options.use_cuda_vox, 1, options.recon_net_type in ['genre_given_depth_4_channel', 'genre_given_depth_4_channel_predict_other', 'genre_given_depth_4_channel_predict_other_stability'], options.extrusion_baseline, options.make_video, options.seed, 2/10.0, 0, options.stability_experiement])
    pool = mp.Pool(processes=options.num_cpu, maxtasksperchild=1)
    args=tuple(args_list[0])
    parallel_runs = [pool.apply_async(run_MPPI,
                                     args=(args_list[i])) for i in range(options.run_num, len(args_list), options.total_runs)]
         
         
    all_done=False
    while not all_done:  
        all_done=True
        for p in parallel_runs:
            all_done=all_done&p.ready()
            if '_value' in p.__dict__ and p.__dict__['_value']!=None:
                print('thread error', p.__dict__['_value'])
            
        #process seg requests
        try:
            seg_inputs=seg_input_queue.get(timeout=0.02)
            mini_batch={'rgb' : data_augmentation.array_to_tensor(np.expand_dims(seg_inputs[1], 0)), 'xyz' : data_augmentation.array_to_tensor(np.expand_dims(seg_inputs[2], 0))}
            fg_masks, direction_predictions, initial_masks, plane_masks, distance_from_table, seg_masks = tabletop_segmentor.run_on_batch(mini_batch)
            seg_masks = seg_masks.cpu().numpy()[0]
            seg_output_queues[seg_inputs[0]].put([seg_masks])
        except Empty:
            u=0
            
        #process completion requests
        try:
            recon_inputs=recon_input_queue.get(timeout=0.02)
            pred_voxelss=point_completion_model.forward_with_gt_depth(recon_inputs[1], recon_inputs[2])
#             if len(pred_voxelss.shape)==4:
#                 pred_voxelss=pred_voxelss[:, None, :, :, :]
            recon_output_queues[recon_inputs[0]].put([pred_voxelss])
        except Empty:
            u=0
            
    print('finished gpu processing')
    results = [p.get() for p in parallel_runs]
    
#     processes = []
#     for rank in range(min(len(args_list), options.num_cpu)):
#         p = mp.Process(target=run_MPPI, args=args_list[rank])
#         # We first train the model across `num_processes` processes
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
    
    # try:
    #     results = [p.get() for p in parallel_runs]
    # except Exception as e:
    #     print(str(e))
    #     print("Timeout Error raised!")
    #     pool.close()
    #     pool.terminate()
    #     pool.join()
    
    # pool.close()
    # pool.terminate()
    # pool.join()
