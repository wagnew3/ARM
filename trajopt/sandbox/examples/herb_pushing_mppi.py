import os
from optparse import OptionParser
import scipy
import time
import glob

dir_name=os.path.dirname(__file__)
parser = OptionParser()
parser.add_option("--top_dir", dest="top_dir", default='/home/willie/workspace/SSC', help="directory code is in, ex. /home/user/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity")
parser.add_option("--H", type="int", dest="H", default=5, help="MPPI lookahead")
parser.add_option("--paths_per_cpu", dest="paths_per_cpu", type="int", default=25, help="number MPPI paths per step")
parser.add_option("--steps", dest="steps", type="int", default=200, help="number MPPI steps")
parser.add_option("--vis_while_running", dest="vis_while_running", type="int", default=0, help="visualize experiments while running")
parser.add_option("--num_cpu", dest="num_cpu", type="int", default=1, help="number fo cpus to use")
parser.add_option("--save_dir", dest="save_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/sss_saved/', help="where to save experiment results to")
parser.add_option("--run_num", dest="run_num", type="int", default=0, help="experiment number for saving results")
parser.add_option("--save_trajectories", dest="save_trajectories", type="int", default=0)
parser.add_option("--recon_net", dest="recon_net", default='/home/willie/workspace/GenRe-ShapeHD/logs/genre_given_depth_shapenet_table_norm_2/60/nets/0013.pt', help="path to saved net weights")
parser.add_option("--recon_net_type", dest="recon_net_type", default='genre_given_depth_4_channel', help="reconstruciton net type: 'genre_given_depth' for baselines, 'genre_given_depth_4_channel' for ARM nets")
parser.add_option("--extrusion_baseline", dest="extrusion_baseline", type="int", default=0, help="run with extrusion baseline")
parser.add_option("--ground_truth", dest="ground_truth", type="int", default=0, help="use ground truth models to plan")
parser.add_option("--make_video", dest="make_video", type="int", default=1, help="make videos fo manipulations")
parser.add_option("--remove_old_runs", dest="remove_old_runs", type="int", default=0, help="remove old experiments")
parser.add_option("--seed", dest="seed", type="int", default=0)
parser.add_option("--use_custom_recon", dest="use_custom_recon", type="int", default=0)
parser.add_option("--stability_experiement", dest="stability_experiement", type="int", default=0, help="run reconstruction stability experiments")
parser.add_option("--make_images", dest="make_images", type="int", default=0, help="make simulation images while not moving robot")

(options, args) = parser.parse_args()
print(options)

os.environ["OMP_NUM_THREADS"]="1"

import torch
import multiprocessing as mp
from queue import Empty
from importlib import reload
import traceback

from genre import models
from genre.options import options_test
from genre.loggers import loggers
import genre.util.util_loadlib as loadlib

import data_augmentation
import data_loader
import segmentation; segmentation = reload(segmentation)

from trajopt.algos.mppi import MPPI
from tqdm import tqdm
import time as timer
import numpy as np
import pickle
from trajopt.envs.herb_pushing_env import HerbEnv
import cv2
from pose_model_estimator import pose_model_estimator
import trimesh
import random
from trajopt.mujoco_utils import convex_decomp_target_object_env
from trajopt.sandbox.examples.custom_recon_net import Custom_Recon_Net

ycb_objects=["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can","006_mustard_bottle","007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", 
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "035_power_drill",
    "036_wood_block",
    "053_mini_soccer_ball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "061_foam_brick",
    "077_rubiks_cube"]

downloaded_objects=[]

#experiment thread method
def run_MPPI(seg_input_queue, seg_output_queue, recon_input_queue, recon_output_queue, task, steps, num_id, use_gt, reconstruction_quality_level,
              quality_type, target_object_name, top_dir, H, paths_per_cpu, vis_while_running, save_dir, use_cuda_vox, stability_loss,
               four_channel, extrusion_baseline, make_video, seed, occluded_frac, scene_num, stability_experiement, custom_recon, seg_model=None, point_completion_model=None):
    np.random.seed(options.seed)
    random.seed(options.seed)
    
    stability_exp_num_steps=51
    if stability_experiement:
        steps=stability_exp_num_steps
    
    if target_object_name in ycb_objects:
        mesh_file=f'herb_reconf/assets/ycb_objects/{target_object_name}/google_16k/nontextured.stl'
    else:
        mesh_file=f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{target_object_name}/scene.stl'

    scene_name=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/{task}_{target_object_name}_{scene_num}_{occluded_frac}_scene.xml')
    
    print('scene_name', scene_name) 
    if not os.path.exists(scene_name):
        print('scene not found!', scene_name)
        return
    
    print('starting', num_id)
    ENV_NAME = f'herb-{task}_40_50_{steps}_{num_id}_{use_gt}_{reconstruction_quality_level}_{quality_type}_{target_object_name}_{stability_loss}_{scene_num}_{occluded_frac}'
    PICKLE_FILE = os.path.join(save_dir+'/results/saved_mppi', ENV_NAME + '_mppi.pickle')
    STATS_FILE=os.path.join(save_dir+'/results/mppi_run_stats', ENV_NAME + '_stats.pickle')
    
    if stability_experiement:
        stats_files=glob.glob(os.path.join(save_dir+'/results/mppi_run_stats', f'herb-{task}_40_50_{steps}_*_{use_gt}_{reconstruction_quality_level}_{quality_type}_{target_object_name}_{stability_loss}_{scene_num}_{occluded_frac}' + '_stats.pickle'))
        if len(stats_files)>0:
            run_info=pickle.load(open(stats_files[0], 'rb'))
        if (len(stats_files)>0
            and len(run_info)>0
            and run_info['loss'].shape[0]>0
            and ((task=='grasping' and (run_info['loss'].shape[0]==199)) 
                 or (task!='grasping' and (run_info['loss'].shape[0]==199)))):
            print(STATS_FILE, 'already exists!')
            return
    else:
        stats_files=glob.glob(os.path.join(save_dir+'/results/mppi_run_stats', f'herb-{task}_40_50_{steps}_*_{use_gt}_{reconstruction_quality_level}_{quality_type}_{target_object_name}_{stability_loss}_{scene_num}_{occluded_frac}' + '_stats.pickle'))
        if len(stats_files)>0:
            run_info=pickle.load(open(stats_files[0], 'rb'))
        if (len(stats_files)>0
            and len(run_info)>0
            and run_info['loss'].shape[0]>0
            and ((task=='grasping' and (run_info['loss'].shape[0]==199)) 
                 or (task!='grasping' and (run_info['loss'].shape[0]==199)))):
            print(STATS_FILE, 'already exists!')
            return
    
    SEED = seed+num_id
    N_ITER = 1
    
    #load environment
    palm_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, 'herb_reconf/assets/bhand_palm_fine-dff1351ac0d212f9757a6bc04b85287b4f13403a.stl'))
    gt_env=HerbEnv(os.path.join(top_dir, scene_name), palm_mesh_vertices, num_id, push_mesh_vertices=trimesh.load(os.path.join(top_dir, mesh_file)), target_mesh_vertices=trimesh.load(os.path.join(top_dir, mesh_file)).vertices, task=task, obs=False)
    gt_env.reset_model(seed=SEED)
    mean = np.zeros(gt_env.action_dim)
    #set mppi action sampling parameters
    sigma = 1.0*np.ones(gt_env.action_dim)
    sigma[0:7]=sigma[0:7]*(gt_env.action_space.high[0:7]-gt_env.action_space.low[0:7])
    sigma[0]=1*sigma[0]
    sigma[1]=0.5*sigma[1]
    sigma[2]=1*sigma[2]
    sigma[3]=2*sigma[3]
    sigma[5]=2*sigma[5]
    sigma[7:]=sigma[7:]*(gt_env.action_space.high[14:22]-gt_env.action_space.low[14:22])
    sigma[7:]=50*sigma[7:]
    sigma[5]=5*sigma[5]
    sigma[-2:]=20*sigma[-2:]
    filter_coefs = [sigma, 0.25, 0.8, 0.0, np.concatenate((gt_env.action_space.high[0:7]-gt_env.action_space.low[0:7], gt_env.action_space.high[14:22]-gt_env.action_space.low[14:22])), np.concatenate((gt_env.action_space.low[0:7], gt_env.action_space.low[14:22])), np.concatenate((gt_env.action_space.high[0:7], gt_env.action_space.high[14:22]))]
    
    #make conv decomp of environment for simulation
    conv_decomp_env_file, target_decomp_ind, num_decomps, _=convex_decomp_target_object_env(scene_name, 
                                                                                                   'gen_body_0',  
                                                                                                   os.path.join(top_dir, f'herb_reconf/cluttered_scenes/'), num_id, top_dir)
    conv_decomp_env=HerbEnv(conv_decomp_env_file, palm_mesh_vertices, num_id, push_mesh_vertices=trimesh.load(os.path.join(top_dir, mesh_file)), target_mesh_vertices=trimesh.load(os.path.join(top_dir, mesh_file)).vertices, task=task, obs=False)
    conv_decomp_env.reset_model(seed=SEED)
    
    agent = MPPI(num_id, conv_decomp_env, os.path.join(top_dir, scene_name), task, H=H, paths_per_cpu=paths_per_cpu, num_cpu=1,
                 kappa=1000.0, gamma=1.0, mean=mean, filter_coefs=filter_coefs,
                 default_act='repeat', seed=SEED)
    obs_states=[]

    pose_estimator=pose_model_estimator(gt_env.model, seg_input_queue, seg_output_queue, recon_input_queue, recon_output_queue, top_dir, os.path.join(top_dir, scene_name), num_id, use_cuda_vox, extrusion_baseline, model=agent.env.model, simulate_model_quality=(quality_type!='none'), model_quality=reconstruction_quality_level, quality_type=quality_type, four_channel=four_channel)
    pose_estimator.tabletop_segmentor=seg_model
    pose_estimator.point_completion_model=point_completion_model
    
    palm_mesh_vertices=pose_estimator.palm_mesh_verts
    conv_decomp_env.set_palm_verts(palm_mesh_vertices)
    gt_env.set_palm_verts(palm_mesh_vertices)
    
    agent.num_cpu=1
    reuse_sim_env=True
    losses=[]

    try:
        ts = timer.time()
        stepped_sim_env=None
        pred_object_positions=None
        pred_rotationss=None
        body_name=None
        geom_names=None
        
        for t in tqdm(range(steps-len(agent.sol_state))):#
            s_time=time.time()
            depth=gt_env.model.render(height=480, width=640, camera_id=1, depth=True)
            rgb=gt_env.model.render(height=480, width=640, camera_id=1, depth=False)

            if vis_while_running:
                cv2.imshow('rbg', rgb)
                im_depth=depth-np.amin(depth)
                im_depth=im_depth*255/np.amax(im_depth)
                im_depth=im_depth.astype(np.uint8)
                im_depth=np.concatenate((im_depth[:,:,None], im_depth[:,:,None], im_depth[:,:,None]), axis=2)
                cv2.waitKey(100)
                
            if not use_gt:     
                cam_pos=agent.env.model.data.cam_xpos[1]
                cam_mat=np.reshape(agent.env.model.data.cam_xmat[1], (3, 3))
                     
                if stepped_sim_env is not None:
                    pred_object_positions=[stepped_sim_env.model.named.data.xpos[body_name]]
                    pred_rotationss=[stepped_sim_env.model.named.data.geom_xmat[geom_name] for geom_name in geom_names]
                
                if not reuse_sim_env or t==0:
                    gt_target_mesh=trimesh.load(os.path.join(top_dir, mesh_file))#trimesh.primitives.Box(extents=np.array([0.11,0.11,0.11]))##
                     
                    transform=np.eye(4)
                    transform[:3,:3]=np.reshape(agent.env.model.named.data.xmat["gen_body_0"], (3,3))
                    gt_target_mesh.apply_transform(transform)
                     
                    transform=np.eye(4)
                    transform[:3,3]=agent.env.model.named.data.xpos["gen_body_0"]
                    gt_target_mesh.apply_transform(transform) 
                     
                    #make reconstructed environment
                    segs=gt_env.model.render(height=480, width=640, camera_id=1, depth=False, segmentation=True)
                    cv2.imwrite('/home/willie/workspace/Predictive_RL_NIPS2017/test_imgs/rgb.png', rgb)
                    np.save('/home/willie/workspace/Predictive_RL_NIPS2017/test_imgs/depth.npy', depth)
                    _,color_seg_masks,obs_xml_path,body_name,geom_names=pose_estimator.estiamte_poses(rgb, depth, gt_env.model, cam_pos, cam_mat, pred_object_positions, pred_rotationss, segs, task, stability_loss, step=t, gt_mesh=gt_target_mesh)
                    print(f'{target_object_name} chamfer loss', pose_estimator.obs_cds[-1]['chamfer'])
                    if (quality_type=='chamfer') and abs(pose_estimator.cd-reconstruction_quality_level)>0.05:
                        break
                if vis_while_running:             
                    cv2.imshow('color_seg_masks', color_seg_masks)
            if vis_while_running:
                cv2.imshow('rbg', rgb)
                cv2.waitKey(20)
            
            if stability_experiement:
                if t==0:
                    if use_gt:
                        step_env=agent.env
                    else:
                        step_env=HerbEnv(obs_xml_path, palm_mesh_vertices, num_id, task=task, obs=True, push_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, f'herb_reconf/temp_{num_id}/target_mesh.stl')), target_mesh_vertices=trimesh.load_mesh(os.path.join(top_dir, f'herb_reconf/temp_{num_id}/target_mesh.stl')).vertices)
                
                if t==0:
                    all_poses=step_env.data.qpos.ravel().copy()
                    start_poses=[]
                    for ind in range(22, all_poses.shape[0], 7):
                        start_poses.append(all_poses[ind:ind+3])
                    start_poses=np.array(start_poses)
                if t==49:
                    all_poses=step_env.data.qpos.ravel().copy()
                    end_poses=[]
                    for ind in range(22, all_poses.shape[0], 7):
                        end_poses.append(all_poses[ind:ind+3])
                    end_poses=np.array(end_poses)
                    STABILITY_STATS_FILE=os.path.join(save_dir+'/results/stability_stats', ENV_NAME + '_stats.pickle')
                    pickle.dump((start_poses, end_poses), open(STABILITY_STATS_FILE, 'wb'))
                    
                if True:
                    net_type='ground_truth'
                    timesteps=[0, 1, 2,4,6,8, 10, 40]
                    if t in timesteps:
                        front=gt_env.model.render(height=480, width=640, camera_id=1, depth=False)
                        scipy.misc.imsave(os.path.join(top_dir, f'plots/raw_images/front_{target_object_name}_{use_gt}_{net_type}_2_{t}.jpg'), front)
                        back=gt_env.model.render(height=480, width=640, camera_id=0, depth=False)
                        scipy.misc.imsave(os.path.join(top_dir, f'plots/raw_images/back_{target_object_name}__{use_gt}_{net_type}_2_{t}.jpg'), back)
                    
                step_env.set_env_state(step_env.get_env_state().copy())
                base_act=step_env.get_env_state().copy()['qp'][:15]
                step_env.step(base_act, compute_stats=False)
                continue
            
            #run MPPI step
            elif use_gt:
                stepped_sim_env=agent.train_step(conv_decomp_env_file, num_id, top_dir, mesh_file, palm_mesh_vertices, niter=N_ITER, gt=use_gt, vis=vis_while_running, last_env=stepped_sim_env, cur_env=agent.env)#
            else:
                if stepped_sim_env==None:
                    obs_states+=[None]
                else:
                    obs_states+=[stepped_sim_env.get_env_state().copy()]
                stepped_sim_env=agent.train_step(obs_xml_path, num_id, top_dir, mesh_file, palm_mesh_vertices, gt_env=gt_env, niter=N_ITER, vis=vis_while_running, use_last_state=(reuse_sim_env & (t>0)), last_state=obs_states[-1], last_env=stepped_sim_env)
                
            print("Trajectory reward = %f" % agent.sol_reward[-1])
            if t % 1 == 0 and t > 0 and options.save_trajectories:
                print("==============>>>>>>>>>>> saving progress ")
                pickle.dump((agent, obs_states), open(PICKLE_FILE, 'wb'))
            
            #compute loss    
            if task=='grasping':
                block_pos_1 = agent.env.model.named.data.xpos[agent.env.block_sid_1]
                block_orientation_1=np.reshape(agent.env.model.named.data.xmat[agent.env.block_sid_1], (3,3))
                trans_push_mesh_vertices=np.matmul(block_orientation_1, agent.env.push_mesh_vertices.vertices.T).T
                trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
                loss=np.linalg.norm(np.mean(trans_push_mesh_vertices, axis=0)-np.mean(agent.env.target_mesh_vertices, axis=0))
                print('loss', loss)
                losses.append(loss)
            elif task=='hard_pushing':
                block_pos_1 = agent.env.model.named.data.xpos[agent.env.block_sid_1]
                block_orientation_1=np.reshape(agent.env.model.named.data.xmat[agent.env.block_sid_1], (3,3))
                trans_push_mesh_vertices=np.matmul(block_orientation_1, agent.env.push_mesh_vertices.vertices.T).T
                trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
                loss=np.linalg.norm(np.mean(trans_push_mesh_vertices, axis=0)-np.mean(agent.env.target_mesh_vertices, axis=0))
                 
                print('loss', loss)
                losses.append(loss)
  
            elif task=='easy_pushing':
                block_pos_1 = agent.env.model.named.data.xpos[agent.env.block_sid_1]
                block_orientation_1=np.reshape(agent.env.model.named.data.xmat[agent.env.block_sid_1], (3,3))
                trans_push_mesh_vertices=np.matmul(block_orientation_1, agent.env.push_mesh_vertices.vertices.T).T
                trans_push_mesh_vertices=trans_push_mesh_vertices+block_pos_1
                loss=np.linalg.norm(np.mean(trans_push_mesh_vertices, axis=0)-np.mean(agent.env.target_mesh_vertices, axis=0))
                print('loss', loss)
                losses.append(loss)
   
            print('time', time.time()-s_time, t+1, task, quality_type, reconstruction_quality_level, target_object_name, scene_num, occluded_frac)
    except:
        print('main loop error!!!', task, steps, num_id, use_gt, reconstruction_quality_level, quality_type, target_object_name)
        traceback.print_exc()
    
    
    if not stability_experiement:
        #save losses and rewards                    
        if options.save_trajectories:
            pickle.dump((agent, obs_states), open(PICKLE_FILE, 'wb'))
        if use_gt:
            pickle.dump({'reward': np.array(agent.sol_reward), 'loss': np.array(losses)}, open(STATS_FILE, 'wb'))
        else:
            pickle.dump({'reward': np.array(agent.sol_reward), 'loss': np.array(losses), 'cd': np.array(pose_estimator.obs_cds)}, open(STATS_FILE, 'wb'))

        print("Trajectory reward = %f" % np.sum(agent.sol_reward))
        print("Time for trajectory optimization = %f seconds" %(timer.time()-ts))
        if make_video:
            try:
                agent.env=gt_env
                agent.animate_result(ENV_NAME, top_dir, save_dir, task, num_id, use_gt, obs_states)
            except:
                print('video creation error!!!')
                traceback.print_exc()
            
    if os.path.exists(conv_decomp_env_file):
        os.remove(conv_decomp_env_file)
    pose_estimator.cleanup_files()

if __name__ == '__main__':
    #clean save dirs if specified
    options.save_dir=os.path.join(options.save_dir, str(options.seed))
    if options.remove_old_runs:
        if options.stability_experiement:
            txt = input(f"removing stability results at {os.path.join(options.save_dir, str(options.seed))} confirm y/n:")
            if txt!='y':
                print('terminating')
                exit()
            results_dir=os.path.join(options.save_dir, 'results')
            if not os.path.exists(options.save_dir):
                os.mkdir(options.save_dir)
                os.mkdir(results_dir)
                os.mkdir(os.path.join(results_dir, 'videos'))
                os.mkdir(os.path.join(results_dir, 'mppi_run_stats'))
                os.mkdir(os.path.join(results_dir, 'saved_mppi'))
            if not os.path.exists(os.path.join(results_dir, 'stability_stats')):
                os.mkdir(os.path.join(results_dir, 'stability_stats'))
                
            
            for f in os.listdir(os.path.join(results_dir, 'stability_stats')):
                f_path=os.path.join(os.path.join(results_dir, 'stability_stats'), f)
                os.remove(f_path)
        else:
            txt = input(f"removing robot performance results at {os.path.join(options.save_dir, str(options.seed))} confirm y/n:")
            if txt!='y':
                print('terminating')
                exit()
            if not os.path.exists(options.save_dir):
                os.mkdir(options.save_dir)
                results_dir=os.path.join(options.save_dir, 'results')
                os.mkdir(results_dir)
                os.mkdir(os.path.join(results_dir, 'videos'))
                os.mkdir(os.path.join(results_dir, 'mppi_run_stats'))
                os.mkdir(os.path.join(results_dir, 'saved_mppi'))
            
            for f in os.listdir(os.path.join(options.save_dir, 'results/videos')):
                f_path=os.path.join(os.path.join(options.save_dir, 'results/videos'), f)
                os.remove(f_path)
            for f in os.listdir(os.path.join(options.save_dir, 'results/mppi_run_stats')):
                f_path=os.path.join(os.path.join(options.save_dir, 'results/mppi_run_stats'), f)
                os.remove(f_path)
    
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
        'skip_pixels' : 10, 
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
    if not options.use_custom_recon:
        #set up genre
        opt = options_test.parse()
        opt.full_logdir = None
        opt.net_file=options.recon_net
        opt.out_channels=1
        opt.output_dir=''
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
    else:
        point_completion_model=Custom_Recon_Net()    
    
    #list of objects and tasks
    tasks=['grasping', 'hard_pushing', 'easy_pushing']
    target_objects=["maytoni",
        "lamp_1",
        "lamp_2",
        "cup_1",
        "vase_3",
        'lamp_3',
        'lamp_4',
        'glass_1',
        'glass_2',
        'glass_3',
        'cup_2',
        'trophy_1',
        "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can","006_mustard_bottle",
    "013_apple",
    "017_orange",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "036_wood_block",
    "056_tennis_ball",
    "077_rubiks_cube"]
    
    #compute object convex decomps
    for target_object_name in target_objects:
        if target_object_name in ycb_objects:
            mesh_file=f'herb_reconf/assets/ycb_objects/{target_object_name}/google_16k/nontextured.stl'
        else:
            mesh_file=f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{target_object_name}/scene.stl'
        decomps_folder=os.path.join(options.top_dir, f'herb_reconf/assets/decomps/{target_object_name}/')
        if not os.path.exists(decomps_folder):
            os.mkdir(decomps_folder)
            mesh=trimesh.load_mesh(os.path.join(options.top_dir, mesh_file))
            decomps=trimesh.decomposition.convex_decomposition(mesh, maxNumVerticesPerCH=1025, concavity=0.01, resolution=100000)
            if not isinstance(decomps, list):
                decomps=[decomps]
            c_decomps=[]
            for decmop in decomps:
                if decmop.faces.shape[0]>4 and decmop.mass>10e-8:
                    c_decomps.append(decmop)
            decomps=c_decomps
            mesh_filenames=[]
            mesh_masses=[]
            for decomp_ind in range(len(decomps)):
                mesh_filename=os.path.join(decomps_folder, f'{decomp_ind}.stl')
                mesh_filenames.append(mesh_filename)
                decomps[decomp_ind].export(mesh_filename)
                mesh_masses.append(decomps[decomp_ind].mass)
            if len(mesh_filenames)>25:
                heavy_inds=np.argsort(np.array(mesh_masses))
                new_mesh_names=[]
                for ind in range(25):
                    new_mesh_names.append(mesh_filenames[heavy_inds[-ind]])
                mesh_filenames=new_mesh_names
                    
    chamfer_dists=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    
    args_list=[]
    
    m = mp.Manager()
    seg_input_queue=m.Queue()
    seg_output_queues={}
    recon_input_queue=m.Queue()
    recon_output_queues={}
    
    #specify experiments
    num_id=0
    for task in tasks:
        for target_object in target_objects:
            for occluded_frac in range(0,11):
                for scene_num in range(0,3):
                    task_id=num_id+1000000*options.run_num
                    if options.ground_truth:
                        seg_output_queues[task_id]=m.Queue()
                        recon_output_queues[task_id]=m.Queue()
                        args_list.append([seg_input_queue, seg_output_queues[task_id], recon_input_queue, recon_output_queues[task_id], task, options.steps, task_id, 1, 0, None, target_object, options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.save_dir, 1, options.stability_loss, 0, options.extrusion_baseline, options.make_video, options.seed, occluded_frac/10.0, scene_num, options.stability_experiement, options.use_custom_recon])
                        num_id+=1
                    else:
                        seg_output_queues[num_id+1000000*options.run_num]=m.Queue()
                        recon_output_queues[num_id+1000000*options.run_num]=m.Queue()
                        args_list.append([seg_input_queue, seg_output_queues[task_id], recon_input_queue, recon_output_queues[task_id], task, options.steps, task_id, 0, 0, 'none', target_object, options.top_dir, options.H, options.paths_per_cpu, options.vis_while_running, options.save_dir, 1, 1, options.recon_net_type in ['genre_given_depth_4_channel', 'genre_given_depth_4_channel_predict_other', 'genre_given_depth_4_channel_predict_other_stability'], options.extrusion_baseline, options.make_video, options.seed, occluded_frac/10.0, scene_num, options.stability_experiement, options.use_custom_recon])
                        num_id+=1
    
    #run experiments multithreaded
    
    run_MPPI(args_list[0][0], args_list[0][1], args_list[0][2], args_list[0][3], args_list[0][4], args_list[0][5], args_list[0][6], args_list[0][7], args_list[0][8],
              args_list[0][9], args_list[0][10], args_list[0][11], args_list[0][12], args_list[0][13], args_list[0][14], args_list[0][15], args_list[0][16], args_list[0][17],
               args_list[0][18], args_list[0][19], args_list[0][20], args_list[0][21], args_list[0][22], args_list[0][23], args_list[0][24], args_list[0][25], seg_model=None, point_completion_model=None)
    
    pool = mp.Pool(processes=options.num_cpu, maxtasksperchild=1)
    args=tuple(args_list[0])
    parallel_runs = [pool.apply_async(run_MPPI, args=(args_list[i])) for i in range(len(args_list))]
                 
    all_done=False
    while not all_done:  
        all_done=True
        for p_ind in range(len(parallel_runs)):
            p=parallel_runs[p_ind]
            all_done=all_done&p.ready()
        try:
            seg_inputs=seg_input_queue.get(timeout=0.02)
            mini_batch={'rgb' : data_augmentation.array_to_tensor(np.expand_dims(seg_inputs[1], 0)), 'xyz' : data_augmentation.array_to_tensor(np.expand_dims(seg_inputs[2], 0))}
            fg_masks, direction_predictions, initial_masks, plane_masks, distance_from_table, seg_masks = tabletop_segmentor.run_on_batch(mini_batch)
            seg_masks = seg_masks.cpu().numpy()[0]
            seg_output_queues[seg_inputs[0]].put([seg_masks])
        except Empty:
            u=0
        try:
            recon_inputs=recon_input_queue.get(timeout=0.02)
            if not options.use_custom_recon:
                pred_voxelss=point_completion_model.forward_with_gt_depth(recon_inputs[1], recon_inputs[2])
            else:
                pred_voxelss=point_completion_model.forward_with_gt_depth(recon_inputs[1], recon_inputs[2], recon_inputs[3], recon_inputs[4], recon_inputs[5], recon_inputs[6], recon_inputs[7])
            recon_output_queues[recon_inputs[0]].put([pred_voxelss])
        except Empty:
            u=0
    
    try:
        results = [p.get() for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised!")
        pool.close()
        pool.terminate()
        pool.join()
