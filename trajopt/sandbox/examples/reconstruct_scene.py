import os
from optparse import OptionParser
import scipy
import time
import glob

dir_name=os.path.dirname(__file__)
parser = OptionParser()
parser.add_option("--rgb_recon_image", dest="rgb_recon_image", help="rgb image of scene to reconstruct")
parser.add_option("--depth_recon_image", dest="depth_recon_image", help="rgb image of scene to reconstruct")
parser.add_option("--robot_sim", dest="robot_sim", help="mujoco sim of robot to reconstruct into")
parser.add_option("--top_dir", dest="top_dir", default='/home/willie/workspace/SSC', help="directory code is in, ex. /home/user/Amodal-3D-Reconstruction-for-Robotic-Manipulationvia-Stability-and-Connectivity")
parser.add_option("--save_dir", dest="save_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/sss_saved/', help="where to save scene reconstruction to")
parser.add_option("--recon_net", dest="recon_net", default='/home/willie/workspace/SSC/genre/logs/0/0013.pt', help="path to saved ARM net weights")
(options, args) = parser.parse_args()
print(options)

if not options.rgb_recon_image:
    parser.error('rgb_recon_image not given')
if not options.rgb_recon_image:
    parser.error('depth_recon_image not given')
if not options.robot_sim:
    parser.error('robot_sim not given')

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
    #set up genre
    opt = options_test.parse()
    opt.gpu=str(options.gpu_num)
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
    
    gt_env=HerbEnv(os.path.join(options.top_dir, 'recon'), np.zeros((1,3)), 0, obs=False)
    pose_estimator=pose_model_estimator(gt_env.model, None, None, None, None, options.top_dir, os.path.join(options.top_dir, 'recon'), 0, False, False, model=gt_env.model, simulate_model_quality=False, model_quality=0, four_channel=True)
    pose_estimator.tabletop_segmentor=tabletop_segmentor
    pose_estimator.point_completion_model=point_completion_model
    cam_pos=gt_env.model.data.cam_xpos[1]
    cam_mat=np.reshape(gt_env.model.data.cam_xmat[1], (3, 3))
    depth=gt_env.model.render(height=480, width=640, camera_id=1, depth=True)
    rgb=gt_env.model.render(height=480, width=640, camera_id=1, depth=False)
    _,color_seg_masks,obs_xml_path,body_name,geom_names=pose_estimator.estiamte_poses(rgb, depth, gt_env.model, cam_pos, cam_mat, None, None, None, '', False, single_threaded=True, use_gt_segs=False)
    
    print('saved reconstruction to {obs_xml_path}')
    