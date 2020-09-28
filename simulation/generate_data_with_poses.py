""" This code is a python script that essentially replicates the code in simulating_data.ipynb

    As a script, this can be called many times with a bash script. This helps when I need to kill
    the process due to the weird memory leak in PyBullet.

    To call this script:
    $:~ python generate_data.py <start_scene> <end_scene>

    Keep end_scene - start_scene <= 400 in order to not get hung by memory leak
"""

import time
import os, sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy

# my libraries
sys.path.insert(0, os.path.abspath('..'))
import simulation_util as sim_util

# pybullet
import pybullet as p
import pybullet_data

# suncg
import pybullet_suncg.simulator as suncg_sim
from pyquaternion import Quaternion

root_dir='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98'

def save_img_dict(img_dict, view_num, save_dir):
    # RGB
    rgb_filename = save_dir + f"rgb_{view_num:05d}.jpeg"
    cv2.imwrite(rgb_filename, cv2.cvtColor(img_dict['rgb'], cv2.COLOR_RGB2BGR))

    # Depth
    depth_filename = save_dir + f"depth_{view_num:05d}.png"
    cv2.imwrite(depth_filename, sim_util.saveable_depth_image(img_dict['depth']))

    # Segmentation
    seg_filename = save_dir + f"segmentation_{view_num:05d}.png"
    sim_util.imwrite_indexed(seg_filename, img_dict['seg'].astype(np.uint8))

def printout(string):
    print(string, file=open('whoa.txt', 'a'))

def main():
    args = [0, 1, 'train']#
    #args=sys.argv[1:]
    
    start_scene = int(args[0])
    print('start_scene', start_scene)
    end_scene = int(args[1])
    train_or_test = args[2]

    ##### Load SUNCG stuff #####

    suncg_dir = root_dir+'/data/suncg/v1/'

    # House lists
    training_houses_filename = root_dir+'/data/tabletop_dataset_v5/training_suncg_houses.json'
    test_houses_filename = root_dir+'/data/tabletop_dataset_v5/test_suncg_houses.json'
    train_houses = json.load(open(training_houses_filename))
    test_houses = json.load(open(test_houses_filename))

    # Room types I'm considering
    valid_room_types = set(['Living_Room', 'Kitchen', 'Room', 'Dining Room', 'Office'])

    # Room objects to filter out
    nyuv2_40_classes_filter_list = ['desk', 'chair', 'table', 'person', 'otherstructure', 'otherfurniture']
    coarse_grained_classes_filter_list = ['desk', 'chair', 'table', 'person', 'computer', 'bench_chair', 
                                          'ottoman', 'storage_bench', 'pet']





    ##### Load ShapeNet stuff #####

    shapenet_filepath = root_dir+'/data/ShapeNetCore.v2/'

    # Create a dictionary of name -> synset_id
    temp = json.load(open(shapenet_filepath + 'taxonomy.json'))
    taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}

    # weirdly, the synsets in the taxonomy file are not the same as what's in the ShapeNetCore.v2 directory. Filter this out
    synsets_in_dir = os.listdir(shapenet_filepath)
    synsets_in_dir.remove('taxonomy.json')
    #synsets_in_dir.remove('README.txt')

    taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synsets_in_dir}

    # useful synsets for simulation
    useful_named_synsets = [
        'ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin',
        'bag,traveling bag,travelling bag,grip,suitcase',
        'birdhouse',
        'bottle',
        'bowl',
        'camera,photographic camera',
        'can,tin,tin can',
        'cap',
        'clock',
        'computer keyboard,keypad',
        'dishwasher,dish washer,dishwashing machine',
        'display,video display',
        'helmet',
        'jar',
        'knife',
        'laptop,laptop computer',
        'loudspeaker,speaker,speaker unit,loudspeaker system,speaker system',
        'microwave,microwave oven',
        'mug',
        'pillow',
        'printer,printing machine',
        'remote control,remote',
        'telephone,phone,telephone set',
        'cellular telephone,cellular phone,cellphone,cell,mobile phone',
        'washer,automatic washer,washing machine'
    ]
    print("Number of synsets: {0}".format(len(useful_named_synsets)))

    # List of train/test tables
    training_tables_filename = root_dir+'/data/tabletop_dataset_v5/training_shapenet_tables.json'
    test_tables_filename = root_dir+'/data/tabletop_dataset_v5/test_shapenet_tables.json'
    train_tables = json.load(open(training_tables_filename))
    test_tables = json.load(open(test_tables_filename))

    # List of train/test object instances
    training_instances_filename = root_dir+'/data/tabletop_dataset_v5/training_shapenet_objects.json'
    test_instances_filename = root_dir+'/data/tabletop_dataset_v5/test_shapenet_objects.json'
    train_models = json.load(open(training_instances_filename))
    test_models = json.load(open(test_instances_filename))






    ##### Simulation Parameters #####
    house_ids = train_houses if train_or_test == 'train' else test_houses
    valid_tables = train_tables if train_or_test == 'train' else test_tables
    object_ids = train_models if train_or_test == 'train' else test_models

    simulation_params = {
        'is_shapenetsem' : False,
        
        # scene stuff
        'min_num_objects_per_scene' : 10,
        'max_num_objects_per_scene' : 25,
        'simulation_steps' : 1000,

        # House stuff
        'house_ids' : house_ids, # test_houses

        # room stuff
        'valid_room_types' : valid_room_types,
        'min_xlength' : 3.0, # Note: I believe this is in meters
        'min_ylength' : 3.0, 

        # table stuff
        'valid_tables' : valid_tables,
        'max_table_height' : 1.0, # measured in meters
        'min_table_height' : 0.75, 
        'table_init_factor' : 0.9, # this multiplicative factor limits how close you can initialize to wall

        # object stuff
        'object_ids' : object_ids,
        'max_xratio' : 1/4,
        'max_yratio' : 1/4,
        'max_zratio' : 1/3,
        'delta' : 1.0,

        # stuff
        'max_initialization_tries' : 100,

        # Camera/Frustum parameters
        'img_width' : 640, 
        'img_height' : 480,
        'near' : 0.01,
        'far' : 100,
        'fov' : 45, # vertical field of view in angles

        # other camera stuff
        'max_camera_rotation' : np.pi / 15., # Max rotation in radians

        # other stuff
        'taxonomy_dict' : taxonomy_dict,
        'nyuv2_40_classes_filter_list' : nyuv2_40_classes_filter_list,
        'coarse_grained_classes_filter_list' : coarse_grained_classes_filter_list,                   
        
    }






    #### Actual Data Generation #####

    views_per_scene = 7
    save_path = root_dir+'/data/tabletop_dataset_v5/' + \
                ('training_set/' if train_or_test == 'train' else 'test_set/')
                
    new_save_path = root_dir+'/data/tabletop_dataset_poses2/' + \
                ('training_set/' if train_or_test == 'train' else 'test_set/')

    scene_num = start_scene # start from here
    while scene_num < end_scene:
        
        # Load scene
        sim = suncg_sim.Simulator(mode='gui', 
                                  suncg_data_dir_base=suncg_dir, 
                                  shapenet_data_dir_base=root_dir, 
                                  params=simulation_params, 
                                  verbose=False)
        scene_description = json.load(open(save_path + f"scene_{scene_num:05d}/scene_description.txt"))
        save_dir = save_path + f"scene_{scene_num:05d}/"
        new_save_dir=new_save_path+ f"scene_{scene_num:05d}/"

#         scene_description = json.load(open('/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/tabletop_dataset_with_poses/debug/cube/scene_description.txt'))
#         save_dir = '/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/tabletop_dataset_with_poses/debug/cube/'
#         new_save_dir='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/tabletop_dataset_with_poses/debug/cube/'

        # Dictionary to save views
        scene_description['views'] = {}

        # Background-only view
        sim.reset()
        #sim.load_house_room(scene_description)
         
        valid_background_view = False
        num_tries = 0
#         while not valid_background_view:
#               
#             if num_tries > simulation_params['max_initialization_tries']:
#                 break # this will force the entire scene to start over
#             num_tries += 1
#               
#             # Sample the view
#             img_dict, _ = sim.sample_room_view()
#   
#             # Make sure it's valid. MUST have at least 2 SUNCG objects (e.g. walls/floor/fridge/couch)
#             unique_labels = np.unique(img_dict['orig_seg_img'])
#             valid_background_view = unique_labels.shape[0] >= 2
#   
#         if not valid_background_view:
#             printout("No valid background views...")
#             sim.disconnect()
#             scene_num += 1
#             continue
#         else:
#             if not os.path.exists(os.path.dirname(new_save_dir)):
#                 os.makedirs(new_save_dir)
#             save_img_dict(img_dict, 0, new_save_dir)
#             scene_description['views']['background'] = img_dict['view_params']
  
        # Background-table view
        sim.load_table(scene_description)
        valid_table_view = False
        num_tries = 0
#         while not valid_table_view:
#               
#             if num_tries > simulation_params['max_initialization_tries']:
#                 break # this will force the entire scene to start over
#             num_tries += 1
#               
#             # Sample the view
#             img_dict, _, _, _ = sim.sample_table_view()
#               
#             # Make sure it's valid
#             unique_labels = np.unique(img_dict['seg'])
#             valid_table_view = 1 in unique_labels and np.count_nonzero(img_dict['seg'] == 1) > 75
#               
#         if not valid_table_view:
#             printout("No valid table views...")
#             sim.disconnect()
#             scene_num += 1
#             continue
#         else:
#             save_img_dict(img_dict, 1, new_save_dir)
#             scene_description['views']['background+table'] = img_dict['view_params']
            
        # Sample background-table-object views and save 
        sim.load_objects(scene_description)
        scene_description['views']['background+table+objects'] = []
        valid_views = False
        view_num = 2; num_tries = 0    
        while not valid_views: #view_num < views_per_scene:
            
            if num_tries > simulation_params['max_initialization_tries']:
                break # this will force the entire scene to start over
            num_tries += 1

            # Sample the view
            img_dict, bid_to_seglabel_mapping, camera_pos, lookat_pos = sim.sample_table_view()
            
            seg_label_to_bid={bid_to_seglabel_mapping[i]: i for i in bid_to_seglabel_mapping}
            bid_to_shape_ind={}
            for bid in bid_to_seglabel_mapping:
                if bid_to_seglabel_mapping[bid]>=2:
                    for obj_id in sim._obj_id_to_body:
                        if sim._obj_id_to_body[obj_id].bid==bid:
                            obj_ind=sim.shapenet_obj_stuff['obj_ids'].index(obj_id)
                            bid_to_shape_ind[bid]=obj_ind
                            break
            
            # Make sure it's valid
            unique_labels = np.unique(img_dict['seg'])
            unique_object_labels = set(unique_labels).difference({0,1})
            valid = (0 in unique_labels and # background is in view
                     1 in unique_labels and # table is in view
                     len(unique_object_labels) >= 1 # at least 1 objects in view
                    )
            for label in unique_object_labels: # Make sure these labels are large enough
                if np.count_nonzero(img_dict['seg'] == label) < 75:
                    valid = False
            
            print('skipping valid check gen_data_with_poses')
#             if not valid:
#                 continue # sample another scene

            label_ind=0
            for label in unique_object_labels:
                if label_ind==4:
                    u=0
                single_object_seg=np.where(img_dict['seg']==label, 1, 0)
                where_label=np.argwhere(single_object_seg==1)
                seg_filename = new_save_dir + f"segmentation_{view_num:05d}_{label_ind}.png"
                cv2.imwrite(seg_filename, 255*single_object_seg)
                
                object_pose_info={}
                object_pose_info['model']=sim.shapenet_obj_stuff['obj_mesh_filenames'][bid_to_shape_ind[seg_label_to_bid[label]]]
                object_pose_info['cam_R_m2c']=np.ndarray.tolist(np.zeros(9, dtype=float)) #don't try to predict rotation for now
                object_pose=sim.shapenet_obj_stuff['cog'][bid_to_shape_ind[seg_label_to_bid[label]]]
                translation_vector=object_pose
                camera_translation_vector=sim.transform_to_camera_vector(translation_vector, camera_pos, lookat_pos,img_dict['view_params']['camera_up_vector'])
                
#                 camera_direction=lookat_pos-camera_pos
#                 z_basis=camera_direction
#                 y_basis=img_dict['view_params']['camera_up_vector']  
#                 x_basis=np.cross(y_basis, z_basis)
#                 x_basis/=np.linalg.norm(x_basis)
#                 proj_x_comp=np.dot(translation_vector, x_basis)
#                 proj_y_comp=np.dot(translation_vector, y_basis)
#                 proj_z_comp=np.dot(translation_vector, z_basis)
#                 translation_projection=np.array([proj_x_comp, proj_y_comp, proj_z_comp])
                
                
                object_pose_info['cam_t_m2c']=np.ndarray.tolist((1000.0*camera_translation_vector).astype(float))
                a=np.argwhere(single_object_seg==1)
                upper_left=[int(np.min(np.argwhere(single_object_seg==1)[:, 0])), int(np.min(np.argwhere(single_object_seg==1)[:, 1]))]
                lower_right=[np.max(np.argwhere(single_object_seg==1)[:, 0]), np.max(np.argwhere(single_object_seg==1)[:, 1])]
                object_pose_info['obj_bb']=upper_left+[int(lower_right[0]-upper_left[0]), int(lower_right[1]-upper_left[1])]
                obj_pose_filename = new_save_dir + f"pose_info_{view_num:05d}_{label_ind}.png"
                if not os.path.exists(os.path.dirname(obj_pose_filename)):
                    os.makedirs(os.path.dirname(obj_pose_filename))
                with open(obj_pose_filename, 'w') as save_file:
                    json.dump(object_pose_info, save_file) 
                label_ind+=1
            ### Save stuff ###
            save_img_dict(img_dict, view_num, new_save_dir)
            scene_description['views']['background+table+objects'].append(img_dict['view_params'])
            
            # increment    
            view_num += 1
            if view_num >= views_per_scene:
                valid_views = True
            
        if not valid_views:
            printout("Tried to sample view too many times...")
            sim.disconnect()
            scene_num += 1
            continue    
        
        # Scene Description
        scene_description_filename = new_save_dir + 'scene_description.txt'
        with open(scene_description_filename, 'w') as save_file:
            json.dump(scene_description, save_file)    

        # increment
        scene_num += 1
        if scene_num % 10 == 0:
            printout(f"Generated scene {scene_num}!")
        sim.disconnect()

if __name__ == '__main__':
    main()

