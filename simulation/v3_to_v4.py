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

# my libraries
sys.path.insert(0, os.path.abspath('..'))
import simulation_util as sim_util

# pybullet
import pybullet as p
import pybullet_data

# suncg
import pybullet_suncg.simulator as suncg_sim


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

    args = sys.argv[1:]
    start_scene = int(args[0])
    end_scene = int(args[1])
    train_or_test = args[2]

    ##### Load SUNCG stuff #####

    suncg_dir = '/data/suncg/v1/'

    # House lists
    training_houses_filename = '/data/tabletop_dataset_v4/training_suncg_houses.json'
    test_houses_filename = '/data/tabletop_dataset_v4/test_suncg_houses.json'
    train_houses = json.load(open(training_houses_filename))
    test_houses = json.load(open(test_houses_filename))

    # Room types I'm considering
    valid_room_types = set(['Living_Room', 'Kitchen', 'Room', 'Dining Room', 'Office'])

    # Room objects to filter out
    nyuv2_40_classes_filter_list = ['desk', 'chair', 'table', 'person', 'otherstructure', 'otherfurniture']
    coarse_grained_classes_filter_list = ['desk', 'chair', 'table', 'person', 'computer', 'bench_chair', 
                                          'ottoman', 'storage_bench', 'pet']





    ##### Load ShapeNet stuff #####

    shapenet_filepath = '/data/ShapeNetCore.v2/'

    # Create a dictionary of name -> synset_id
    temp = json.load(open(shapenet_filepath + 'taxonomy.json'))
    taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}

    # weirdly, the synsets in the taxonomy file are not the same as what's in the ShapeNetCore.v2 directory. Filter this out
    synsets_in_dir = os.listdir(shapenet_filepath)
    synsets_in_dir.remove('taxonomy.json')
    synsets_in_dir.remove('README.txt')

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
    training_tables_filename = '/data/tabletop_dataset_v4/training_shapenet_tables.json'
    test_tables_filename = '/data/tabletop_dataset_v4/test_shapenet_tables.json'
    train_tables = json.load(open(training_tables_filename))
    test_tables = json.load(open(test_tables_filename))

    # List of train/test object instances
    training_instances_filename = '/data/tabletop_dataset_v4/training_shapenet_objects.json'
    test_instances_filename = '/data/tabletop_dataset_v4/test_shapenet_objects.json'
    train_models = json.load(open(training_instances_filename))
    test_models = json.load(open(test_instances_filename))






    ##### Simulation Parameters #####
    house_ids = train_houses if train_or_test == 'train' else test_houses
    valid_tables = train_tables if train_or_test == 'train' else test_tables
    object_ids = train_models if train_or_test == 'train' else test_models

    simulation_params = {
        # scene stuff
        'min_num_objects_per_scene' : 5,
        'max_num_objects_per_scene' : 15,

        # House stuff
        'house_ids' : train_houses, # test_houses

        # room stuff
        'valid_room_types' : valid_room_types,
        'min_xlength' : 3.0, # Note: I believe this is in meters
        'min_ylength' : 3.0, 

        # table stuff
        'valid_tables' : train_tables, # test_tables
        'max_table_height' : 1.0, # measured in meters
        'min_table_height' : 0.75, 
        'table_init_factor' : 0.9, # this multiplicative factor limits how close you can initialize to wall

        # object stuff
        'object_ids' : train_models, # test_models
        'max_xratio' : 1/3,
        'max_yratio' : 1/3,
        'max_zratio' : 1/4,
        'delta' : 0.2,

        # stuff
        'max_initialization_tries' : 100,

        # Camera/Frustum parameters
        'img_width' : 640, 
        'img_height' : 480,
        'near' : 0.01,
        'far' : 100,
        'fov' : 60, # vertical field of view in angles

        # other camera stuff
        'max_camera_rotation' : np.pi / 10., # Max rotation in radians

        # other stuff
        'taxonomy_dict' : taxonomy_dict,
        'nyuv2_40_classes_filter_list' : nyuv2_40_classes_filter_list,
        'coarse_grained_classes_filter_list' : coarse_grained_classes_filter_list,                   
        
    }






    #### Actual Data Generation #####

    views_per_scene = 7
    save_path = '/data/tabletop_dataset_v4/' + \
                ('training_set/' if train_or_test == 'train' else 'test_set/')

    scene_num = start_scene # start from here
    while scene_num < end_scene:
        printout("SCENE NUM " + str(scene_num))
        
        # Load scene
        sim = suncg_sim.Simulator(mode='gui', 
                                  suncg_data_dir_base=suncg_dir, 
                                  shapenet_data_dir_base=shapenet_filepath, 
                                  params=simulation_params, 
                                  verbose=False)
        scene_description = json.load(open(save_path.replace('_v4', '_v3') + f"scene_{scene_num:05d}/scene_description.txt"))

        # Make directory
        save_dir = save_path + f"scene_{scene_num:05d}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)




        # Dictionary to save views
        scene_description['views'] = {}

        # Background-only view
        sim.reset()
        sim.load_house_room(scene_description)
        
        valid_background_view = False
        num_tries = 0
        while not valid_background_view:
            
            if num_tries > simulation_params['max_initialization_tries']:
                break # this will force the entire scene to start over
            num_tries += 1
            
            # Sample the view
            img_dict = sim.sample_room_view()

            # Make sure it's valid. MUST have at least 2 SUNCG objects (e.g. walls/floor/fridge/couch)
            unique_labels = np.unique(img_dict['orig_seg_img'])
            valid_background_view = unique_labels.shape[0] >= 2

        if not valid_background_view:
            printout("No valid background views...")
            sim.disconnect()
            scene_num += 1
            continue
        else:
            save_img_dict(img_dict, 0, save_dir)
            scene_description['views']['background'] = img_dict['view_params']

        # Background-table view
        sim.load_table(scene_description)
        valid_table_view = False
        num_tries = 0
        while not valid_table_view:
            
            if num_tries > simulation_params['max_initialization_tries']:
                break # this will force the entire scene to start over
            num_tries += 1
            
            # Sample the view
            img_dict = sim.sample_table_view()
            
            # Make sure it's valid
            unique_labels = np.unique(img_dict['seg'])
            valid_table_view = 1 in unique_labels and np.count_nonzero(img_dict['seg'] == 1) > 75
            
        if not valid_table_view:
            printout("No valid table views...")
            sim.disconnect()
            scene_num += 1
            continue
        else:
            save_img_dict(img_dict, 1, save_dir)
            scene_description['views']['background+table'] = img_dict['view_params']
            
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
            img_dict = sim.sample_table_view()
            
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
            if not valid:
                continue # sample another scene

            ### Save stuff ###
            save_img_dict(img_dict, view_num, save_dir)
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
        scene_description_filename = save_dir + 'scene_description.txt'
        with open(scene_description_filename, 'w') as save_file:
            json.dump(scene_description, save_file)    

        # increment
        scene_num += 1
        if scene_num % 10 == 0:
            printout(f"Generated scene {scene_num}!")
        sim.disconnect()

if __name__ == '__main__':
    main()

