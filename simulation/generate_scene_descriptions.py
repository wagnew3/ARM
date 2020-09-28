""" This code is a python script that essentially replicates the code in simulating_data.ipynb

    As a script, this can be called many times with a bash script. This helps when I need to kill
    the process due to the weird memory leak in PyBullet.

    To call this script:
    $:~ python generate_data.py <start_scene> <end_scene>

    Keep end_scene - start_scene <= 400 in order to not get hung by memory leak

    NOTE: THIS SCRIPT ONLY GENERATES SCENE DESCRIPTIONS, so that it can be run in parallel w/out GUI.
          generate_data.py will read the scene descriptions and generate RGBD + Segmentation images.
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
    training_houses_filename = '/data/tabletop_dataset_v5/training_suncg_houses.json'
    test_houses_filename = '/data/tabletop_dataset_v5/test_suncg_houses.json'
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
    training_tables_filename = '/data/tabletop_dataset_v5/training_shapenet_tables.json'
    test_tables_filename = '/data/tabletop_dataset_v5/test_shapenet_tables.json'
    train_tables = json.load(open(training_tables_filename))
    test_tables = json.load(open(test_tables_filename))

    # List of train/test object instances
    training_instances_filename = '/data/tabletop_dataset_v5/training_shapenet_objects.json'
    test_instances_filename = '/data/tabletop_dataset_v5/test_shapenet_objects.json'
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
    save_path = '/data/tabletop_dataset_v5/' + \
                ('training_set/' if train_or_test == 'train' else 'test_set/')

    scene_num = start_scene # start from here
    while scene_num < end_scene:
        
        # Sample scene
        try:
            sim = suncg_sim.Simulator(mode='direct', 
                                      suncg_data_dir_base=suncg_dir, 
                                      shapenet_data_dir_base=shapenet_filepath, 
                                      params=simulation_params, 
                                      verbose=False)
            scene_description = sim.generate_scenes(1)[0]
        except TimeoutError as e: # Scene took longer than 45 seconds to generate, or errored out
            printout(str(e))
            sim.disconnect()
            continue
        except:
            printout("Errored out. Not due to timer, but something else...")
            sim.disconnect()
            continue
            
        # Make directory
        save_dir = save_path + f"scene_{scene_num:05d}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
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

