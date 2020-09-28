import json
import os
os.environ["OMP_NUM_THREADS"]="1"
# os.environ["MUJOCO_GL"]="osmesa"
import numpy as np
import multiprocessing as mp
import math
from trajopt.envs.mujoco_env import MujocoEnv
import trimesh
from pose_model_estimator import get_mesh_list, compute_mujoco_int_transform
import shutil
from xml.dom import minidom
from trajopt.utils import generate_perturbed_actions
from trajopt.sandbox.examples.herb_pushing_mppi import convex_decomp_target_object_env
import random
import cv2
from pyquaternion import Quaternion
import pickle
from optparse import OptionParser
import traceback
import pybullet as p
from dm_control.mujoco.engine import Camera
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from trajopt.envs.herb_pushing_env import HerbEnv

target_objects=["maytoni",
        "potted_plant_2",
        "lamp_1",
        "cup_1",
        "vase_1",
        "vase_2"]

target_objects+=["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can","006_mustard_bottle","007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", 
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

parser = OptionParser()
#path to shapenet dataset
parser.add_option("--shapenet_filepath", dest="shapenet_filepath", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/ShapeNetCore.v2/')
#filepath to convex decompositions of shapenet objects. I posted this in the slack channel
parser.add_option("--shapenet_decomp_filepath", dest="shapenet_decomp_filepath", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/shapenet_conv_decmops/')
#root project dir
parser.add_option("--top_dir", dest="top_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/cluttered_manipulation_scenes/')
#roo project dir+/inhand_datagen
parser.add_option("--instances_dir", dest="instances_dir", default='/home/willie/workspace/SSC/inhand_datagen')
#where to save generated data to
parser.add_option("--save_dir", dest="save_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/cluttered_manipulation_scenes/')
parser.add_option("--num_threads", dest="num_threads", type="int", default=12)
(options, args) = parser.parse_args()

def add_object(scene_name, object_name, mesh_name, xpos, y_pos, size, color, rot, other_objects, run_id, top_dir):
    print('1')
    geom_args=[]
    
    if object_name in target_objects[6:]:
        mesh_filename=os.path.join(top_dir, f'herb_reconf/assets/ycb_objects/{mesh_name}/google_16k/nontextured.stl')
        type='ycb'
    else:
        mesh_filename=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{mesh_name}/scene.stl')
        type='downloaded'
    
    print('1a')
    
    mujoco_center, _=compute_mujoco_int_transform(mesh_filename, run_id, size=size)
    
    print('1b')
    
    mic=mujoco_center[2]
    mesh=trimesh.load(mesh_filename)
    #lower_z=-mic
    z_offset=0.3-mesh.bounds[0,2]
    
    print('1c')
    
    contact_geom_list=[
        ("herb/wam_1/bhand//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_16", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_15", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_14", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_12", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_10", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_8", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_7", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_6", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_4", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_3", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_2", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_1", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_21", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_17", "4 4 0.2 0.04 0.04"),
        ("table_plane", "0.5 0.5 0.005 0.0001 0.0001")
    ]

    xmldoc = minidom.parse(scene_name)
    
    print('2')
    
    assets = xmldoc.getElementsByTagName('asset')[0]
    new_mesh=xmldoc.createElement('mesh')
    new_mesh.setAttribute('name', f'gen_mesh_{object_name}')
    new_mesh.setAttribute('class', 'geom0')
    new_mesh.setAttribute('scale', f'{size} {size} {size}')
    if type=='ycb':
        new_mesh.setAttribute('file', f'ycb_objects/{mesh_name}/google_16k/nontextured.stl')
    elif type=='downloaded':
        new_mesh.setAttribute('file', f'downloaded_assets/{mesh_name}/scene.stl')
    assets.appendChild(new_mesh)
    
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('body')
    body_name=f'gen_body_{object_name}'
    new_body.setAttribute('name', body_name)
    new_body.setAttribute('pos', f'{xpos} {y_pos} {z_offset}')
    new_body.setAttribute('euler', f'{rot[0]} {rot[1]} {rot[2]}')
    
    geom_names=[]
    new_geom=xmldoc.createElement('geom')
    geom_name=f'gen_geom_{object_name}'
    geom_names.append(geom_name)
    new_geom.setAttribute('name', geom_name)
    new_geom.setAttribute('class', '/')
    new_geom.setAttribute('type', 'mesh')
    new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
    new_geom.setAttribute('mesh', f'gen_mesh_{object_name}')
    for geom_arg in geom_args:
        new_geom.setAttribute(geom_arg[0], geom_arg[1])
    
    new_body.appendChild(new_geom)
    
    new_joint=xmldoc.createElement('joint')
    new_joint.setAttribute('name', f'gen_joint_{object_name}')
    new_joint.setAttribute('class', '/')
    new_joint.setAttribute('type', 'free')
    #new_joint.setAttribute('damping', '0.001')
    new_body.appendChild(new_joint)
    
    world_body.appendChild(new_body)
  
    print('3')
#     contact = xmldoc.getElementsByTagName('contact')[0]
#     for contact_geom in contact_geom_list:
#         new_contact=xmldoc.createElement('pair')
#         geom_name=f'gen_geom_{object_name}'
#         new_contact.setAttribute('geom1', geom_name)
#         new_contact.setAttribute('geom2', contact_geom[0])
#         new_contact.setAttribute('friction', contact_geom[1])
#         new_contact.setAttribute('solref', "0.01 1")
#         new_contact.setAttribute('solimp', "0.999 0.999 0.01")
#         new_contact.setAttribute('condim', "4")
#         contact.appendChild(new_contact)
#     for added_object_name in other_objects:
#         new_contact=xmldoc.createElement('pair')
#         geom_name=f'gen_geom_{object_name}'
#         geom2_name=f'gen_geom_{added_object_name}'
#         new_contact.setAttribute('geom1', geom_name)
#         new_contact.setAttribute('geom2', geom2_name)
#         new_contact.setAttribute('friction', "0.5 0.5 0.005 0.0001 0.0001")
#         new_contact.setAttribute('solref', "0.01 1")
#         new_contact.setAttribute('solimp', "0.999 0.999 0.01")
#         new_contact.setAttribute('condim', "4")
#         contact.appendChild(new_contact)
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)
    
    return body_name, geom_names

def transform_to_camera_vector(vector, camera_pos, lookat_pos, camera_up_vector):
    view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
    view_matrix = np.array(view_matrix).reshape(4,4, order='F')
    vector=np.concatenate((vector, np.array([1])))
    transformed_vector=view_matrix.dot(vector)
    return transformed_vector[:3]

#transform robot hand meshes into current pose
def make_known_meshes(known_meshes, physics, geom_names):
    transformed_known_meshes=[]
    for known_mesh_ind in range(len(known_meshes)):
        
        transformed_known_mesh=known_meshes[known_mesh_ind].copy()
        transform=np.eye(4)
        transform[0:3,0:3]=np.reshape(physics.named.data.geom_xmat[geom_names[known_mesh_ind]],(3,3))
        transformed_known_mesh.apply_transform(transform)
        transform=np.eye(4)
        transform[0:3,3]=physics.named.data.geom_xpos[geom_names[known_mesh_ind]]
        transformed_known_mesh.apply_transform(transform)
        transformed_known_meshes.append(transformed_known_mesh)

    return transformed_known_meshes

#enable/disable gravity in xml
def set_gravity(scene_name, set_unset):
    xmldoc = minidom.parse(scene_name)
    options = xmldoc.getElementsByTagName('option')[0]
    if set_unset:
        options.setAttribute('gravity', "0 0 -9.81")
    else:
        options.setAttribute('gravity', "0 0 0")
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

def add_camera(scene_name, cam_name, cam_pos, cam_target, cam_id):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('camera')
    new_body.setAttribute('name', cam_name)
    new_body.setAttribute('mode', 'targetbody')
    new_body.setAttribute('pos', f'{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}')
    new_body.setAttribute('target', f'added_cam_target_{cam_id}')
    world_body.appendChild(new_body)
    
    new_body=xmldoc.createElement('body')
    new_body.setAttribute('name', f'added_cam_target_{cam_id}')
    new_body.setAttribute('pos', f'{cam_target[0]} {cam_target[1]} {cam_target[2]}')
    new_geom=xmldoc.createElement('geom')
    geom_name=f'added_cam_target_geom_{cam_id}'
    new_geom.setAttribute('name', geom_name)
    new_geom.setAttribute('class', '/')
    new_geom.setAttribute('type', 'box')
    new_geom.setAttribute('contype', '0')
    new_geom.setAttribute('conaffinity', '0')
    new_geom.setAttribute('group', '1')
    new_geom.setAttribute('size', "1 1 1")
    new_geom.setAttribute('rgba', f'0 0 0 0')
    new_body.appendChild(new_geom)
    world_body.appendChild(new_body)
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

def add_objects(scene_name, object_name, mesh_names, pos, size, color, rot, run_id, other_objects):

    xmldoc = minidom.parse(scene_name)
    
    assets = xmldoc.getElementsByTagName('asset')[0]
    for mesh_ind in range(len(mesh_names)):
        new_mesh=xmldoc.createElement('mesh')
        new_mesh.setAttribute('name', f'gen_mesh_{object_name}_{mesh_ind}')
        new_mesh.setAttribute('class', 'geom0')
        new_mesh.setAttribute('scale', f'{size} {size} {size}')
        new_mesh.setAttribute('file', mesh_names[mesh_ind])
        assets.appendChild(new_mesh)
    
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('body')
    body_name=f'gen_body_{object_name}'
    new_body.setAttribute('name', body_name)
    new_body.setAttribute('pos', f'{pos[0]} {pos[1]} {pos[2]}')
    new_body.setAttribute('euler', f'{rot[0]} {rot[1]} {rot[2]}')
    
    geom_names=[]
    for geom_ind in range(len(mesh_names)):
        new_geom=xmldoc.createElement('geom')
        geom_name=f'gen_geom_{object_name}_{geom_ind}'
        other_objects.append(f'gen_geom_{object_name}_{geom_ind}')
        geom_names.append(geom_name)
        new_geom.setAttribute('name', geom_name)
        new_geom.setAttribute('class', '/')
        new_geom.setAttribute('type', 'mesh')
        new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
        new_geom.setAttribute('mesh', f'gen_mesh_{object_name}_{geom_ind}')
        new_body.appendChild(new_geom)
    
    new_joint=xmldoc.createElement('joint')
    new_joint.setAttribute('name', f'gen_joint_{object_name}')
    new_joint.setAttribute('class', '/')
    new_joint.setAttribute('type', 'free')
    new_joint.setAttribute('damping', '0.001')
    new_body.appendChild(new_joint)
    
    world_body.appendChild(new_body)
    
#     contact_geom_list=[
#         ("herb/wam_1/bhand//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_16", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_15", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_14", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_12", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_10", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_8", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_7", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_6", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_4", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_3", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_2", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1/bhand//unnamed_geom_1", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1//unnamed_geom_21", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
#         ("herb/wam_1//unnamed_geom_17", "4 4 0.2 0.04 0.04"),
#         ("table_plane", "0.5 0.5 0.005 0.0001 0.0001")
#     ]
#     
#     contact = xmldoc.getElementsByTagName('contact')[0]
#     for geom_ind in range(len(mesh_names)):
#         geom_name=f'gen_geom_{object_name}_{geom_ind}'
#         for contact_geom in contact_geom_list:
#             new_contact=xmldoc.createElement('pair')
#             new_contact.setAttribute('geom1', geom_name)
#             new_contact.setAttribute('geom2', contact_geom[0])
#             new_contact.setAttribute('friction', contact_geom[1])
#             new_contact.setAttribute('solref', "0.01 1")
#             new_contact.setAttribute('solimp', "0.999 0.999 0.01")
#             new_contact.setAttribute('condim', "4")
#             contact.appendChild(new_contact)
#         for added_object_name in other_objects:
#             if added_object_name!=geom_name:
#                 new_contact=xmldoc.createElement('pair')
#                 new_contact.setAttribute('geom1', geom_name)
#                 new_contact.setAttribute('geom2', added_object_name)
#                 new_contact.setAttribute('friction', "0.5 0.5 0.005 0.0001 0.0001")
#                 new_contact.setAttribute('solref', "0.01 1")
#                 new_contact.setAttribute('solimp', "0.999 0.999 0.01")
#                 new_contact.setAttribute('condim', "4")
#                 contact.appendChild(new_contact)
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)
    
    return body_name, geom_names

def move_object_in_xml(scene_name, object_name, object_pos, object_rot):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    for ind in range(len(world_body.childNodes)):
        if isinstance(world_body.childNodes[ind], minidom.Element) and world_body.childNodes[ind]._attrs['name'].nodeValue==object_name:
            break
    world_body.childNodes[ind].setAttribute('pos', f'{object_pos[0]} {object_pos[1]} {object_pos[2]}')
    world_body.childNodes[ind].setAttribute('quat', f'{object_rot[0]} {object_rot[1]} {object_rot[2]} {object_rot[3]}')
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

def move_object(e, ind, pos):
    all_poses=e.data.qpos.ravel().copy()
    all_vels=e.data.qvel.ravel().copy()
    all_poses[22+7*ind:22+7*ind+3]=pos
    all_vels[21+6*ind:21+6*ind+6]=0
    e.set_state(all_poses, all_vels)
    
def get_visible_pixels(env, object_num):
    #env = HerbEnv(scene_name, np.zeros((1,3)), task='grasping', obs=False)
    #env.reset_model(seed=2)
    segs=env.model.render(height=480, width=640, camera_id=1, depth=False, segmentation=True)
    return np.sum(segs==object_num)

def get_random_object_params(upright_chance):
    min_size=0.5
    max_size=4.0
    object_color=np.random.uniform(size=3)
    object_size=np.random.uniform(low=min_size, high=max_size, size=1)[0]
    if random.random()<upright_chance:
        object_rot=np.array([0,0,np.random.uniform(low=0, high=2*math.pi, size=1)[0]])
    else:
        object_rot=np.random.uniform(low=0, high=2*math.pi, size=3)
    return object_size, object_color, object_rot

def get_num_contacts(e, body_num):
    contacts=e.model._data.contact
    num_contacts=0
    for contact in contacts:
        geom_name=e.model.model.id2name(contact[10], "geom")
        if 'gen_geom_object' in geom_name and body_num==int(geom_name.split('_')[3]):
            num_contacts+=1
        geom_name=e.model.model.id2name(contact[11], "geom")
        if 'gen_geom_object' in geom_name and body_num==int(geom_name.split('_')[3]):
            num_contacts+=1
    return num_contacts

#@profile
def gen_data(scene_num, shapenet_filepath, shapenet_decomp_filepath, instances_dir, top_dir, save_dir, target_object, task, num_generated, run_id):
    global_gauss_std=np.array([0.25, 0.25])
    if task=='hard_pushing' or task=='grasping':
        global_gauss_center=np.array([0.0, 0])
    else:
        global_gauss_center=np.array([0.05, -0.35])
    prob_upright=0.8
    max_height=0.35
    lib_type='downloaded' if target_object in target_objects[0:6] else 'ycb'
    
    np.random.seed(scene_num)
    target_obj_geom_id=72
    train_or_test='training_set'
    num_images=10000000
    
    box=trimesh.creation.box(np.array([0.1, 0.1, 0.1]))
    
    min_object_scale=1.0
    max_object_scale=4.0
    
    print(f'generating {train_or_test} dataset')
    
    training_instances_filename = os.path.join(instances_dir, 'training_instances.json')
    test_instances_filename = os.path.join(instances_dir, 'novel_class_test_instances.json')
    train_models = json.load(open(training_instances_filename))
    test_models = json.load(open(test_instances_filename))
    object_ids = train_models if train_or_test == 'training_set' else test_models
    
    training_tables_filename = os.path.join(instances_dir, 'training_shapenet_tables.json')
    test_tables_filename = os.path.join(instances_dir, 'test_shapenet_tables.json')
    train_tables = json.load(open(training_tables_filename))
    test_tables = json.load(open(test_tables_filename))
    valid_tables = train_tables if train_or_test == 'training_set' else test_tables
    
    
    
    new_object_ids=[]
    for cat in object_ids:
        for obj_id in object_ids[cat]:
            new_object_ids.append((cat, obj_id))
    object_ids=new_object_ids
    
    temp = json.load(open(shapenet_filepath + 'taxonomy.json'))
    taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}
    
    # weirdly, the synsets in the taxonomy file are not the same as what's in the ShapeNetCore.v2 directory. Filter this out
    synsets_in_dir = os.listdir(shapenet_filepath)
    synsets_in_dir.remove('taxonomy.json')
    
    taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synsets_in_dir}
    
    # selected_index = np.random.randint(0, object_ids.shape[0])
    
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
    
    included_meshes=[]
    geom_names=[]
    pred_obj_meshes=[]
    
    if os.path.exists(save_dir+f'/{target_object}/'):
        shutil.rmtree(save_dir+f'/{target_object}/')
    os.mkdir(save_dir+f'/{target_object}/')
    
#     num_generated=thread_num
#     view_num=thread_num
    generated=False
    occlusions=[]
    thread_num=scene_num
    env_info={} #name: (occlusion level, start poses)
    
    print('a')
    
    while not generated:#num_generated<num_images/num_threads:
        try:
            scene_xml_file=os.path.join(top_dir, f'herb_reconf/{task}_scene.xml')
            decomp_scene_xml_file=os.path.join(save_dir, f'{target_object}_{task}_{num_generated}_decomp_scene.xml')
            shutil.copyfile(scene_xml_file, decomp_scene_xml_file)
            gen_scene_xml_file=os.path.join(save_dir, f'{target_object}_{task}_{num_generated}_scene.xml')
            shutil.copyfile(scene_xml_file, gen_scene_xml_file)
            
            #print('b')
            
            if target_object in target_objects[6:]:
                mesh_filename=os.path.join(top_dir, f'herb_reconf/assets/ycb_objects/{target_object}/google_16k/nontextured.stl')
            else:
                mesh_filename=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{target_object}/scene.stl')
            
            #choose num objects, pos dist center (on table)
            num_objects=random.randint(3,10)
            _, color, rot=get_random_object_params(prob_upright)
            if task=='hard_pushing' or task=="grasping":
                #add_object(decomp_scene_xml_file, '0', target_object, 0, 0, 1, color, rot, [], id, top_dir)
                add_object(gen_scene_xml_file, '0', target_object, 0, 0, 1, color, rot, [], id, top_dir)
                #scene_file, _, other_objects=convex_decomp_target_object_env(gen_scene_xml_file, 'gen_body_0', mesh_filename, save_dir, run_id, top_dir, new_scene_name=decomp_scene_xml_file)
            elif task=='easy_pushing':
                #add_object(decomp_scene_xml_file, '0', target_object, -0.05,-0.35, 1, color, rot, [], id, top_dir)
                add_object(gen_scene_xml_file, '0', target_object, -0.05,-0.35, 1, color, rot, [], id, top_dir)
            scene_file, _, other_objects=convex_decomp_target_object_env(gen_scene_xml_file, 'gen_body_0', mesh_filename, save_dir, run_id, top_dir, new_scene_name=decomp_scene_xml_file)
            #other_objects=['gen_geom_0']
            
            print('c')
            
            #drop one by one onto table
            if target_object in target_objects[6:]:
                push_object=trimesh.load(os.path.join(top_dir, f'herb_reconf/assets/ycb_objects/{target_object}/google_16k/nontextured.stl'))
            else:
                push_object=trimesh.load(os.path.join(top_dir, f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{target_object}/scene.stl'))  
            e=HerbEnv(decomp_scene_xml_file, box, task=task, obs=False, push_mesh_vertices=box, skip=1)
            sigma=1.0*np.ones(e.action_dim)
            sigma[0:7]=sigma[0:7]*(e.action_space.high[0:7]-e.action_space.low[0:7])
            sigma[0]=1*sigma[0]
            sigma[1]=0.5*sigma[1]
            sigma[2]=2*sigma[1]
            sigma[3]=0.5*sigma[3]
            sigma[7:]=sigma[7:]*(e.action_space.high[14:22]-e.action_space.low[14:22])
            sigma[7:]=50*sigma[7:]
            sigma[5]=sigma[5]
            sigma[-2:]=20*sigma[-2:]
            filter_coefs = [sigma, 0.25, 0.8, 0.0, np.concatenate((e.action_space.high[0:7]-e.action_space.low[0:7], e.action_space.high[14:22]-e.action_space.low[14:22])), np.concatenate((e.action_space.low[0:7], e.action_space.low[14:22])), np.concatenate((e.action_space.high[0:7], e.action_space.high[14:22]))]
    
            state=e.get_env_state().copy()
            e.set_env_state(state)
            base_act=np.repeat(np.expand_dims(state['qp'][:15], axis=0), 1000, axis=0)
            act, vel=generate_perturbed_actions(state, base_act, filter_coefs, 0.5, 0.15, state['qp'][4], 1.59, hand_open=0, move=False)
            for added_object_ind in range(num_objects):
                for step in range(100):
    #                     if step%50==0:
    #                         rgb=e.model.render(height=480, width=640, camera_id=0, depth=False, segmentation=False)
    #                         cv2.imshow('rbg', rgb)
    #                         cv2.waitKey(20)
                    e.step(act[step])
            state=e.get_env_state().copy() 
            real_env=HerbEnv(gen_scene_xml_file, box, task=task, obs=False, push_mesh_vertices=box, skip=1)
            real_env.set_env_state(state)
            real_env.sim.physics.forward()
            
            print('d')
            
            rgb=real_env.model.render(height=480, width=640, camera_id=1, depth=False, segmentation=False)
            cv2.imshow('first rbg', rgb)
            cv2.waitKey(20)
            
            visible_pix=get_visible_pixels(real_env, 72)
            unobscured_visible_pix=visible_pix
            
            #choose objects, add to scene
            added_objects=0
            obj_mesh_filenames=[]
            obj_initial_positions=[]
            obj_colors=[]
            obj_scales=[]
            
            print('e')
            
            while added_objects<num_objects:
                selected_index = np.random.randint(0, len(object_ids))
                obj_id = object_ids[selected_index]
                obj_cat=taxonomy_dict[obj_id[0]]
                obj_mesh_filename = shapenet_filepath + f'/{obj_cat}/{obj_id[1]}/models/model_normalized.obj'
                object_mesh=trimesh.load(obj_mesh_filename)
                stl_obj_mesh_filename=os.path.join(top_dir, f'assets/model_normalized_{thread_num}_{added_objects}.stl')
                object_mesh.export(stl_obj_mesh_filename)
                object_color=np.random.uniform(size=3)
                object_size=np.random.uniform(low=min_object_scale, high=max_object_scale, size=1)[0]
                diag=np.sqrt(np.sum(np.square(object_mesh.bounds[0]-object_mesh.bounds[1])))
                object_size*=0.1/diag
                object_rot=np.random.uniform(low=0, high=2*math.pi, size=3)
                if random.random()<prob_upright:
                    object_rot[:2]=0
                object_drop_pos=np.random.normal(loc=global_gauss_center, scale=global_gauss_std)
                object_mesh=trimesh.load(stl_obj_mesh_filename)
                if object_mesh.faces.shape[0]>200000:
                    print('too many mesh faces!')
                    continue
                obj_mesh_filenames+=[obj_mesh_filename]
                obj_initial_positions.append(object_drop_pos)
                obj_colors.append(object_color)
                obj_scales.append(object_size)
                
                #load conv decomp meshes
                mesh_names=[]
                mesh_masses=[]
                decomp_shapenet_decomp_filepath=os.path.join(shapenet_decomp_filepath, f'{obj_cat}/{obj_id[1]}')
                for mesh_file in os.listdir(decomp_shapenet_decomp_filepath):
                    decomp_object_mesh=trimesh.load(os.path.join(decomp_shapenet_decomp_filepath, mesh_file))
                    trimesh.repair.fix_inversion(decomp_object_mesh)
                    if decomp_object_mesh.faces.shape[0]>5 and decomp_object_mesh.mass>10e-8:
                        obj_mesh_filename=os.path.join(shapenet_decomp_filepath, f'{obj_cat}/{obj_id[1]}', mesh_file[:-3]+'stl')
                        decomp_object_mesh.export(obj_mesh_filename)
                        mesh_names.append(obj_mesh_filename)
                        mesh_masses.append(decomp_object_mesh.mass)
                if len(mesh_names)>25:
                    heavy_inds=np.argsort(np.array(mesh_masses))
                    new_mesh_names=[]
                    for ind in range(25):
                        new_mesh_names.append(mesh_names[heavy_inds[-ind]])
                    mesh_names=new_mesh_names
                add_objects(decomp_scene_xml_file, f'object_{added_objects}_{thread_num}', mesh_names, [50,50,-5-added_objects], object_size, object_color, object_rot, thread_num, other_objects)
                add_objects(gen_scene_xml_file, f'object_{added_objects}_{thread_num}', [stl_obj_mesh_filename], [50,50,-5-added_objects], object_size, object_color, object_rot, thread_num, other_objects)
                added_objects+=1
            #drop one by one onto table
            e=HerbEnv(decomp_scene_xml_file, box, task=task, obs=False, push_mesh_vertices=box, skip=1)
            new_state=e.get_env_state().copy()
            new_state['qp'][:state['qp'].shape[0]]=state['qp']
            e.set_env_state(new_state)
            e.sim.physics.forward()
            for added_object_ind in range(num_objects):
                obj_drop_pos=obj_initial_positions[added_object_ind]
                max_h=0.75+max_height
                min_h=max_height
                move_object(e, added_object_ind, [obj_drop_pos[0], obj_drop_pos[1], max_h])
                e.sim.physics.forward()
                a=e.model._data.contact
                init_num_contacts=get_num_contacts(e, added_object_ind)
                best_h=max_h
                for i in range(10):
                    height=(max_h+min_h)/2.0
                    move_object(e, added_object_ind, [obj_drop_pos[0], obj_drop_pos[1], height])
                    e.sim.physics.forward()
                    num_contacts=get_num_contacts(e, added_object_ind)
                    if num_contacts>init_num_contacts:
                        min_h=height
                        best_h=min_h
                    else:
                        max_h=height
                move_object(e, added_object_ind, [obj_drop_pos[0], obj_drop_pos[1], height])
                for step in range(100):
                    if step%10==0:
                        rgb=e.model.render(height=480, width=640, camera_id=0, depth=False, segmentation=False)
                        cv2.imshow('rbg', rgb)
                        cv2.waitKey(20)
                    e.step(act[step])
            state=e.get_env_state().copy() 
            
            real_env=HerbEnv(gen_scene_xml_file, box, task=task, obs=False, push_mesh_vertices=box, skip=1)
            real_env.set_env_state(state)
            real_env.sim.physics.forward()
            rgb=real_env.model.render(height=480, width=640, camera_id=1, depth=False, segmentation=False)
            cv2.imshow('final rbg', rgb)
            cv2.waitKey(20)
            
            rgb=e.model.render(height=480, width=640, camera_id=1, depth=False, segmentation=False)
            cv2.imshow('final decomp', rgb)
            cv2.imwrite(save_dir+f'/{target_object}_{task}_{num_generated}_img.png', rgb)
            cv2.waitKey(20)
            
            visible_pix=get_visible_pixels(real_env, 72)
            
            with open(save_dir+f'/{target_object}_{task}_{num_generated}_scene_info.p', 'wb') as save_file:
                pickle.dump((visible_pix/unobscured_visible_pix, state), save_file)   
            
            num_generated+=1
        except:
            print('gen error!')
            traceback.print_exc()

        

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        raise    

if __name__ == '__main__':
    gen_data(0, options.shapenet_filepath, options.shapenet_decomp_filepath, options.instances_dir, options.top_dir, options.save_dir, "maytoni", 'hard_pushing', 0, 0)
    
#     num_processes=options.num_threads
#     pool = mp.Pool(processes=num_processes, maxtasksperchild=1)
#     for target_object_ind in range(len(target_objects)):
#         for scene_num in range(1000):
#             abortable_func = partial(abortable_worker, gen_data, timeout=240)
#             pool.apply_async(abortable_func, args=(scene_num, options.shapenet_filepath, options.shapenet_decomp_filepath, options.instances_dir, options.top_dir, options.save_dir, 0.0))
#     pool.close()
#     pool.join()
    
#     parallel_runs = [pool.apply_async(gen_data, args= for i in range(num_processes)]   
#     results = [p.get() for p in parallel_runs]
        
        
        
        