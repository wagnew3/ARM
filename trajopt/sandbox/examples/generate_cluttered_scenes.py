#generate cluttered manipulation scenes

import numpy as np
import os
os.environ["OMP_NUM_THREADS"]="1"
import pickle
os.environ["MUJOCO_GL"]="osmesa"
import shutil
import random
import math
from trajopt.envs.herb_pushing_env import HerbEnv
from xml.dom import minidom
from pose_model_estimator import compute_mujoco_int_transform
import trimesh
import cv2
import multiprocessing as mp
from trajopt.utils import generate_perturbed_actions
import time
from optparse import OptionParser
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import traceback
from trajopt.sandbox.examples.herb_pushing_mppi import convex_decomp_target_object_env
from trajopt.mujoco_utils import add_object, remove_object, make_global_contacts
                        
gtasks=['grasping', 'hard_pushing', 'easy_pushing']
tasks=['hard_pushing', 'grasping', 'easy_pushing']#['hard_pushing']

target_objects=["maytoni",
        "potted_plant_2",
        "lamp_1",
        "lamp_2",
        "cup_1",
        "vase_1",
        "vase_2",
        "cardboard_box",
        "vase_3",
        'lamp_3', 'lamp_4', 'glass_1', 'glass_2', 'glass_3', 'cup_2', 'trophy_1']

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

def get_visible_pixels(env, object_num):
    segs=env.model.render(height=480, width=640, camera_id=1, depth=False, segmentation=True)
    return np.sum(segs==object_num)
    
def get_random_ycb_object():
    min_size=0.5
    max_size=2.0
    object_name=target_objects[random.randint(16, len(target_objects)-1)]
    object_color=np.random.uniform(size=3)
    object_size=np.random.uniform(low=min_size, high=max_size, size=1)[0]
    object_rot=np.random.uniform(low=0, high=2*math.pi, size=1)[0]
    return object_name, object_size, object_color, object_rot

def move_object(e, ind, pos):
    all_poses=e.data.qpos.ravel().copy()
    all_vels=e.data.qvel.ravel().copy()
    all_poses[22+7*ind:22+7*ind+3]=pos
    all_vels[21+6*ind:21+6*ind+6]=0
    e.set_state(all_poses, all_vels)

lib_type='downloaded'

def generate_scene(target_object_ind, task, scene_num, top_dir, run_id, min_clutter, max_clutter):
    try:
        random.seed(run_id)
        np.random.seed(run_id)
        global_gauss_center=np.array([0.3, 0])        
        num_starting_global_objects=random.randint(0,6)

        if task=='hard_pushing' or task=='grasping':
            global_gauss_center=np.array([0.1, 0.0])
        else:
            global_gauss_center=np.array([-0.0, -0.35])
        glabal_gauss_std=np.array([0.1, 0.0])    
             
        target_object=target_objects[target_object_ind]

        for num_attempts in range(5):
            #copy fromb ase scene and add target manipulation object
            scene_xml_file=os.path.join(top_dir, f'herb_reconf/{task}_scene.xml')
            temp_scene_xml_file=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/temp_{task}_{target_object}_{scene_num}_{min_clutter}_scene.xml')
            shutil.copyfile(scene_xml_file, temp_scene_xml_file)
            temp_decomp_scene_xml_file=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/temp_decomp_{task}_{target_object}_{scene_num}_{min_clutter}_scene.xml')
            shutil.copyfile(scene_xml_file, temp_decomp_scene_xml_file)
            _, _, color, rot=get_random_ycb_object()
            if task=='hard_pushing' or task=="grasping":
                add_object(temp_scene_xml_file, '0', target_object, 0, 0, 1, color, rot, [], run_id, top_dir, target_objects, type=lib_type) #
            elif task=='easy_pushing':
                add_object(temp_scene_xml_file, '0', target_object, -0.05,-0.35, 1, color, rot, [], run_id, top_dir, target_objects, type=lib_type)
            cluttered_scene_xml_file=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/{task}_{target_object}_{scene_num}_{min_clutter}_scene.xml')
            shutil.copyfile(temp_scene_xml_file, cluttered_scene_xml_file)
            added_objects=['0']
            t_added_objects=['0']
            
            #add clutter objects
            num_attempts=0
            num_objects=40
            object_infos=[]
            for object_num in range(num_objects):
                #print(num_attempts)
                num_attempts+=1
                name, size, color, rot=get_random_ycb_object()
                _, _, z_offset=add_object(temp_scene_xml_file, f'gadded_{object_num+1}', name, 50, 50, size, color, rot, [], run_id, top_dir, target_objects, z_pos=-50*(object_num+1))
                t_added_objects.append(f'gadded_{object_num+1}')
                if name in target_objects[16:]:
                    mesh_filename=os.path.join(top_dir, f'herb_reconf/assets/ycb_objects/{name}/google_16k/nontextured.stl')
                    type='ycb'
                else:
                    mesh_filename=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{name}/scene.stl')
                    type='downloaded' 
                mujoco_center, _=compute_mujoco_int_transform(mesh_filename, run_id, size=size)
                mic=mujoco_center[2]
                mesh=trimesh.load(mesh_filename)
                lower_z=-mic
                z_offset=0.3-(mesh.bounds[0,2]*size+(mujoco_center[2]-mesh.centroid[2]*size))#(mujoco_center[2]-mesh.centroid)-mesh.bounds[0,2]*size
                object_infos.append((name, size, color, rot, z_offset))
                
            scene_file, _, _, other_objects=convex_decomp_target_object_env(temp_scene_xml_file, 'gen_body_0', os.path.join(top_dir, f'herb_reconf/cluttered_scenes/'), run_id, top_dir, new_scene_name=temp_decomp_scene_xml_file, add_contacts=False)
            make_global_contacts(scene_file)
            
            added_objs=0
            env = HerbEnv(scene_file, None, run_id, task=task, obs=False)
            real_env = HerbEnv(temp_scene_xml_file, None, run_id, task=task, obs=False)
            env.reset_model(seed=0)
            real_env.reset_model(seed=0)
            visible_pix=get_visible_pixels(real_env, 72)
            unobscured_visible_pix=visible_pix
            past_visible_pix=unobscured_visible_pix
            added_nums=[]
            max_attempts=2000
            #randomly place cluttered objects until desired occlusion level is reached
            for attempt_num in range(max_attempts):
                object_num=attempt_num%num_objects
                name, size, color, rot, z_offset=object_infos[object_num]
                pos=np.random.normal(loc=global_gauss_center, scale=glabal_gauss_std, size=2)
                move_object(env, object_num, np.array([pos[0], pos[1], z_offset]))
                env.sim.physics.forward()
                move_object(real_env, object_num, np.array([pos[0], pos[1], z_offset]))
                real_env.sim.physics.forward()
                contacting_other_objs=0
                for contact_ind in range(len(env.model._data.contact)):
                    if (env.model._data.contact[contact_ind][10]>4 and env.model._data.contact[contact_ind][11]>=72) or (env.model._data.contact[contact_ind][10]>=72 and env.model._data.contact[contact_ind][11]>4):                    
                        a=env.model._data.contact[contact_ind][10]
                        b=env.model._data.contact[contact_ind][11]
                        contacting_other_objs=1
                        break
                if contacting_other_objs:
                    move_object(env, object_num, [50, 50, -50*(object_num+1)])
                    move_object(real_env, object_num, [50, 50, -50*(object_num+1)])
                else:
                    visible_pix=get_visible_pixels(real_env, 72)
                    if visible_pix/unobscured_visible_pix>=min_clutter and visible_pix/unobscured_visible_pix<max_clutter:
                        past_visible_pix=visible_pix
                        added_nums.append(object_num)
                        add_object(cluttered_scene_xml_file, f'gadded_{added_objs+1}', name, pos[0], pos[1], size, color, rot, added_objects, run_id, top_dir, target_objects, z_pos=z_offset)
                        added_objects.append(f'gadded_{added_objs+1}')
                        added_objs+=1
                        break
                    else:
                        move_object(env, object_num, [50, 50, -50*(object_num+1)])
                        move_object(real_env, object_num, [50, 50, -50*(object_num+1)])
            if visible_pix/unobscured_visible_pix>=min_clutter and visible_pix/unobscured_visible_pix<max_clutter:
                break
        
        
        #place unused clutter objects until target number of objects is reached
        glabal_gauss_std=np.array([0.25, 0.25])        
        for object_num in range(num_objects):
            if object_num in added_nums:
                continue
            name, size, color, rot, z_offset=object_infos[object_num]
            pos=np.random.normal(loc=global_gauss_center, scale=glabal_gauss_std, size=2)
            move_object(env, object_num, np.array([pos[0], pos[1], z_offset]))
            env.sim.physics.forward()
            move_object(real_env, object_num, np.array([pos[0], pos[1], z_offset]))
            real_env.sim.physics.forward()
            contacting_other_objs=0
            for contact_ind in range(len(env.model._data.contact)):
                if (env.model._data.contact[contact_ind][10]>4 and env.model._data.contact[contact_ind][11]>=72) or (env.model._data.contact[contact_ind][10]>=72 and env.model._data.contact[contact_ind][11]>4):                    
                    contacting_other_objs=1
                    break
            if contacting_other_objs:
                #remove_object(temp_scene_xml_file, f'gadded_{added_objs+1}')
                move_object(env, object_num, [50, 50, -50*(object_num+1)])
                move_object(real_env, object_num, [50, 50, -50*(object_num+1)])
            else:
                visible_pix=get_visible_pixels(real_env, 72)
                print(visible_pix/unobscured_visible_pix)
                if visible_pix/unobscured_visible_pix>=min_clutter and visible_pix/unobscured_visible_pix<max_clutter:
                    past_visible_pix=visible_pix
                    added_nums.append(object_num)
                    add_object(cluttered_scene_xml_file, f'gadded_{added_objs+1}', name, pos[0], pos[1], size, color, rot, added_objects, run_id, top_dir, target_objects, z_pos=z_offset)
                    added_objects.append(f'gadded_{added_objs+1}')
                    added_objs+=1
                    if added_objs>=num_starting_global_objects:
                        break
                else:
                    move_object(env, object_num, [50, 50, -50*(object_num+1)])
                    move_object(real_env, object_num, [50, 50, -50*(object_num+1)])
        os.remove(temp_scene_xml_file)
        os.remove(scene_file)
        print(past_visible_pix/unobscured_visible_pix)
        if not (past_visible_pix/unobscured_visible_pix>=min_clutter and past_visible_pix/unobscured_visible_pix<max_clutter):
            print(f'failed {task} {target_object} {min_clutter} {max_clutter}')
        data_info_path=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/{task}_{target_object}_{scene_num}_{min_clutter}_scene_info.p')
        with open(data_info_path, 'wb') as save_file:
            pickle.dump((visible_pix/unobscured_visible_pix, added_objs), save_file) 
        print(f'generated {task} {target_object}')
    except:
        print('gen error!')
        traceback.print_exc()
        return False

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        raise  

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--top_dir", dest="top_dir", default='/home/willie/workspace/SSC')
    parser.add_option("--num_cpu", dest="num_cpu", type="int", default=10)
    parser.add_option("--run_num", dest="run_num", type="int", default=0)
    parser.add_option("--total_runs", dest="total_runs", type="int", default=1)
    (options, args) = parser.parse_args()

    run_id=options.run_num
    pool = mp.Pool(processes=options.num_cpu, maxtasksperchild=1)
    for task in tasks:
        for target_object_ind in range(9,17):
            for occluded_frac in range(options.run_num, 11, options.total_runs):
                for scene_num in range(3):
                    abortable_func = partial(abortable_worker, generate_scene, timeout=3600)
                    pool.apply_async(abortable_func, args=(target_object_ind, task, scene_num, options.top_dir, run_id, occluded_frac/10.0, (occluded_frac+1)/10.0))
                    run_id+=options.total_runs
    pool.close()
    pool.join()
        
        
        