import json
import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MUJOCO_GL"]="osmesa"
import numpy as np
import multiprocessing as mp
import math
from trajopt.envs.mujoco_env import MujocoEnv
import trimesh
import shutil
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
from trajopt.mujoco_utils import add_camera, add_objects

parser = OptionParser()
#path to shapenet dataset
parser.add_option("--shapenet_filepath", dest="shapenet_filepath", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/ShapeNetCore.v2/')
#filepath to convex decompositions of shapenet objects. I posted this in the slack channel
parser.add_option("--shapenet_decomp_filepath", dest="shapenet_decomp_filepath", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/shapenet_conv_decmops/')
#root project dir
parser.add_option("--top_dir", dest="top_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/cluttered_datasets/')
#roo project dir+/inhand_datagen
parser.add_option("--instances_dir", dest="instances_dir", default='/home/willie/workspace/SSC/inhand_datagen')
#where to save generated data to
parser.add_option("--save_dir", dest="save_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/cluttered_datasets/')
parser.add_option("--train_or_test", dest="train_or_test", default='training_set')
parser.add_option("--num_scenes", dest="num_scenes", type="int", default=3000)
parser.add_option("--num_threads", dest="num_threads", type="int", default=6)
(options, args) = parser.parse_args()
training_instances_filename = os.path.join(options.instances_dir, 'training_instances.json')
test_instances_filename = os.path.join(options.instances_dir, 'novel_class_test_instances.json')
train_models = json.load(open(training_instances_filename))
test_models = json.load(open(test_instances_filename))

def transform_to_camera_vector(vector, camera_pos, lookat_pos, camera_up_vector):
    view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
    view_matrix = np.array(view_matrix).reshape(4,4, order='F')
    vector=np.concatenate((vector, np.array([1])))
    transformed_vector=view_matrix.dot(vector)
    return transformed_vector[:3]

def move_object(e, ind, pos, rot):
    all_poses=e.data.qpos.ravel().copy()
    all_vels=e.data.qvel.ravel().copy()
    all_poses[7+7*ind+3:7+7*ind+7]=rot
    all_poses[7+7*ind:7+7*ind+3]=pos
    all_vels[6+6*ind:6+6*ind+6]=0
    e.set_state(all_poses, all_vels)

#@profile
def gen_data(scene_num, shapenet_filepath, shapenet_decomp_filepath, instances_dir, top_dir, save_dir, obj_pos_dist_std, train_or_test):
    np.random.seed(scene_num)
    
    min_object_scale=0.5
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
    generated=False
    occlusions=[]
    thread_num=scene_num
    while not generated:
        try:
            #make temp datagen scene
            scene_xml_file=os.path.join(top_dir, f'base_scene.xml')
            temp_scene_xml_file=os.path.join(top_dir, f'temp_data_gen_scene_{thread_num}.xml')
            shutil.copyfile(scene_xml_file, temp_scene_xml_file)
            #choose table and table sclae and add to sim
            table_id=valid_tables[np.random.randint(0, len(valid_tables))]
            table_size=np.random.uniform(low=0.67, high=2, size=1)[0]
            
            table_mesh_filename = shapenet_filepath + f'/04379243/{table_id}/models/model_normalized.obj'
            object_mesh=trimesh.load(table_mesh_filename)
            
            #rotate table
            scale_mat=np.eye(4)
            scale_mat[2,2]=0.0
            scale_mat[2,1]=1.0
            scale_mat[1,2]=1.0
            scale_mat[1,1]=0.0
            object_mesh.apply_transform(scale_mat)
            
            stl_table_mesh_filename=os.path.join(top_dir, f'assets/table_{thread_num}.stl')
            object_mesh.export(stl_table_mesh_filename)
            table_color=np.random.uniform(size=3)
            
            table_height=-object_mesh.bounds[0,2]*table_size
            max_height=object_mesh.bounds[1,2]*table_size+table_height
            drop_xy=np.random.uniform(low=object_mesh.bounds[0,:2]*table_size, high=object_mesh.bounds[1,:2]*table_size, size=2)
            drop_std=np.array([obj_pos_dist_std,obj_pos_dist_std])
            
            add_objects(temp_scene_xml_file, 'table', [stl_table_mesh_filename], [0,0,table_height], table_size, table_color, [1,0,0,0], thread_num, add_contacts=False)
            
            #choose num objects, pos dist center (on table)
            num_objects=random.randint(3,20)
            
            #choose objects, add to scene
            added_objects=0
            obj_mesh_filenames=[]
            obj_initial_positions=[]
            obj_initial_rotations=[]
            obj_colors=[]
            obj_scales=[]
            while added_objects<num_objects:
                selected_index = np.random.randint(0, len(object_ids))
                obj_id = object_ids[selected_index]
                obj_cat=taxonomy_dict[obj_id[0]]
                obj_mesh_filename = shapenet_filepath + f'/{obj_cat}/{obj_id[1]}/models/model_normalized.obj'
                object_mesh=trimesh.load(obj_mesh_filename)
                scale_mat=np.eye(4)
                scale_mat[2,2]=0.0
                scale_mat[2,1]=1.0
                scale_mat[1,2]=1.0
                scale_mat[1,1]=0.0
                object_mesh.apply_transform(scale_mat)
                stl_obj_mesh_filename=os.path.join(top_dir, f'assets/model_normalized_{thread_num}_{added_objects}.stl')
                object_mesh.export(stl_obj_mesh_filename)
                object_color=np.random.uniform(size=3)
                object_size=np.random.uniform(low=min_object_scale, high=max_object_scale, size=1)[0]
                diag=np.sqrt(np.sum(np.square(object_mesh.bounds[0]-object_mesh.bounds[1])))
                object_size*=0.05
                if diag<0.001:   
                    object_size*=1000
                object_rot=np.zeros(4)
                object_rot[1:4]=np.random.uniform(low=0, high=1, size=3)
                object_rot[0]=np.random.uniform(low=0, high=2*math.pi, size=1)
                if random.random()<0.8:
                    object_rot[0]=1
                    object_rot[1:4]=0
                object_drop_pos=np.random.normal(loc=drop_xy, scale=drop_std)
                object_mesh=trimesh.load(stl_obj_mesh_filename)
                if object_mesh.faces.shape[0]>200000:
                    print('too many mesh faces!')
                    continue
                obj_mesh_filenames+=[obj_mesh_filename]
                obj_initial_positions.append(object_drop_pos)
                obj_initial_rotations.append(object_rot)
                obj_colors.append(object_color)
                obj_scales.append(object_size)
                
                comb_mesh=None
                decomp_shapenet_decomp_filepath=os.path.join(shapenet_decomp_filepath, f'{obj_cat}/{obj_id[1]}')
                for mesh_file in os.listdir(decomp_shapenet_decomp_filepath):
                    decomp_object_mesh=trimesh.load(os.path.join(decomp_shapenet_decomp_filepath, mesh_file))
                    if comb_mesh==None:
                        comb_mesh=decomp_object_mesh
                    else:
                        comb_mesh+=decomp_object_mesh
                comb_mesh.apply_transform(scale_mat)
                trimesh.repair.fix_inversion(comb_mesh)
                meshes=comb_mesh.split()
                #load conv decomp meshes
                mesh_names=[]
                mesh_masses=[]
                
                combined_mesh=None
                mesh_file_ind=0
                for mesh_file in os.listdir(decomp_shapenet_decomp_filepath):
                    decomp_object_mesh=meshes[mesh_file_ind]
                    if decomp_object_mesh.faces.shape[0]>10 and decomp_object_mesh.mass>10e-7:
                        obj_mesh_filename=os.path.join(decomp_shapenet_decomp_filepath, mesh_file[:-3]+'stl')
                        
                        decomp_object_mesh.export(obj_mesh_filename)
                        mesh_names.append(obj_mesh_filename)
                        mesh_masses.append(decomp_object_mesh.mass)
                        if combined_mesh==None:
                            combined_mesh=decomp_object_mesh
                        else:
                            combined_mesh+=decomp_object_mesh
                    mesh_file_ind+=1
                    if mesh_file_ind>=len(meshes):
                        break
                    
                if len(mesh_names)>100:
                    heavy_inds=np.argsort(np.array(mesh_masses))
                    new_mesh_names=[]
                    for ind in range(100):
                        new_mesh_names.append(mesh_names[heavy_inds[-ind]])
                    mesh_names=new_mesh_names
                add_objects(temp_scene_xml_file, f'object_{added_objects}_{thread_num}', mesh_names, [50,50,added_objects], object_size, object_color, object_rot, thread_num, add_contacts=False)
                added_objects+=1
            #drop one by one onto table
            e=MujocoEnv(temp_scene_xml_file, 1, has_robot=False)
            u=0
            for added_object_ind in range(num_objects):
                obj_drop_pos=obj_initial_positions[added_object_ind]
                obj_rot=obj_initial_rotations[added_object_ind]
                max_drop_height=max_height+0.2
  
                move_object(e, added_object_ind, [obj_drop_pos[0], obj_drop_pos[1], max_drop_height], obj_rot)
                for step in range(2000):
                    e.model.step()
            
            #take cam views, compute occlusion
            
            #replace conv decomps with real meshes
            cam_temp_scene_xml_file=os.path.join(top_dir, f'cam_temp_data_gen_scene_{thread_num}.xml')
            shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)
            add_objects(cam_temp_scene_xml_file, 'table', [stl_table_mesh_filename], [0,0,table_height], table_size, table_color, [1,0,0,0], thread_num, add_contacts=False)
            for added_object_ind in range(num_objects):
                add_objects(cam_temp_scene_xml_file, f'object_{added_object_ind}_{thread_num}', [os.path.join(top_dir, f'assets/model_normalized_{thread_num}_{added_object_ind}.stl')], [50,50,added_objects], obj_scales[added_object_ind], obj_colors[added_object_ind], object_rot, thread_num, add_contacts=False)
            
            coms=[]
            scene_description={}
            scene_description['table']={'mesh_filename':table_mesh_filename, 'position': [0,0,table_height], 'orientation': [1,0,0,0], 'scale':table_size}
            scene_description['object_descriptions']=[]
            scene_description['views']={"background+table+objects": []}
            off_table_inds=[]
            all_meshses=None
            for added_object_ind in range(num_objects):
                object_description={}
                object_description['mesh_filename']=obj_mesh_filenames[added_object_ind]
                object_description['position']=e.data.qpos.ravel()[7+7*added_object_ind:7+7*added_object_ind+3].copy()
                object_description['orientation']=e.data.qpos.ravel()[7+7*added_object_ind+3:7+7*added_object_ind+7].copy()
                mesh=trimesh.load(obj_mesh_filenames[added_object_ind])
                scale_mat=np.eye(4)
                scale_mat=scale_mat*obj_scales[added_object_ind]
                scale_mat[3,3]=1.0
                mesh.apply_transform(scale_mat)
                
                transform=np.eye(4)
                transform[0:3,0:3]=Quaternion(object_description['orientation']).rotation_matrix
                mesh.apply_transform(transform)
                
                transform=np.eye(4)
                transform[0:3,3]=object_description['position']
                mesh.apply_transform(transform)
                
                if mesh.bounds[0,2]<max_height*0.9:
                    off_table_inds.append(added_object_ind)
                object_description['cog']=mesh.centroid
                coms.append(mesh.centroid)
                object_description['scale']=obj_scales[added_object_ind]
                        
                scene_description['object_descriptions'].append(object_description)
            
            if os.path.exists(save_dir+f'/{train_or_test}/scene_{scene_num:06}'):
                shutil.rmtree(save_dir+f'/{train_or_test}/scene_{scene_num:06}')
            os.mkdir(save_dir+f'/{train_or_test}/scene_{scene_num:06}')
            
            #add 20 different views (cameras)
            views_tried=0
            view_dir_num=0
            
            camera_xyzs=[]
            camera_lookats=[]
            for cam_num in range(20):
                camera_r=random.uniform(0.125, 1.5)
                camera_porps=np.random.uniform(size=3)
                camera_porps=camera_porps/np.sum(camera_porps)**0.5
                for ind in range(2):
                    if random.random()<0.5:
                        camera_porps[ind]=-camera_porps[ind]
                camera_xyz=camera_porps*camera_r**0.5
                camera_xyz[2]+=max_height
                
                camera_r=random.uniform(0, min(camera_r/2, 0.25))
                camera_porps=np.random.uniform(size=3)
                camera_porps=camera_porps/np.sum(camera_porps)**0.5
                for ind in range(len(camera_porps)):
                    if random.random()<0.5:
                        camera_porps[ind]=-camera_porps[ind]
                camera_lookat=camera_porps*camera_r**0.5+np.array([drop_xy[0], drop_xy[1], max_height])

                camera_direction = camera_lookat - camera_xyz
                camera_distance = np.linalg.norm(camera_direction)
                camera_direction = camera_direction / camera_distance
                
                add_camera(cam_temp_scene_xml_file, f'gen_cam_{cam_num}', camera_xyz, camera_lookat, cam_num)
                camera_xyzs.append(camera_xyz)
                camera_lookats.append(camera_lookat)
            
            state=e.get_env_state().copy()   
            e=MujocoEnv(cam_temp_scene_xml_file, 1, has_robot=False)
            e.set_env_state(state)
            e.sim.physics.forward()
            #render views, use views with at least 1000 object pixels to generate data
            for cam_num in range(20):
                
                scene_description['views']["background+table+objects"].append({'camera_pos': camera_xyzs[cam_num], 'lookat_pos': camera_lookats[cam_num], 'camera_up_vector': [0,1,0], 'cam_x_mat': e.model.data.cam_xmat[1+cam_num]})
                depth=e.model.render(height=480, width=640, camera_id=1+cam_num, depth=True, segmentation=False)
                depth=(depth*1000).astype(np.uint16)
                cv2.imwrite(save_dir+f'/{train_or_test}/scene_{scene_num:06}/depth_{(cam_num+2):05}.png', depth)
                rgb=e.model.render(height=480, width=640, camera_id=1+cam_num, depth=False, segmentation=False)
                cv2.imwrite(save_dir+f'/{train_or_test}/scene_{scene_num:06}/rgb_{(cam_num+2):05}.jpeg', rgb)
                
                camera = Camera(physics=e.model, height=480, width=640, camera_id=1+cam_num)
                segs=camera.render(segmentation=True)[:,:,0]

                occluded_geom_id_to_seg_id={camera.scene.geoms[geom_ind][3]: camera.scene.geoms[geom_ind][8] for geom_ind in range(camera.scene.geoms.shape[0])}

                cv2.imwrite(save_dir+f'/{train_or_test}/scene_{scene_num:06}/segmentation_{(cam_num+2):05}.png', segs)
                present_in_view_ind=0
                for added_object_ind in range(num_objects):
                    if added_object_ind in off_table_inds: #don't make examples from objects that have fallen off the table
                        continue
                    target_id=e.model.model.name2id(f'gen_geom_object_{added_object_ind}_{thread_num}_0', "geom")
                    segmentation=segs==occluded_geom_id_to_seg_id[target_id]
                    target_obj_pix=np.argwhere(segmentation).shape[0]
                    if target_obj_pix>50:
                        for move_obj_ind in range(num_objects):
                            if move_obj_ind!=added_object_ind:
                                move_object(e, move_obj_ind, [50, 50, move_obj_ind], [1,0,0,0])
                        e.sim.physics.forward()
                        unocc_target_id=e.model.model.name2id(f'gen_geom_object_{added_object_ind}_{thread_num}_0', "geom")
                        unoccluded_camera = Camera(physics=e.model, height=480, width=640, camera_id=1+cam_num)
                        unoccluded_segs=unoccluded_camera.render(segmentation=True)
                        e.set_env_state(state)
                        e.sim.physics.forward()
                        unoccluded_geom_id_to_seg_id={unoccluded_camera.scene.geoms[geom_ind][3]: unoccluded_camera.scene.geoms[geom_ind][8] for geom_ind in range(unoccluded_camera.scene.geoms.shape[0])}

                        unoccluded_segs=np.concatenate((unoccluded_segs[:,:,0:1],unoccluded_segs[:,:,0:1],unoccluded_segs[:,:,0:1]), axis=2).astype(np.uint8)

                        unoccluded_segmentation=unoccluded_segs[:,:,0]==unoccluded_geom_id_to_seg_id[unocc_target_id]
                        num_unoccluded_pix=np.argwhere(unoccluded_segmentation).shape[0]
                        segmentation=np.logical_and(segmentation, unoccluded_segmentation)
                        
                        if np.argwhere(segmentation).shape[0]>0:
                            cv2.imwrite(save_dir+f'/{train_or_test}/scene_{scene_num:06}/segmentation_{(cam_num+2):05}_{present_in_view_ind}.png', segmentation.astype(np.uint8))
                            
                            rgb=e.model.render(height=480, width=640, camera_id=1+cam_num, depth=False, segmentation=False)

                            where_seen=np.where(segmentation, 255, 0).astype(np.uint8)
                            where_seen=np.concatenate((where_seen[:,:,None],where_seen[:,:,None],where_seen[:,:,None]), axis=2).astype(np.uint8)
                            unocc_where_seen=np.where(unoccluded_segmentation, 255, 0).astype(np.uint8)
                            unocc_where_seen=np.concatenate((unocc_where_seen[:,:,None],unocc_where_seen[:,:,None],unocc_where_seen[:,:,None]), axis=2).astype(np.uint8)
                            occlusion=1.0-np.argwhere(segmentation).shape[0]/num_unoccluded_pix
                            translation_vector=coms[added_object_ind]
                            camera_translation_vector=transform_to_camera_vector(translation_vector, camera_xyzs[cam_num], camera_lookats[cam_num], np.array([0,1,0]))
                            
                            object_pose_info={}
                            object_pose_info['model']=obj_mesh_filenames[added_object_ind]
                            object_pose_info['cam_R_m2c']=np.ndarray.tolist(np.zeros(9, dtype=float))
                            object_pose_info['cam_t_m2c']=np.ndarray.tolist((1000.0*camera_translation_vector).astype(float))
                            upper_left=[int(np.min(np.argwhere(segmentation)[:, 0])), int(np.min(np.argwhere(segmentation)[:, 1]))]
                            lower_right=[np.max(np.argwhere(segmentation)[:, 0]), np.max(np.argwhere(segmentation)[:, 1])]
                            object_pose_info['obj_bb']=upper_left+[int(lower_right[0]-upper_left[0]), int(lower_right[1]-upper_left[1])]
                            object_pose_info['occlusion']=occlusion
                            
                            with open(save_dir+f'/{train_or_test}/scene_{scene_num:06}/pose_info_{(cam_num+2):05}_{present_in_view_ind}.p', 'wb') as save_file:
                                pickle.dump(object_pose_info, save_file)    
                            
                            generated=True
                            #num_generated+=1
                            view_dir_num+=1
                            present_in_view_ind+=1
                            if scene_num%100==0:
                                print('num_generated:', scene_num)
            
            with open(save_dir+f'/{train_or_test}/scene_{scene_num:06}/scene_description.p', 'wb') as save_file:
                pickle.dump(scene_description, save_file)    
            
            views_tried+=1
        except:
            print('gen error!')
            traceback.print_exc()
    
    with open(save_dir+f'/occlusions.p', 'wb') as save_file:
        pickle.dump(occlusions, save_file)    

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
    cats=""
    for key in train_models:
        cats+=key+", "
    print(cats)
    cats=""
    print()
    for key in test_models:
        cats+=key+", "
    print(cats)
    
    u=0
#     num_processes=options.num_threads
#     pool = mp.Pool(processes=num_processes, maxtasksperchild=1)
#     for scene_num in range(options.num_scenes):
#         abortable_func = partial(abortable_worker, gen_data, timeout=600)
#         pool.apply_async(abortable_func, args=(scene_num, options.shapenet_filepath, options.shapenet_decomp_filepath, options.instances_dir, options.top_dir, options.save_dir, 0.1, options.train_or_test))
#     pool.close()
#     pool.join()
        
        