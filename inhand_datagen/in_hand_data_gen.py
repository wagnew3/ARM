import json
import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MUJOCO_GL"]="osmesa"
import numpy as np
import multiprocessing as mp
import math
from trajopt.envs.herb_pushing_env import HerbEnv
import trimesh
from pose_model_estimator import get_mesh_list, compute_mujoco_int_transform, make_known_meshes
import shutil
from xml.dom import minidom
from trajopt.utils import generate_perturbed_actions
import random
import cv2
from pyquaternion import Quaternion
import pickle
from optparse import OptionParser
import traceback
from trajopt.mujoco_utils import set_gravity, add_objects, add_camera

#@profile
def gen_data(thread_num, num_threads, shapenet_filepath, shapenet_decomp_filepath, instances_dir, top_dir, save_dir):
    target_obj_geom_id=72
    train_or_test='train'
    num_images=10000000
    
    min_object_scale=0.5
    max_object_scale=4.0
    
    print(f'generating {train_or_test} dataset')
    
    training_instances_filename = os.path.join(instances_dir, 'training_instances.json')
    test_instances_filename = os.path.join(instances_dir, 'novel_class_test_instances.json')
    
    train_models = json.load(open(training_instances_filename))
    test_models = json.load(open(test_instances_filename))
    
    object_ids = train_models if train_or_test == 'train' else test_models
    
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
    
    mj_scene_xml=os.path.join(top_dir, f'herb_reconf/easy_pushing_scene.xml')
    e = HerbEnv(mj_scene_xml, np.zeros((0,3)), task='data_gen', obs=False, state_arm_pos=np.zeros(1))
    model=e.model
    
    mesh_list, mesh_name_to_file, name_to_scale_dict=get_mesh_list(mj_scene_xml)[:70]
    mesh_list=mesh_list[:69]
    included_meshes=[]
    geom_names=[]
    pred_obj_meshes=[]
    
    sigma = 1.0*np.ones(e.action_dim)
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
    
    #for mesh_ind in range(model.model.geom_dataid.shape[0]):
    palm_mesh_off_trans=None
    palm_mesh_off_rot=None
    for geom_name in model.named.data.geom_xpos.axes.row.names:
        num_voxels=256
        geom_id=model.model.name2id(geom_name, "geom")
        if geom_id<71:#geom_id!=71 and geom_id!=72:
            if model.model.geom_dataid[geom_id]>-1:
                mesh=trimesh.load_mesh(mesh_name_to_file[model.model.id2name(model.model.geom_dataid[geom_id], "mesh")])
                mesh_off_trans, mesh_off_rot=compute_mujoco_int_transform(mesh_name_to_file[model.model.id2name(model.model.geom_dataid[geom_id], "mesh")], thread_num)
                
                if model.model.id2name(model.model.geom_dataid[geom_id], "mesh")=="herb/wam_1/bhand/bhand_palm_fine":
                    palm_mesh_off_trans=np.copy(mesh_off_trans)
                    palm_mesh_off_rot=np.copy(mesh_off_rot)
                    
                
                trans_mat=np.eye(4)
                trans_mat[0:3, 3]=-mesh_off_trans
                mesh.apply_transform(trans_mat)
                
                trans_mat=np.eye(4)
                trans_mat[0:3, 0:3]=mesh_off_rot.transpose()
                mesh.apply_transform(trans_mat)
                included_meshes.append(mesh)
                geom_names.append(geom_name)
            elif geom_id<len(mesh_list) and mesh_list[geom_id] is not None:
                mesh=mesh_list[geom_id]
                included_meshes.append(mesh)
                geom_names.append(geom_name)
    
    num_generated=thread_num
    view_num=thread_num
    while num_generated<num_images/num_threads:
        try:
            #choose shapenet object and rotation, color, and size
            selected_index = np.random.randint(0, len(object_ids))
            obj_id = object_ids[selected_index]
            obj_cat=taxonomy_dict[obj_id[0]]
            obj_mesh_filename = shapenet_filepath + f'/{obj_cat}/{obj_id[1]}/models/model_normalized.obj'
            object_mesh=trimesh.load(obj_mesh_filename)
            obj_mesh_filename=os.path.join(top_dir, f'herb_reconf/assets/model_normalized_{thread_num}.stl')
            object_mesh.export(obj_mesh_filename)
            
            object_color=np.random.uniform(size=3)
            object_size=np.random.uniform(low=min_object_scale, high=max_object_scale, size=1)[0]
            object_mesh=trimesh.load(obj_mesh_filename)
            if object_mesh.faces.shape[0]>200000:
                print('too many mesh faces!')
                continue
            
            diag=np.sqrt(np.sum(np.square(object_mesh.bounds[0]-object_mesh.bounds[1])))
            object_size*=0.1/diag
            
            object_rot=np.random.uniform(low=0, high=2*math.pi, size=3)
        
            #choose herb arm/wrist pose
            arm_pos=np.random.uniform(low=e.action_space.low[0:7], high=e.action_space.high[0:7])
            #arm_pos[6]=0
            e = HerbEnv(os.path.join(top_dir, f'herb_reconf/easy_pushing_scene.xml'), np.zeros((0,3)), task='data_gen', obs=False, state_arm_pos=arm_pos)
            
            state=e.get_env_state().copy()
            state['qp'][:7]=arm_pos
            e.set_env_state(state)
            robot_meshes=make_known_meshes(included_meshes, e.model, geom_names)
            
            mesh=trimesh.load(os.path.join(top_dir, 'herb_reconf/assets/bhand_palm_fine-dff1351ac0d212f9757a6bc04b85287b4f13403a.stl'))
            transform=np.eye(4)
            transform[0:3,0:3]=np.reshape(e.model.named.data.geom_xmat["herb/wam_1/bhand//unnamed_geom_0"],(3,3))
            mesh.apply_transform(transform)
            
            transform=np.eye(4)
            transform[0:3,3]=e.model.named.data.geom_xpos["herb/wam_1/bhand//unnamed_geom_0"]
            mesh.apply_transform(transform)
            
            #sample shapenet object positions until no conflict is found
            position_cube=trimesh.primitives.Box(extents=np.array([0.05,0.05,0.05]))
            
            transform=np.eye(4)
            transform[2,3]=0.125
            position_cube.apply_transform(transform)
            
            trans_mat=np.eye(4)
            trans_mat[0:3, 3]=-palm_mesh_off_trans
            position_cube.apply_transform(trans_mat)
            
            trans_mat=np.eye(4)
            trans_mat[0:3, 0:3]=palm_mesh_off_rot.transpose()
            position_cube.apply_transform(trans_mat)
            
            transform=np.eye(4)
            transform[0:3,0:3]=np.reshape(e.model.named.data.geom_xmat["herb/wam_1/bhand//unnamed_geom_0"],(3,3))
            position_cube.apply_transform(transform)
            
            transform=np.eye(4)
            transform[0:3,3]=e.model.named.data.geom_xpos["herb/wam_1/bhand//unnamed_geom_0"]
            position_cube.apply_transform(transform)
            
            contacting=True
            num_attempts=0
            
            while contacting and num_attempts<20:
                object_mesh=trimesh.load(obj_mesh_filename)
                object_pos=np.random.uniform(low=position_cube.bounds[0], high=position_cube.bounds[1])
                
                scale_mat=np.eye(4)
                scale_mat=scale_mat*object_size
                scale_mat[3,3]=1.0
                object_mesh.apply_transform(scale_mat)
                
                transform=np.eye(4)
                transform[0:3,0:3]=Quaternion(axis=object_rot, radians=math.pi/2).rotation_matrix
                object_mesh.apply_transform(transform)
                
                transform=np.eye(4)
                transform[0:3,3]=object_pos
                object_mesh.apply_transform(transform)
                
                combined_mesh=object_mesh+position_cube
                for known_mesh in robot_meshes:
                    contacting=np.any(known_mesh.ray.contains_points(object_mesh.vertices).astype(int))
                    if contacting:
                        break
                num_attempts+=1
            
            if not contacting:   
                #close hand around object with zero graviity for 30 timesteps
                scene_xml_file=os.path.join(top_dir, f'herb_reconf/data_gen_scene.xml')
                temp_scene_xml_file=os.path.join(top_dir, f'herb_reconf/temp_data_gen_scene_{thread_num}.xml')
                shutil.copyfile(scene_xml_file, temp_scene_xml_file)
                set_gravity(temp_scene_xml_file, False)
                
                mesh_names=[]
                decomp_shapenet_decomp_filepath=os.path.join(shapenet_decomp_filepath, f'{obj_cat}/{obj_id[1]}')
                for mesh_file in os.listdir(decomp_shapenet_decomp_filepath):
                    decomp_object_mesh=trimesh.load(os.path.join(decomp_shapenet_decomp_filepath, mesh_file))
                    trimesh.repair.fix_inversion(decomp_object_mesh)
                    if decomp_object_mesh.faces.shape[0]>10 and decomp_object_mesh.mass>10e-7:
                        obj_mesh_filename=os.path.join(decomp_shapenet_decomp_filepath, mesh_file[:-3]+'stl')
                        decomp_object_mesh.export(obj_mesh_filename)
                        mesh_names.append(obj_mesh_filename)
                
                if len(mesh_names)==0:
                    print('no large meshes')
                    continue
                add_objects(temp_scene_xml_file, "target", mesh_names, object_pos, object_size, object_color, object_rot, 0)
                
                e=HerbEnv(temp_scene_xml_file, np.zeros((0,3)), task='data_gen', obs=False, state_arm_pos=arm_pos)
                state=e.get_env_state().copy()
                state['qp'][:7]=arm_pos
                e.set_env_state(state)
                base_act=np.repeat(np.expand_dims(state['qp'][:15], axis=0), 30, axis=0)
                act, vel=generate_perturbed_actions(state, base_act, filter_coefs, 0.5, 0.15, state['qp'][4], 1.59, hand_open=0, move=False)
                step_error=False
                for step in range(30):
                    try:
                        e.step(act[step])
                    except:
                        print('step error!')
                        step_error=True
                if step_error:
                    continue
                
                #run gravity for 10 timesteps
                state=e.get_env_state().copy()
                set_gravity(temp_scene_xml_file, True)
                e = HerbEnv(temp_scene_xml_file, np.zeros((0,3)), task='data_gen', obs=False, state_arm_pos=arm_pos)
                    
                e.set_env_state(state)
                base_act=np.repeat(np.expand_dims(state['qp'][:15], axis=0), 30, axis=0)
                act, vel=generate_perturbed_actions(state, base_act, filter_coefs, 0.5, 0.15, state['qp'][4], 1.59, hand_open=0, move=False)
                for step in range(10):
                    try:
                        e.step(act[step])
                    except:
                        print('step error!')
                        step_error=True
                if step_error:
                    continue
                
                state=e.get_env_state().copy()
                #if object is still in contact, choose n camera positions in shell around hand center, camera lookat angles in sphere around hand center, camera rotation angles the same as ssc sim data
                #accept if at least part of object is visible
                contacting_table=False
                for contact_ind in range(len(e.model._data.contact)):
                    if (e.model._data.contact[contact_ind][10]!=3 and e.model._data.contact[contact_ind][11]==72) or (e.model._data.contact[contact_ind][10]==72 and e.model._data.contact[contact_ind]!=3):                    
                        contacting_table=True
                
                if contacting_table:
                    scene_description={'state': state, 'mesh': {'file': obj_mesh_filename, 'scale': object_size, 'rotation': object_rot, 'color': object_color, 'position': object_pos}, 'views':[]}
                    if not os.path.exists(save_dir+f'/{train_or_test}/view_{view_num:06}'):
                        os.mkdir(save_dir+f'/{train_or_test}/view_{view_num:06}')
                    
                    #add 20 different views (cameras)
                    views_tried=0
                    view_dir_num=0
                    can_see_obj=False
                    cam_temp_scene_xml_file=os.path.join(top_dir, f'herb_reconf/cam_temp_easy_pushing_scene_{thread_num}.xml')
                    shutil.copyfile(scene_xml_file, cam_temp_scene_xml_file)
                    add_objects(cam_temp_scene_xml_file, "target", [obj_mesh_filename], object_pos, object_size, object_color, object_rot, 0)
                    
                    camera_xyzs=[]
                    camera_lookats=[]
                    for cam_num in range(20):
                        camera_r=random.uniform(0.25, 3.0)
                        camera_porps=np.random.uniform(size=3)
                        camera_porps=camera_porps/np.sum(camera_porps)**0.5
                        for ind in range(len(camera_porps)):
                            if random.random()<0.5:
                                camera_porps[ind]=-camera_porps[ind]
                        camera_xyz=camera_porps*camera_r**0.5+object_mesh.centroid
                        
                        camera_r=random.uniform(0, 0.5)
                        camera_porps=np.random.uniform(size=3)
                        camera_porps=camera_porps/np.sum(camera_porps)**0.5
                        for ind in range(len(camera_porps)):
                            if random.random()<0.5:
                                camera_porps[ind]=-camera_porps[ind]
                        camera_lookat=camera_porps*camera_r**0.5+object_mesh.centroid
                        
                        camera_direction = camera_lookat - camera_xyz
                        camera_distance = np.linalg.norm(camera_direction)
                        camera_direction = camera_direction / camera_distance                        
                        
                        add_camera(cam_temp_scene_xml_file, f'gen_cam_{cam_num}', camera_xyz, camera_lookat, cam_num)
                        camera_xyzs.append(camera_xyz)
                        camera_lookats.append(camera_lookat)
                        
                    e = HerbEnv(cam_temp_scene_xml_file, np.zeros((0,3)), task='data_gen', obs=False, state_arm_pos=arm_pos)
                    e.set_env_state(state)
                    
                    #render views, use views with at least 1000 object pixels to generate data
                    for cam_num in range(20):
                        segs=e.model.render(height=480, width=640, camera_id=2+cam_num, depth=False, segmentation=True)
                        target_obj_pix=np.sum(segs==target_obj_geom_id)

                        if target_obj_pix>1000:
                            #save rbg, depth, segmentation, camera pose/lookat, mesh pose and size
                            segmentation=(segs[:,:,0]==target_obj_geom_id)
                            cv2.imwrite(save_dir+f'/{train_or_test}/view_{view_num:06}/{view_dir_num:02}_segmentation.png', segmentation.astype(np.uint8))
                            depth=e.model.render(height=480, width=640, camera_id=2+cam_num, depth=True, segmentation=False)
                            cv2.imwrite(save_dir+f'/{train_or_test}/view_{view_num:06}/{view_dir_num:02}_depth.png', depth)
                            rgb=e.model.render(height=480, width=640, camera_id=2+cam_num, depth=False, segmentation=False)
                            cv2.imwrite(save_dir+f'/{train_or_test}/view_{view_num:06}/{view_dir_num:02}_rgb.jpeg', rgb)
                            num_generated+=1
                            view_dir_num+=1
                            
                            scene_description['views'].append({'cam_xyz': camera_xyzs[cam_num], 'cam_lookat': camera_lookats[cam_num]})
                            
                            if num_generated%10==0:
                                print('num_generated:', num_generated)
                    
                    with open(save_dir+f'/{train_or_test}/view_{view_num:06}/scene_description.json', 'wb') as save_file:
                        pickle.dump(scene_description, save_file)    
                    
                    views_tried+=1
                    view_num+=num_threads
        except:
            print('gen error!')
            traceback.print_exc()
            
if __name__ == '__main__':
    parser = OptionParser()
    #path to shapenet dataset
    parser.add_option("--shapenet_filepath", dest="shapenet_filepath", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/ShapeNetCore.v2/')
    #filepath to convex decompositions of shapenet objects. I posted this in the slack channel
    parser.add_option("--shapenet_decomp_filepath", dest="shapenet_decomp_filepath", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/shapenet_conv_decmops/')
    #root project dir
    parser.add_option("--top_dir", dest="top_dir", default='/home/willie/workspace/SSC')
    #roo project dir+/inhand_datagen
    parser.add_option("--instances_dir", dest="instances_dir", default='/home/willie/workspace/SSC/inhand_datagen')
    #where to save generated data to
    parser.add_option("--save_dir", dest="save_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/in_hand_dataset')
    parser.add_option("--num_threads", dest="num_threads", type="int", default=12)
    (options, args) = parser.parse_args()
    
    gen_data(0, 1, options.shapenet_filepath, options.shapenet_decomp_filepath, options.instances_dir, options.top_dir, options.save_dir)
    
    num_processes=options.num_threads
    pool = mp.Pool(processes=num_processes, maxtasksperchild=1)
    parallel_runs = [pool.apply_async(gen_data, args=(i, num_processes, options.shapenet_filepath, options.shapenet_decomp_filepath, options.instances_dir, options.top_dir, options.save_dir)) for i in range(num_processes)]   
    results = [p.get() for p in parallel_runs]
        
        
        
        