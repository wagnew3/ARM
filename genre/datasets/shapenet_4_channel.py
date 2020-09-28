from os.path import join
import random
import numpy as np
import torch.utils.data as data
import trimesh
import os
from PIL import Image
from pyquaternion import Quaternion
import pybullet as p
import open3d as o3d
#import pcl
import time
#import pymesh
from genre.voxelization import voxel
import pickle
import traceback
from trimesh.util import append_faces
from pose_model_estimator import select_points_in_cube_voxelize_sphr_proj, make_pointcloud_all_points

class Dataset(data.Dataset):

    @classmethod
    def add_arguments(cls, parser):
        return parser, set()

    @classmethod
    def read_bool_status(cls, status_file):
        with open(join(cls.list_root, status_file)) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')[:-1]]

    def __init__(self, opt, loaded_meshes, mode='train', model=None):
        assert mode in ('train', 'vali', 'test')
        self.mode = mode
        self.model=model
        self.baseline_remove_seen=opt.baseline_remove_seen
        if model is None:
            required = ['rgb']
            self.preproc = None
        else:
            required = model.requires
            self.preproc = model.preprocess
            
        self.loaded_meshes=loaded_meshes
        self.cached_samples={}
        self.shapenet_root=opt.shapenet_root
        self.exp_id=opt.expr_id

        ind=0
        samples=[]
        input_root=opt.dataset_root
        self.four_channel=opt.net=="genre_given_depth_4_channel" or opt.baseline_remove_seen
        
        if mode=='train':
            input_root+='/training_set'
        elif mode=='vali':
            input_root+='/validation_set'
        elif mode=='test':
            input_root+='/test_set'
        
        num_loaded=0
        for dir_2 in os.listdir(input_root):
            ind+=1

            if os.path.isfile(input_root+'/'+dir_2+'/'+'scene_description.p'):
                try:
                    scene_description=pickle.load(open(input_root+'/'+dir_2+'/'+'scene_description.p', 'rb'))
                except:
                    print('shapenet_4_channel dataloader failed to load scene', dir_2)
                    continue

                pose_infos={}
                successfully_loaded_object_pose_info=True
                for file_2 in os.listdir(input_root+'/'+dir_2):
                    if file_2.startswith('pose_info'):
                        view_num=file_2.split('_')[2]
                        if view_num not in pose_infos:
                            pose_infos[view_num]={}
                        try:
                            object_pose_info=pickle.load(open(input_root+'/'+dir_2+'/'+file_2, 'rb'))
                        except:
                            print('shapenet_4_channel dataloader failed to object pose info', dir_2, file_2)
                            successfully_loaded_object_pose_info=False
                            break
                        pose_infos[view_num][object_pose_info["model"]]=object_pose_info
                
                if not successfully_loaded_object_pose_info:
                    continue
                
                for file_2 in os.listdir(input_root+'/'+dir_2):
                    if file_2.startswith('pose_info'):
                        view_num=file_2.split('_')[2]
                        sample={}
                        object_pose_info=pickle.load(open(input_root+'/'+dir_2+'/'+file_2, 'rb'))
                        sample['cam_dist']=[]
                        sample['table']=scene_description['table']
                        
                        sample['obj_pos']=[]
                        sample['obj_cog']=[]
                        sample['obj_rot']=[]
                        sample['obj_scale']=[]
                        sample['mesh_filename']=[]
                        added_obj_ind=0
                        for obj_dict_ind in range(len(scene_description["object_descriptions"])):
                            obj_dict=scene_description["object_descriptions"][obj_dict_ind]
                            if obj_dict["mesh_filename"] in pose_infos[view_num]:
                                sample['obj_pos'].append(np.array(obj_dict["position"]))
                                sample['obj_cog'].append(np.array(obj_dict["cog"]))
                                sample['obj_rot'].append(obj_dict["orientation"])
                                sample['obj_scale'].append(obj_dict["scale"])
                                sample['mesh_filename'].append(obj_dict["mesh_filename"][obj_dict["mesh_filename"].index('ShapeNetCore.v2'):])
                                sample['cam_dist'].append(np.array(pose_infos[view_num][obj_dict["mesh_filename"]]["cam_t_m2c"])/1000.0)
                                if obj_dict["mesh_filename"]==object_pose_info["model"]:
                                    sample['target_ind']=added_obj_ind
                                    sample['occlusion']=object_pose_info['occlusion']
                                added_obj_ind+=1
                            
                        if 'target_ind' not in sample:
                            continue
                        sample['cam_pos']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["camera_pos"]
                        sample['cam_x_mat']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["cam_x_mat"]
                        sample['lookat_pos']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["lookat_pos"]
                        sample['up_vector']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["camera_up_vector"]
                        sample['depth_file']=input_root+'/'+dir_2+'/'+'depth_'+file_2[10:15]+'.png'
                        sample['mask_file']=input_root+'/'+dir_2+'/'+'segmentation_'+file_2[10:-2]+'.png'                        
                        samples+=[sample]
            if len(samples)>300000:
                break
            if ind%100==0:
                print(ind)

        # If validation, dataloader shuffle will be off, so need to DETERMINISTICALLY
        # shuffle here to have a bit of every class
        if opt.manual_seed:
            seed = opt.manual_seed
        else:
            seed = 0
        random.Random(seed).shuffle(samples)
        self.samples = samples
    
    def make_voxels(self, pointcloud):
        pass
    
    #@profile
    #get an object mesh transformed into camera space
    def get_object_mesh_gt_voxels(self, mesh_name, obj_rot, cam_pos, obj_pos, lookat_pos, up_vector, obj_cog, cam_dist, obj_scale, i, b_pos, scale, cam_mat, table=False):
        sample_loaded = {}
        # sample structure:
        #    'cam_pos': camera position
        #    'depth_file': depth file
        #    'mask_file': mask file
        #    'obj_pos': object position
        #    'obj_rot': object world rotation
        #    'obj model': object model file
        # load steps:
        # 1. load all files
        # 2. compute object rotation relative to camera
        # 3. compute pointcloud from object mesh
        # 4. rotate pointcloud
        # 5. compute spherical projection of rotated pointcloud
        

        w_r_mat=Quaternion(np.array(obj_rot))
        
        cam_pos=np.array(cam_pos)
        obj_pos=np.array(obj_pos)
        
        lookat_pos=lookat_pos
        up_vector=up_vector
        
        w_trans_mat=w_r_mat.transformation_matrix
        
        
        camera_rotation_matrix=np.asarray(p.computeViewMatrix(cam_pos, lookat_pos, up_vector)).reshape(4,4, order='F')
        camera_rotation_matrix[0:3, 3]=0.0
        
        cam_mat=np.linalg.inv(np.reshape(cam_mat, (3,3)))
        cog_pos_dist=obj_pos-obj_cog
        cog_pos_dist=cam_mat.dot(cog_pos_dist)
        net_translation=cam_dist
        net_translation[2]=net_translation[2]
        net_translation[1]=net_translation[1]
        net_translation=net_translation
        
        scale_mat=np.eye(4)
        scale_mat=scale_mat*obj_scale
        scale_mat[3,3]=1.0

        trans_mat=np.eye(4)
        trans_mat[0:3, 3]=net_translation

        if mesh_name in self.loaded_meshes:
            t_mesh=self.loaded_meshes[mesh_name].copy()
        else:
            t_mesh=trimesh.load_mesh(os.path.join(self.shapenet_root, mesh_name))
            self.loaded_meshes[mesh_name]=t_mesh.copy()
        grid_size=128
            
        scale_mat=np.eye(4)
        scale_mat=scale_mat*obj_scale
        scale_mat[3,3]=1.0
        t_mesh.apply_transform(scale_mat)
        
        flip_mat=np.zeros((4,4))
        flip_mat[2,1]=1.0
        flip_mat[1,2]=1.0
        flip_mat[0,0]=1.0
        if table:
            t_mesh.apply_transform(flip_mat)
        
        trans_mat=np.eye(4)
        trans_mat[0:3, 3]=obj_pos-cam_pos
        
        t_mesh.apply_transform(w_trans_mat)
        t_mesh.apply_transform(trans_mat)

        cam_trans=np.eye(4)
        cam_trans[:3,:3]=cam_mat
        t_mesh.apply_transform(cam_trans)

        return t_mesh
    
#     #@profile
#     def select_points_in_cube_voxelize_sphr_proj(self, all_points, i, grid_size=128, fill_type=None, estimate_table=False, sub_vox=0):
#         low=np.array([-0.5,-0.5,-0.5])
#         hi=np.array([0.5,0.5,0.5])
#         points=all_points[np.argwhere(np.all(np.logical_and(all_points>=low, all_points<=hi), axis=1))][:,0,:]
#         
#         voxels=np.zeros((grid_size,grid_size,grid_size))
#         inds=np.floor((points + 0.5) * grid_size).astype(int)
#         if sub_vox!=0:
#             inds[:,2]=inds[:,2]-sub_vox/(128/grid_size)
#             az_inds=np.argwhere(inds[:,2]>=0)
#             inds=inds[az_inds[:,0]]
#             
#         inds=np.clip(inds, 0, grid_size-1)
#         voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
#         
#         if estimate_table:
#             more_points=all_points[np.argwhere(np.all(np.logical_and(all_points>=np.array([-2,-2,-0.5]), all_points<=np.array([2,2,0.5])), axis=1))][:,0,:]
#             
#             more_inds=np.floor((more_points + 0.5) * grid_size).astype(int)
#             if more_inds.shape[0]>0:
#                 max_inds=scipy.stats.mode(more_inds[:,2], axis=None)[0][0]
#             else:
#                 max_inds=0
#             inds[:,2]=inds[:,2]-max_inds
#             az_inds=np.argwhere(inds[:,2]>=0)
#             inds=inds[az_inds[:,0]]
#             voxels=np.zeros((grid_size,grid_size,grid_size))
#             voxels[:,:,0]=1
#             voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
#         
#         if fill_type=='bfh':
#             scipy.ndimage.morphology.binary_fill_holes(voxels)
#         elif fill_type=='cc3d':
#             components=cc3d.connected_components(voxels==0, connectivity=6)
#             component_labels=np.unique(components)
#             largest_component_num=-1
#             largest_num_components=-1
#             for component_num in component_labels:
#                 if component_num>0:
#                     num_comps=np.sum(components==component_num)
#                     if num_comps>largest_num_components:
#                         largest_num_components=num_comps
#                         largest_component_num=component_num
#             voxels=np.where(np.logical_and(components!=0, components!=largest_component_num), 1, voxels)
# 
#         no_points=False
#         
#         
#         try:
#             verts, faces, normals, values = measure.marching_cubes_lewiner(
#                 voxels, spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size))
#             mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces, vertex_normals=normals)
#             proj=util_sph.proj_spherical(mesh)
#         except:
#             print('no voxels!')
#             proj=np.zeros((1,1,160,160), dtype=np.float32)
#             no_points=True
#         
#         if grid_size!=128:
#             full_voxels=np.zeros((128,128,128))
#             voxels=np.zeros((128,128,128))
#             inds=np.floor((points + 0.5) * 128).astype(int)
#             inds=np.clip(inds, 0, 128-1)
#             
#             if sub_vox!=0:
#                 inds[:,2]=inds[:,2]-sub_vox
#                 az_inds=np.argwhere(inds[:,2]>=0)
#                 inds=inds[az_inds[:,0]]
#             
#             full_voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
#             voxels=full_voxels
#         
#         if estimate_table:
#             return voxels, proj, no_points, max_inds
#         else:
#             return voxels, proj, no_points
    
    #transform vector into camera space
    def transform_to_camera_vector(self, vector, camera_pos, lookat_pos, camera_up_vector):
        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
        view_matrix = np.array(view_matrix).reshape(4,4, order='F')
        vector=np.concatenate((vector, np.array([1])))
        transformed_vector=view_matrix.dot(vector)
        return transformed_vector[:3]
    
    #trimesh mesh concate bug workaround
    def no_visual_mesh_concat(self, mesh_1, mesh_2):
        if mesh_1==None:
            return mesh_2
        meshes=[mesh_1, mesh_2]
        vertices, faces = append_faces(
        [m.vertices.copy() for m in meshes],
        [m.faces.copy() for m in meshes])

        # only save face normals if already calculated
        face_normals = None
        if all('face_normals' in m._cache for m in meshes):
            face_normals = np.vstack([m.face_normals
                                      for m in meshes])
    
        # create the mesh object
        mesh = trimesh.Trimesh(vertices=vertices,
                            faces=faces,
                            face_normals=face_normals,
                            process=False)
        
        return mesh
    
    #@profile
    #return a single sameple
    def __getitem__(self, i):
        try:
            sample_loaded = {}
            
            s_time=time.time()
            # sample structure:
            #    'cam_pos': camera position
            #    'depth_file': depth file
            #    'mask_file': mask file
            #    'obj_pos': object position
            #    'obj_rot': object world rotation
            #    'obj model': object model file
            # load steps:
            # 1. load all files
            # 2. compute object rotation relative to camera
            # 3. compute pointcloud from object mesh
            # 4. rotate pointcloud
            # 5. compute spherical projection of rotated pointcloud
            
            #create and scale pointclud from depth image
            sample=self.samples[i]
            depth = np.array(Image.open(sample['depth_file']))
            label = np.array(Image.open(sample['mask_file']))
            
            obj_points_inds=np.where(label, depth, 0.0).flatten().nonzero()[0]
            other_points_inds=np.argwhere(np.where(label, depth, 0.0).flatten()==0)[:,0]
            obs_ptcld=make_pointcloud_all_points(depth)
            obs_ptcld=obs_ptcld/1000.0
            
            inv_proj_mat=np.reshape(sample['cam_x_mat'], (3,3))
            z_line=np.zeros((100,3))
            z_line[:,2]=np.arange(100)

            obj_pointcloud=obs_ptcld[obj_points_inds]
            translation=np.mean(obj_pointcloud, axis=0)

            obs_ptcld_min=np.amin(obj_pointcloud, axis=0)
            obs_ptcld_max=np.amax(obj_pointcloud, axis=0)
            #four channel voxel cube scale
            scale=4.0*float(np.max(obs_ptcld_max-obs_ptcld_min))

            #load scene meshes
            target_mesh=None
            other_meshes=None
            for ind in range(len(sample['mesh_filename'])):
                gt_mesh=self.get_object_mesh_gt_voxels(sample['mesh_filename'][ind], sample['obj_rot'][ind], sample['cam_pos'], sample['obj_pos'][ind], sample['lookat_pos'], sample['up_vector'], sample['obj_cog'][ind], sample['cam_dist'][ind], sample['obj_scale'][ind], i, translation, scale, sample['cam_x_mat'], table=True)
                if ind!=sample['target_ind']:
                    other_meshes=self.no_visual_mesh_concat(other_meshes, gt_mesh)
                else:
                    target_mesh=gt_mesh
            table_cam_dist=self.transform_to_camera_vector(sample['table']['position'], sample['cam_pos'], sample['lookat_pos'], sample['up_vector'])
            table_mesh=self.get_object_mesh_gt_voxels(sample['table']['mesh_filename'][sample['table']['mesh_filename'].index('ShapeNetCore.v2')::], sample['table']['orientation'], sample['cam_pos'], sample['table']['position'], sample['lookat_pos'], sample['up_vector'], sample['table']['position'], table_cam_dist, sample['table']['scale'], i, translation, scale, sample['cam_x_mat'], table=True)
            other_meshes=self.no_visual_mesh_concat(other_meshes, table_mesh)

            #find unknown points by projecting a line from camera past filled voxels
            line_points=1+np.arange(1,400)/300.0
            near_obs_ptcld=obs_ptcld[np.argwhere(np.logical_and(np.logical_and(obs_ptcld[:,0]>=obs_ptcld_min[0]-scale/2, np.logical_and(obs_ptcld[:,1]>=obs_ptcld_min[1]-scale/2, obs_ptcld[:,2]>=obs_ptcld_min[2]-scale/2)),np.logical_and(obs_ptcld[:,0]<=obs_ptcld_max[0]+scale/2, np.logical_and(obs_ptcld[:,1]<=obs_ptcld_max[1]+scale/2, obs_ptcld[:,2]<=obs_ptcld_max[2]+scale/2))))][:,0,:]
            unk_points=np.reshape(near_obs_ptcld[:, None, :]*line_points[:, None], (-1, 3))
            
            other_pointcloud=obs_ptcld[other_points_inds]
            
            #translate and scale pointclouds into camera space
            unk_points=(unk_points-translation)/scale
            obj_pointcloud=(obj_pointcloud-translation)/scale
            other_pointcloud=(other_pointcloud-translation)/scale
            
            unk_points=inv_proj_mat.dot(unk_points.T).T
            obj_pointcloud=inv_proj_mat.dot(obj_pointcloud.T).T
            other_pointcloud=inv_proj_mat.dot(other_pointcloud.T).T
            
            u_target_mesh=target_mesh.copy()
            trans_mat=np.eye(4)
            trans_mat[0:3, 3]=-translation
            other_meshes.apply_transform(trans_mat)
            scale_mat=np.eye(4)
            scale_mat=scale_mat*(1.0/scale)
            scale_mat[3,3]=1.0
            other_meshes.apply_transform(scale_mat)
            inv_proj_mat_4=np.eye(4)
            inv_proj_mat_4[:3,:3]=inv_proj_mat
            other_meshes.apply_transform(inv_proj_mat_4)
            gt_other_voxels=voxel.voxelize_model_binvox(other_meshes, 128, i+1000000*self.exp_id)
            gt_other_voxels=np.argwhere(gt_other_voxels)/128.0-0.5

            trans_mat=np.eye(4)
            trans_mat[0:3, 3]=-translation
            target_mesh.apply_transform(trans_mat)
            scale_mat=np.eye(4)
            scale_mat=scale_mat*(1.0/scale)
            scale_mat[3,3]=1.0
            target_mesh.apply_transform(scale_mat)
            target_mesh.apply_transform(inv_proj_mat_4)
            gt_obj_voxels=voxel.voxelize_model_binvox(target_mesh, 128, i+1000000*self.exp_id)
            gt_obj_voxels=np.argwhere(gt_obj_voxels)/128.0-0.5
            
            #create four channel rep (voxels and spherical projection) by slicing different voxel channels in cube surrounding target object, also normalize to table for four channel rep
            if self.four_channel:
                occupied_voxels, occupied_proj, no_points, top_height=select_points_in_cube_voxelize_sphr_proj(other_pointcloud, i+1000000*self.exp_id, grid_size=128, estimate_table=True)
                unk_voxels, unk_proj, _=select_points_in_cube_voxelize_sphr_proj(unk_points, i+1000000*self.exp_id, grid_size=64, sub_vox=top_height)       
                obj_voxels, obs_proj, _=select_points_in_cube_voxelize_sphr_proj(obj_pointcloud, i+1000000*self.exp_id, grid_size=128, sub_vox=top_height)
                gt_occupied_voxels, gt_occupied_proj, _=select_points_in_cube_voxelize_sphr_proj(gt_other_voxels, i+10000*self.exp_id, grid_size=128, fill_type='cc3d', sub_vox=top_height)
                c_gt_obj_voxels, gt_obj_proj, no_points=select_points_in_cube_voxelize_sphr_proj(gt_obj_voxels, i+10000*self.exp_id, grid_size=128, fill_type='cc3d', sub_vox=top_height)
            else:
                occupied_voxels, occupied_proj, no_points=select_points_in_cube_voxelize_sphr_proj(other_pointcloud, i+1000000*self.exp_id, grid_size=128)
                unk_voxels, unk_proj, _=select_points_in_cube_voxelize_sphr_proj(unk_points, i+1000000*self.exp_id, grid_size=64)       
                obj_voxels, obs_proj, _=select_points_in_cube_voxelize_sphr_proj(obj_pointcloud, i+1000000*self.exp_id, grid_size=128)
                gt_occupied_voxels, gt_occupied_proj, _=select_points_in_cube_voxelize_sphr_proj(gt_other_voxels, i+10000*self.exp_id, grid_size=128, fill_type='cc3d')
                c_gt_obj_voxels, gt_obj_proj, no_points=select_points_in_cube_voxelize_sphr_proj(gt_obj_voxels, i+10000*self.exp_id, grid_size=128, fill_type='cc3d')

            unoccupied_voxels=np.clip(np.ones((128,128,128))-(unk_voxels+obj_voxels+occupied_voxels), 0, 1)
            unoccupied_voxels=np.logical_and(unoccupied_voxels, gt_occupied_voxels)

            if self.four_channel:
                if self.baseline_remove_seen:
                    sample_loaded['gt_voxels']=np.concatenate((c_gt_obj_voxels[None,:,:,:], gt_occupied_voxels[None,:,:,:], unoccupied_voxels[None,:,:,:])).astype(np.float32)
                else:
                    sample_loaded['gt_voxels']=c_gt_obj_voxels[None,:,:,:].astype(np.float32)
                sample_loaded['gt_proj_sphr_img']=np.concatenate((gt_obj_proj[0], gt_occupied_proj[0]), axis=0).astype(np.float32)
                sample_loaded['unknown_voxels']=unk_voxels[None,:,:,:].astype(np.float32)
                sample_loaded['obs_other']=occupied_voxels[None,:,:,:].astype(np.float32)
                sample_loaded['obs_voxels']=np.concatenate((obj_voxels[None,:,:,:], occupied_voxels[None,:,:,:], unk_voxels[None,:,:,:]), axis=0).astype(np.float32)
                sample_loaded['obs_proj']=np.concatenate((obs_proj[0], occupied_proj[0], unk_proj[0]), axis=0).astype(np.float32)
                sample_loaded['scale']=scale
                sample_loaded['occlusion']=sample['occlusion']
                return sample_loaded
            else:
                sample_loaded['gt_voxels']=np.expand_dims(c_gt_obj_voxels, 0).astype(np.float32)
                sample_loaded['gt_proj_sphr_img']=gt_obj_proj[0].cpu().numpy().astype(np.float32)
                sample_loaded['obs_voxels']=np.expand_dims(obj_voxels, 0).astype(np.float32)
                sample_loaded['obs_proj']=obs_proj[0].cpu().numpy().astype(np.float32)
                sample_loaded['scale']=scale
                sample_loaded['occlusion']=sample['occlusion']
                return sample_loaded
        except:
            traceback.print_exc()
            print(f'dataloader error on sample {i}!')
            if self.four_channel:
                sample_loaded={}
                sample_loaded['gt_voxels']=np.zeros((1,128,128,128), dtype=np.float32)
                sample_loaded['unknown_voxels']=np.zeros((1,128,128,128), dtype=np.float32)
                sample_loaded['gt_proj_sphr_img']=np.zeros((2,160,160), dtype=np.float32)
                sample_loaded['obs_other']=np.zeros((1,128,128,128), dtype=np.float32)
                sample_loaded['obs_voxels']=np.zeros((3,128,128,128), dtype=np.float32)
                sample_loaded['obs_proj']=np.zeros((3,160,160), dtype=np.float32)
                sample_loaded['scale']=1.0
                sample_loaded['occlusion']=1.0
                return sample_loaded
            else:
                sample_loaded={}
                sample_loaded['gt_voxels']=np.zeros((1,128,128,128), dtype=np.float32)
                sample_loaded['gt_proj_sphr_img']=np.zeros((1,160,160), dtype=np.float32)
                sample_loaded['obs_voxels']=np.zeros((1,128,128,128), dtype=np.float32)
                sample_loaded['obs_proj']=np.zeros((1,160,160), dtype=np.float32)
                sample_loaded['scale']=1.0
                sample_loaded['occlusion']=1.0
                return sample_loaded
    
    def make_o3d_pcd(self, points, color):
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points)
        pcd2.paint_uniform_color(np.array(color))
        return pcd2
    
    @staticmethod
    def convert_to_float32(sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def get_classes(self):
        return self._class_str
