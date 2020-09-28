from os.path import join
import random
import numpy as np
from scipy.io import loadmat
import torch.utils.data as data
from genre.util import util_sph
import trimesh
import os
import json
from PIL import Image
from pyquaternion import Quaternion
import math
import pybullet as p
from scipy.misc import imresize
import open3d as o3d
from pyntcloud import PyntCloud
from skimage import measure
import cv2
#import pcl
import time
#import pymesh
from genre.voxelization import voxel
from genre.voxelization import binvox_rw
import copy
import torch
import pickle
import cc3d
import scipy

class Dataset(data.Dataset):
    data_root = '/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/shapenet_cars_chairs_planes_20views'
    list_root = join(data_root, 'status')
    status_and_suffix = {
        'rgb': {
            'status': 'rgb.txt',
            'suffix': '_rgb.png',
        },
        'depth': {
            'status': 'depth.txt',
            'suffix': '_depth.png',
        },
        'depth_minmax': {
            'status': 'depth_minmax.txt',
            'suffix': '.npy',
        },
        'silhou': {
            'status': 'silhou.txt',
            'suffix': '_silhouette.png',
        },
        'normal': {
            'status': 'normal.txt',
            'suffix': '_normal.png'
        },
        'voxel': {
            'status': 'vox_rot.txt',
            'suffix': '_gt_rotvox_samescale_128.npz'
        },
        'spherical': {
            'status': 'spherical.txt',
            'suffix': '_spherical.npz'
        },
        'voxel_canon': {
            'status': 'vox_canon.txt',
            'suffix': '_voxel_normalized_128.mat'
        },
    }
    class_aliases = {
        'drc': '03001627+02691156+02958343',
        'chair': '03001627',
        'table': '04379243',
        'sofa': '04256520',
        'couch': '04256520',
        'cabinet': '03337140',
        'bed': '02818832',
        'plane': '02691156',
        'car': '02958343',
        'bench': '02828884',
        'monitor': '03211117',
        'lamp': '03636649',
        'speaker': '03691459',
        'firearm': '03948459+04090263',
        'cellphone': '02992529+04401088',
        'watercraft': '04530566',
        'hat': '02954340',
        'pot': '03991062',
        'rocket': '04099429',
        'train': '04468005',
        'bus': '02924116',
        'pistol': '03948459',
        'faucet': '03325088',
        'helmet': '03513137',
        'clock': '03046257',
        'phone': '04401088',
        'display': '03211117',
        'vessel': '04530566',
        'rifle': '04090263',
        'small': '03001627+04379243+02933112+04256520+02958343+03636649+02691156+04530566',
        'all-but-table': '02691156+02747177+02773838+02801938+02808440+02818832+02828884+02843684+02871439+02876657+02880940+02924116+02933112+02942699+02946921+02954340+02958343+02992529+03001627+03046257+03085013+03207941+03211117+03261776+03325088+03337140+03467517+03513137+03593526+03624134+03636649+03642806+03691459+03710193+03759954+03761084+03790512+03797390+03928116+03938244+03948459+03991062+04004475+04074963+04090263+04099429+04225987+04256520+04330267+04401088+04460130+04468005+04530566+04554684',
        'all-but-chair': '02691156+02747177+02773838+02801938+02808440+02818832+02828884+02843684+02871439+02876657+02880940+02924116+02933112+02942699+02946921+02954340+02958343+02992529+03046257+03085013+03207941+03211117+03261776+03325088+03337140+03467517+03513137+03593526+03624134+03636649+03642806+03691459+03710193+03759954+03761084+03790512+03797390+03928116+03938244+03948459+03991062+04004475+04074963+04090263+04099429+04225987+04256520+04330267+04379243+04401088+04460130+04468005+04530566+04554684',
        'all': '02691156+02747177+02773838+02801938+02808440+02818832+02828884+02843684+02871439+02876657+02880940+02924116+02933112+02942699+02946921+02954340+02958343+02992529+03001627+03046257+03085013+03207941+03211117+03261776+03325088+03337140+03467517+03513137+03593526+03624134+03636649+03642806+03691459+03710193+03759954+03761084+03790512+03797390+03928116+03938244+03948459+03991062+04004475+04074963+04090263+04099429+04225987+04256520+04330267+04379243+04401088+04460130+04468005+04530566+04554684',
    }
    class_list = class_aliases['all'].split('+')

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
        if model is None:
            required = ['rgb']
            self.preproc = None
        else:
            required = model.requires
            self.preproc = model.preprocess
            
        self.loaded_meshes=loaded_meshes
        self.cached_samples={}
        self.shapenet_root=opt.shapenet_root
        self.upsample=1.0
        self.xmap = np.array([[j for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])
        self.ymap = np.array([[i for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])

        ind=0
        samples=[]
        input_root=opt.dataset_root
        
#         for i in range(100):
#             input_root='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/tabletop_dataset_with_poses/debug/'
#             dir_2='cube'
#             for file_2 in os.listdir(input_root+'/'+dir_2):
#                 if file_2.startswith('pose_info'):
#                     sample={}
#                       
#                     scene_description=json.load(open(input_root+'/'+dir_2+'/'+'scene_description.txt', 'r'))
#                     object_pose_info=json.load(open(input_root+'/'+dir_2+'/'+file_2, 'r'))
#                     sample['cam_dist']=np.array(object_pose_info["cam_t_m2c"])/1000.0
#                       
#                     for obj_dict in scene_description["object_descriptions"]:
#                         if obj_dict["mesh_filename"]==object_pose_info["model"]:
#                             sample['obj_pos']=np.array(obj_dict["position"])
#                             sample['obj_cog']=np.array(obj_dict["cog"])
#                             sample['obj_rot']=obj_dict["orientation"]
#                             sample['obj_scale']=obj_dict["scale"]
#                             sample['mesh_filename']=obj_dict["mesh_filename"][obj_dict["mesh_filename"].index('ShapeNetCore.v2'):]
#                       
#                     sample['cam_pos']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["camera_pos"]
#                     sample['lookat_pos']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["lookat_pos"]
#                     sample['up_vector']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["camera_up_vector"]
#                     sample['depth_file']=input_root+'/'+dir_2+'/'+'depth_'+file_2[10:15]+'.png'
#                     sample['mask_file']=input_root+'/'+dir_2+'/'+'segmentation_'+file_2[10:17]+'.png'
#                     samples+=[sample]
        
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
    #                     file_2='pose_info_00006_1.png'
                        view_num=file_2.split('_')[2]
                        sample={}
                        object_pose_info=pickle.load(open(input_root+'/'+dir_2+'/'+file_2, 'rb'))
                        sample['cam_dist']=[]#np.array(object_pose_info["cam_t_m2c"])/1000.0
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
        if self.mode == 'vali':
            if opt.manual_seed:
                seed = opt.manual_seed
            else:
                seed = 0
            random.Random(seed).shuffle(samples)
        self.samples = samples
        self.copied_samples=copy.deepcopy(samples)
        
    def get_object_mesh_gt_voxels(self, mesh_name, obj_rot, cam_pos, obj_pos, lookat_pos, up_vector, obj_cog, cam_dist, obj_scale, i, b_pos, scale, cam_mat, table=False):
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
        
        if mesh_name in self.loaded_meshes:
            t_mesh=self.loaded_meshes[mesh_name].copy()
        else:
            t_mesh=trimesh.load_mesh(os.path.join(self.shapenet_root, mesh_name))
            self.loaded_meshes[mesh_name]=t_mesh.copy()

        w_r_mat=Quaternion(np.array(obj_rot))
        
        cam_pos=np.array(cam_pos)
        obj_pos=np.array(obj_pos)
        
        lookat_pos=lookat_pos
        up_vector=up_vector
        
        w_trans_mat=w_r_mat.transformation_matrix
        
        
        camera_rotation_matrix=np.asarray(p.computeViewMatrix(cam_pos, lookat_pos, up_vector)).reshape(4,4, order='F')
        camera_rotation_matrix[0:3, 3]=0.0
        
        cam_mat=np.linalg.inv(np.reshape(cam_mat, (3,3)))
        cog_pos_dist=obj_pos-obj_cog#+(obj_pos-b_pos)#-b_pos#-obj_cog+(obj_pos-b_pos)
        cog_pos_dist=cam_mat.dot(cog_pos_dist)
        net_translation=cam_dist#+cog_pos_dist
        net_translation[2]=net_translation[2]
        net_translation[1]=net_translation[1]
        net_translation=net_translation
        
        grid_size=128
        if os.path.exists(os.path.join(self.shapenet_root, mesh_name[:-4]+'.solid.binvox')):
            voxels=binvox_rw.read_file_as_3d_array(open(os.path.join(self.shapenet_root, mesh_name[:-4]+'.solid.binvox'), 'rb'))
            t_binvox_points=(np.argwhere(voxels.data)/128.0)*voxels.scale+voxels.translate
        else:
            print('no binvox found!')
            voxels=voxel.voxelize_model_binvox(t_mesh, 128, i+10000)
            t_binvox_points=np.argwhere(voxels)#
        scale_mat=np.eye(4)
        scale_mat=scale_mat*obj_scale
        scale_mat[3,3]=1.0

        
        flip_mat=np.zeros((3,3))
        flip_mat[2,1]=1.0
        flip_mat[1,2]=1.0
        flip_mat[0,0]=1.0
        if table:
            t_binvox_points=flip_mat.dot(t_binvox_points.T).T
        t_binvox_points=t_binvox_points*obj_scale
        t_binvox_points=w_trans_mat[:3,:3].dot(t_binvox_points.T).T
        t_binvox_points=t_binvox_points+obj_pos
        
        t_binvox_points=t_binvox_points-cam_pos
        t_binvox_points=cam_mat.dot(t_binvox_points.T).T
        
        #transform mesh
        scale_mat=np.eye(4)
        scale_mat=scale_mat*obj_scale
        scale_mat[3,3]=1.0
        t_mesh.apply_transform(scale_mat)
        
        trans_mat=np.eye(4)
        trans_mat[:3,:3]=w_trans_mat[:3,:3]
        t_mesh.apply_transform(trans_mat)
        
        trans_mat=np.eye(4)
        trans_mat[0:3, 3]=obj_pos-cam_pos
        t_mesh.apply_transform(trans_mat)
        
        trans_mat=np.eye(4)
        trans_mat[:3,:3]=cam_mat[:3,:3]
        t_mesh.apply_transform(trans_mat)
        
#         flip_mat=np.array([[0,-1,0],
#                    [1,0,0],
#                    [0,0,1]])
#         t_binvox_points=flip_mat.dot(t_binvox_points.T).T
#         t_binvox_points=self.transform_points(t_binvox_points, scale_mat)
#         t_binvox_points=self.transform_points(t_binvox_points, w_trans_mat)
#         
#         t_binvox_points=self.transform_points(t_binvox_points, camera_rotation_matrix)
#         #table_grid=self.transform_points(table_grid, camera_rotation_matrix)
# 
#         trans_mat=np.eye(4)
#         trans_mat[0:3, 3]=net_translation
#         t_binvox_points=self.transform_points(t_binvox_points, trans_mat)
        #table_grid=self.transform_points(table_grid, trans_mat)
    
#             scale_mat=np.eye(4)
#             scale_mat=scale_mat*(1.0/scale)
#             scale_mat[3,3]=1.0
#             t_binvox_points=self.transform_points(t_binvox_points, scale_mat)
        
#         gt_voxels=np.zeros((grid_size,grid_size,grid_size))
#         inds=np.floor((t_binvox_points + 0.5) * grid_size).astype(int)
#         inds=np.clip(inds, 0, 127)
#         gt_voxels[inds[:,0], inds[:,1], inds[:,2]] = 1.0
#            t_binvox_points=np.zeros((0,3))
#            print('no voxelization!')
        if t_binvox_points.shape[0]==0:
            print('no points', os.path.exists(os.path.join(self.shapenet_root, mesh_name[:-4]+'.solid.binvox')))
        return t_binvox_points, t_mesh
    
    def voxelize_subdivide(self, mesh,
                           pitch,
                           max_iter=10,
                           edge_factor=2.0):
        """
        Voxelize a surface by subdividing a mesh until every edge is
        shorter than: (pitch / edge_factor)
        Parameters
        -----------
        mesh:        Trimesh object
        pitch:       float, side length of a single voxel cube
        max_iter:    int, cap maximum subdivisions or None for no limit.
        edge_factor: float,
        Returns
        -----------
        VoxelGrid instance representing the voxelized mesh.
        """
        max_edge = pitch / edge_factor
    
        if max_iter is None:
            longest_edge = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] -
                                          mesh.vertices[mesh.edges[:, 1]],
                                          axis=1).max()
            max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)
    
        # get the same mesh sudivided so every edge is shorter
        # than a factor of our pitch
        v, f = trimesh.remesh.subdivide_to_size(mesh.vertices,
                                        mesh.faces,
                                        max_edge=max_edge,
                                        max_iter=max_iter)
    
        # convert the vertices to their voxel grid position
        hit = v / pitch
    
        # Provided edge_factor > 1 and max_iter is large enough, this is
        # sufficient to preserve 6-connectivity at the level of voxels.
        hit = np.round(hit).astype(int)
    
        # remove duplicates
        unique, inverse = trimesh.grouping.unique_rows(hit)
    
        # get the voxel centers in model space
        occupied_index = hit[unique]
    
        origin_index = occupied_index.min(axis=0)
        origin_position = origin_index * pitch
    
        return trimesh.voxel.VoxelGrid(
            trimesh.voxel.SparseBinaryEncoding(occupied_index - origin_index),
            transform=tr.scale_and_translate(
                scale=pitch, translate=origin_position))
    
    #@profile  
    def make_pointcloud_densefusion(self, depth_image):
#         depth_scale=np.amax(depth_image)
#         depth_image=(depth_image/depth_scale)
#         min_depth=np.amin(np.where(depth_image==0.0, 1.0, depth_image))
#         depth_image=np.reshape(depth_image, (depth_image.shape[0], depth_image.shape[1], 1))
#         depth_image=np.concatenate((depth_image,depth_image,depth_image), axis=2)
#         depth_image = cv2.resize(depth_image, (int(self.upsample*depth_image.shape[1]), int(self.upsample*depth_image.shape[0])), interpolation=cv2.INTER_LINEAR)#imresize(depth_image, float(upsample), mode='F', interp='nearest')#
#         depth_image=depth_image[:,:,0]
#         depth_image=np.where(depth_image<min_depth, 0.0, depth_image)
#         depth_image=depth_image*depth_scale
        
        #depth_image=depth_image*1000.0
#         cv2.imshow('depth_image', depth_image)
#         cv2.waitKey(20)
        
        cam_scale = 1.0
        
        
        cam_cx = 320.0
        cam_cy = 240.0
        camera_params={'fx':579.411255, 'fy':579.411255, 'img_width':640, 'img_height': 480}
        
        depth_masked = depth_image.flatten()[:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[:, np.newaxis].astype(np.float32)
        
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked/self.upsample - cam_cx) * pt2 / (camera_params['fx'])
        pt1 = (xmap_masked/self.upsample - cam_cy) * pt2 / (camera_params['fy'])
        cloud = np.concatenate((pt0, -pt1, -pt2), axis=1)
        
        #seg_cloud=cloud[close[:,0]]#[choose]
        #cloud_bounds=[np.amin(seg_cloud, axis=0), np.amax(seg_cloud, axis=0)]
        
#         p = pcl.PointCloud(cloud)
#         fil = p.make_statistical_outlier_filter()
#         fil.set_mean_k(50)
#         fil.set_std_dev_mul_thresh(5)
#         fil_cloud=np.asarray(fil.filter())
        
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(seg_cloud)
#         cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
#         fil_cloud=pcd.select_down_sample(ind)
#         fil_pcd = o3d.geometry.PointCloud()
#         fil_pcd.points = o3d.utility.Vector3dVector(cloud-np.mean(fil_cloud, axis=0))
#         o3d.visualization.draw_geometries([fil_cloud])
        return cloud
    
    def make_voxels(self, pointcloud):
        pass
    
    def transform_points(self, points,
                     matrix,
                     translate=True):
        """
        Returns points rotated by a homogeneous
        transformation matrix.
        If points are (n, 2) matrix must be (3, 3)
        If points are (n, 3) matrix must be (4, 4)
        Parameters
        ----------
        points : (n, D) float
          Points where D is 2 or 3
        matrix : (3, 3) or (4, 4) float
          Homogeneous rotation matrix
        translate : bool
          Apply translation from matrix or not
        Returns
        ----------
        transformed : (n, d) float
          Transformed points
        """
        points = np.asanyarray(
            points, dtype=np.float64)
        # no points no cry
        if len(points) == 0:
            return points.copy()
    
        matrix = np.asanyarray(matrix, dtype=np.float64)
        if (len(points.shape) != 2 or
                (points.shape[1] + 1 != matrix.shape[1])):
            raise ValueError('matrix shape ({}) doesn\'t match points ({})'.format(
                matrix.shape,
                points.shape))
    
        # check to see if we've been passed an identity matrix
        identity = np.abs(matrix - np.eye(matrix.shape[0])).max()
        if identity < 1e-8:
            return np.ascontiguousarray(points.copy())
    
        dimension = points.shape[1]
        column = np.zeros(len(points)) + int(bool(translate))
        stacked = np.column_stack((points, column))
        transformed = np.dot(matrix, stacked.T).T[:, :dimension]
        transformed = np.ascontiguousarray(transformed)
        return transformed
    
    #@profile
    def __getitem__(self, i):
        #print('loaded', i)
#         if i in self.cached_samples:
#             return self.cached_samples[i]
        low=np.array([-0.5,-0.5,-0.5])
        hi=np.array([0.5,0.5,0.5])
        
        grid_size=128
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
        
        sample=self.samples[i]
        target_ind=sample['target_ind']
        
        if sample['mesh_filename'][target_ind] in self.loaded_meshes:
            t_mesh=self.loaded_meshes[sample['mesh_filename'][target_ind]].copy()
        else:
            t_mesh=trimesh.load_mesh(os.path.join(self.shapenet_root, sample['mesh_filename'][target_ind]))
            self.loaded_meshes[sample['mesh_filename'][target_ind]]=t_mesh.copy()
            
        inv_proj_mat=np.reshape(sample['cam_x_mat'], (3,3))
        #t_mesh=trimesh.load_mesh(os.path.join(self.shapenet_root, sample['mesh_filename']))
        #t_mesh.show()
        
        #process obs
        depth = np.array(Image.open(sample['depth_file']))
        label = np.array(Image.open(sample['mask_file']))
        obj_points_inds=np.where(label, depth, 0.0).flatten().nonzero()[0]
        other_points_inds=np.argwhere(np.where(label, depth, 0.0).flatten()==0)[:,0]
        obs_ptcld=self.make_pointcloud_densefusion(depth)
        obs_ptcld=obs_ptcld/1000.0
        
        obj_pointcloud=obs_ptcld[obj_points_inds]
        
        translation=np.mean(obj_pointcloud, axis=0)
        obj_pointcloud=obj_pointcloud-translation
#         
        obs_ptcld_min=np.amin(obj_pointcloud, axis=0)
        obs_ptcld_max=np.amax(obj_pointcloud, axis=0)
        scale=4.0*float(np.max(obs_ptcld_max-obs_ptcld_min))
        obj_pointcloud=obj_pointcloud/scale
        obj_pointcloud=inv_proj_mat.dot(obj_pointcloud.T).T
        obj_pointcloud=obj_pointcloud[np.argwhere(np.all(np.logical_and(obj_pointcloud>=low, obj_pointcloud<=hi), axis=1))][:,0,:]
        obj_inds=np.floor((obj_pointcloud + 0.5) * grid_size).astype(int)
        
        other_pointcloud=obs_ptcld[other_points_inds]
        other_pointcloud=(other_pointcloud-translation)/scale
        other_pointcloud_all=inv_proj_mat.dot(other_pointcloud.T).T
        other_pointcloud=other_pointcloud_all[np.argwhere(np.all(np.logical_and(other_pointcloud_all>=np.array([-2,-2,-0.5]), other_pointcloud_all<=np.array([2,2,0.5])), axis=1))][:,0,:]
#         other_voxels=np.zeros((grid_size, grid_size, grid_size))    
        other_inds=np.floor((other_pointcloud + 0.5) * grid_size).astype(int)
        #other_voxels[other_inds[:, 0], other_inds[:, 1], other_inds[:, 2]] = 1.0
        
        if other_inds.shape[0]>0:
            sub_vox=scipy.stats.mode(other_inds[:,2], axis=None)[0][0]
        else:
            sub_vox=0

        obj_inds[:,2]=obj_inds[:,2]-sub_vox
        az_inds=np.argwhere(obj_inds[:,2]>=0)
        obj_inds=obj_inds[az_inds[:,0]]
        obs_voxels=np.zeros((grid_size,grid_size,grid_size))
        obs_voxels[obj_inds[:, 0], obj_inds[:, 1], obj_inds[:, 2]] = 1.0

        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
            obs_voxels, spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size))
            obs_mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
            #obs_mesh.show()
            obs_proj=util_sph.proj_spherical(obs_mesh)
        except:
            obs_proj=torch.from_numpy(np.zeros((1,1,160,160), dtype=np.float32))
#             obj_pcd=self.make_o3d_pcd(other_inds, [1,0,0])
#             gt_pcd=self.make_o3d_pcd(np.floor((obj_pointcloud + 0.5) * grid_size).astype(int), [0,1,0])
#             o3d.visualization.draw_geometries([gt_pcd, obj_pcd])
            print('marching cubes error!')
        #obs_proj=np.zeros((1,1,160,160), dtype=np.float32)
        #process ground truth
        gt_points, gt_mesh=self.get_object_mesh_gt_voxels(sample['mesh_filename'][target_ind], sample['obj_rot'][target_ind], sample['cam_pos'], sample['obj_pos'][target_ind], 
                                                 sample['lookat_pos'], sample['up_vector'], sample['obj_cog'][target_ind], sample['cam_dist'][target_ind], 
                                                 sample['obj_scale'][target_ind], i, translation, scale, sample['cam_x_mat'], table=True)

        gt_points=(gt_points-translation)/scale
        gt_points=inv_proj_mat.dot(gt_points.T).T
        gt_points=gt_points[np.argwhere(np.all(np.logical_and(gt_points>=low, gt_points<=hi), axis=1))][:,0,:]
        gt_voxels=np.zeros((grid_size,grid_size,grid_size))
        inds=np.floor((gt_points + 0.5) * grid_size).astype(int)
        inds[:,2]=inds[:,2]-sub_vox
        az_inds=np.argwhere(inds[:,2]>=0)
        inds=inds[az_inds[:,0]]
        inds=np.clip(inds, 0, grid_size-1)
        gt_voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
        components=cc3d.connected_components(gt_voxels==0, connectivity=6)
        component_labels=np.unique(components)
        largest_component_num=-1
        largest_num_components=-1
        for component_num in component_labels:
            if component_num>0:
                num_comps=np.sum(components==component_num)
                if num_comps>largest_num_components:
                    largest_num_components=num_comps
                    largest_component_num=component_num
        gt_voxels=np.where(np.logical_and(components!=0, components!=largest_component_num), 1, gt_voxels)

        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                    gt_voxels, spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size))
            gt_mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces, vertex_normals=normals)
            
            if verts.shape[0]>50000:
                gt_mesh.export(f'/dev/shm/shapenet_temp_mesh_conv_{i}.ply')
                o3d_mesh=o3d.io.read_triangle_mesh(f'/dev/shm/shapenet_temp_mesh_conv_{i}.ply')
                o3d_mesh=o3d_mesh.simplify_vertex_clustering(0.05)#.filter_smooth_taubin(number_of_iterations=20)
                gt_mesh=trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles), face_normals=np.asarray(o3d_mesh.triangle_normals), process=False)
                os.remove(f'/dev/shm/shapenet_temp_mesh_conv_{i}.ply')
        
        #trimesh.repair.fix_inversion(gt_mesh)
            gt_proj_sphr_img=util_sph.proj_spherical(gt_mesh)
        except:
            gt_proj_sphr_img=torch.from_numpy(np.zeros((1,1,160,160), dtype=np.float32))
            print('marching cubes error!')

        

#         gt_voxels[:,:,0]=1
#         obj_pcd=self.make_o3d_pcd(np.argwhere(obs_voxels), [1,0,0])
#         gt_pcd=self.make_o3d_pcd(np.argwhere(gt_voxels), [0,1,0])
#         o3d.visualization.draw_geometries([gt_pcd, obj_pcd])
        
#         obj_pcd=self.make_o3d_pcd(np.argwhere(obs_voxels), [1,0,0])
#         gt_pcd=self.make_o3d_pcd(np.argwhere(gt_voxels), [0,1,0])
#         o3d.visualization.draw_geometries([gt_pcd, obj_pcd])

        sample_loaded['gt_voxels']=np.expand_dims(gt_voxels, 0)
        sample_loaded['gt_proj_sphr_img']=gt_proj_sphr_img[0]
        sample_loaded['obs_voxels']=np.expand_dims(obs_voxels, 0)
        sample_loaded['obs_proj']=obs_proj[0]
        sample_loaded['scale']=scale
        sample_loaded['occlusion']=sample['occlusion']

        self.convert_to_float32(sample_loaded)
        
        e_time=time.time()
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
