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
import pymesh
from genre.voxelization import voxel
from genre.voxelization import binvox_rw
import copy
import torch
import pickle

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
        assert mode in ('train', 'vali')
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
            input_root+='/training_set'
         
        for dir_2 in os.listdir(input_root)[:10]:
            ind+=1
            if os.path.exists(input_root+'/'+dir_2+'/'+'scene_description.txt'):
                scene_description=json.load(open(input_root+'/'+dir_2+'/'+'scene_description.txt', 'r'))
                for file_2 in os.listdir(input_root+'/'+dir_2):
                    if file_2.startswith('pose_info'):
    #                     file_2='pose_info_00006_1.png'
                        sample={}
                            
                        
                        object_pose_info=json.load(open(input_root+'/'+dir_2+'/'+file_2, 'r'))
                        sample['cam_dist']=np.array(object_pose_info["cam_t_m2c"])/1000.0
                            
                        for obj_dict in scene_description["object_descriptions"]:
                            if obj_dict["mesh_filename"]==object_pose_info["model"]:
                                sample['obj_pos']=np.array(obj_dict["position"])
                                sample['obj_cog']=np.array(obj_dict["cog"])
                                sample['obj_rot']=obj_dict["orientation"]
                                sample['obj_scale']=obj_dict["scale"]
                                sample['mesh_filename']=obj_dict["mesh_filename"][obj_dict["mesh_filename"].index('ShapeNetCore.v2'):]
                            
                        sample['cam_pos']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["camera_pos"]
                        sample['lookat_pos']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["lookat_pos"]
                        sample['up_vector']=scene_description["views"]["background+table+objects"][int(file_2[10:15])-2]["camera_up_vector"]
                        sample['depth_file']=input_root+'/'+dir_2+'/'+'depth_'+file_2[10:15]+'.png'
                        sample['mask_file']=input_root+'/'+dir_2+'/'+'segmentation_'+file_2[10:-4]+'.png'
                        
                        if dir_2=='scene_25816' and file_2=='pose_info_00006_1.png':
                            print(sample)
                        
                        samples+=[sample]
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
        choose=depth_image.flatten().nonzero()[0]
        
        cam_cx = 320.0
        cam_cy = 240.0
        camera_params={'fx':579.411255, 'fy':579.411255, 'img_width':640, 'img_height': 480}
        
        depth_masked = depth_image.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
        
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked/self.upsample - cam_cx) * pt2 / (camera_params['fx'])
        pt1 = (xmap_masked/self.upsample - cam_cy) * pt2 / (camera_params['fy'])
        cloud = np.concatenate((pt0, -pt1, -pt2), axis=1)
        
#         p = pcl.PointCloud(cloud)
#         fil = p.make_statistical_outlier_filter()
#         fil.set_mean_k(50)
#         fil.set_std_dev_mul_thresh(5)
#         fil_cloud=np.asarray(fil.filter())
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        fil_cloud=pcd.select_down_sample(ind)
#         fil_pcd = o3d.geometry.PointCloud()
#         fil_pcd.points = o3d.utility.Vector3dVector(cloud-np.mean(fil_cloud, axis=0))
#         o3d.visualization.draw_geometries([fil_cloud])
        return np.asarray(fil_cloud.points)
    
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
        #print('start', i)
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
        
#         print(sample['mask_file'])
        if sample['mesh_filename'] in self.loaded_meshes:
            t_mesh=self.loaded_meshes[sample['mesh_filename']].copy()
        else:
            t_mesh=trimesh.load_mesh(os.path.join(self.shapenet_root, sample['mesh_filename']))
            self.loaded_meshes[sample['mesh_filename']]=t_mesh.copy()
        #t_mesh=trimesh.load_mesh(os.path.join(self.shapenet_root, sample['mesh_filename']))
        #t_mesh.show()
        
        depth = np.array(Image.open(sample['depth_file']))
        label = np.array(Image.open(sample['mask_file']))
        
        #pred=self.model.forward_with_gt_depth(depth, label)
        
        obs_ptcld=self.make_pointcloud_densefusion(np.where(label, depth, 0.0))
        obs_ptcld=obs_ptcld/1000.0
        
        translation=np.mean(obs_ptcld, axis=0)
        obs_ptcld=obs_ptcld-translation
#         
        obs_ptcld_min=np.amin(obs_ptcld, axis=0)
        obs_ptcld_max=np.amax(obs_ptcld, axis=0)
        scale=3.0*float(np.max(obs_ptcld_max-obs_ptcld_min))
        obs_ptcld=obs_ptcld/scale
        
        obs_ptcld_min=np.amin(obs_ptcld, axis=0)
        obs_ptcld_max=np.amax(obs_ptcld, axis=0)
        
#         m=np.mean(obs_ptcld, axis=0)
#         mesh = o3d.io.read_triangle_mesh(os.path.join(self.shapenet_root, sample['mesh_filename']))
#         o3d_mesh=o3d.geometry.TriangleMesh()
#         mesh.vertex_colors=o3d_mesh.vertex_colors
#         gt_voxel_ptcld=np.argwhere(np.asarray(t_binvox.matrix))
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(gt_voxel_ptcld)
#         pcd.paint_un 
    
        
        #t_binvox=trimesh.exchange.binvox.load_binvox(open(os.path.join(self.shapenet_root, sample['mesh_filename'][:-4]+'.solid.binvox'), 'rb'))

        #print(len(self.loaded_meshes))
        #t_mesh.show()
        #mesh_ptcld=mesh_ptcld*
        #res=mesh.register(obs_ptcld)
        #mesh.show()
        w_r_mat=Quaternion(np.array(sample['obj_rot']))
        
        cam_pos=np.array(sample['cam_pos'])
        obj_pos=np.array(sample['obj_pos'])
        
        delta=cam_pos-obj_pos
        
        lookat_pos=sample['lookat_pos']
        up_vector=sample['up_vector']
        
        w_trans_mat=w_r_mat.transformation_matrix
        
        
        camera_rotation_matrix=np.asarray(p.computeViewMatrix(cam_pos, lookat_pos, up_vector)).reshape(4,4, order='F')
        camera_rotation_matrix[0:3, 3]=0.0
        

        
        cog_pos_dist=sample['obj_pos']-sample['obj_cog']
        cog_pos_dist=camera_rotation_matrix[:3,:3].dot(cog_pos_dist)
        net_translation=sample['cam_dist']+cog_pos_dist
        net_translation[2]=net_translation[2]
        net_translation[1]=net_translation[1]
        net_translation=net_translation-translation
        
        scale_mat=np.eye(4)
        scale_mat=scale_mat*sample['obj_scale']
        scale_mat[3,3]=1.0
        t_mesh.apply_transform(scale_mat)
        t_mesh.apply_transform(w_trans_mat)
        
#         world_mesh=t_mesh.copy()
#         world_transl_mat=np.eye(4)
#         world_transl_mat[0:3, 3]=cog_pos_dist=sample['obj_pos']
#         world_mesh.apply_transform(world_transl_mat)
        
        t_mesh.apply_transform(camera_rotation_matrix)
        
        #t_binvox_points=self.transform_points(t_binvox_points, scale_mat)
        #t_binvox.apply_transform(scale_mat)
        #t_binvox_points=self.transform_points(t_binvox_points, w_trans_mat)
        #t_binvox.apply_transform(w_trans_mat)
        #t_binvox_points=self.transform_points(t_binvox_points, camera_rotation_matrix)
        #t_binvox.apply_transform(camera_rotation_matrix)
        

        
        trans_mat=np.eye(4)
        trans_mat[0:3, 3]=net_translation
        t_mesh.apply_transform(trans_mat)
        #t_binvox_points=self.transform_points(t_binvox_points, trans_mat)
        #t_binvox.apply_transform(trans_mat)
        
        scale_mat=np.eye(4)
        scale_mat=scale_mat*(1.0/scale)
        scale_mat[3,3]=1.0
        t_mesh.apply_transform(scale_mat)
        
        A=np.array([[1,0,0],[0,0,1],[0,1,0]])
        C=camera_rotation_matrix[0:3,0:3]
        R=np.linalg.inv(np.matmul(C, A))
        trans_mat=np.eye(4)
        trans_mat[0:3,0:3]=R
        t_mesh.apply_transform(trans_mat)
        
        zero_trans_mat=np.eye(4)
        zero_trans_mat[0:3, 3]=np.array([0, 0, -0.5-t_mesh.bounds[0,2]])
        t_mesh.apply_transform(zero_trans_mat)
        #t_binvox_points=self.transform_points(t_binvox_points, scale_mat)
        #t_binvox.apply_transform(scale_mat)
        #t_binvox.show()
        
#         gt_voxel_ptcld2=np.argwhere(np.asarray(t_binvox.matrix))
#         pcd2 = o3d.geometry.PointCloud()
#         pcd2.points = o3d.utility.Vector3dVector(gt_voxel_ptcld2)
#         pcd2.paint_uniform_color(np.array([0,1,0]))
#         o3d.visualization.draw_geometries([pcd, pcd2])
        
#         mesh = pymesh.form_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
#         grid = pymesh.VoxelGrid(1.0/128.0, mesh.dim);
#         grid.insert_mesh(mesh);
#         grid.create_grid();
#         print('grid.mesh.vertices.shape', grid.mesh.vertices.shape)
#         out_mesh = grid.mesh;
        
#         o3d_mesh.vertices=o3d.utility.Vector3iVector(np.asarray(t_mesh.vertices)
#         o3d_mesh.triangles=o3d.utility.Vector3iVector(t_mesh.faces)
#         o3d_mesh.triangle_normals=o3d.utility.Vector3iVector(t_mesh.face_normals)
#         o3d.visualization.draw_geometries([o3d_mesh])

        
#         gt_voxel_ptcld_extents=[np.amin(np.asarray(t_mesh.vertices), axis=0), np.amax(np.asarray(t_mesh.vertices), axis=0)]
#         obs_voxel_ptcld_extents=[np.amin(obs_ptcld, axis=0), np.amax(obs_ptcld, axis=0)]
#         
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(obs_ptcld)
#         pcd.paint_uniform_color(np.array([1,0,0]))
#         mpcd = o3d.geometry.PointCloud()
#         mpcd.points = o3d.utility.Vector3dVector(np.asarray(t_mesh.vertices))
#         mpcd.paint_uniform_color(np.array([0,1,0]))
#         o3d.visualization.draw_geometries([pcd, mpcd])
        
        #t_mesh.show()
        
        #gt_voxels=np.asarray(t_binvox.matrix)#voxel.voxelize_model_binvox(t_mesh, 128, i, binvox_add_param='-bb -.5 -.5 -.5 .5 .5 .5')#
        
#         mesh.scale(sample['obj_scale'])
#         mesh.transform(w_trans_mat)
#         mesh.transform(camera_rotation_matrix)
#         
#         mesh.translate(net_translation)
#         mesh.scale(1.0/scale)
#         
#         m_vertices=np.asarray(mesh.vertices)
#         m_vertices_bounds=[np.amin(m_vertices, axis=0), np.amax(m_vertices, axis=0)]
#         mesh.translate(t_mesh.bounds[0]-m_vertices_bounds[0])
#         
#         m_vertices=np.asarray(mesh.vertices)
#         m_vertices_bounds=[np.amin(m_vertices, axis=0), np.amax(m_vertices, axis=0)]
#         
#         mesh_voxelized=o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 1.0/128.0)
#         voxels=np.asarray(mesh_voxelized.voxels)
# 
# #         mesh.compute_vertex_normals()
# #         o3d.io.write_triangle_mesh('/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/temp/conv_mesh.obj', mesh)
# #     
        grid_size=128
        if os.path.exists(os.path.join(self.shapenet_root, sample['mesh_filename'][:-4]+'.solid.binvox')):
            voxels=binvox_rw.read_as_3d_array(open(os.path.join(self.shapenet_root, sample['mesh_filename'][:-4]+'.solid.binvox'), 'rb'))
            t_binvox_points=trimesh.voxel.Voxel(voxels.data, 1.0/128.0, np.array([0,0,0])).points*voxels.scale+voxels.translate
            scale_mat=np.eye(4)
            scale_mat=scale_mat*sample['obj_scale']
            scale_mat[3,3]=1.0
            t_binvox_points=self.transform_points(t_binvox_points, scale_mat)
            t_binvox_points=self.transform_points(t_binvox_points, w_trans_mat)
            t_binvox_points=self.transform_points(t_binvox_points, camera_rotation_matrix)

        
            trans_mat=np.eye(4)
            trans_mat[0:3, 3]=net_translation
            t_binvox_points=self.transform_points(t_binvox_points, trans_mat)
        
            scale_mat=np.eye(4)
            scale_mat=scale_mat*(1.0/scale)
            scale_mat[3,3]=1.0
            t_binvox_points=self.transform_points(t_binvox_points, scale_mat)
            
#             gt_voxels=np.zeros((grid_size,grid_size,grid_size))
#             inds=np.floor((t_binvox_points + 0.5) * grid_size).astype(int)
#             inds=np.clip(inds, 0, 127)
#             gt_voxels[inds[:,0], inds[:,1], inds[:,2]] = 1.0
#             gt_voxels[:,0,:]=1
#             verts, faces, normals, values = measure.marching_cubes_lewiner(gt_voxels, 0.5, spacing=(1 / 128, 1 / 128, 1 / 128))
#             mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
#             trimesh.repair.fix_inversion(mesh)
#             mesh.show()
            #
            #
            trans_mat=np.eye(4)
            trans_mat[0:3,0:3]=R
            t_binvox_points=self.transform_points(t_binvox_points, trans_mat)
            
            t_binvox_points=self.transform_points(t_binvox_points, zero_trans_mat)
            
            gt_voxels=np.zeros((grid_size,grid_size,grid_size))
            inds=np.floor((t_binvox_points + 0.5) * grid_size).astype(int)
            inds=np.clip(inds, 0, 127)
            gt_voxels[inds[:,0], inds[:,1], inds[:,2]] = 1.0
#             gt_voxels[:,:,0]=1
#             verts, faces, normals, values = measure.marching_cubes_lewiner(gt_voxels, 0.5, spacing=(1 / 128, 1 / 128, 1 / 128))
#             mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
#             trimesh.repair.fix_inversion(mesh)
#             mesh.show()
        else:
            gt_voxels=voxel.voxelize_model_binvox(t_mesh, 128, i)

#         gt_voxels=np.zeros((grid_size,grid_size,grid_size))
# #         cloud = PyntCloud.from_instance("open3d", mesh)
# #         voxelgrid_id = cloud.add_structure("voxelgrid", size_x=1.0/128.0, size_y=1.0/128.0, size_z=1.0/128.0)
# #         voxelgrid = cloud.structures[voxelgrid_id]
# #         inds=np.floor((voxelgrid._points + 0.5) * grid_size).astype(int)
# #         inds=np.clip(inds, 0, 127)
# #         gt_voxels[voxelgrid.voxel_x, voxelgrid.voxel_y, voxelgrid.voxel_z] = 1.0
# #         u=0
# # 
# #         mesh_cloud=cloud.get_sample("mesh_random", n=1000)
# #         mesh_cloud=mesh_cloud.to_numpy()
# #         
# #         a=np.asarray(mesh.vertices)
# #         b=np.asarray(mesh.triangles)
# #         c=np.asarray(mesh.triangle_normals)
# #         #t_mesh=trimesh.load_mesh('/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/temp/conv_mesh.obj')
# #         t_mesh= trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
# #                        faces=np.asarray(mesh.triangles), face_normals=np.asarray(mesh.triangle_normals), process=False)
#         #t_mesh.show()
#         
#         
# #         mesh_voxelized=t_mesh.voxelized(1.0/128.0)
# #         inds=np.floor((mesh_voxelized.points + 0.5) * grid_size).astype(int)
# #         gt_voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
#         
# #         mesh_voxelized_points_expents=[np.amin(mesh_voxelized.points, axis=0), np.amax(mesh_voxelized.points, axis=0)]
#        
# #         s_time=time.time()
# #         a=mesh_voxelized.points
# #         e_time=time.time()
# #         print('time', e_time-s_time)
#         for voxel in voxels:
# #             inds=np.floor((voxel.grid_index + 0.5) * grid_size).astype(int)
# #             inds=np.clip(inds, 0, 127)
# #             gt_voxels[inds[0], inds[1], inds[2]] = 1.0
#               
#             gt_voxels[int(np.floor((voxel.grid_index[0]/128.0+mesh_voxelized.origin[0]+0.5)*128.0)), 
#                       int(np.floor((voxel.grid_index[1]/128.0+mesh_voxelized.origin[1]+0.5)*128.0)), 
#                       int(np.floor((voxel.grid_index[2]/128.0+mesh_voxelized.origin[2]+0.5)*128.0))] = 1.0
        
        
        
        gt_proj_sphr_img=util_sph.proj_spherical(t_mesh) 
        
#         np_gt_proj_sphr_img=gt_proj_sphr_img.detach().cpu().numpy()
#         cv2.imshow('gt_proj_sphr_img', np_gt_proj_sphr_img.astype(np.float32)[0,0,:,:])
#         np_gt_proj_sphr_img=np_gt_proj_sphr_img-1.0
#         np_gt_proj_sphr_img=np_gt_proj_sphr_img/np.amax(np_gt_proj_sphr_img)
        
        
        tdf = np.zeros([grid_size, grid_size, grid_size]) / grid_size
        cnt = np.zeros([grid_size, grid_size, grid_size])
        
        obs_ptcld=np.matmul(R, obs_ptcld.T).T
        obs_ptcld=obs_ptcld+zero_trans_mat[0:3, 3]
        inds2=np.floor((obs_ptcld + 0.5) * grid_size).astype(int)
        inds2=np.clip(inds2, 0, 127)
        tdf[inds2[:, 0], inds2[:, 1], inds2[:, 2]] = 1.0
#         tdf[:,:,0]=1
        
#         for pts in obs_ptcld:
#             pt = pts  # np.array([-pts[2], -pts[0], pts[1]])
#             ids = np.floor((pt + 0.5) * grid_size).astype(int)
#             if np.any(np.abs(pt) >= 0.5):
#                 continue
#             center = ((ids + 0.5) * 1 / grid_size) - 0.5
#             dist = ((center[0] - pt[0])**2 + (center[1] - pt[1])
#                     ** 2 + (center[2] - pt[2])**2)**0.5
#             n = cnt[ids[0], ids[1], ids[2]]
#             tdf[ids[0], ids[1], ids[2]] = 1.0
#             cnt[ids[0], ids[1], ids[2]] += 1
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
            tdf, spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size))
            verts=verts+zero_trans_mat[0:3, 3]
            obs_mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
#             trimesh.repair.fix_inversion(obs_mesh)
#             obs_mesh.show()
            obs_proj=util_sph.proj_spherical(obs_mesh)
        except:
            obs_proj=np.zeros((1,160,160))
            print('marching cubes error!')
#         np_gt_proj_sphr_img=obs_proj.detach().cpu().numpy()
#         cv2.imshow('obs_proj', np_gt_proj_sphr_img.astype(np.float32)[0,0,:,:])
        obs_voxels=tdf
        
#         gt_voxel_ptcld=np.argwhere(gt_voxels)
#         obs_voxel_ptcld=np.argwhere(obs_voxels)
# #         
#         gt_voxel_ptcld_extents2=[np.amin(gt_voxel_ptcld, axis=0), np.amax(gt_voxel_ptcld, axis=0)]
#         obs_voxel_ptcld_extents2=[np.amin(obs_voxel_ptcld, axis=0), np.amax(obs_voxel_ptcld, axis=0)]
#             
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(gt_voxel_ptcld)
#         pcd.paint_uniform_color(np.array([1,0,0]))
#         mpcd = o3d.geometry.PointCloud()
#         mpcd.points = o3d.utility.Vector3dVector(np.asarray(obs_voxel_ptcld))
#         mpcd.paint_uniform_color(np.array([0,1,0]))
#         o3d.visualization.draw_geometries([pcd, mpcd])

#         gt_voxel_locs=np.argwhere(gt_voxels)
#         min_z=np.min(gt_voxel_locs[:, 2])
#         gt_voxel_locs[:, 2]=gt_voxel_locs[:, 2]-min_z
#         obs_voxel_locs=np.argwhere(obs_voxels)
#         obs_voxel_locs[:, 2]=obs_voxel_locs[:, 2]-min_z
        
        #         
        sample_loaded['gt_voxels']=np.expand_dims(gt_voxels, 0)
        sample_loaded['gt_proj_sphr_img']=gt_proj_sphr_img.detach().cpu().numpy()[0]
        sample_loaded['obs_voxels']=np.expand_dims(obs_voxels, 0)
        sample_loaded['obs_proj']=obs_proj.detach().cpu().numpy()[0]
#         print('obs_proj mean', np.mean(sample_loaded['obs_proj']))
#         
#         if np.sum(sample_loaded['gt_voxels'])==0:
#             print('gt voxels all zero!')
#         if np.amin(sample_loaded['gt_proj_sphr_img'])==np.amax(sample_loaded['gt_proj_sphr_img']):
#             print('gt_proj_sphr_img min==max')
#         if np.amax(np.abs(sample_loaded['gt_proj_sphr_img']))>10:
#             print('large gt_proj_sphr_img', np.amax(np.abs(sample_loaded['gt_proj_sphr_img'])))
#             print('depth file', sample['mask_file'])
#             print('mesh_filename', sample['mesh_filename'])
#             print('mask_file', sample['mask_file'])
#             print(sample)
#             print('\n')
#             print(self.copied_samples[i])
#         if np.sum(sample_loaded['obs_voxels'])==0:
#             print('obs voxels all zero!')
#         if np.amin(sample_loaded['obs_proj'])==np.amax(sample_loaded['obs_proj']):
#             print('obs proj min==max')
#         if np.amax(np.abs(sample_loaded['obs_proj']))>10:
#             print('large obs_proj', np.amax(np.abs(sample_loaded['obs_proj'])))
#             print('depth file', sample['mask_file'])
#             print('mesh_filename', sample['mesh_filename'])
#             print('mask_file', sample['mask_file'])
#             print(sample)
#             print('\n')
#             print(self.copied_samples[i])
        
        # convert all types to float32 for better copy speed
        self.convert_to_float32(sample_loaded)
        
        e_time=time.time()
        #print(f'loaded: {i}', e_time-s_time)
        #print('loaded', i)
#         self.cached_samples[i]=sample_loaded

        #self.model.eval()
        
        #print('img mean', np.mean(torch.from_numpy(np.expand_dims(sample_loaded['obs_proj'], 0)).repeat(8, 1, 1, 1).float().cuda().detach().cpu().numpy()))
        #inputs=[torch.from_numpy(np.expand_dims(sample_loaded['obs_proj'], 0)).repeat(8, 1, 1, 1).float(), torch.from_numpy(np.expand_dims(sample_loaded['obs_voxels'], 0)).repeat(8, 1, 1, 1, 1).float()]
        #pickle.dump(inputs, open('/home/willie/workspace/GenRe-ShapeHD/scratch/inputs.p', 'wb'))
        #inputs=pickle.load(open('/home/willie/workspace/GenRe-ShapeHD/scratch/inputs.p', 'rb'))
        #print(inputs)
        #inputs[0]=inputs[0].cuda()
        #inputs[1]=inputs[1].cuda()
        
        #pred=self.model.net.forward(inputs)
        
        #np_pred_voxel=pred['pred_voxel'].detach().cpu().numpy()
        #np_pred_voxel=1 / (1 + np.exp(-np_pred_voxel))
#         for ind in range(1):
#             verts, faces, normals, values = measure.marching_cubes_lewiner(
#                 np_pred_voxel[ind, 0, :, :, :], 0.5, spacing=(1 / 128, 1 / 128, 1 / 128))
#             verts=verts-0.5
#             mesh = trimesh.Trimesh(vertices=verts, faces=faces)
#             mesh.show()

        return sample_loaded

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
