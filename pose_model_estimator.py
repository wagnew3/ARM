import shutil
import util.util as util_
import os
import cv2
import open3d as o3d
import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
import trimesh
from skimage import measure
import scipy
from sklearn.neighbors import KDTree
from scipy.ndimage.measurements import label
import data_augmentation
from genre.voxelization import voxel
import traceback
from genre.util import util_sph
from scipy import stats
from dm_control.mujoco.engine import Camera
from trajopt.mujoco_utils import add_object_to_mujoco, remove_objects_from_mujoco, get_mesh_list, compute_mujoco_int_transform

mesh_level=0.5

def chamfer_distance(pcd_1, pcd_2):
        pcd_tree = KDTree(pcd_2)
        nearest_distances_1, _=pcd_tree.query(pcd_1)
        
        pcd_tree = KDTree(pcd_1)
        nearest_distances_2, _=pcd_tree.query(pcd_2)
        
        return np.sum(nearest_distances_1)/pcd_1.shape[0]+np.sum(nearest_distances_2)/pcd_2.shape[0]

#return outher shell of voxel shape 
def hollow_dense_pointcloud(ptcld):
    conv=scipy.ndimage.convolve(ptcld, np.ones((3,3,3)))
    ptcld=np.where(conv<27, ptcld, 0)
    return ptcld

def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l) # pixels per meter
        focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = util_.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    
    return xyz_img

upsample=1
xmap = np.array([[j for i in range(int(upsample*640))] for j in range(int(upsample*480))])
ymap = np.array([[i for i in range(int(upsample*640))] for j in range(int(upsample*480))])

#make pointcloud from depth image
def make_pointcloud_all_points(depth_image):
        cam_scale = 1.0

        cam_cx = 320.0
        cam_cy = 240.0
        camera_params={'fx':579.411255, 'fy':579.411255, 'img_width':640, 'img_height': 480}
        
        depth_masked = depth_image.flatten()[:, np.newaxis].astype(np.float32)
        xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
        ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)
        
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked/upsample - cam_cx) * pt2 / (camera_params['fx'])
        pt1 = (xmap_masked/upsample - cam_cy) * pt2 / (camera_params['fy'])
        cloud = np.concatenate((pt0, -pt1, -pt2), axis=1)
        return cloud
   
def color_code_objects(frame, state_id_to_model_pixels, display=False):
    #generate object color mapping
    labels=np.unique(frame)
    exec_dir=os.path.dirname(os.path.realpath(__file__))
    color_map_file_name=exec_dir+'/data/object_color_maps/object_color_map_size_'+str(labels.shape[0])+'.p'
    if os.path.isfile(color_map_file_name):
        object_color_map=pickle.load(open(color_map_file_name, "rb" ))
    else:
        self.object_color_map=glasbey.get_colors(len(state_id_to_model_pixels))
        pickle.dump(self.object_color_map, open(color_map_file_name, "wb" ))
    
    #create labelled image
    labelled_frame=np.zeros((frame.shape[0], frame.shape[1], 3))
    for label in range(labels.shape[0]):
        object_pixel_positions_exact=np.argwhere(frame==label)
        object_pixel_positions_exact_in_bounds=object_pixel_positions_exact.astype(int)
        if len(object_pixel_positions_exact_in_bounds.shape)==2 and object_pixel_positions_exact_in_bounds.shape[0]>0 and object_pixel_positions_exact_in_bounds.shape[1]==2:
            object_color=object_color_map[label]
            labelled_frame[object_pixel_positions_exact_in_bounds[:, 0], object_pixel_positions_exact_in_bounds[:, 1]]=object_color
                
    if display:
        cv2.imshow('object labels', labelled_frame)
        cv2.waitKey(20)
        
    return labelled_frame

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
def get_bbox(bbx):
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

#transform robot meshes into current position
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

#select voxel points in cube around target object, also compute table surface height
def select_points_in_cube_voxelize_sphr_proj(self, all_points, i, grid_size=128, estimate_table=False, sub_vox=0, min_z=None, unocc=None):
    low=np.array([-0.5,-0.5,-0.5])
    hi=np.array([0.5,0.5,0.5])
    points=all_points[np.argwhere(np.all(np.logical_and(all_points>=low, all_points<=hi), axis=1))][:,0,:]
    
    voxels=np.zeros((grid_size,grid_size,grid_size))
    inds=np.floor((points + 0.5) * grid_size).astype(int)
    if sub_vox!=0:
        inds[:,2]=inds[:,2]-sub_vox/(128/grid_size)
        az_inds=np.argwhere(inds[:,2]>=0)
        inds=inds[az_inds[:,0]]
        
    inds=np.clip(inds, 0, grid_size-1)
    if inds.shape[0]==0:
        if estimate_table:
            return np.zeros((128,128,128)), np.zeros((160,160)), None, 0
        else:
            return np.zeros((128,128,128)), np.zeros((160,160)), None
    voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
    if unocc is not None:
        voxels=np.clip(voxels-unocc, 0, 1)
    
    if estimate_table:
        more_points=all_points[np.argwhere(np.all(np.logical_and(all_points>=np.array([-3,-3,-1]), all_points<=np.array([2,2,min_z+0.01])), axis=1))][:,0,:]

        more_inds=np.floor((more_points + 0.5) * grid_size).astype(int)
        a=more_inds[:,2]
        max_inds=scipy.stats.mode(more_inds[:,2], axis=None)[0][0]
        inds[:,2]=inds[:,2]-max_inds
        az_inds=np.argwhere(inds[:,2]>=0)
        inds=inds[az_inds[:,0]]
        voxels=np.zeros((grid_size,grid_size,grid_size))
        voxels[:,:,0]=1
        voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
    
    no_points=False

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        voxels, spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size))
    mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces, vertex_normals=normals)
    
    trimesh.repair.fix_inversion(mesh)
    if verts.shape[0]>50000:
        mesh.export(f'/dev/shm/temp_mesh_conv_{i}.ply')
        o3d_mesh=o3d.io.read_triangle_mesh(f'/dev/shm/temp_mesh_conv_{i}.ply')
        o3d_mesh=o3d_mesh.simplify_vertex_clustering(0.05)
        mesh=trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles), face_normals=np.asarray(o3d_mesh.triangle_normals), process=False)
        os.remove(f'/dev/shm/temp_mesh_conv_{i}.ply')
    proj=util_sph.proj_spherical(mesh)

    if grid_size!=128:
        full_voxels=np.zeros((128,128,128))
        voxels=np.zeros((128,128,128))
        inds=np.floor((points + 0.5) * 128).astype(int)
        inds=np.clip(inds, 0, 128-1)
        if sub_vox!=0:
            inds[:,2]=inds[:,2]-sub_vox
            az_inds=np.argwhere(inds[:,2]>=0)
            inds=inds[az_inds[:,0]]
        
        full_voxels[inds[:, 0], inds[:, 1], inds[:, 2]] = 1.0
        voxels=full_voxels
    
    if estimate_table:
        return voxels, proj, no_points, max_inds
    else:
        return voxels, proj, no_points

class pose_model_estimator():
    
    def __init__(self, physics, seg_send, seg_receive, recon_send, recon_receive, project_dir, mj_scene_xml, save_id, use_cuda_vox, extrusion_baseline, custom_recon_net=False, max_known_body_id=70, voxels_per_meter=256, model=None, simulate_model_quality=False, model_quality=0, quality_type='', four_channel=False):#
        self.obs_cds=[]
        self.past_filled_voxels=None

        self.top_dir=project_dir
        self.scene_xml_file=mj_scene_xml
        self.save_id=save_id
        self.simulate_model_quality=simulate_model_quality
        self.model_quality=model_quality
        self.quality_type=quality_type
        self.four_channel=four_channel
        self.custom_recon_net=custom_recon_net
        
        self.seg_send=seg_send
        self.seg_receive=seg_receive
        self.recon_send=recon_send
        self.recon_receive=recon_receive
        
        #1: load and voxelize all known meshes        
        mesh_list, self.mesh_name_to_file, self.name_to_scale_dict=get_mesh_list(mj_scene_xml)[:max_known_body_id]
        mesh_list=mesh_list[:69]
        self.included_meshes=[]
        self.geom_names=[]
        self.pred_obj_meshes=[]
        self.use_cuda_vox=use_cuda_vox
        self.extrusion_baseline=extrusion_baseline
        self.upsample=1
        
        self.palm_mesh_verts=None
        
        self.xmap = np.array([[j for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])
        self.geom_ids=[]
        self.ymap = np.array([[i for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])
        for geom_name in model.named.data.geom_xpos.axes.row.names:
            num_voxels=256
            geom_id=model.model.name2id(geom_name, "geom")
            self.geom_ids.append(geom_id)
            if geom_id<71:
                if model.model.geom_dataid[geom_id]>-1:
                    mesh_name=model.model.id2name(model.model.geom_dataid[geom_id], "mesh")
                    mesh=trimesh.load_mesh(self.mesh_name_to_file[model.model.id2name(model.model.geom_dataid[geom_id], "mesh")])
                    mesh_off_trans, mesh_off_rot=compute_mujoco_int_transform(self.mesh_name_to_file[model.model.id2name(model.model.geom_dataid[geom_id], "mesh")], save_id)
                    
                    if geom_name=="herb/wam_1/bhand//unnamed_geom_0":
                        c_mesh=mesh.convex_hull
                        scale=2*np.amax(np.abs(c_mesh.bounds))
                        print('cmesh bounds', c_mesh.bounds)
                        scale_mat=np.eye(4)
                        scale_mat=scale_mat/scale
                        scale_mat[3,3]=1.0
                        s_palm_mesh_vertices=c_mesh.copy().apply_transform(scale_mat)
                        self.palm_mesh_verts=voxel.voxelize_model_binvox(s_palm_mesh_vertices, 32, self.save_id, binvox_add_param='-bb -.5 -.5 -.5 .5 .5 .5', use_cuda_vox=False)
                        a=np.argwhere(self.palm_mesh_verts)
                        self.palm_mesh_verts=(np.argwhere(hollow_dense_pointcloud(self.palm_mesh_verts))/32.0-0.5)*scale
                        self.palm_mesh_verts=self.palm_mesh_verts[np.argwhere(self.palm_mesh_verts[:,2]>0.075)][:,0,:]
                        self.palm_mesh_verts=self.palm_mesh_verts[np.argwhere(np.abs(self.palm_mesh_verts[:,1])<0.025)][:,0,:]
                        self.palm_mesh_verts=self.palm_mesh_verts[np.argwhere(np.abs(self.palm_mesh_verts[:,0])<0.02)][:,0,:]
                        self.palm_mesh_verts=np.matmul(mesh_off_rot.T, (self.palm_mesh_verts-mesh_off_trans).T).T
                    trans_mat=np.eye(4)
                    trans_mat[0:3, 3]=-mesh_off_trans
                    mesh.apply_transform(trans_mat)
                    
                    trans_mat=np.eye(4)
                    trans_mat[0:3, 0:3]=mesh_off_rot.transpose()
                    mesh.apply_transform(trans_mat)
                    self.included_meshes.append(mesh)
                    self.geom_names.append(geom_name)
                elif geom_id<len(mesh_list) and mesh_list[geom_id] is not None:
                    mesh=mesh_list[geom_id]
                    self.included_meshes.append(mesh)
                    self.geom_names.append(geom_name)
        
        #2: transform voxel grids into scene, round into global voxel grid
        #3: transform predicted voxel into global grid, comput intersection with known voxels, return non-intersecting
        #4: marching cubes to convert voxels to mesh
        
        self.camera_params={'fx':579.4112549695428, 'fy':579.4112549695428, 'img_width':640, 'img_height': 480}
        
        self.past_poses=None
        self.past_voxels_scales_translations=None
        self.tracking_max_distance=0.25
        make_known_meshes(self.included_meshes, physics, self.geom_names)
    
    #remove intersections between predicted objects
    def subtract_mesh_hull_no_stability_loss(self, resolve_dense_pointcloud, meshes_sub, cam_mat, translation, cam_pos, scale, inv_cm):
        world_translation=np.matmul(cam_mat, translation)
        
        dense_ptcld_cpy=resolve_dense_pointcloud-cam_pos-world_translation
        dense_ptcld_cpy=np.round((dense_ptcld_cpy/scale+0.5)*128.0).astype(int)
        voxels=np.zeros((128,128,128), dtype=int)
        voxels[dense_ptcld_cpy[:, 0],dense_ptcld_cpy[:, 1],dense_ptcld_cpy[:, 2]]=1
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            voxels, mesh_level, spacing=(1 / 128, 1 / 128, 1 / 128))
        verts=(verts-0.5)*scale
        verts=verts+cam_pos+world_translation

        outside_mesh=np.ones(resolve_dense_pointcloud.shape[0])
        for known_mesh in meshes_sub:
            outside_mesh=np.logical_and(outside_mesh, 1-known_mesh.ray.contains_points(resolve_dense_pointcloud).astype(int))
            pcd_tree = KDTree(known_mesh.vertices)
            gt_pred_nn_dists, gt_pred_nn_inds=pcd_tree.query(resolve_dense_pointcloud)
            outside_mesh=np.where(np.ndarray.flatten(gt_pred_nn_dists)<0.05, 0, outside_mesh)

        dense_ptcld=resolve_dense_pointcloud[np.argwhere(outside_mesh)[:, 0]]
        dense_ptcld=dense_ptcld-cam_pos-world_translation
        dense_ptcld=np.round((dense_ptcld/scale+0.5)*128.0).astype(int)
        voxels=np.zeros((128,128,128), dtype=int)
        voxels[dense_ptcld[:, 0],dense_ptcld[:, 1],dense_ptcld[:, 2]]=1

        if dense_ptcld.shape[0]>0:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                voxels, mesh_level, spacing=(1 / 128, 1 / 128, 1 / 128))
            verts=(verts-0.5)*scale       
        else:
            verts=np.zeros([0,3])
            faces=np.zeros([0,3])
            normals=np.zeros([0,3])
        
        if verts.shape[0]>0:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            trimesh.repair.fix_inversion(mesh)
            mesh.visual.face_colors = [109, 95, 119, 255]
            if mesh.faces.shape[0]>0 and mesh.mass>10e-6:
                combined_mesh=None
                decomps=trimesh.decomposition.convex_decomposition(mesh, maxNumVerticesPerCH=1024, concavity=0.0025, resolution=500000)
                if not isinstance(decomps, list):
                    decomps=[decomps]
                new_decomps=[]
                num_vertices=0
                for decomp in decomps:
                    if decomp.mass>10e-9:
                        new_decomps.append(decomp)
                        num_vertices+=decomp.vertices.shape[0]
                        if combined_mesh==None:
                            combined_mesh=decomp
                        else:
                            combined_mesh+=decomp
                print('num_vertices', num_vertices)
                decomps=new_decomps
                return decomps, combined_mesh
            else:
                return None, None
        else:
            return None, None
    
    def projection_baseline(self, pointcloud):
        projected_pointcloud=np.copy(pointcloud)
        occupied_z_inds=np.argwhere(np.any(pointcloud, axis=2))
        projected_pointcloud[occupied_z_inds[:,0], occupied_z_inds[:,1], 0]=1
        return projected_pointcloud
    
    #remove predicted mesh itnersections with robot and table
    def refine_mesh_no_stability_loss(self, cam_mat, translation, gt_mesh, cam_pos, scale, pred_voxels, inv_cm, known_meshes, refine=False):
        world_translation=np.matmul(cam_mat, translation)
        transform_mesh=gt_mesh.copy()
        transform=np.eye(4)
        transform[:3,3]=-cam_pos-world_translation
        transform_mesh.apply_transform(transform)
        inv_cam_mat=np.linalg.inv(cam_mat)
        transform=np.eye(4)
        transform[:3, :3]=inv_cam_mat
        transform_mesh.apply_transform(transform)
        scale_mat=np.eye(4)
        scale_mat=scale_mat/scale
        scale_mat[3,3]=1.0
        transform_mesh.apply_transform(scale_mat)
        ground_truth_voxels=voxel.voxelize_model_binvox(transform_mesh, 128, self.save_id, binvox_add_param='-bb -.5 -.5 -.5 .5 .5 .5', use_cuda_vox=self.use_cuda_vox)
        mesh_losses={}
        try:          
            gt_points=np.argwhere(ground_truth_voxels)
            pred_voxels=pred_voxels[0]
            if self.simulate_model_quality:   
                pred_voxels, self.cd=self.change_model_quality(pred_voxels, ground_truth_voxels, scale/128.0)
            thres_pred_voxels=pred_voxels>=mesh_level
            thres_pred_points=np.argwhere(thres_pred_voxels)
            
            if refine:
                pcd_tree = KDTree(gt_points)
                pred_gt_nn_dists, pred_gt_nn_inds=pcd_tree.query(thres_pred_points)
                pred_gt_nn_dists=(pred_gt_nn_dists/128.0)*scale
                pred_gt_nn_inds=pred_gt_nn_inds[:,0] 
                
                pcd_tree = KDTree(thres_pred_points)
                gt_pred_nn_dists, gt_pred_nn_inds=pcd_tree.query(gt_points)
                gt_pred_nn_dists=(gt_pred_nn_dists/128.0)*scale
                gt_pred_nn_inds=gt_pred_nn_inds[:,0]

                pg_loss=np.sum(pred_gt_nn_dists)/thres_pred_points.shape[0]
                gp_loss=np.sum(gt_pred_nn_dists)/gt_points.shape[0]
                mesh_losses['chamfer']=pg_loss+gp_loss
        except:
            print('gt voxels projection error!')
            traceback.print_exc()
            thres_pred_points=np.argwhere(pred_voxels>=mesh_level)

        dense_ptcld=(thres_pred_points/128.0-0.5)*scale
        dense_ptcld=dense_ptcld+world_translation+cam_pos

        outside_mesh=np.ones(dense_ptcld.shape[0])
        for known_mesh in known_meshes:
            outside_mesh=np.logical_and(outside_mesh, 1-known_mesh.ray.contains_points(dense_ptcld).astype(int))

        dense_ptcld=dense_ptcld[np.argwhere(outside_mesh)[:, 0]]
        dense_ptcld=dense_ptcld[np.argwhere(dense_ptcld[:,2]>=0.3)[:, 0]]
        resolve_dense_ptcld=np.copy(dense_ptcld)

        dense_ptcld=dense_ptcld-cam_pos-world_translation
        dense_ptcld=np.round((dense_ptcld/scale+0.5)*128.0).astype(int)
        voxels=np.zeros((128,128,128), dtype=int)
        voxels[dense_ptcld[:, 0],dense_ptcld[:, 1],dense_ptcld[:, 2]]=1
        if dense_ptcld.shape[0]>0:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                voxels, mesh_level, spacing=(1 / 128, 1 / 128, 1 / 128))
            verts=(verts-0.5)*scale        
        else:
            verts=np.zeros([0,3])
            faces=np.zeros([0,3])
            normals=np.zeros([0,3])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        trimesh.repair.fix_inversion(mesh)
        try:
            trimesh.repair.fix_inversion(mesh)
        except:
            print('reconstruction error!')
        decomps=[mesh]
        return decomps, mesh_losses, mesh, resolve_dense_ptcld
            
    #main method, reconstruct meshes in image
    def estiamte_poses(self, rbg, depth, physics, cam_pos, cam_mat, pred_object_positions, pred_rotationss, object_labels, task, stability_loss, step=0, gt_mesh=None, use_gt_segs=True, single_threaded=False):
        stability_loss=False
        #preprocess data
        target_id=72
        background_id=0
        table_id=4
        
        rbg_seg_standardized=data_augmentation.standardize_image(rbg)
        xyz_img=compute_xyz(depth, self.camera_params)
        
        temp_scene_xml_file=os.path.join(self.top_dir, f'herb_reconf/temp_scene_{self.save_id}_{step}.xml')
        shutil.copyfile(self.scene_xml_file, temp_scene_xml_file)
        removed_objects=remove_objects_from_mujoco(temp_scene_xml_file, 9)
        removed_mesh_name_to_body_name={removed_object[1]: removed_object[0] for removed_object in removed_objects}
        removed_meshes=[]

        poses=[]
        world_translations=[]
        ind=0
        seg_to_ind={}
        segs=[]
        depths=[]
        rots=[]
        
        target_ind=0
        
        if single_threaded:
            u=0
        else:
            self.seg_send.put([self.save_id, rbg_seg_standardized, xyz_img])
        
        if use_gt_segs: #use gt segmetnations (mujoco sim only)
            camera=Camera(physics=physics, height=480, width=640, camera_id=1)
            object_labels=camera.render(segmentation=True)
            seg_id_to_geom_id={camera.scene.geoms[geom_ind][8]: camera.scene.geoms[geom_ind][3] for geom_ind in range(camera.scene.geoms.shape[0])}
            seg_masks=np.copy(object_labels[:, :, 0])
            for seg_label in np.unique(object_labels[:, :, 0]):
                seg_inds=np.argwhere(object_labels[:, :, 0]==seg_label)
                if seg_inds.shape[0]<100:
                    seg_masks[seg_inds[:,0], seg_inds[:,1]]=0
        else: #use UOIS segs
            seg_masks=self.seg_receive.get(timeout=120)[0]
            
        #remove small/disjoint segmentations
        for seg_label in np.unique(seg_masks):
            if seg_label>0: #don't estiamte background pose
                seg=seg_masks==seg_label
                seg_inds=np.argwhere(seg)
                object_ids=object_labels[seg_inds[:, 0], seg_inds[:,1], 0]
                robot_pix=np.argwhere(np.logical_and(object_ids<target_id, np.logical_and(object_ids!=background_id, object_ids!=table_id)))
                if robot_pix.shape[0]/float(seg_inds.shape[0])<0.5:
                    connected_segs=label(seg, structure=np.array([[1,1,1],[1,1,1],[1,1,1]]))[0]
                    largest_seg=None
                    num_largest_label=0
                    for seg_d in np.unique(connected_segs):
                        if seg_d>0: #no background
                            connected_label=connected_segs==seg_d
                            num_labels=np.sum(connected_label)
                            if num_labels>num_largest_label:
                                num_largest_label=num_labels
                                largest_seg=connected_label
                    if num_largest_label>100:
                        seg_to_ind[ind]=seg_label
                        o_id=stats.mode(object_ids)[0][0]
                        if o_id==72:
                            target_ind=len(segs)
                        
                        if physics.model.id2name(physics.model.geom_dataid[o_id], "mesh") in self.mesh_name_to_file:
                            mesh=trimesh.load_mesh(self.mesh_name_to_file[physics.model.id2name(physics.model.geom_dataid[o_id], "mesh")])
                            
                            scale=self.name_to_scale_dict[physics.model.id2name(physics.model.geom_dataid[o_id], "mesh")]
                            scale_mat=np.eye(4)
                            scale_mat=scale_mat*scale
                            scale_mat[3,3]=1.0
                            mesh.apply_transform(scale_mat)
                            
                            transform=np.eye(4)
                            transform[:3,:3]=np.reshape(physics.named.data.xmat[removed_mesh_name_to_body_name[physics.model.id2name(physics.model.geom_dataid[o_id], "mesh")]], (3,3))
                            mesh.apply_transform(transform)
                              
                            transform=np.eye(4)
                            transform[:3,3]=physics.named.data.xpos[removed_mesh_name_to_body_name[physics.model.id2name(physics.model.geom_dataid[o_id], "mesh")]]
                            mesh.apply_transform(transform)
                            
                            removed_meshes.append(mesh)
                            
                            segs+=[seg]
                            depths+=[1000.0*depth]
                            ind+=1
        
        if len(depths)>0:
            projections=None
            voxels=None
            scales=[]
            translations=[]
            sub_voxes=[]
            accepted_segs=[]
            for sample_ind in range(len(depths)):
                #compute mesh reconstructions
                inputs, scale, translation, rot, sub_vox=self.predict_preprocess(depths[sample_ind], segs[sample_ind], cam_mat, cam_pos, stability_loss=stability_loss)
                if not inputs is None:
                    accepted_segs.append(segs[sample_ind])
                    scales.append(scale)
                    translations.append(translation)
                    rots.append(rot)
                    sub_voxes.append(sub_vox)
                    if projections is None:
                        projections=np.expand_dims(inputs['obs_proj'], 0)
                        voxels=np.expand_dims(inputs['obs_voxels'], 0)
                    else:
                        projections=np.concatenate((projections, np.expand_dims(inputs['obs_proj'], 0)), axis=0)
                        voxels=np.concatenate((voxels, np.expand_dims(inputs['obs_voxels'], 0)), axis=0)
            if not voxels is None:
                if single_threaded:
                    if self.custom_recon_net:
                        pred_voxelss=self.point_completion_model.forward_with_gt_depth((projections, voxels, rbg, depth, accepted_segs, cam_mat, cam_pos))
                    else:
                        pred_voxelss=self.point_completion_model.forward_with_gt_depth(projections, voxels)
                else:
                    self.recon_send.put([self.save_id, projections, voxels, stability_loss])
                    pred_voxelss=self.recon_receive.get(timeout=120)[0]
                for pred_voxels_ind in range(len(pred_voxelss)):
                    if self.extrusion_baseline:
                        grid_inds=np.tile(np.arange(128)[None, None, :], (128, 128, 1))
                        thres_voxels=pred_voxelss[pred_voxels_ind][0]>=mesh_level
                        z_inds=np.argmax(thres_voxels, axis=2).astype(np.int)
                        grid_inds=grid_inds-z_inds[:,:,None]
                        pred_voxelss[pred_voxels_ind][0]=np.where(grid_inds<0, 1, thres_voxels)
                    shifted_voxels=np.zeros((1, 128,128,128))
                    if sub_voxes[pred_voxels_ind]>0:
                        shifted_voxels[:, :,:,sub_voxes[pred_voxels_ind]:]=pred_voxelss[pred_voxels_ind][:,:,:,:-sub_voxes[pred_voxels_ind]]
                    else:
                        shifted_voxels=pred_voxelss[pred_voxels_ind]
                        
                    if self.four_channel: #translate from table
                        pcd = o3d.geometry.PointCloud()
                        a=np.argwhere(shifted_voxels[0]>0.5)
                        pcd.points = o3d.utility.Vector3dVector(np.argwhere(shifted_voxels[0]>0.5).astype(np.float32))
                        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=5)#(nb_neighbors=100, std_ratio=1)#
                        fil_cloud=pcd.select_down_sample(ind)
                        other_pointcloud=np.array(fil_cloud.points).astype(np.int)
                        shifted_voxels[0,:,:,:]=0
                        shifted_voxels[0,other_pointcloud[:,0], other_pointcloud[:,1], other_pointcloud[:,2]]=1
                    
                    pred_voxelss[pred_voxels_ind]=shifted_voxels
            else:
                pred_voxelss=[]
                scales=[]
                translations=[]
        else:
            pred_voxelss=[]
            scales=[]
            translations=[]
             
        known_meshes=make_known_meshes(self.included_meshes, physics, self.geom_names)
        
        inv_cm=np.linalg.inv(cam_mat)
        
        past_voxels_scales_translations=[]
        for ind in range(len(pred_voxelss)):
            past_voxels_scales_translations.append((pred_voxelss[ind], scales[ind], translations[ind]))
            if stability_loss:
                world_translation=np.matmul(cam_mat, translations[ind])
                poses+=[cam_pos+world_translation]
                world_translations+=[world_translation]
            else:   
                world_translation=np.matmul(cam_mat, translations[ind])
                poses+=[cam_pos+world_translation]
                world_translations+=[cam_pos+world_translation]
        
        poses=np.array(poses)
        world_translations=np.array(world_translations)
        
        #mesh tracking (not very important for these experiments
        if self.past_poses is None:
            if task in ['hard_pushing', 'grasping']:
                target_center=np.array([0,0])
            elif task=='easy_pushing':
                target_center=np.array([-0.05, -0.35])
            abs_distances=np.linalg.norm(poses[:, :2]-target_center, axis=1)
            
            sorted_inds=np.argsort(abs_distances)
            min_z_ind=np.argmin(abs_distances)
            sorted_inds=np.ndarray.tolist(sorted_inds)
            sorted_inds.remove(min_z_ind)
            sorted_inds=[min_z_ind]+sorted_inds
            sorted_inds=np.array(sorted_inds)
            poses=poses[sorted_inds]
            self.past_poses=poses
            
            #match colors
            removed_object_poses=[np.array(removed_object_info[2]) for removed_object_info in removed_objects]
            removed_object_colors=np.array([np.array(removed_object_info[3]) for removed_object_info in removed_objects])
            removed_object_poses=np.array(removed_object_poses)
            cost_matrix=np.zeros((removed_object_poses.shape[0], poses.shape[0]))
            for past_pos_ind in range(removed_object_poses.shape[0]):
                for pos_ind in range(poses.shape[0]):
                    cost=np.linalg.norm(poses[pos_ind]-removed_object_poses[past_pos_ind])
                    cost_matrix[past_pos_ind, pos_ind]=cost
            minimum_matching=linear_sum_assignment(cost_matrix)
            removed_object_colors=removed_object_colors[minimum_matching[1]]
        else:
            sorted_inds=np.zeros(max(poses.shape[0], self.past_poses.shape[0]), dtype=np.int)
            cost_matrix=np.zeros((self.past_poses.shape[0], poses.shape[0]))
            for past_pos_ind in range(self.past_poses.shape[0]):
                for pos_ind in range(poses.shape[0]):
                    cost=np.linalg.norm(self.past_poses[past_pos_ind]-poses[pos_ind])
                    if cost<self.tracking_max_distance:
                        cost_matrix[past_pos_ind, pos_ind]=cost
                    else:
                        cost_matrix[past_pos_ind, pos_ind]=1000000000.0
            minimum_matching=linear_sum_assignment(cost_matrix)
            self.past_poses[minimum_matching[0]]=poses[minimum_matching[1]]#np.zeros(poses.shape)
        self.past_voxels_scales_translations=[]
        for ind in range(len(sorted_inds)):
            self.past_voxels_scales_translations.append(past_voxels_scales_translations[sorted_inds[ind]])

        #resolve intersections between predicted meshes and robot/table
        target_body_name=None
        target_geom_names=None
        target_mesh=None
        target_cd=None
        new_objects=[]
        new_meshes=[]
        new_mesh_inds=[]
        mesh_volumes=[]
        resolve_dense_pointclouds=[]
        for ind in range(min(len(pred_voxelss), len(removed_meshes))):
            mesh, cd, whole_mesh, resolve_dense_ptcld=self.refine_mesh_no_stability_loss(cam_mat, translations[sorted_inds[ind]], removed_meshes[sorted_inds[ind]], cam_pos, scales[sorted_inds[ind]], pred_voxelss[sorted_inds[ind]], inv_cm, known_meshes, refine=True)
            for mesh_ind in range(len(mesh)):
                resolve_dense_pointclouds.append(resolve_dense_ptcld)
                if mesh[mesh_ind].faces.shape[0]>=1:
                    mesh_copy=mesh[mesh_ind].copy()
                    transform=np.eye(4)
                    transform[0:3,3]=self.past_poses[ind]
                    mesh_copy.apply_transform(transform)
                    new_meshes.append(mesh_copy)
                    new_mesh_inds.append(ind)
                    mesh_volumes.append(whole_mesh.volume)
                       
            if sorted_inds[ind]==target_ind:
                target_cd=cd
        
        #resolve intersections between predicted meshes
        mesh_volumes=np.array(mesh_volumes)
        mesh_volumes_inds=np.ndarray.tolist(np.argsort(mesh_volumes))
        for new_mesh_ind in reversed(mesh_volumes_inds):
            cur_mesh=new_meshes[new_mesh_ind]
            other_meshes=new_meshes[:new_mesh_ind]
            if new_mesh_ind<len(new_meshes):
                other_meshes+=new_meshes[new_mesh_ind+1:]
            ind=new_mesh_inds[new_mesh_ind]
            cur_mesh, single_mesh=self.subtract_mesh_hull_no_stability_loss(resolve_dense_pointclouds[ind], other_meshes, cam_mat, translations[sorted_inds[ind]], cam_pos, scales[sorted_inds[ind]], inv_cm)
            if cur_mesh!=None:
                body_name, geom_names=add_object_to_mujoco(temp_scene_xml_file, cur_mesh, self.past_poses[ind], os.path.join(self.top_dir, f'herb_reconf/temp_{self.save_id}/'), ind, step, new_objects, joint=True, include_collisions=True, color=removed_object_colors[ind])#
                new_objects+=geom_names
                transform=np.eye(4)
                transform[0:3,3]=self.past_poses[ind]
                single_mesh_c=single_mesh.copy()
                single_mesh_c.apply_transform(transform)
                new_meshes[new_mesh_ind]=single_mesh_c
                if sorted_inds[ind]==target_ind:
                    target_mesh=single_mesh
                    target_body_name=body_name
                    target_geom_names=geom_names
                
        self.obs_cds.append(target_cd)
        
        combined_mesh=target_mesh

        combined_mesh.export(os.path.join(self.top_dir, f'herb_reconf/temp_{self.save_id}/target_mesh.stl'))
        
        self.pred_obj_meshes=[combined_mesh]
        
        color_seg_masks=color_code_objects(seg_masks, self.state_id_to_model_pixels)
        return self.past_poses[0]+self.pred_obj_meshes[0].center_mass, color_seg_masks, temp_scene_xml_file, target_body_name, target_geom_names
    
    def cleanup_files(self):
        if os.path.exists(os.path.join(self.top_dir, f'herb_reconf/temp_{self.save_id}/target_mesh.stl')):
            os.remove(os.path.join(self.top_dir, f'herb_reconf/temp_{self.save_id}/target_mesh.stl'))
        if os.path.exists(os.path.join(self.top_dir, f'herb_reconf/temp_{self.save_id}')):
            shutil.rmtree(os.path.join(self.top_dir, f'herb_reconf/temp_{self.save_id}'))
        if os.path.exists(os.path.join(self.top_dir, f'herb_reconf/temp_scene_{self.save_id}_{0}.xml')):
            os.remove(os.path.join(self.top_dir, f'herb_reconf/temp_scene_{self.save_id}_{0}.xml'))

    def convert_to_float32(self, sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)

    def make_o3d_pcd(self, points, color):
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points)
        pcd2.paint_uniform_color(np.array(color))
        return pcd2

    #preprocess images into four channel representation
    def predict_preprocess(self, depth, label, cam_mat, cam_pos, stability_loss=False):
        obj_points_inds=np.where(label, depth, 0.0).flatten().nonzero()[0]
        other_points_inds=np.argwhere(np.where(label, depth, 0.0).flatten()==0)[:,0]
        obs_ptcld=make_pointcloud_all_points(depth)
        obs_ptcld=obs_ptcld/1000.0
        
        obj_pointcloud=obs_ptcld[obj_points_inds]
        other_pointcloud=obs_ptcld[other_points_inds]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pointcloud)
        pcd.paint_uniform_color(np.array([0,1,0]))
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)
        obj_pointcloud=np.array(pcd.select_down_sample(ind).points)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(other_pointcloud)
        pcd.paint_uniform_color(np.array([0,1,0]))
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.5)
        other_pointcloud=np.array(pcd.select_down_sample(ind).points)
        inv_proj_mat=np.reshape(cam_mat, (3,3))
        proj_mat=np.linalg.inv(inv_proj_mat)
        other_pointcloud=inv_proj_mat.dot(other_pointcloud.T).T

        other_pointcloud=proj_mat.dot(other_pointcloud.T).T
        rot=None
        ground_plane=[]
        for x in range(100):
            for y in range(100):
                ground_plane.append([x/100.0,y/100.0,0.35])
        ground_plane=np.array(ground_plane)

        obs_ptcld_min=np.amin(obj_pointcloud, axis=0)
        obs_ptcld_max=np.amax(obj_pointcloud, axis=0)
        scale=4.0*float(np.max(obs_ptcld_max-obs_ptcld_min))
        translation=np.mean(obj_pointcloud, axis=0)

        obj_pointcloud=(obj_pointcloud-translation)/scale
        other_pointcloud=(other_pointcloud-translation)/scale
        
        
        obj_pointcloud=inv_proj_mat.dot(obj_pointcloud.T).T
        other_pointcloud=inv_proj_mat.dot(other_pointcloud.T).T

        if self.four_channel:
            min_object_z=np.min(obj_pointcloud[:,2])
            occupied_voxels, occupied_proj, no_points, sub_vox=self.select_points_in_cube_voxelize_sphr_proj(other_pointcloud, self.save_id, grid_size=128, estimate_table=True, min_z=min_object_z)
            if occupied_voxels is None:
                return None, None, None, None, None
            obj_voxels, obs_proj, _=self.select_points_in_cube_voxelize_sphr_proj(obj_pointcloud, self.save_id, grid_size=128, sub_vox=sub_vox)
            if obj_voxels is None:
                return None, None, None, None, None
        
            line_points=np.arange(50,200)/200.0
            near_obs_ptcld=obs_ptcld[np.argwhere(np.logical_and(np.logical_and(obs_ptcld[:,0]>=translation[0]-0.5*scale, np.logical_and(obs_ptcld[:,1]>=translation[1]-0.5*scale, obs_ptcld[:,2]>=translation[2]-0.5*scale)),
                                                                np.logical_and(obs_ptcld[:,0]<=translation[0]+0.5*scale, np.logical_and(obs_ptcld[:,1]<=translation[1]+0.5*scale, obs_ptcld[:,2]<=translation[0]+0.5*scale))))][:,0,:]
            unoccupied_points=np.reshape(near_obs_ptcld[:, None, :]*line_points[:, None], (-1, 3))
            empty_points=(unoccupied_points-translation)/scale
            empty_points=inv_proj_mat.dot(empty_points.T).T
            empty_voxels, empty_proj, _=self.select_points_in_cube_voxelize_sphr_proj(empty_points, self.save_id, grid_size=128, sub_vox=sub_vox) 
            
            away_line_points=1+np.arange(1,200)/100.0
            occupied_points=np.reshape(near_obs_ptcld[:, None, :]*away_line_points[:, None], (-1, 3))
            unk_points=(occupied_points-translation)/scale
            unk_points=inv_proj_mat.dot(unk_points.T).T
            unk_voxels, unk_proj, _=self.select_points_in_cube_voxelize_sphr_proj(unk_points, self.save_id, grid_size=128, sub_vox=sub_vox, unocc=np.logical_or(occupied_voxels,np.logical_or(obj_voxels, empty_voxels))) 

            sample_loaded={}
            sample_loaded['obs_voxels']=np.concatenate((obj_voxels[None,:,:,:], occupied_voxels[None,:,:,:], unk_voxels[None,:,:,:]), axis=0).astype(np.float32)
            sample_loaded['obs_proj']=np.concatenate((obs_proj[0], occupied_proj[0], unk_proj[0]), axis=0).astype(np.float32)
        else:
            sub_vox=0
            occupied_voxels, occupied_proj, no_points=self.select_points_in_cube_voxelize_sphr_proj(other_pointcloud, self.save_id, grid_size=128)
            if occupied_voxels is None:
                return None, None, None, None, None
            obj_voxels, obs_proj, _=self.select_points_in_cube_voxelize_sphr_proj(obj_pointcloud, self.save_id, grid_size=128)
            if obj_voxels is None:
                return None, None, None, None, None

            sample_loaded={}
            sample_loaded['obs_voxels']=obj_voxels[None,:,:,:].astype(np.float32)
            sample_loaded['obs_proj']=obs_proj.detach().cpu().numpy()[0]
        
        self.convert_to_float32(sample_loaded)

        return sample_loaded, scale, translation, rot, sub_vox
