import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.morphology import binary_erosion
from genre.models.netinterface import data_parallel_decorator
from genre.models.depth_pred_with_sph_inpaint import Net as Depth_inpaint_net
from genre.models.depth_pred_with_sph_inpaint import Model as DepthInpaintModel
from genre.networks.networks import Unet_3D
from genre.toolbox.cam_bp.cam_bp.modules.camera_backprojection_module import Camera_back_projection_layer
from genre.toolbox.cam_bp.cam_bp.functions import SphericalBackProjection
from genre.toolbox.spherical_proj import gen_sph_grid
from os import makedirs
from os.path import join
from genre.util import util_img
from genre.util import util_sph
import torch.nn.functional as F
from genre.toolbox.spherical_proj import sph_pad
from scipy.misc import imresize
import cv2
import trimesh
from skimage import measure
import open3d as o3d
import time
from torch.autograd import Function
import math
import multiprocessing as mp
import scipy
from sklearn.neighbors import KDTree
import cc3d

#stability loss computations
class Log_Stability_Loss(Function):

    @staticmethod
    def forward(ctx, pred, scale):
        obs_unk=pred[:,2] #unknown voxels
        obs_other=pred[:,1:2] #voxels of other objects
        pred=pred[:,0:1] #predicted voxels of target object
        
        pred=torch.nn.functional.sigmoid(pred)
        pred_obj_positions=pred[:,0]
        pred_positions=torch.from_numpy(np.argwhere(np.zeros(pred_obj_positions.shape)>-float('inf'))).float().cuda()
        pred_positions=pred_positions.view(pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4], 4)
        center=torch.zeros(2).cuda()
        center[0]=(pred.shape[2]-1)/2.0
        center[1]=(pred.shape[3]-1)/2.0
        
        e_pred=pred_obj_positions.view(pred_obj_positions.shape[0], pred_obj_positions.shape[1], pred_obj_positions.shape[2], pred_obj_positions.shape[3], 1)
        weighted_pred_positions=(pred_positions)*e_pred
        cdf_mean=torch.sum(weighted_pred_positions.view(pred.shape[0], -1, 4), 1)/torch.sum(pred_obj_positions, (1,2,3))[:, None]
        
        loss_derivatives=torch.zeros((pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4])).cuda()
        stability_loss=0
        num_directions=25
        
        #for each direction
        for direction_ind in range(num_directions):
            #compute normals to direction
            normal_theta=(2.0*math.pi*direction_ind)/num_directions
            normal_x=math.cos(normal_theta)
            normal_y=math.sin(normal_theta)
            normal_vec=torch.zeros(2).cuda()
            normal_vec[0]=normal_x
            normal_vec[1]=normal_y
            
            cam_start_point=torch.zeros(2).cuda()
            cam_start_point[0]=10000*normal_y
            cam_start_point[1]=-10000*normal_x
            
            #project voxels along direction
            t=normal_vec[0]*(pred_positions[:, :, :, :, 1]-63)+normal_vec[1]*(pred_positions[:, :, :, :, 2]-63)
            projected_positions=torch.zeros((pred_positions.shape[0],pred_positions.shape[1],pred_positions.shape[2],pred_positions.shape[3],2)).cuda()
            projected_positions[:, :, :, :, 0]=pred_positions[:, :, :, :, 1]-t*normal_vec[0]
            projected_positions[:, :, :, :, 1]=pred_positions[:, :, :, :, 2]-t*normal_vec[1]
            projected_positions=torch.pow(torch.pow(projected_positions[:, :, :, :, 0]-cam_start_point[0], 2)+torch.pow(projected_positions[:, :, :, :, 1]-cam_start_point[1], 2), 0.5)#-projected_center
            
            #project cdf function along direction
            t=normal_x*(cdf_mean[:, 1]-63)+normal_y*(cdf_mean[:, 2]-63)
            projected_mean=torch.zeros(cdf_mean.shape[0], 2).cuda()
            projected_mean[:, 0]=cdf_mean[:, 1]-t*normal_vec[0]
            projected_mean[:, 1]=cdf_mean[:, 2]-t*normal_vec[1]
            projected_mean=torch.pow(torch.pow(projected_mean[:, 0]-cam_start_point[0], 2.0)+torch.pow(projected_mean[:, 1]-cam_start_point[1], 2.0), 0.5)
            
            #compute cdf at each projected voxel
            diff_reshape=projected_positions-projected_mean[:, None, None, None]
            e_pred_reshape=e_pred[:, :, :, :, :].view(e_pred.shape[0], e_pred.shape[1], e_pred.shape[2], e_pred.shape[3])
            cdf_std_dir=torch.pow(torch.sum(diff_reshape*diff_reshape*e_pred_reshape*(1.0-e_pred_reshape), (1,2,3)), 0.5)/torch.sum(e_pred[:, :, :, :, :], (1,2,3,4))
            com_dist=torch.distributions.normal.Normal(projected_mean[:, None, None, None], (cdf_std_dir+0.001)[:, None, None, None])
            cdfs=com_dist.cdf(projected_positions[:,:,:,:])
            
            #compute supporting voxels
            neighbor_occupied=torch.zeros(pred[:,0].shape).cuda()
            neighbor_occupied[:,:,:,1:]+=(obs_other[:,0,:,:,:-1]>0.5).type(torch.cuda.FloatTensor)
            if normal_x>0:
                neighbor_occupied[:,:-1,:,:]+=(obs_other[:,0,1:,:,:]>0.5).type(torch.cuda.FloatTensor)
            elif normal_x<0:
                neighbor_occupied[:,1:,:,:]+=(obs_other[:,0,:-1,:,:]>0.5).type(torch.cuda.FloatTensor)
            if normal_y>0:
                neighbor_occupied[:,:,:-1,:]+=(obs_other[:,0,:,1:,:]>0.5).type(torch.cuda.FloatTensor)
            elif normal_y<0:
                neighbor_occupied[:,:,1:,:]+=(obs_other[:,0,:,:-1,:]>0.5).type(torch.cuda.FloatTensor)
            neighbor_occupied=((neighbor_occupied>0)*(1-obs_other[:,0,:,:,:]>0.5)).type(torch.cuda.FloatTensor)

            #compute stability loss and derivative
            lse=torch.exp(torch.sum(torch.log(1.0+1e-6-cdfs*((pred[:,0,:,:,:]>0.5).type(torch.cuda.FloatTensor)*neighbor_occupied*obs_unk)), (1,2,3)))
            lse=torch.clamp(lse, 0, 1)
            loss_derivatives+=-obs_unk*(neighbor_occupied.type(torch.cuda.FloatTensor))*cdfs*(lse[:,None,None,None]/(1.01-pred[:,0,:,:,:]*neighbor_occupied*cdfs))/(1.01-lse[:,None,None,None])
            
            stability_loss+=-torch.log(1.0+1e-6-lse)
        
        #clip for (training) stability
        loss_derivatives=torch.clamp(loss_derivatives, -1, 1)
        
        stability_loss=stability_loss*scale
        loss_derivatives=loss_derivatives*scale  

        nloss_derivatives=loss_derivatives.detach().cpu().numpy()
        ctx.save_for_backward(loss_derivatives)
        return stability_loss

    @staticmethod
    def backward(ctx, grad_output):
        loss_derivatives=ctx.saved_tensors[0]
        derivatives=torch.zeros((loss_derivatives.shape[0],3,128,128,128)).cuda()
        derivatives[:,0,:,:,:]=loss_derivatives
        return derivatives, torch.zeros(1).float().cuda()
    
def node_num_to_pos(node_num, side_len):
    pos=np.zeros(3, dtype=np.int)
    pos[2]=node_num//side_len**2
    pos[1]=(node_num%side_len**2)//side_len
    pos[0]=node_num%side_len
    return pos
    
def get_comps(components, coarsening_factor, vert_pos):
    vert_pos=(vert_pos*coarsening_factor).astype(int)
    comps=np.unique(components[vert_pos[0]:vert_pos[0]+coarsening_factor, vert_pos[1]:vert_pos[1]+coarsening_factor, vert_pos[2]:vert_pos[2]+coarsening_factor])
    return comps

#multithreaded connectivity loss computation sub-method
def connectivity_on_batch(b_npred, coarsening_factor, vert_nums, b_np_coarse_preds, node_num_to_vert_nums, max_np_course_preds, unk_vox):
    #don't connect very small components
    components=cc3d.connected_components(b_npred[0]>=0.5, connectivity=6)
    counts=np.bincount(np.ndarray.flatten(components))
    new_probs=np.clip(b_npred[0], 0, 0.49)
    large_comps=np.argwhere(counts[1:]>=25)[:,0]+1
    new_probs=np.where(np.isin(components, large_comps), b_npred, new_probs)
    b_npred[0]=new_probs
    where_verts_present=np.argwhere(b_npred[0]>=0.5)
    comp_labels=components[where_verts_present[:, 0], where_verts_present[:, 1], where_verts_present[:, 2]]
    
    if comp_labels.shape[0]==0 or np.amin(comp_labels)==np.amax(comp_labels):
        return 0, np.zeros((b_npred.shape[1], b_npred.shape[2], b_npred.shape[3]))
    
    where_verts_present=np.floor(where_verts_present/coarsening_factor).astype(np.int64)
    where_verts_present=vert_nums[where_verts_present[:, 0], where_verts_present[:, 1], where_verts_present[:, 2]]
    where_verts_present, unique_inds=np.unique(where_verts_present, return_index=True)
    comp_labels=comp_labels[unique_inds]
    
    if where_verts_present.shape[0]>128:
        where_verts_present=where_verts_present[:128]
        comp_labels=comp_labels[:128]
    
    #compute adjecency matrix
    adjacency_matrix=np.zeros((b_np_coarse_preds.shape[1]*b_np_coarse_preds.shape[2]*b_np_coarse_preds.shape[3], b_np_coarse_preds.shape[1]*b_np_coarse_preds.shape[2]*b_np_coarse_preds.shape[3]))
    for x in range(b_np_coarse_preds.shape[1]):
        for y in range(b_np_coarse_preds.shape[2]):
            for z in range(b_np_coarse_preds.shape[3]):
                adjacency_matrix[vert_nums[x,y,z], vert_nums[max(x-1, 0):min(x+2, b_np_coarse_preds.shape[1]),max(y-1, 0):min(y+2, b_np_coarse_preds.shape[2]),max(z-1, 0):min(z+2, b_np_coarse_preds.shape[3])]]=b_np_coarse_preds[0, max(x-1, 0):min(x+2, b_np_coarse_preds.shape[1]),max(y-1, 0):min(y+2, b_np_coarse_preds.shape[2]),max(z-1, 0):min(z+2, b_np_coarse_preds.shape[3])]
                adjacency_matrix[vert_nums[x,y,z], vert_nums[x,y,z]]=0
     
    #compute all pairs most (log) probable paths
    adjacency_matrix=np.where(adjacency_matrix<=0, 10e-100, adjacency_matrix)
    adjacency_matrix=-np.log(adjacency_matrix)
    dists, paths=scipy.sparse.csgraph.dijkstra(adjacency_matrix, return_predecessors=True, limit=10, indices=where_verts_present)#, indices=np.arange(0, 100, dtype=np.int64)

    probs=np.zeros((b_np_coarse_preds.shape[1]*b_np_coarse_preds.shape[2]*b_np_coarse_preds.shape[3]))
    for x in range(b_np_coarse_preds.shape[1]):
        for y in range(b_np_coarse_preds.shape[2]):
            for z in range(b_np_coarse_preds.shape[3]):
                probs[vert_nums[x,y,z]]=b_np_coarse_preds[0,x,y,x]
    
    dists=np.exp(-dists)

    path_pairs=np.argwhere(np.zeros((where_verts_present.shape[0], where_verts_present.shape[0]))>-float('inf'))
    both_not_same_comp=np.argwhere(comp_labels[path_pairs[:, 0]]!=comp_labels[path_pairs[:, 1]])
             
    pair_both_not_base_inds=np.argwhere(both_not_same_comp)
    path_pairs=path_pairs[pair_both_not_base_inds[:, 0]]
    
    #compute connectivity loss and derivatives
    probs=max_np_course_preds[0, node_num_to_vert_nums[where_verts_present, 0], node_num_to_vert_nums[where_verts_present, 1], node_num_to_vert_nums[where_verts_present, 2]][:, None]
    pair_probs=np.matmul(probs, probs.T)
    pair_probs=pair_probs[path_pairs[:, 0], path_pairs[:, 1]]
    w_dists=np.copy(dists)
    w_dists_3D_1=np.transpose(w_dists[None, :, :], (2, 1, 0))
    w_dists_3D_2=np.transpose(w_dists_3D_1, (0,2,1))
    shortest_paths_through=np.matmul(w_dists_3D_1, w_dists_3D_2)
    shortest_paths_through=shortest_paths_through[:, path_pairs[:, 0], path_pairs[:, 1]]
    e_connected=np.copy(shortest_paths_through)
    
    ww_dists=dists[path_pairs[:, 0], where_verts_present[path_pairs[:, 1]]]
    shortest_paths_through=np.where(shortest_paths_through!=ww_dists, shortest_paths_through*(1-ww_dists), shortest_paths_through)
    shortest_paths_through=shortest_paths_through/b_np_coarse_preds[0, node_num_to_vert_nums[:, 0], node_num_to_vert_nums[:, 1], node_num_to_vert_nums[:, 2]][:, None]
    shortest_paths_through=shortest_paths_through*pair_probs
    
    e_connected=np.where(e_connected!=ww_dists, 1.0-(1.0-e_connected)*(1.0-ww_dists), e_connected)
    
    loss=pair_probs*e_connected+1.0-pair_probs
    derivatives=shortest_paths_through/(loss)
    derivatives=np.sum(derivatives, axis=1)
    tcoarse_derivatives=derivatives[vert_nums]
    
    full_derivatives=np.zeros((b_npred.shape[1], b_npred.shape[2], b_npred.shape[3]))
    for x in range(tcoarse_derivatives.shape[0]):
        for y in range(tcoarse_derivatives.shape[1]):
            for z in range(tcoarse_derivatives.shape[2]):
                full_derivatives[coarsening_factor*x:coarsening_factor*(x+1), coarsening_factor*y:coarsening_factor*(y+1), coarsening_factor*z:coarsening_factor*(z+1)]=tcoarse_derivatives[x,y,z]    
    
    full_derivatives=full_derivatives*unk_vox
    
    return np.sum(loss), full_derivatives

#multithreaded connectivity loss main methods
class Connectivity_Loss(Function):

    @staticmethod
    def forward(ctx, pred, scale):
        
        unk_vox=pred[:,1]
        pred_vox=pred[:,0:1]
        pred_vox=pred_vox.view(pred_vox.shape[0], 1, pred_vox.shape[2], pred_vox.shape[3], pred_vox.shape[4])
        
        pred_vox=torch.nn.functional.sigmoid(pred_vox)
    
        unk_vox=unk_vox.cpu().detach().numpy()
        
        #coarsen predictions
        coarsening_factor=8
        
        weights=torch.ones(1, 1, coarsening_factor, coarsening_factor, coarsening_factor)/float(coarsening_factor**3)
        weights=weights.cuda()
        coarse_preds=F.conv3d(pred_vox, weights, stride=weights.shape[2])
        np_coarse_preds=coarse_preds.cpu().detach().numpy()
        np_coarse_preds=np_coarse_preds.astype(np.float64)
        max_np_course_preds=np.zeros(np_coarse_preds.shape)
        pred_vox=pred_vox.cpu().detach().numpy()
    
        full_path_probs=np.zeros((pred_vox.shape[2], pred_vox.shape[3], pred_vox.shape[4]))
        for batch_ind in range(pred_vox.shape[0]):
            for x in range(np_coarse_preds.shape[2]):
                for y in range(np_coarse_preds.shape[3]):
                    for z in range(np_coarse_preds.shape[4]):
                        max_np_course_preds[batch_ind, 0, x,y,z]=np.amax(pred_vox[batch_ind, 0, coarsening_factor*x:coarsening_factor*(x+1), coarsening_factor*y:coarsening_factor*(y+1), coarsening_factor*z:coarsening_factor*(z+1)])

        pred_positions=np.argwhere(np.zeros(np_coarse_preds[:, 0, :, :, :].shape)>-float('inf'))
        pred_positions=pred_positions.reshape(coarse_preds.shape[0], coarse_preds.shape[2], coarse_preds.shape[3], coarse_preds.shape[4], 4)
        vert_nums=pred_positions[0,:,:,:,1]+pred_positions.shape[1]*(pred_positions[0,:,:,:,2]+pred_positions.shape[1]*pred_positions[0,:,:,:,3])
        node_nums=np.arange(0, np_coarse_preds.shape[2]*np_coarse_preds.shape[3]*np_coarse_preds.shape[4])
        node_num_to_vert_nums=np.zeros((node_nums.shape[0], 3), dtype=np.int64)
        node_num_to_vert_nums[:, 2]=np.floor(node_nums/pred_positions.shape[1]**2)
        node_num_to_vert_nums[:, 1]=np.floor(np.mod(node_nums, pred_positions.shape[1]**2)/pred_positions.shape[1])
        node_num_to_vert_nums[:, 0]=np.mod(node_nums, pred_positions.shape[1])
        
        loss=torch.zeros(pred_vox.shape[0], 1).cuda()
        derivatives=np.zeros(pred_vox.shape)
        
        #setup and call multithreaded sub-methods
        args_list=[]
        for batch_ind in range(np_coarse_preds.shape[0]):
            args_list.append((pred_vox[batch_ind], coarsening_factor, vert_nums, np_coarse_preds[batch_ind], node_num_to_vert_nums, max_np_course_preds[batch_ind], unk_vox[batch_ind]))
        
        pool = mp.Pool(processes=np_coarse_preds.shape[0], maxtasksperchild=1)
        parallel_runs = [pool.apply_async(connectivity_on_batch, args=(args_list[i])) for i in range(np_coarse_preds.shape[0])]
        results = [p.get(timeout=100000) for p in parallel_runs]
        pool.terminate()
          
        for batch_ind in range(len(results)):
            loss[batch_ind, 0]=results[batch_ind][0]
            derivatives[batch_ind, 0]=results[batch_ind][1]
            
        derivatives=torch.from_numpy(derivatives).float().cuda()

        derivatives=torch.clamp(derivatives, -1, 1)
        derivatives=derivatives*scale
        loss=loss*scale
        
        ctx.save_for_backward(derivatives)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        loss_derivatives=ctx.saved_tensors[0]
        derivatives=torch.zeros((loss_derivatives.shape[0], 2,128,128,128))
        derivatives[:,:,:,:,:]=loss_derivatives[:,:,:,:,:]
        
        return derivatives.cuda(), torch.zeros(1).float().cuda()

def chamfer_distance(pcd_1, pcd_2):
    if pcd_1.shape[0]==0:
        print('pcd 1 empty!')
        if pcd_2.shape[0]>0:
            return np.mean(np.linalg.norm(pcd_2, axis=1))
        else:
            return 0
    if pcd_2.shape[0]==0:
        print('pcd 2 empty!')
        return 0
    
    pcd_tree = KDTree(pcd_2)
    nearest_distances_1, _=pcd_tree.query(pcd_1)
    
    pcd_tree = KDTree(pcd_1)
    nearest_distances_2, _=pcd_tree.query(pcd_2)
    
    return np.sum(nearest_distances_1)/pcd_1.shape[0]+np.sum(nearest_distances_2)/pcd_2.shape[0]

class Model(DepthInpaintModel):
    @classmethod
    def add_arguments(cls, parser):
        parser, unique_params = DepthInpaintModel.add_arguments(parser)
        parser.add_argument('--inpaint_path', default=None, type=str,
                            help="path to pretrained inpainting module")
        parser.add_argument('--surface_weight', default=1.0, type=float,
                            help="weight for voxel surface prediction")
        unique_params_model = {'surface_weight', 'joint_train', 'inpaint_path'}
        return parser, unique_params.union(unique_params_model)

    def __init__(self, opt, logger):
        super(Model, self).__init__(opt, logger)
        self.joint_train = opt.joint_train
        self.requires=['silhou', 'depth', 'voxel', 'rgb']
        self.gt_names=['gt_voxels', 'gt_proj_sphr_img', 'scale', 'occlusion', 'obs_other', 'unknown_voxels']
        self._metrics += ['loss', 'voxel_loss', 'surface_loss', 'stability_loss']
        self.net = data_parallel_decorator(Net)(opt, Model)#
        self.optimizer = self.adam(
            self.net.parameters(),
            lr=opt.lr,
            **self.optim_params
        )
        self.cd_occlusion=[]
        self.stability_loss=opt.stability_loss
        self.connectivity_loss=opt.connectivity_loss
        
        print("Using", torch.cuda.device_count(), "GPUs!")
        self.compute_chamfer_dist=opt.compute_chamfer_dist
        self._nets = [self.net]
        self._optimizers = [self.optimizer]
        self.init_vars(add_path=True)
        self.requires=['silhou', 'depth', 'voxel', 'rgb']
        if not self.joint_train:
            self.init_weight(self.net.refine_net)
        self.upsample=1
        self.xmap = np.array([[j for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])
        self.ymap = np.array([[i for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])

    def __str__(self):
        string = "Full model of GenRe."
        if self.joint_train:
            string += ' Jointly training all the modules.'
        else:
            string += ' Only training the refinement module'
        return string

    #@profile
    def compute_loss(self, pred):
        loss = 0
        loss_data = {}
        if self.joint_train:
            loss, loss_data = super(Model, self).compute_loss(pred)

        voxel_loss = F.binary_cross_entropy_with_logits(pred['pred_voxel'], self._gt.gt_voxels[:,0:1])
        sigmoid_voxel = torch.sigmoid(pred['pred_voxel'])
        surface_loss = F.binary_cross_entropy(sigmoid_voxel * self._gt.gt_voxels[:,0:1], self._gt.gt_voxels[:,0:1])
        
        if self.compute_chamfer_dist:
            np_pred_voxel=sigmoid_voxel.detach().cpu().numpy()
            np_gt_voxels=self._gt.gt_voxels.detach().cpu().numpy()
            chamfer_loss=0
            for batch_ind in range(self._gt.gt_voxels.shape[0]):
                scale=float(self._gt.scale[batch_ind].detach().cpu().numpy())
                np_pred_points=(np.argwhere(np_pred_voxel[batch_ind,0]>=0.5)/128.0-0.5)*scale
                np_gt_points=(np.argwhere(np_gt_voxels[batch_ind,0])/128.0-0.5)*scale
                if np_gt_points.shape[0]==0:
                    print('no gt points', batch_ind)
                cd=chamfer_distance(np_pred_points, np_gt_points)
                chamfer_loss+=cd
                self.cd_occlusion.append((self._gt.occlusion.detach().cpu().numpy()[batch_ind], cd))
            loss_data['chamfer_dist']=chamfer_loss/self._gt.gt_voxels.shape[0]
        
        loss += voxel_loss.mean()
        loss += surface_loss.mean() * self.opt.surface_weight
        
        loss_data['voxel_loss'] = voxel_loss.mean().item()
        loss_data['surface_loss'] = surface_loss.mean().item() * self.opt.surface_weight
        
        #stability loss call
        if self.stability_loss>0:
            scale=torch.tensor(np.array([self.stability_loss])).float().cuda()
            scale.requires_grad=False
            stability_loss=Log_Stability_Loss.apply(torch.cat((pred['pred_voxel'][:,0:1], self._gt.obs_other, self._gt.unknown_voxels),1), scale)
            loss += stability_loss.mean()
            loss_data['stability_loss']=stability_loss.mean().item()
        
        #connectivity loss call
        if self.connectivity_loss>0:
            scale=torch.tensor(np.array([self.connectivity_loss])).float().cuda()
            scale.requires_grad=False
            connectivity_loss=Connectivity_Loss.apply(torch.cat((pred['pred_voxel'][:,0:1], self._gt.unknown_voxels),1), scale)
            loss += connectivity_loss.mean()      
            loss_data['connectivity_loss']=connectivity_loss.mean().item()
            

        loss_data['loss'] = loss.mean().item()
        return loss, loss_data

    def pack_output(self, pred, batch, add_gt=True):
        pack = {}
        if self.joint_train:
            pack = super(Model, self).pack_output(pred, batch, add_gt=add_gt)
        pack['pred_voxel'] = pred['pred_voxel'].cpu().numpy()
        pack['pred_proj_sph_partial'] = pred['pred_voxel'].cpu().numpy()
        pack['pred_proj_sph_full'] = pred['pred_proj_sph_full'].cpu().numpy()
        if add_gt:
            pack['gt_voxel'] = batch['gt_voxels'].numpy()
        return pack

    @classmethod
    def preprocess(cls, data, mode='train'):
        dataout = DepthInpaintModel.preprocess(data, mode)
        if 'voxel' in dataout:
            val = dataout['voxel'][0, :, :, :]
            val = np.transpose(val, (0, 2, 1))
            val = np.flip(val, 2)
            voxel_surface = val - binary_erosion(val, structure=np.ones((3, 3, 3)), iterations=2).astype(float)
            voxel_surface = voxel_surface[None, ...]
            voxel_surface = np.clip(voxel_surface, 0, 1)
            dataout['voxel'] = voxel_surface
        return dataout
    
    #@profile
    def forward_with_gt_depth(self, inputs, scale, translation):
        self.eval()
        with torch.no_grad():
            inputs=[torch.from_numpy(np.expand_dims(inputs['obs_proj'], 0)).repeat(8, 1, 1, 1).float().cuda(), torch.from_numpy(np.expand_dims(inputs['obs_voxels'], 0)).repeat(8, 1, 1, 1, 1).float().cuda()]
            preds=self.net.forward(inputs)
    
            np_pred_voxel=preds['pred_voxel'].detach().cpu().numpy()
            np_pred_voxel=1 / (1 + np.exp(-np_pred_voxel))
            np_pred_voxel=np.where(np_pred_voxel>=0.5, 1, 0)
        
        return np_pred_voxel, scale, translation, inputs['obs_voxels']
    
    

    def make_pointcloud_densefusion(self, depth_image):
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
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        fil_cloud=pcd.select_down_sample(ind)
        fil_pcd = o3d.geometry.PointCloud()
        fil_pcd.points = o3d.utility.Vector3dVector(cloud-np.mean(fil_cloud, axis=0))
        o3d.visualization.draw_geometries([fil_cloud])
        return np.asarray(fil_cloud.points)

    def convert_to_float32(self, sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)

    def predict_preprocess(self, depth, label):
        obs_ptcld=self.make_pointcloud_densefusion(np.where(label, depth, 0.0))
        obs_ptcld=obs_ptcld/1000.0
        
        translation=np.mean(obs_ptcld, axis=0)
        obs_ptcld=obs_ptcld-translation
         
        obs_ptcld_min=np.amin(obs_ptcld, axis=0)
        obs_ptcld_max=np.amax(obs_ptcld, axis=0)
        scale=3.0*float(np.max(obs_ptcld_max-obs_ptcld_min))
        obs_ptcld=obs_ptcld/scale
        
        obs_ptcld_min=np.amin(obs_ptcld, axis=0)
        obs_ptcld_max=np.amax(obs_ptcld, axis=0)
        
        grid_size=128
        tdf = np.zeros([grid_size, grid_size, grid_size]) / grid_size
        cnt = np.zeros([grid_size, grid_size, grid_size])
        
        inds2=np.floor((obs_ptcld + 0.5) * grid_size).astype(int)
        tdf[inds2[:, 0], inds2[:, 1], inds2[:, 2]] = 1.0

        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
            tdf, spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size))
            obs_mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
            #obs_mesh.show()
            obs_proj=util_sph.proj_spherical(obs_mesh)
        except:
            obs_proj=np.zeros((1,160,160))
            print('marching cubes error!')
        obs_voxels=tdf
                 
        sample_loaded={}
        sample_loaded['obs_voxels']=np.expand_dims(obs_voxels, 0)
        sample_loaded['obs_proj']=obs_proj.detach().cpu().numpy()[0]
        
        self.convert_to_float32(sample_loaded)

        return sample_loaded, scale, translation


class Net(nn.Module):
    def __init__(self, opt, base_class):
        super().__init__()
        self.base_class = base_class
        self.depth_and_inpaint = Depth_inpaint_net(opt, base_class, in_channels=3, out_channels=2)
        self.refine_net = Unet_3D(in_channel=5, out_channel=1)
        self.proj_depth = Camera_back_projection_layer()
        self.joint_train = opt.joint_train
        self.register_buffer('grid', gen_sph_grid())
        self.grid = self.grid.expand(1, -1, -1, -1, -1)
        self.proj_spherical = SphericalBackProjection().apply
        self.margin = opt.padding_margin
        if opt.inpaint_path is not None:
            state_dicts = torch.load(opt.inpaint_path)
            self.depth_and_inpaint.load_state_dict(state_dicts['nets'][0])

    #@profile
    def forward(self, input_struct):
        pred_sph=self.depth_and_inpaint.net2(input_struct[0])
        pred_proj_sph_target = self.backproject_spherical(pred_sph['spherical'][:,0,:,:][:,None,:,:])
        pred_proj_sph_other = self.backproject_spherical(pred_sph['spherical'][:,1,:,:][:,None,:,:])
        
        proj_depth = torch.clamp(input_struct[1], 1e-5, 1 - 1e-5)
        refine_input = torch.cat((pred_proj_sph_target, pred_proj_sph_other, proj_depth), dim=1)
        pred_voxel = self.refine_net(refine_input)
        out_1={}
        out_1['pred_voxel'] = pred_voxel
        out_1['pred_proj_sph_full'] = pred_sph['spherical']

        return out_1

    def backproject_spherical(self, sph):
        batch_size, _, h, w = sph.shape
        grid = self.grid[0, :, :, :, :]
        grid = grid.expand(batch_size, -1, -1, -1, -1)
        crop_sph = sph[:, :, self.margin:h - self.margin, self.margin:w - self.margin]
        proj_df, cnt = self.proj_spherical(1 - crop_sph, grid, 128)
        mask = torch.clamp(cnt.detach(), 0, 1)
        proj_df = (-proj_df + 1 / 128) * 128
        proj_df = proj_df * mask
        return proj_df

class Model_test(Model):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        self.requires = ['rgb', 'mask']  # mask for bbox cropping only
        self.input_names = ['rgb']
        self.init_vars(add_path=True)
        self.load_state_dict(opt.net_file, load_optimizer='auto')
        self.output_dir = opt.output_dir
        self.input_names.append('silhou')
        self.upsample=8
        self.xmap = np.array([[j for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])
        self.ymap = np.array([[i for i in range(int(self.upsample*640))] for j in range(int(self.upsample*480))])

    def __str__(self):
        return "Testing GenRe"

    @classmethod
    def preprocess_wrapper(cls, in_dict):
        silhou_thres = 0.95
        in_size = 480
        pad = 85
        im = in_dict['rgb']
        mask = in_dict['silhou']
        bbox = util_img.get_bbox(mask, th=silhou_thres)
        im_crop = util_img.crop(im, bbox, in_size, pad, pad_zero=False)
        silhou_crop = util_img.crop(in_dict['silhou'], bbox, in_size, pad, pad_zero=False)
        in_dict['rgb'] = im_crop
        in_dict['silhou'] = silhou_crop
        # Now the image is just like those we rendered
        out_dict = cls.preprocess(in_dict, mode='test')
        return out_dict

    def test_on_batch(self, batch_i, batch, use_trimesh=True):
        outdir = join(self.output_dir, 'batch%04d' % batch_i)
        makedirs(outdir, exist_ok=True)
        if not use_trimesh:
            pred = self.predict(batch, load_gt=False, no_grad=True)
        else:
            assert self.opt.batch_size == 1
            pred = self.forward_with_trimesh(batch)
        output = self.pack_output(pred, batch, add_gt=False)
        self.visualizer.visualize(output, batch_i, outdir)
        np.savez(outdir + '.npz', **output)

    def pack_output(self, pred, batch, add_gt=True):
        pack = {}
        pack['pred_voxel'] = pred['pred_voxel'].cpu().numpy()
        if add_gt:
            pack['gt_voxel'] = batch['voxel'].numpy()
        return pack

    #@profile
    def forward_with_gt_depth(self, projections, voxels):
        inputs=[torch.from_numpy(projections).float(), torch.from_numpy(voxels).float()]
        
        repeated=False
        if inputs[0].shape[0]==1:
            repeated=True
            inputs[0]=inputs[0].repeat(2, 1, 1, 1)
            inputs[1]=inputs[1].repeat(2, 1, 1, 1, 1)
        inputs[0]=inputs[0].cuda()  
        inputs[1]=inputs[1].cuda()
        preds=self.net.forward(inputs)

        np_pred_voxel=preds['pred_voxel'].detach().cpu().numpy()
        np_pred_voxel=1 / (1 + np.exp(-np_pred_voxel))
        
        meshes=[]
        if repeated:
            num_meshes=1
        else:
            num_meshes=np_pred_voxel.shape[0]
        for ind in range(num_meshes):
#             verts, faces, normals, values = measure.marching_cubes_lewiner(
#                 np_pred_voxel[ind, 0, :, :, :], 0.25, spacing=(1 / 128, 1 / 128, 1 / 128))
#             verts=verts-0.5     
#             mesh = trimesh.Trimesh(vertices=verts, faces=faces)
#             trimesh.repair.fix_inversion(mesh)
#             meshes.append(mesh)
            meshes.append(np_pred_voxel[ind, 0])
        #mesh.show()
        return meshes#, scales, translations
    
    #@profile
    def make_pointcloud_densefusion(self, depth_image):
        depth_scale=np.amax(depth_image)
        depth_image=(depth_image/depth_scale)
        min_depth=np.amin(np.where(depth_image==0.0, 1.0, depth_image))
        depth_image=np.reshape(depth_image, (depth_image.shape[0], depth_image.shape[1], 1))
        depth_image=np.concatenate((depth_image,depth_image,depth_image), axis=2)
        depth_image = cv2.resize(depth_image, (int(self.upsample*depth_image.shape[1]), int(self.upsample*depth_image.shape[0])), interpolation=cv2.INTER_LINEAR)#imresize(depth_image, float(upsample), mode='F', interp='nearest')#
        depth_image=depth_image[:,:,0]
        depth_image=np.where(depth_image<min_depth, 0.0, depth_image)
        depth_image=depth_image*depth_scale
        
        cam_scale = 1.0
        choose=depth_image.flatten().nonzero()[0]
        
        cam_cx = 320.0
        cam_cy = 240.0
        camera_params={'fx':579.4112549695428, 'fy':579.4112549695428, 'img_width':640, 'img_height': 480}
        
        depth_masked = depth_image.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
        
        pt2 = depth_masked / cam_scale
        a=np.amin(pt2)
        b=np.amax(pt2)
        pt0 = (ymap_masked/self.upsample - cam_cx) * pt2 / (camera_params['fx'])
        pt1 = (xmap_masked/self.upsample - cam_cy) * pt2 / (camera_params['fy'])
        cloud = np.concatenate((pt0, -pt1, -pt2), axis=1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.paint_uniform_color(np.array([0,1,0]))
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1)
        fil_cloud=pcd.select_down_sample(ind)
        fil_pcd = o3d.geometry.PointCloud()
        fil_pcd.paint_uniform_color(np.array([1,0,0]))
        
        
        points=np.asarray(fil_cloud.points)
        diff=np.linalg.norm(points-np.mean(points, axis=0), axis=1)
        std=np.mean(diff)
        under_std=np.argwhere(diff<2.5*std)
        points=points[under_std[:, 0]]
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points=o3d.utility.Vector3dVector(points)
        return points

    def convert_to_float32(self, sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)

    #@profile
    def predict_preprocess(self, depth, label):
        obs_ptcld=self.make_pointcloud_densefusion(np.where(label, depth, 0.0))
        obs_ptcld=obs_ptcld/1000.0
        
        translation=np.mean(obs_ptcld, axis=0)
        obs_ptcld=obs_ptcld-translation
         
        obs_ptcld_min=np.amin(obs_ptcld, axis=0)
        obs_ptcld_max=np.amax(obs_ptcld, axis=0)
        scale=3.0*float(np.max(obs_ptcld_max-obs_ptcld_min))
        obs_ptcld=obs_ptcld/scale
        
        obs_ptcld_min=np.amin(obs_ptcld, axis=0)
        obs_ptcld_max=np.amax(obs_ptcld, axis=0)
        
        grid_size=128
        tdf = np.zeros([grid_size, grid_size, grid_size]) / grid_size
        inds2=np.floor((obs_ptcld + 0.5) * grid_size).astype(int)
        tdf[inds2[:, 0], inds2[:, 1], inds2[:, 2]] = 1.0

        try:
            grid_size=128
            tdf_hr = np.zeros([grid_size, grid_size, grid_size]) / grid_size
            inds_hr=np.floor((obs_ptcld + 0.5) * grid_size).astype(int)
            tdf_hr[inds_hr[:, 0], inds_hr[:, 1], inds_hr[:, 2]] = 1.0
        
            verts, faces, normals, values = measure.marching_cubes_lewiner(
            tdf_hr, spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size))
            obs_mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces, vertex_normals=normals)
            trimesh.repair.fix_inversion(obs_mesh)
            trimesh.repair.fill_holes(obs_mesh)
            #obs_mesh.show()
            
            obs_proj=util_sph.proj_spherical(obs_mesh)
        except:
            obs_proj=np.zeros((1,160,160))
            print('marching cubes error!')
        obs_voxels=tdf
                 
        sample_loaded={}
        sample_loaded['obs_voxels']=np.expand_dims(obs_voxels, 0)
        sample_loaded['obs_proj']=obs_proj.detach().cpu().numpy()[0]
        
        self.convert_to_float32(sample_loaded)

        return sample_loaded, scale, translation