import numpy as np
import torch
import torch.nn as nn
from genre.models.marrnet1 import Model as DepthModel
from genre.networks.uresnet import Net_inpaint as Uresnet
from genre.toolbox.cam_bp.cam_bp.modules.camera_backprojection_module import Camera_back_projection_layer
from genre.toolbox.spherical_proj import render_spherical, sph_pad
import torch.nn.functional as F
import time


class Model(DepthModel):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--pred_depth_minmax', action='store_true', default=True,
                            help="GenRe needs minmax prediction")
        parser.add_argument('--load_offline', action='store_true',
                            help="load offline prediction results")
        parser.add_argument('--joint_train', action='store_true',
                            help="joint train net1 and net2")
        parser.add_argument('--net1_path', default=None, type=str,
                            help="path to pretrained net1")
        parser.add_argument('--padding_margin', default=16, type=int,
                            help="padding margin for spherical maps")
        unique_params = {'joint_train'}
        return parser, unique_params

    def __init__(self, opt, logger):
        super(Model, self).__init__(opt, logger)
        self.joint_train = opt.joint_train
        if not self.joint_train:
            self.requires = ['silhou', 'rgb', 'spherical']
            self.gt_names = ['spherical_object']
            self._metrics = ['spherical']
        else:
            self.requires.append('spherical')
            self.gt_names = ['depth', 'silhou', 'normal', 'depth_minmax', 'spherical_object']
            self._metrics.append('spherical')
        self.input_names = ['obs_voxels', 'obs_proj']
        self.net = Net(opt, Model)
        self.optimizer = self.adam(
            self.net.parameters(),
            lr=opt.lr,
            **self.optim_params
        )
        self._nets = [self.net]
        self._optimizers = [self.optimizer]
        self.init_vars(add_path=True)
        self.init_weight(self.net.net2)

    def __str__(self):
        string = "Depth Prediction with Spherical Refinement"
        if self.joint_train:
            string += ' Jointly training all the modules.'
        else:
            string += ' Only training the inpainting module.'
        return string

    #@profile
    def compute_loss(self, pred):
        loss_data = {}
        loss = 0
#         if self.joint_train:
#             loss, loss_data = super(Model, self).compute_loss(pred)
        
        
#         d=torch.zeros((16,1,160,160), dtype=torch.float32).cuda()
        sph_loss=torch.mean(torch.pow(pred['pred_proj_sph_full']-self._gt.gt_proj_sphr_img, 2))#F.mse_loss(pred['pred_proj_sph_full'], self._gt.gt_proj_sphr_img)
        loss_data['spherical'] = sph_loss.mean().item()
        loss = sph_loss
        loss_data['loss'] = loss.mean().item()
        return loss, loss_data

    def pack_output(self, pred, batch, add_gt=True):
        pack = {}
#         if self.joint_train:
#             pack = super(Model, self).pack_output(pred, batch, add_gt=False)
        pack['pred_spherical_full'] = pred['pred_proj_sph_full'].data.cpu().numpy()
        #pack['pred_spherical_partial'] = pred['pred_sph_partial'].data.cpu().numpy()
        #pack['proj_depth'] = pred['proj_depth'].data.cpu().numpy()
        #pack['rgb_path'] = batch['rgb_path']
        if add_gt:
            pack['gt_spherical_full'] = batch['gt_proj_sphr_img'].numpy()
        return pack

    @classmethod
    def preprocess(cls, data, mode='train'):
        dataout = DepthModel.preprocess(data, mode)
        if 'spherical_object' in dataout.keys():
            val = dataout['spherical_object']
            assert(val.shape[1] == val.shape[2])
            assert(val.shape[1] == 128)
            sph_padded = np.pad(val, ((0, 0), (0, 0), (16, 16)), 'wrap')
            sph_padded = np.pad(sph_padded, ((0, 0), (16, 16), (0, 0)), 'edge')
            dataout['spherical_object'] = sph_padded
        return dataout


class Net(nn.Module):
    def __init__(self, opt, base_class=Model, in_channels=1, out_channels=1):
        super().__init__()
#         self.net1 = Net1(
#             [3, 1, 1],
#             ['normal', 'depth', 'silhou'],
#             pred_depth_minmax=True)
        self.net2 = Uresnet([1], ['spherical'], input_planes=in_channels, out_channels=out_channels)
        self.base_class = base_class
        self.proj_depth = Camera_back_projection_layer()
        self.render_spherical = render_spherical()
        self.joint_train = opt.joint_train
        self.load_offline = opt.load_offline
        self.padding_margin = opt.padding_margin
        if opt.net1_path:
            state_dicts = torch.load(opt.net1_path)
            self.net1.load_state_dict(state_dicts['nets'][0])

    #@profile
    def forward(self, input_struct):
        if not self.joint_train:
            with torch.no_grad():
                out_1 = self.net1(input_struct)
        else:
            out_1 = self.net1(input_struct)
        pred_abs_depth = self.get_abs_depth(out_1, input_struct)
        proj = self.proj_depth(pred_abs_depth)
        
#         depth_max=.25
#         depth_min=-25
#         pred_abs_depth=input_struct.depth
#         pred_abs_depth/=100
# #         pred_abs_depth=pred_abs_depth* (depth_max - depth_min + 1e-4) + depth_min
#         silhou=input_struct.silhou
#         pred_abs_depth[silhou < 0.5] = 0
#         np_pred_abs_depth=pred_abs_depth.detach().cpu().numpy()
#         proj = self.proj_depth(pred_abs_depth)
#         
# 
#         np_proj=proj.detach().cpu().numpy()
#         pmin=np.amin(np_proj)
#         pmax=np.amax(np_proj)
#         v_mesh=trimesh.voxel.base.VoxelGrid(np_proj[0, 0])
#         v_mesh.as_boxes().show()
        
        if self.load_offline:
            sph_in = input_struct.spherical_depth
        else:
            sph_in = self.render_spherical(torch.clamp(pred_abs_depth, 1e-5, 1 - 1e-5))
        # pad sph_in to approximate boundary conditions
        sph_in = sph_pad(sph_in, self.padding_margin)
        out_2 = self.net2(sph_in)
        out_1['proj_depth'] = proj * 50
        out_1['pred_sph_partial'] = sph_in

        
        out_1['pred_sph_full'] = out_2['spherical']
        return out_1

    def get_abs_depth(self, pred, input_struct):
        pred_depth = pred['depth']
        pred_depth = self.base_class.postprocess(pred_depth)
        pred_depth_minmax = pred['depth_minmax'].detach()
        pred_abs_depth = self.base_class.to_abs_depth(1 - pred_depth, pred_depth_minmax)
        silhou = self.base_class.postprocess(input_struct.silhou).detach()
        pred_abs_depth[silhou < 0.5] = 0
        pred_abs_depth = pred_abs_depth.permute(0, 1, 3, 2)
        pred_abs_depth = torch.flip(pred_abs_depth, [2])
        return pred_abs_depth
