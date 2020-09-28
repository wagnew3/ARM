
import sys, os
from time import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

import util.util as util_
import util.base_networks as base_networks

class CELossWeighted(nn.Module):
    """ Computes BCE with logits using a mask
    """

    def __init__(self):
        super(CELossWeighted, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target):
        """ Computes weighted cross entropy

            @param x: a [N x C x H x W] torch.FloatTensor of values
            @param target: a [N x H x W] torch.LongTensor of values
        """
        temp = self.CrossEntropyLoss(x, target) # Shape: [N x H x W]

        # Compute pixel weights
        weight_mask = torch.zeros_like(target).float() # Shape: [N x H x W]. weighted mean over pixels
        unique_object_labels = torch.unique(target)
        for obj in unique_object_labels:
            num_pixels = torch.sum(target == obj, dtype=torch.float)
            weight_mask[target == obj] = 1 / num_pixels # inversely proportional to number of pixels

        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 
        return loss

class CosineSimilarityLossMasked(nn.Module):
    """ Computes Cosine Similarity loss using a mask
    """
    def __init__(self, weighted=False):
        super(CosineSimilarityLossMasked, self).__init__()
        self.CosineSimilarity = nn.CosineSimilarity(dim=1)
        self.weighted = weighted

    def forward(self, x, target, mask=None):
        """ Computes masked cosine similarity loss

            @param x: a [N x C x H x W] torch float tensor of values
            @param target: a [N x C x H x W] torch float tensor of values
            @param mask: a [N x H x W] torch tensor with values in {0, 1, 2, ..., K+1}, where K is number of objects. 
                                       Could also be None
        """
        temp = .5 * (1 - self.CosineSimilarity(x, target)) # Shape: [N x H x W]. values are in [0, 1]
        if mask is None:
            return torch.sum(temp) / target.numel() # return mean

        # Compute binary object mask
        binary_object_mask = (mask.clamp(0,2).long() == 2) # Shape: [N x H x W]

        if torch.sum(binary_object_mask) > 0:
            if self.weighted:
                # Compute pixel weights
                weight_mask = torch.zeros_like(mask) # Shape: [N x H x W]. weighted mean over pixels
                unique_object_labels = torch.unique(mask)
                unique_object_labels = unique_object_labels[unique_object_labels >= 2]
                for obj in unique_object_labels:
                    num_pixels = torch.sum(mask == obj, dtype=torch.float)
                    weight_mask[mask == obj] = 1 / num_pixels # inversely proportional to number of pixels
            else:
                weight_mask = binary_object_mask.float() # mean over observed pixels
            loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 
        else:
            print("all gradients are 0...")
            loss = torch.tensor(0., dtype=torch.float, device=x.device) # just 0. all gradients will be 0

        bg_mask = ~binary_object_mask
        if torch.sum(bg_mask) > 0:
            bg_loss = 0.1 * torch.sum(temp * bg_mask.float()) / torch.sum(bg_mask.float())
        else:
            bg_loss = torch.tensor(0., dtype=torch.float, device=x.device) # just 0

        return loss + bg_loss

class BCEWithLogitsLossWeighted(nn.Module):
    """ Computes Cosine Similarity loss using a mask
    """
    def __init__(self, weighted=False):
        super(BCEWithLogitsLossWeighted, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.weighted = weighted

    def forward(self, x, target):
        """ Computes masked cosine similarity loss

            @param x: a [N x H x W] torch float tensor of foreground logits
            @param target: a [N x H x W] torch float tensor of values in {0, 1}
        """
        temp = self.BCEWithLogitsLoss(x, target) # Shape: [N x H x W]. values are in [0, 1]

        if self.weighted:
            # Compute pixel weights
            weight_mask = torch.zeros_like(target) # Shape: [N x H x W]. weighted mean over pixels
            unique_object_labels = torch.unique(target) # Should be {0, 1}
            for obj in unique_object_labels:
                num_pixels = torch.sum(target == obj, dtype=torch.float)
                weight_mask[target == obj] = 1 / num_pixels # inversely proportional to number of pixels
        else:
            weight_mask = torch.ones_like(target) # mean over observed pixels
        loss = torch.sum(temp * weight_mask) / torch.sum(weight_mask) 

        return loss



class UNet(nn.Module):
    
    def __init__(self, input_channels, output_channels, coordconv=False):
        """ Creates the UODNetwork (feedforward network only, no losses)

        """
        super(UNet, self).__init__()
        self.ic = input_channels
        self.oc = output_channels
        self.ng = output_channels # number of groups for GroupNorm module
        self.coordconv = coordconv
        self.build_network()
        
    def build_network(self):
        """ Build encoder-decoder network
            Uses a U-Net-like architecture
        """

        ### Encoder ###
        self.layer1 = base_networks.Conv2d_GN_ReLUx2(self.ic, self.oc, self.ng)
        self.layer2 = base_networks.Conv2d_GN_ReLUx2(self.oc, self.oc*2, self.ng)
        self.layer3 = base_networks.Conv2d_GN_ReLUx2(self.oc*2, self.oc*4, self.ng)
        self.layer4 = base_networks.Conv2d_GN_ReLUx2(self.oc*4, self.oc*8, self.ng)

        # Increase channels in this low-resolution space
        self.layer5 = base_networks.Conv2d_GN_ReLU(self.oc*8, self.oc*16, self.ng)

        ### Decoder ###
        self.layer6 = base_networks.Upsample_Concat_Conv2d_GN_ReLU(self.oc*16, self.oc*8, self.ng)
        self.layer7 = base_networks.Upsample_Concat_Conv2d_GN_ReLU(self.oc*8, self.oc*4, self.ng)
        self.layer8 = base_networks.Upsample_Concat_Conv2d_GN_ReLU(self.oc*4, self.oc*2, self.ng)
        self.layer9 = base_networks.Upsample_Concat_Conv2d_GN_ReLU(self.oc*2, self.oc, self.ng)

        # Final layer
        self.layer10 = base_networks.Conv2d_GN_ReLU(self.oc, self.oc, self.ng)

        if self.coordconv:
            # Extra 1x1 Conv layers for CoordConv
            self.layer11 = base_networks.Conv2d_GN_ReLUx2(self.oc+2, self.oc, self.ng, ksize=1)
            self.layer12 = base_networks.Conv2d_GN_ReLUx2(self.oc, self.oc, self.ng, ksize=1)

        # This puts features everywhere, not just nonnegative orthant
        self.last_conv = nn.Conv2d(self.fd, self.fd, kernel_size=3,
                                   stride=1, padding=1, bias=True)

    def forward(self, images):

        x1 = self.layer1(images)
        mp_x1 = base_networks.maxpool2x2(x1)
        x2 = self.layer2(mp_x1)
        mp_x2 = base_networks.maxpool2x2(x2)
        x3 = self.layer3(mp_x2)
        mp_x3 = base_networks.maxpool2x2(x3)
        x4 = self.layer4(mp_x3)
        mp_x4 = base_networks.maxpool2x2(x4)

        x5 = self.layer5(mp_x4)

        out = self.layer6(x5, x4)
        out = self.layer7(out, x3)
        out = self.layer8(out, x2)
        out = self.layer9(out, x1)

        out = self.layer10(out)

        if self.coordconv:
            out = util_.concatenate_spatial_coordinates(out)
            out = self.layer11(out)
            out = self.layer12(out)

        out = self.last_conv(out)

        return out


class UNet_Encoder(nn.Module):
    
    def __init__(self, input_channels, feature_dim):
        super(UNet_Encoder, self).__init__()
        self.ic = input_channels
        self.fd = feature_dim
        self.reduction_factor = 16 # 4 max pools
        self.build_network()
        
    def build_network(self):
        """ Build encoder network
            Uses a U-Net-like architecture
        """

        ### Encoder ###
        self.layer1 = base_networks.Conv2d_GN_ReLUx2(self.ic, self.fd, self.fd)
        self.layer2 = base_networks.Conv2d_GN_ReLUx2(self.fd, self.fd*2, self.fd)
        self.layer3 = base_networks.Conv2d_GN_ReLUx2(self.fd*2, self.fd*4, self.fd)
        self.layer4 = base_networks.Conv2d_GN_ReLUx2(self.fd*4, self.fd*8, self.fd)
        self.last_layer = base_networks.Conv2d_GN_ReLU(self.fd*8, self.fd*16, self.fd)


    def forward(self, images):

        x1 = self.layer1(images)
        mp_x1 = base_networks.maxpool2x2(x1)
        x2 = self.layer2(mp_x1)
        mp_x2 = base_networks.maxpool2x2(x2)
        x3 = self.layer3(mp_x2)
        mp_x3 = base_networks.maxpool2x2(x3)
        x4 = self.layer4(mp_x3)
        mp_x4 = base_networks.maxpool2x2(x4)
        x5 = self.last_layer(mp_x4)

        return x5, [x1, x2, x3, x4]

class UNet_Decoder(nn.Module):

    def __init__(self, num_encoders, feature_dim, coordconv=False):
        super(UNet_Decoder, self).__init__()
        self.ne = num_encoders
        self.fd = feature_dim
        self.coordconv = coordconv
        self.reduction_factor = 1 # full resolution features
        self.build_network()

    def build_network(self):
        """ Build a decoder network
            Uses a U-Net-like architecture
        """

        # Fusion layer
        self.fuse_layer = base_networks.Conv2d_GN_ReLU(self.fd*16 * self.ne, self.fd*16, self.fd, ksize=1)

        # Decoding
        self.layer1 = base_networks.Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*16, self.fd*8, self.fd, self.ne)
        self.layer2 = base_networks.Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*8, self.fd*4, self.fd, self.ne)
        self.layer3 = base_networks.Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*4, self.fd*2, self.fd, self.ne)
        self.layer4 = base_networks.Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(self.fd*2, self.fd, self.fd, self.ne)

        # Final layer
        self.layer5 = base_networks.Conv2d_GN_ReLU(self.fd, self.fd, self.fd)

        if self.coordconv:
            # Extra 1x1 Conv layers for CoordConv
            self.layer6 = base_networks.Conv2d_GN_ReLUx2(self.fd+2, self.fd, self.fd, ksize=1)
            self.layer7 = base_networks.Conv2d_GN_ReLUx2(self.fd, self.fd, self.fd, ksize=1)        

        # This puts features everywhere, not just nonnegative orthant
        self.last_conv = nn.Conv2d(self.fd, self.fd, kernel_size=3,
                                   stride=1, padding=1, bias=True)

    def forward(self, encoder_list):
        """ Forward module

            @param encoder_list: a list of tuples
                                 each tuple includes 2 elements:
                                    - output of encoder: an [N x C x H x W] torch tensor
                                    - list of intermediate outputs: a list of 4 torch tensors

        """

        # Apply fusion layer to the concatenation of encoder outputs
        out = torch.cat([x[0] for x in encoder_list], dim=1) # Concatenate on channels dimension
        out = self.fuse_layer(out)

        out = self.layer1(out, [x[1][3] for x in encoder_list])
        out = self.layer2(out, [x[1][2] for x in encoder_list])
        out = self.layer3(out, [x[1][1] for x in encoder_list])
        out = self.layer4(out, [x[1][0] for x in encoder_list])

        out = self.layer5(out)

        if self.coordconv:
            out = util_.concatenate_spatial_coordinates(out)
            out = self.layer6(out)
            out = self.layer7(out)

        out = self.last_conv(out)

        return out
