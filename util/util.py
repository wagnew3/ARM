import sys, os
import json
from itertools import compress
import torch

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import cv2
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def multidim_argmax(M):
    return np.unravel_index(M.argmax(), M.shape)

def normalize(M):
    """ Take all values of M and normalize it to the range [0,1]
    """
    M = M.astype(np.float32)
    return (M - M.min()) / (M.max() - M.min())

def resize_image(img, new_size, interpolation='zoom'):
    """ Resizes an image to new_size.

        Default interpolation uses cv2.INTER_LINEAR (good for zooming)
    """
    if interpolation == 'zoom':
        interp = cv2.INTER_LINEAR
    elif interpolation == 'shrink':
        interp = cv2.INTER_AREA
    elif interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    else:
        raise Exception("Interpolation should be one of: ['zoom', 'shrink', 'nearest']")
    return cv2.resize(img, new_size, interpolation=interp)

def load_rgb_image(imagefile):
    image = cv2.imread(imagefile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_rgb_image_with_resize(imagefile, new_size, interp='zoom'):
    """ Load image and resize
    """
    image = load_rgb_image(imagefile)
    image = resize_image(image, new_size, interpolation=interp)
    return image

def get_color_mask(object_index, nc=None):
    """ Colors each index differently. Useful for visualizing semantic masks

        @param object_index: a [H x W] numpy array of ints from {0, ..., nc-1}
        @param nc: total number of colors. If None, this will be inferred by masks
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255
        
    return color_mask

def encode_one_hot_tensor(labels):
    """ Takes a torch tensor of integers and encodes it into a one-hot tensor.
        Let K be the number of labels

        @param labels: a [T x H x W] torch tensor with values in {0, ..., K-1}

        @return: a [T x K x H x W] torch tensor of 0's and 1's
    """
    T, H, W = labels.shape
    K = int(torch.max(labels).item() + 1)
    
    # Encode the one hot tensor
    one_hot_tensor = torch.zeros((T, K, H, W), device=labels.device)
    one_hot_tensor.scatter_(1, labels.long().unsqueeze(1), 1)

    return one_hot_tensor

def build_matrix_of_indices(height, width):
    """ Builds a [height, width, 2] numpy array containing coordinates.

        @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)

def torch_moi(h, w, device='cpu'):
    """ Torch function to do the same thing as above function, but returns CHW format
        Also, B[0, ...] is x-coordinates
        
        @return: 3d torch tensor B s.t. B[0, ...] contains x-coordinates, B[1, ...] contains y-coordinates
    """
    ys = torch.arange(h, device=device).view(-1,1).expand(h,w)
    xs = torch.arange(w, device=device).view(1,-1).expand(h,w)
    return torch.stack([xs,ys], dim=0).float()

def concatenate_spatial_coordinates(feature_map):
    """ Adds x,y coordinates as channels to feature map

        @param feature_map: a [T x C x H x W] torch tensor
    """
    T, C, H, W = feature_map.shape

    # build matrix of indices. then replicated it T times
    MoI = build_matrix_of_indices(H, W) # Shape: [H, W, 2]
    MoI = np.tile(MoI, (T, 1, 1, 1)) # Shape: [T, H, W, 2]
    MoI[..., 0] = MoI[..., 0] / (H-1) * 2 - 1 # in [-1, 1]
    MoI[..., 1] = MoI[..., 1] / (W-1) * 2 - 1
    MoI = torch.from_numpy(MoI).permute(0,3,1,2).to(feature_map.device) # Shape: [T, 2, H, W]

    # Concatenate on the channels dimension
    feature_map = torch.cat([feature_map, MoI], dim=1)

    return feature_map

def append_channels_dim(img):
    """ If an image is 2D (shape: [H x W]), add a channels dimensions to bring it to: [H x W x 1]

        This is to be called after cv2.resize, since if you resize a [H x W x 1] image, cv2.resize
            spits out a [new_H x new_W] image
    """
    if img.ndim == 2:
        # append axis dimension
        img = np.expand_dims(img, axis=-1)
        return img
    elif img.ndim == 3:
        # do nothing
        return img
    else:
        # wtf is happening
        raise Exception("This image is a weird shape: {0}".format(img.shape))


def visualize_segmentation(im, masks, nc=None, return_rgb=False, save_dir=None):
    """ Visualize segmentations nicely. Based on code from:
        https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

        @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
        @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
        @param nc: total number of colors. If None, this will be inferred by masks
    """ 
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    # matplotlib stuff
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)

    # Mask
    imgMask = np.zeros(im.shape)


    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)


    # Draw mask contours
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        contour, hier = cv2.findContours(
            e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            if save_dir is None and not return_rgb:
                polygon = Polygon(c.reshape((-1, 2)), fill=False, facecolor=color_mask, edgecolor='w', linewidth=1.2, alpha=0.5)
                ax.add_patch(polygon)
            else:
                cv2.drawContours(im, contour, -1, (255,255,255), 2)


    if save_dir is None and not return_rgb:
        ax.imshow(im)
        return fig
    elif return_rgb:
        return im
    elif save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image
    

### These two functions were adatped from the DAVIS public dataset ###

def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation

def imwrite_indexed(filename,array):
    """ Save indexed png with palette."""

    palette_abspath = '/data/tabletop_dataset_v5/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def compute_table_surface_normal():
    pass

def compute_table_mean():
    pass

def transform_camera_xyz_to_table_xyz(xyz, surface_normals, tabletop_mask):
    """ Transform xyz ordered point cloud from camera coordinate frame to
        table coordinate frame

        @param camera_xyz: a [3 x H x W] torch.FloatTensor ordered point cloud in camera coordinates
        @param surface_normals: a [3 x H x W] torch.FloatTensor of surface normals
        @tabletop_mask: a [H x W] torch.ByteTensor of tabletop labls in {0, 1}

        @return: a [3 x H x W] torch.FloatTensor ordered point cloud in tabletop coordinates
    """

    # Compute average surface normal with weighted average pooling
    nonzero_depth_mask = ~torch.isclose(xyz[2, ...], torch.tensor([0.], device=xyz.device)) # Shape: [H x W]
    tabletop_mask = tabletop_mask & nonzero_depth_mask

    # inflate tabletop_mask so that surface normal computation is correct. we do this because of how surface normal is computed
    tabletop_mask = tabletop_mask & torch.roll(nonzero_depth_mask, -1, dims=0)
    tabletop_mask = tabletop_mask & torch.roll(nonzero_depth_mask, 1,  dims=0)
    tabletop_mask = tabletop_mask & torch.roll(nonzero_depth_mask, -1, dims=1)
    tabletop_mask = tabletop_mask & torch.roll(nonzero_depth_mask, 1,  dims=1)

    # Compute y direction of table
    table_y = torch.mean(surface_normals[:, tabletop_mask], dim=1)
    table_y = table_y / (torch.norm(table_y) + 1e-10)

    # Project camera z-axis onto table plane. NOTE: this is differentiable w.r.t. table_y
    camera_z = torch.tensor([0,0,1], dtype=torch.float, device=xyz.device)
    table_z = camera_z - torch.dot(table_y, camera_z) * table_y
    table_z = table_z / (torch.norm(table_z) + 1e-10)

    # Get table x-axis. NOTE: this is differentiable w.r.t. table_y, table_z, since cross products are differentiable
    # Another note: cross product adheres to the handedness of the coordinate system, which is a left-handed system
    table_x = torch.cross(table_y, table_z)
    table_x = table_x / (torch.norm(table_x) + 1e-10)

    # Transform xyz depth map to table coordinates
    table_mean = torch.mean(xyz[:, tabletop_mask], dim=1)

    x_projected = torch.tensordot(table_x, xyz - table_mean.unsqueeze(1).unsqueeze(2), dims=1)
    y_projected = torch.tensordot(table_y, xyz - table_mean.unsqueeze(1).unsqueeze(2), dims=1)
    z_projected = torch.tensordot(table_z, xyz - table_mean.unsqueeze(1).unsqueeze(2), dims=1)

    new_xyz = torch.stack([x_projected, y_projected, z_projected], dim=0)
    return new_xyz

def batch_transform_camera_xyz_to_table_xyz(xyz_batch, surface_normal_batch, tabletop_mask_batch):
    """ Run above method in a for loop

        @param xyz_batch: a [N x 3 x H x W] torch tensor of ordered point clouds in camera coordinates
        @param surface_normal_batch: a [N x 3 x H x W] torch tensor of surface normals
        @param tabletop_mask_batch: a [N x H x W] torch tensor of tabletop labels in {0, 1}

        @return: a [N x 3 x H x W] torch tensor of ordered point clouds in tabletop coordinates
    """

    N, H, W = tabletop_mask_batch.shape

    table_xyz_batch = torch.zeros_like(surface_normal_batch)
    for n in range(N):
        if torch.sum(tabletop_mask_batch[n]) == 0:
            table_xyz = surface_normal_batch[n]
        else:
            table_xyz = transform_camera_xyz_to_table_xyz(xyz_batch[n],
                                                          surface_normal_batch[n],
                                                          tabletop_mask_batch[n]
                                                         )
        table_xyz_batch[n] = table_xyz

    return table_xyz_batch

def mask_to_tight_box_numpy(mask):
    """ Return bbox given mask

        @param mask: a [H x W] numpy array
    """
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max

def mask_to_tight_box_pytorch(mask):
    """ Return bbox given mask

        @param mask: a [H x W] torch tensor
    """
    a = torch.nonzero(mask)
    bbox = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(a[:, 1]), torch.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max

def mask_to_tight_box(mask):
    if type(mask) == torch.Tensor:
        return mask_to_tight_box_pytorch(mask)
    elif type(mask) == np.ndarray:
        return mask_to_tight_box_numpy(mask)
    else:
        raise Exception(f"Data type {type(mask)} not understood for mask_to_tight_box...")
