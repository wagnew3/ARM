"""
Written by Christopher B. Choy <chrischoy@ai.stanford.edu>
Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction, ECCV 2016
"""
import os
import sys
import numpy as np
from contextlib import redirect_stdout
import subprocess

from tempfile import TemporaryFile
import trimesh
import time
import io
import open3d as o3d

import genre.voxelization.binvox_rw as binvox_rw


def evaluate_voxel_prediction(preds, gt, thresh):
    preds_occupy = preds[:, 1, :, :] >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt[:, 1, :, :]))
    intersection = np.sum(np.logical_and(preds_occupy, gt[:, 1, :, :]))
    union = np.sum(np.logical_or(preds_occupy, gt[:, 1, :, :]))
    num_fp = np.sum(np.logical_and(preds_occupy, gt[:, 0, :, :]))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt[:, 1, :, :]))  # false negative
    return np.array([diff, intersection, union, num_fp, num_fn])

#@profile
def voxelize_model_binvox(mesh, n_vox, file_num, return_voxel=True, binvox_add_param='', use_cuda_vox=False):
    ramdisk='/dev/shm/'#'/home/willie/workspace/SSC/genre/voxelization'#'
    
    file_name=ramdisk+str(file_num)+'.obj'
    mesh.export(file_name)

    use_cuda_vox=True
    if use_cuda_vox:
        cmd = "/home/willie/github/cuda_voxelizer//build/cuda_voxelizer -s %d -f %s -cpu" % (n_vox,
                file_name)
    else:
        cmd = "/home/willie/workspace/SSC/genre/voxelization/binvox -d %d -dc -aw -pb %s -t binvox %s" % (
                n_vox, binvox_add_param, file_name)

    if not os.path.exists(file_name):
        raise ValueError('No obj found : %s' % file_name)

    # Stop printing command line output
    
    result = subprocess.check_output(cmd, shell=True)
    #print(result)
    
    if use_cuda_vox:
        with open('%s.binvox' % (file_name+f"_{n_vox}"), 'rb') as f:
            vox = binvox_rw.read_as_3d_array(f)
        os.remove('%s.binvox' % (file_name+f"_{n_vox}"))
    else:
        # load voxelized model
        with open('%s.binvox' % file_name[:-4], 'rb') as f:
            vox = binvox_rw.read_as_3d_array(f)
        os.remove('%s.binvox' % file_name[:-4])
        
    os.remove(file_name)
    return vox
    
    

# mesh=trimesh.load('/home/willie/workspace/SSC/herb_reconf/assets/ycb_objects/021_bleach_cleanser/google_16k/textured.obj')
# start_time=time.time()
# voxels=voxelize_model_binvox(mesh, 128, 0, use_cuda_vox=True)
# end_time=time.time()
#     
#    
#    
#     
# print('time', end_time-start_time)
# u=0