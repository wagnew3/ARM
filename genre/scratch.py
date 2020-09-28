import trimesh
import open3d as o3d
import numpy as np
from skimage import measure
import scipy

pred_voxels=np.load('/home/willie/workspace/SSC/scratch_save/pred_voxels.npz.npy')
pred_voxels=scipy.special.expit(pred_voxels)
gt_voxels=np.load('/home/willie/workspace/SSC/scratch_save/gt_voxels.npz.npy')
# gt_voxels[:,:,:,:,0]=1.0
for ind in range(16):
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(pred_voxels[ind, 0], 0.5, spacing=(1 / 128, 1 / 128, 1 / 128))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    trimesh.repair.fix_inversion(mesh)
    mesh.show()
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(gt_voxels[ind, 0], 0.5, spacing=(1 / 128, 1 / 128, 1 / 128))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    trimesh.repair.fix_inversion(mesh)
    mesh.show()