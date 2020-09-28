import trimesh
from genre.util.util_img import depth_to_mesh_df, resize
from skimage import measure
import numpy as np
import open3d as o3d
from genre.toolbox.spherical_proj import sph_pad
import torch

def make_sgrid(b, alpha, beta, gamma):
    res = b * 2
    pi = np.pi
    phi = np.linspace(0, 180, res * 2 + 1)[1::2]
    theta = np.linspace(0, 360, res + 1)[:-1]
    grid = np.zeros([res, res, 3])
    for idp, p in enumerate(phi):
        for idt, t in enumerate(theta):
            grid[idp, idt, 2] = np.cos((p * pi / 180))
            proj = np.sin((p * pi / 180))
            grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
            grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
    grid = np.reshape(grid, (res * res, 3))
    return grid

sgrid = make_sgrid(64, 0, 0, 0)

#@profile
def proj_spherical(mesh):
    b = 64
    im_depth = render_model(mesh, sgrid)
    im_depth = im_depth.reshape(2 * b, 2 * b)
    im_depth = np.where(im_depth > 1, 1, im_depth)
    rendered_sph=im_depth[None, None, ...]
    rendered_sph = torch.from_numpy(rendered_sph).float()
    rendered_sph = sph_pad(rendered_sph)
    return rendered_sph
    
#@profile
def render_model(mesh, sgrid):
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))

    grid_hits = sgrid[index_ray]
    dist = np.linalg.norm(grid_hits - loc, axis=-1)
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    im = dist_im
    return im

def render_spherical(data, mask, obj_path=None, debug=False):
    depth_im = data['depth'][0, 0, :, :]
    th = data['depth_minmax']
    depth_im = resize(depth_im, 480, 'vertical')
    im = resize(mask, 480, 'vertical')
    gt_sil = np.where(im > 0.95, 1, 0)
    depth_im = np.where(im > 0.95, depth_im, 0)#depth_im * gt_sil
    depth_im = depth_im[:, :, np.newaxis]
    b = 64
    tdf, pcld = depth_to_mesh_df(depth_im, th, False, 10.0, 2.2)
# #     try:
    #print('tdf', tdf.shape)
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        tdf, 0.999 / 128, spacing=(1 / 128, 1 / 128, 1 / 128))
    mesh = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)
    mesh.show()
    im_depth = render_model(mesh, sgrid)
    im_depth = im_depth.reshape(2 * b, 2 * b)
    im_depth = np.where(im_depth > 1, 1, im_depth)
#     except:
#         im_depth = np.ones([128, 128])
#         return im_depth
    return im_depth, tdf
