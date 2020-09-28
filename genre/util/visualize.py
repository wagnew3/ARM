import trimesh
import pywavefront
from pywavefront import visualization
import numpy as np
import pyglet

# z=np.load('/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/shapenet_cars_chairs_planes_20views/02691156/1a04e3eab45ca15dd86060f189eb133/02691156_1a04e3eab45ca15dd86060f189eb133_view000.npy')
# a=0
# mesh = trimesh.load_mesh('/home/willie/workspace/GenRe-ShapeHD/output/batch0000/0000_12_pred_voxel.obj')
# mesh.show()
window = pyglet.window.Window(1024, 720, caption='Demo', resizable=True)
scene = pywavefront.Wavefront('/home/willie/workspace/GenRe-ShapeHD/output/batch0000/0000_12_pred_voxel.obj')
visualization.draw(scene)
u=0