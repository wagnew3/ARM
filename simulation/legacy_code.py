def playing_with_pybullet_zbuffer():
    pass
"""

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')
p.loadURDF('cube_small.urdf', basePosition=[0.025, 0.025, 0.025])

temp = p.getDebugVisualizerCamera() # Get parameters of camera
M = np.array(temp[3]).reshape(4,4, order='F') # Perspective projection matrix
A = M[2,2] # See A,B defined here: http://web.archive.org/web/20141020172632/http://www.songho.ca/opengl/gl_projectionmatrix_mathml.html
B = M[2,3]
print(A, B)

width = 128
height = 128
far = 1000; near = 0.01

# Get depth values using the OpenGL renderer
# temp = p.getDebugVisualizerCamera() # Get parameters of camera
# view_mat = np.array(temp[2]).reshape(4,4, order='F')
# proj_mat = p.computeProjectionMatrix(0, 1, 0, 1, near, far)
# images = p.getCameraImage(width, height, projectionMatrix=proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
images = p.getCameraImage(width, height, renderer=p.ER_BULLET_HARDWARE_OPENGL)
depth_buffer_opengl = np.reshape(images[3], [height, width])
z_values = far * near / (far - (far - near) * depth_buffer_opengl)
# z_values = 2 * far * near / ( (far - near) * (2 * depth_buffer_opengl - 1) - (far + near) )
# z_values *= -1

# Plot both images - should show depth values of 0.45 over the cube and 0.5 over the plane
plt.imshow(z_values, cmap='gray')
plt.title('OpenGL Renderer')
plt.show()

# Some debugging code
plt.imshow(depth_buffer_opengl)
print(z_values.min(), z_values.max(), z_values.max() - z_values.min())
print(depth_buffer_opengl.min(), depth_buffer_opengl.max())

temp = p.getDebugVisualizerCamera()
camera_yaw, camera_pitch, camera_distance, camera_target = temp[8:]
print("Yaw: {0}".format(camera_yaw))
print("Pitch: {0}".format(camera_pitch))
print("Distance: {0}".format(camera_distance))
print("Target: {0}".format(camera_target))

"""

def simulating_colored_cubes_on_a_table_with_pybullet():
    pass
"""

import time
import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import simulation_util as util
import pybullet as p
import pybullet_data

# Connect to GUI server
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # got some data here

----------------------------------------------------------------------------------------------------

p.setGravity(0,0,-10)
plane_id = p.loadURDF('plane.urdf')
table_id = p.loadURDF("table/table.urdf")

# Get max/min coordinates of table
table_min, table_max = p.getAABB(table_id)
table_min = np.array(table_min)
table_max = np.array(table_max)
table_xmin, table_ymin, table_zmin = table_min
table_xmax, table_ymax, table_zmax = table_max
table_xsize = table_xmax - table_xmin
table_ysize = table_ymax - table_ymin

----------------------------------------------------------------------------------------------------

num_objects = 100
obj_ids = []
for i in range(num_objects):

    # Load object at origin (0,0,0)
    obj_id = p.loadURDF('cube_small.urdf')
    obj_ids.append(obj_id)

    # Get size of thing
    obj_min, obj_max = p.getAABB(obj_id)
    obj_min, obj_max = np.array(obj_min), np.array(obj_max)
    obj_xmin, obj_ymin, obj_zmin = obj_min
    obj_xmax, obj_ymax, obj_zmax = obj_max

    # Sample a random starting location
    delta = 0.1
    random_start_xpos = np.random.uniform(table_xmin - obj_xmin, table_xmax - obj_xmax)
    random_start_ypos = np.random.uniform(table_ymin - obj_ymin, table_ymax - obj_ymax)
    random_start_zpos = np.random.uniform(table_zmax - obj_zmin + delta, 3*table_zmax)
    random_start_pos = np.array([random_start_xpos, random_start_ypos, random_start_zpos])

    # Sample a random starting orientation
    random_orientation = np.random.uniform(0, 2*np.pi, 3) # Euler angles
    random_orientation = p.getQuaternionFromEuler(random_orientation)

    # Set position/orientation
    p.resetBasePositionAndOrientation(obj_id, random_start_pos, random_orientation)
    
    # Randomly color the cube
    obj_color = util.random_color()
    p.changeVisualShape(obj_id, -1, rgbaColor=obj_color)

----------------------------------------------------------------------------------------------------

# Let the objects fall
for i in range(500):
    p.stepSimulation()
    time.sleep(1./240)

----------------------------------------------------------------------------------------------------

# Remove any that are under the table
fallen_obj_ids = []
for obj_id in obj_ids:
    obj_zpos = p.getBasePositionAndOrientation(obj_id)[0][2]
    if obj_zpos < table_zmax:
        fallen_obj_ids.append(obj_id)

for obj_id in fallen_obj_ids:
    obj_ids.remove(obj_id)
    p.removeBody(obj_id)
    
print("Number of fallen objects: {0}".format(len(fallen_obj_ids)))

p.disconnect()

"""

def get_all_object_mesh_filepaths_from_scenes():
    pass
"""
# Get all object mesh filepaths from the scenes

all_object_meshes = []
for i in range(len(scenes)):
    all_object_meshes = all_object_meshes + [x['mesh_filename'] for x in scenes[i]['object_descriptions']]
    
for i in range(len(all_object_meshes)):
    all_object_meshes[i] = all_object_meshes[i].replace('models/model_normalized.obj', '')
    
for i in range(len(all_object_meshes)):
    os.system('cp -r {0} /home/chrisxie/Desktop/few_shapenet_objects/'.format(all_object_meshes[i]))
"""

def test_collision_list():
    pass
"""
# Connect to GUI server
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # got some data here

# Some standard stuff
p.setGravity(0,0,-10)
plane_id = p.loadURDF('plane.urdf')

sim_util = reload(sim_util)
cube1_id = p.loadURDF('cube_small.urdf', basePosition=np.array([0.025, 0.025, 0.025]) + 0.5)
p.changeVisualShape(cube1_id, -1, rgbaColor=sim_util.random_color())
cube2_id = p.loadURDF('cube_small.urdf', basePosition=np.array([0.045, 0.045, 0.045]) + 0.5)
p.changeVisualShape(cube2_id, -1, rgbaColor=sim_util.random_color())
cube3_id = p.loadURDF('cube_small.urdf', basePosition=np.array([0.5, 0.075, 0.045]) + 0.5)
p.changeVisualShape(cube3_id, -1, rgbaColor=sim_util.random_color())

sim_util.get_collision_list(cube1_id, [0,1,2,3]) # Should be [2]
sim_util.get_collision_list(cube2_id, [0,1,2,3]) # Should be [1]
sim_util.get_collision_list(cube3_id, [0,1,2,3]) # Should be []

"""

def count_number_of_valid_shapenet_tables():
    pass
"""
from functools import wraps
import errno
import os, sys
import signal
import json
import glob
import numpy as np
import scipy
import scipy.spatial

# my libraries
sys.path.insert(0, os.path.abspath('..'))
import util.simulation_util as sim_util

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

@timeout(5)
def valid_table_shape(obj_file_name, iou_threshold=0.8):
    "" Computes the xz bounding box of the vertices with the highest y value (and vertical normals)
        and compares this with the xz bounding box of the entire table. If the IoU is high enough,
        the table is mostly flat on top.

        To filter out corner tables, check if vertices make up a convex shape. To do this, we compute 
        the convex hull of the high vertices, and compute the faces that are associated with those vertices.
        Make sure the IoU is close to 1
    ""
    # Load the mesh
    temp = sim_util.load_mesh(obj_file_name)
    vertices = np.array(temp['vertices']) # Shape: num_vertices x 3
    normals = np.array(temp['normals'])   # Shape: num_vertices x 3
    faces = np.array(temp['faces'])       # Shape: num_faces x 3


    ### Bounding Box Comparison ###

    # Get the indices of the vertical normals 
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    vertical_normal_indices = np.where(np.isclose(np.abs(normalized_normals[:,1]), 1, atol=1e-4))[0]

    unique_y_vals = np.unique(vertices[vertical_normal_indices, 1])
    highest_y_val = np.max(unique_y_vals)

    highest_vertex_indices = np.where(np.isclose(vertices[:,1], highest_y_val, atol=1e-2))[0] 
    highest_vertices = vertices[highest_vertex_indices, :] # Shape: num_highest_vertices x 3
    # highest_vertices is a list of vertices with a vertical normal and highest y value

    # Compare the xz bounding box of the highest vertices with the xz bounding box of the entire table. LTRB bbox format
    highest_vertices_xz_rect = np.concatenate([np.min(highest_vertices, axis=0)[[0,2]], np.max(highest_vertices, axis=0)[[0,2]]])
    table_xz_rect = np.concatenate([np.min(vertices, axis=0)[[0,2]], np.max(vertices, axis=0)[[0,2]]])

    iou = sim_util.IoU(highest_vertices_xz_rect, table_xz_rect)
    valid_iou = iou >= iou_threshold



    ### Filtering Out Corner Tables ###

    # Convex hull of highest vertices (xz, not xyz)
    conv_hull = scipy.spatial.ConvexHull(highest_vertices[:,[0,2]])

    ## Deal with duplicate vertices/faces ##

    # Merge duplicate vertices
    unique_highest_vertices, unique_highest_vertices_index = np.unique(highest_vertices[:,[0,2]], axis=0, return_inverse=True)
    # Get the faces corresponding to the highest vertices
    highest_faces = np.array([row for row in faces if set(row).issubset(highest_vertex_indices)])

    # Compute the faces of these vertices
    unique_faces = []
    for face in highest_faces:

        # Find indices of face vertices in highest_vertex_indices list
        face_highest_vertex_indices = [list(highest_vertex_indices).index(face[i]) for i in range(len(face))]

        # Find unique indices of face vertices in unique_highest_vertices (merged highest vertices)
        face_unique_highest_vertex_indices = unique_highest_vertices_index[face_highest_vertex_indices]
        face_unique_highest_vertex_indices = sorted(face_unique_highest_vertex_indices)
        
        # Unique faces
        if face_unique_highest_vertex_indices in unique_faces:
            continue
        else:
            unique_faces.append(face_unique_highest_vertex_indices)

    total_volume = 0
    for face in unique_faces:
        total_volume += sim_util.triangle_area(unique_highest_vertices[face])

    # Compare total volume of unique faces with volume of computed convex hull
    is_convex = np.isclose(total_volume, conv_hull.volume, atol=1e-1)
    # if not is_convex and valid_iou: # debugging
    #     print(total_volume, conv_hull.volume, iou)
    #     if np.isnan(total_volume) or not is_convex:
    #         from IPython import embed; embed()

    return valid_iou and is_convex



### Script starts here ###
shapenet_filepath = '/data/ShapeNetCore.v2/'

# Create a dictionary of name -> synset_id
temp = json.load(open(shapenet_filepath + 'taxonomy.json'))
taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}

# weirdly, the synsets in the taxonomy file are not the same as what's in the ShapeNetCore.v2 directory. Filter this out
synsets_in_dir = os.listdir(shapenet_filepath)
synsets_in_dir.remove('taxonomy.json')
synsets_in_dir.remove('README.txt')

taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synsets_in_dir}


table_dirs = glob.glob(shapenet_filepath + taxonomy_dict['table'] + '/*/images/')
table_dirs = [tdir.replace('images/', '') for tdir in table_dirs]
table_models = [tdir + 'models/model_normalized.obj' for tdir in table_dirs]

# For saving the valid tables
valid_table_filename = '/data/ShapeNetCore.v2/valid_tables2.json'

valid_tables = []
for i, table_mesh in enumerate(table_models):
    try:
        valid_table = valid_table_shape(table_mesh)
    except:
        valid_table = False
        
    if valid_table:
        valid_tables.append(table_mesh.split('/')[4])
        print(i, 'valid!')
    else:
        print(i)
        
    if i % 50 == 0:
        # Serialize this JSON file
        print('Saving JSON file...')
        with open(valid_table_filename, 'w') as save_file:  
            json.dump(valid_tables, save_file)

# Serialize this JSON file
with open(valid_table_filename, 'w') as save_file:  
    json.dump(valid_tables, save_file)
# vts = json.load(open(valid_table_filename))



### Splitting valid tables into train/test ###

np.random.seed(0) # for repeatability
train_percentage = 0.8
test_percentage = 0.2

num_tables_train = int(len(valid_tables) * train_percentage)
num_tables_test = int(len(valid_tables) * test_percentage)
valid_tables_permutation = np.random.permutation(valid_tables).tolist()
train_tables = valid_tables_permutation[:num_tables_train]
test_tables = valid_tables_permutation[num_tables_train:]

print(f"Number of training tables: {len(train_tables)}")
print(f"Number of test tables: {len(test_tables)}")

# Save JSON files
training_tables_filename = '/data/tabletop_dataset/training_shapenet_tables.json'
test_tables_filename = '/data/tabletop_dataset/test_shapenet_tables.json'
with open(training_tables_filename, 'w') as save_file:  
    json.dump(train_tables, save_file)
with open(test_tables_filename, 'w') as save_file:  
    json.dump(test_tables, save_file)
# train_tables = json.load(open(training_tables_filename))
# test_tables = json.load(open(test_tables_filename))

print(f"Number of valid tables: {np.count_nonzero(valid_tables)}")
"""

def separate_suncg_houses_into_train_test_split():
    pass
"""

# List of all houses
house_ids = os.listdir(suncg_dir + 'house/')

np.random.seed(0) # for repeatability
train_percentage = 0.8
test_percentage = 0.2

num_houses_train = int(len(house_ids) * train_percentage)
num_houses_test = int(len(house_ids) * test_percentage)
house_ids_permutation = np.random.permutation(house_ids).tolist()
train_houses = house_ids_permutation[:num_houses_train]
test_houses = house_ids_permutation[num_houses_train:]

print(f"Number of training houses: {len(train_houses)}")
print(f"Number of test houses: {len(test_houses)}")

# Save JSON files
training_houses_filename = '/data/tabletop_dataset/training_suncg_houses.json'
test_houses_filename = '/data/tabletop_dataset/test_suncg_houses.json'
with open(training_houses_filename, 'w') as save_file:  
    json.dump(train_houses, save_file)
with open(test_houses_filename, 'w') as save_file:  
    json.dump(test_houses, save_file)
# train_houses = json.load(open(training_houses_filename))
# test_houses = json.load(open(test_houses_filename))

"""

def separate_shapenet_object_classes_into_train_test_split():
    pass
"""
# IMPORTANT NOTE: Call this after `useful_named_synsets` has been defined

# Randomly separate classes into train/test
train_percentage = 0.8
test_percentage = 0.2

num_classes_train = int(len(useful_named_synsets) * train_percentage)
num_classes_test = int(len(useful_named_synsets) * test_percentage)
classes_permutation = np.random.permutation(useful_named_synsets).tolist()
train_classes = classes_permutation[:num_classes_train]
test_classes = classes_permutation[num_classes_train:]

print(f"Number of training classes: {num_classes_train}")
print(f"Number of test classes: {num_classes_test}")
"""

def separate_shapenet_object_instances_into_train_test_split():
    pass

"""
# IMPORTANT NOTE: Call this after `useful_named_synsets` has been defined

# Randomly separate class instances into train/test
np.random.seed(0) # for repeatability
train_percentage = 0.8
test_percentage = 0.2

train_models = dict()
test_models = dict()
for named_synset in useful_named_synsets:
    
    # Get textured model IDs w/out shapenet filepath
    synset_dir = shapenet_filepath + taxonomy_dict[named_synset] + '/'
    models = glob.glob(synset_dir + '*/images/')
    models = [x.split('/')[4] for x in models]
    
    # Randomize the model ID list
    models = np.random.permutation(models).tolist()
    
    # Get random train/test split
    num_train_models = int(len(models) * train_percentage)
    train_models[named_synset] = models[:num_train_models]
    test_models[named_synset] = models[num_train_models:]
    
#     print(f"Number of {named_synset} models used for training: {len(train_models[named_synset])}")
#     print(f"Number of {named_synset} models used for test: {len(test_models[named_synset])}")
    
print(f"Number of total training instances: {sum([len(x) for x in train_models.values()])}")
print(f"Number of total training instances: {sum([len(x) for x in test_models.values()])}")
    
# Save JSON files
training_instances_filename = '/data/ShapeNetCore.v2/training_instances.json'
test_instances_filename = '/data/ShapeNetCore.v2/test_instances.json'
with open(training_instances_filename, 'w') as save_file:  
    json.dump(train_models, save_file)
with open(test_instances_filename, 'w') as save_file:  
    json.dump(test_models, save_file)
# train_models = json.load(open(training_instances_filename))
# test_models = json.load(open(test_instances_filename))
    
# Reset random seed
np.random.seed(int(time.time()))
"""

def test_correctness_of_xyz_pointcloud_from_depth():
    pass
""" Test correctness of code to produce XYZ point cloud from linear z-depth 

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import sys, os

# my libraries
sys.path.insert(0, os.path.abspath('..'))
import util.simulation_util as sim_util

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF('plane.urdf')
cube_id = p.loadURDF('cube_small.urdf', basePosition=[0.025, 0.025, 0.025])

------------------------------------------------------------------------------------------------------------------------

cube_pos = list(p.getBasePositionAndOrientation(cube_id)[0])
camera_pos = cube_pos[:]
camera_pos[2] = camera_pos[2] + .3

------------------------------------------------------------------------------------------------------------------------

# Getting RGB/Depth/Segmentation images using p.computeViewMatrix()
img_width, img_height = 640, 480

# Frustum parameters
near, far = 0.01, 100
aspect_ratio = img_width/img_height
fov = 60 # field of view in angles
e = 1/(np.tan(np.radians(fov/2.)))
t = near/e; b = -t
r = t*aspect_ratio; l = -r

# Compute view/projection matrices and get camera image
camera_up_vector = np.array([1,0,0])
view_matrix = p.computeViewMatrix(camera_pos, cube_pos, camera_up_vector)
proj_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near, far)
temp = p.getCameraImage(img_width, img_height, 
                        viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL) # tuple of: width, height, rgbPixels, depthPixels, segmentation

plt.figure(1, figsize=(20,60))

# RGB image
rgb_img = np.reshape(temp[2], (img_height, img_width, 4))[..., :3]
plt.subplot(1,3,1)
plt.imshow(rgb_img)
plt.title('RGB')

# Depth image
depth_buffer = np.array(temp[3]).reshape(img_height,img_width)
depth_img = far * near / (far - (far - near) * depth_buffer)
# Note: this gives positive z values. this equation multiplies the actual negative z values by -1
#       Negative z values are because OpenGL camera +z axis points away from image
plt.subplot(1,3,2)
plt.imshow(depth_buffer) # plotting NDC depth is nicer to look at
plt.title('NDC depth')

# Segmentation image
seg_img = np.array(temp[4]).reshape(img_height,img_width) # How to reshape the list of depthPixels to an image
print(np.unique(seg_img))
plt.subplot(1,3,3)
plt.imshow(seg_img)
plt.title('Segmentation')

------------------------------------------------------------------------------------------------------------------------

# virtual camera parameters
alpha = img_width / (r-l) # pixels per meter
focal_length = near * alpha # focal length of virtual camera (frustum camera)

# Compute XYZ depth image
indices = sim_util.build_matrix_of_indices(img_height, img_width)
z_e = depth_img
x_e = (indices[..., 1] - img_width/2) * z_e / focal_length
y_e = (indices[..., 0] - img_height/2) * z_e / focal_length

------------------------------------------------------------------------------------------------------------------------

plt.figure(1, figsize=(20,60))
plt.subplot(1,3,1)
plt.imshow(x_e)
plt.title('x_e')
plt.subplot(1,3,2)
plt.imshow(y_e)
plt.title('y_e')
plt.subplot(1,3,3)
plt.imshow(z_e)
plt.title('z_e')

------------------------------------------------------------------------------------------------------------------------

max_y = camera_pos[2] * np.tan(np.radians(fov/2.))
print(f'Max y: {max_y}, computed: {np.max(y_e)}')
max_x = camera_pos[2] * np.tan(np.radians(fov/2.)) * aspect_ratio
print(f'Max x: {max_x}, computed: {np.max(x_e)}')

"""

def visualize_3d_point_cloud_with_open3d():
    pass
""" Using open3d-python library to visualize 3D xyz point cloud

import open3d
import numpy as np
P = np.loadtxt('points.txt') # Shape: [num_points x 3]
pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(P)
open3d.draw_geometries([pcd])

"""

def extract_tabletop_from_depthmap():
    pass
""" Code for extracting tabletop. some visualization too

view_num = 4

camera_pos = np.array(scene_description['views']['background+table+objects'][view_num]['camera_pos'])
lookat_pos = np.array(scene_description['views']['background+table+objects'][view_num]['lookat_pos'])
camera_up_vector = np.array(scene_description['views']['background+table+objects'][view_num]['camera_up_vector'])

img_dict = sim.get_camera_images(camera_pos, lookat_pos, camera_up_vector)

np.set_printoptions(linewidth=150)
H,W = img_dict['depth'].shape[:2]
view_mat = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
view_mat = np.array(view_mat).reshape(4,4,order='F')
cam_ext = np.linalg.inv(view_mat) # Camera extrinsics matrix


# negative depth because OpenGL camera z-axis faces behind. 
xyz_img = sim_util.compute_xyz(img_dict['depth'], sim.params) # Shape: [H x W x 3]
xyz_img[..., 2] = -1 * xyz_img[..., 2] # negate the depth to get OpenGL camera frame


# Multiply each homogenous xyz point by camera extrinsics matrix to bring it back to world coordinate frame
world_frame_depth = cam_ext.dot(np.concatenate([xyz_img, np.ones((H,W,1))], axis=2).reshape(-1,4).T)
world_frame_depth = world_frame_depth.T.reshape((H,W,4))[..., :3]


# Get tabletop. Compute histogram of y-values and pick mode of histogram. 
# It's kinda like RANSAC in 1 dimension, but using a discretization instead of randomness.
table_mask = img_dict['orig_seg_img'] == sim._obj_id_to_body[sim.table_stuff['table_obj_id']].bid
highest_y_val = round(np.max(world_frame_depth[table_mask, 1]) + 0.05, 2)

# use 1cm bins
bin_count, bin_edges = np.histogram(world_frame_depth[table_mask, 1], 
                                    bins=int(highest_y_val / .01), 
                                    range=(0,highest_y_val))
bin_index = np.argmax(bin_count)
tabletop_y_low = bin_edges[bin_index]
tabletop_y_high = bin_edges[bin_index + 1]

tabletop_mask = np.logical_and(world_frame_depth[..., 1] >= tabletop_y_low, 
                               world_frame_depth[..., 1] <= tabletop_y_high)
tabletop_mask = np.logical_and(tabletop_mask, table_mask) # Make sure tabletop_mask is subset of table


print(bin_count)
print(bin_edges)

# For visualization
if sim._mode == 'gui':
    camera_direction = lookat_pos - camera_pos
    camera_distance = np.linalg.norm(camera_direction)
    camera_direction = camera_direction / camera_distance
    camera_pitch = np.arcsin(camera_direction[1]) * 180 / np.pi
    camera_yaw = np.arctan2(camera_direction[0], camera_direction[2]) * 180 / np.pi
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, lookat_pos)

plt.imshow(world_frame_depth[..., 1])
plt.imshow(table_mask)
plt.imshow(tabletop_mask)
plt.imshow(np.logical_xor(tabletop_mask, table_mask))

"""

def parse_ShapeNetSem_csv_files():
    pass
"""

# From directory: /data/ShapeNetSem/ShapeNetSem_csv_files/

import os
import pandas as pd

# All CSV files
csv_files = [x for x in os.listdir('.') if x.endswith('.csv')]

# Create empty dataframe
shapenetsem_objects = pd.DataFrame(columns=['fullId', 'wnsynset', 'wnlemmas', 'up', 'front', 'name', 'tags'])

for csv in csv_files:
    temp = pd.read_csv(csv)
    shapenetsem_objects = shapenetsem_objects.append(temp)


# Make sure everything exists
for fid in shapenetsem_objects['fullId'].values:
    filename = '/data/ShapeNetSem/models/' + fid.split('.')[1]
    if not os.path.exists(filename + '.obj') or not os.path.exists(filename + '.mtl'):
        print(fid)

"""

def separate_shapenetsem_object_instances_into_train_test_split():
    pass
"""
# Randomly separate class instances into train/test
np.random.seed(0) # for repeatability
train_percentage = 0.9
test_percentage = 0.1

# Load all of the models
shapenetsem_objects_dataframe = sim_util.load_shapenetsem_all_models_csv() # type: pandas dataframe

# Randomize the dataframe rows
shapenetsem_objects_dataframe = shapenetsem_objects_dataframe.sample(frac=1.0) # shuffle
num_train_models = int(shapenetsem_objects_dataframe.shape[0] * train_percentage)

# Split
train_instances = shapenetsem_objects_dataframe[:num_train_models]
test_instances = shapenetsem_objects_dataframe[num_train_models:]

print(f"Number of total training instances: {train_instances.shape[0]}")
print(f"Number of total test instances: {test_instances.shape[0]}")
    
# Save csv files
train_instances.to_csv('/data/tabletop_dataset_v4/training_shapenetsem_objects.csv')
test_instances.to_csv('/data/tabletop_dataset_v4/test_shapenetsem_objects.csv')

# To load:
# train_instances = pd.read_csv('/data/tabletop_dataset_v4/training_shapenetsem_objects.csv')
# test_instances = pd.read_csv('/data/tabletop_dataset_v4/test_shapenetsem_objects.csv')
    
# Reset random seed
np.random.seed(int(time.time()))
"""