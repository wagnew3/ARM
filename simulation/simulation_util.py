import time
import glob

import numpy as np
import scipy
import scipy.spatial
from PIL import Image
import pandas as pd

import os, sys
import errno, signal
from functools import wraps

root_dir='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98'

def load_mesh(mesh_path, strict=False, scale=1.0):
    """ Borrowed from Aaron Walsman's renderpy!!
    """
    
    try:
        obj_vertices = []
        obj_normals = []
        obj_uvs = []
        obj_faces = []
        obj_vertex_colors = []
        
        mesh = {
            'vertices':[],
            'normals':[],
            'uvs':[],
            'vertex_colors':[],
            'faces':[]}
        
        #vertex_uv_mapping = {}
        #vertex_normal_mapping = {}
        vertex_face_mapping = {}
        
        with open(mesh_path) as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if tokens[0] == 'v':
                    # add a vertex
                    if len(tokens) != 4 and len(tokens) != 7:
                        if strict:
                            raise MeshError(
                                    'Vertex must have exactly three '
                                    'or six elements')
                    obj_vertices.append(
                            [float(xyz) * scale for xyz in tokens[1:4]])
                    if len(tokens) > 4:
                        obj_vertex_colors.append(
                                [float(rgb) for rgb in tokens[4:7]])
                
                if tokens[0] == 'vt':
                    # add a uv
                    if len(tokens) != 3 and len(tokens) != 4:
                        raise MeshError(
                                'UV must have two or three elements')
                    obj_uvs.append([float(uv) for uv in tokens[1:3]])
                
                if tokens[0] == 'vn':
                    # add a normal
                    if len(tokens) != 4:
                        raise MeshError(
                                'Normal must have exactly three elements')
                    obj_normals.append([float(xyz) for xyz in tokens[1:]])
                
                if tokens[0] == 'f':
                    # add a face
                    if len(tokens) != 4:
                        raise MeshError(
                                'Only triangle meshes are supported')
                    face = []
                    face_id = len(obj_faces)
                    for i, part_group in enumerate(tokens[1:]):
                        face_parts = part_group.split('/')
                        if len(face_parts) == 1:
                            face_parts = face_parts * 3
                        if len(face_parts) != 3:
                            raise MeshError(
                                    'Each face must contain an vertex, '
                                    'uv and normal')
                        if face_parts[1] == '':
                            face_parts[1] = 0
                        face_parts = [int(part)-1 for part in face_parts]
                        face.append(face_parts)
                        
                        vertex, uv, normal = face_parts
                        vertex_face_mapping.setdefault(vertex, [])
                        vertex_face_mapping[vertex].append((face_id,i))
                    obj_faces.append(face)
                    
                    #vertex_uv_mapping.setdefault(vertex, [])
                    #vertex_normal_mapping.setdefault(vertex, [])
                    #vertex_uv_mapping[vertex].append(uv)
                    #vertex_normal_mapping[vertex].append(normal)
        
        # break up the mesh so that all vertices have the same uv and normal
        for vertex_id, vertex in enumerate(obj_vertices):
            if vertex_id not in vertex_face_mapping:
                if strict:
                    raise MeshError('Vertex %i is not used in any faces'%i)
                else:
                    continue
            
            # find out how many splits need to be made by going through all
            # faces this vertex is used in and finding which normals and uvs
            # are associated with it
            face_combo_lookup = {}
            for face_id, corner_id in vertex_face_mapping[vertex_id]:
                corner = obj_faces[face_id][corner_id]
                combo = corner[1], corner[2]
                face_combo_lookup.setdefault(combo, [])
                face_combo_lookup[combo].append((face_id, corner_id))
            
            for combo in face_combo_lookup:
                uv_id, normal_id = combo
                new_vertex_id = len(mesh['vertices'])
                # fix the mesh faces
                for face_id, corner_id in face_combo_lookup[combo]:
                    obj_faces[face_id][corner_id] = [
                            new_vertex_id, new_vertex_id, new_vertex_id]
                
                mesh['vertices'].append(vertex)
                if len(obj_uvs):
                    mesh['uvs'].append(obj_uvs[uv_id])
                mesh['normals'].append(obj_normals[normal_id])
                if len(obj_vertex_colors):
                    mesh['vertex_colors'].append(obj_vertex_colors[vertex_id])
        
        mesh['faces'] = [[corner[0] for corner in face] for face in obj_faces]
    
    except:
        print('Failed to load %s'%mesh_path)
        raise
    
    return mesh


def triangle_area(pts):
    """ Computes the area of a triangle represented as 3 2D points
    
        @param pts: a [3 x 2] numpy array of xy coordinates
    """
    a = pts[0]
    b = pts[1]
    c = pts[2]
    
    side_a = np.linalg.norm(a - b, 2)
    side_b = np.linalg.norm(b - c, 2)
    side_c = np.linalg.norm(c - a, 2)
    s = 0.5 * ( side_a + side_b + side_c)
    return np.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))


def random_color():
    """ Return a random RGBA color, with alpha = 1 (fully opaque)
        RGB is in [0, 1] range
    """
    color = np.random.choice(range(256), size=3) / 256. # random RGB color
    color = np.append(color, [1]) # alpha = 1
    return color

def model_has_texture(model_dir):
    """ Check if ShapeNetCore provides a texture with this mesh
    """
    directories_in_model_dir = glob.glob(model_dir + '*/')
    has_texture = False
    for directory in directories_in_model_dir:
        if 'images' in directory:
            has_texture = True
            break
    return has_texture
    
def IoU(rect1, rect2):
    """ Calculates IoU of two rectangles.
        Assumes rectanles are in ltrb (left, right, top, bottom) format.
        ltrb is also known as x1y1x2y2 format, whch is two corners
    """
    intersection = max( min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]), 0 ) * \
                   max( min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]), 0 )

    # A1 + A2 - I
    union = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + \
            (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) - \
            intersection 

    return float(intersection) / max(union, .00001)

    
def get_collision_list(obj_id, all_obj_ids):
    """ My own simple collision checking using axis-aligned bounding boxes
    """
    obj_coords = get_object_bbox_coordinates(obj_id)
    objects_in_collision = []

    for other_obj_id in all_obj_ids:
        if other_obj_id == obj_id:
            continue
        other_obj_coords = get_object_bbox_coordinates(other_obj_id)
        
        collision = (min(obj_coords['xmax'], other_obj_coords['xmax']) - max(obj_coords['xmin'], other_obj_coords['xmin']) > 0) and \
                    (min(obj_coords['ymax'], other_obj_coords['ymax']) - max(obj_coords['ymin'], other_obj_coords['ymin']) > 0) and \
                    (min(obj_coords['zmax'], other_obj_coords['zmax']) - max(obj_coords['zmin'], other_obj_coords['zmin']) > 0)
        if collision:
            objects_in_collision.append(other_obj_id)

    return objects_in_collision


def valid_table_shape(obj_file_name, iou_threshold=0.8):
    """ Computes the xz bounding box of the vertices with the highest y value (and vertical normals)
        and compares this with the xz bounding box of the entire table. If the IoU is high enough,
        the table is mostly flat on top.

        To filter out corner tables, check if vertices make up a convex shape. To do this, we compute 
        the convex hull of the high vertices, and compute the faces that are associated with those vertices.
        Make sure the IoU is close to 1
    """
    # Load the mesh
    temp = load_mesh(obj_file_name)
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

    iou = IoU(highest_vertices_xz_rect, table_xz_rect)
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
        total_volume += triangle_area(unique_highest_vertices[face])

    # Compare total volume of unique faces with volume of computed convex hull
    is_convex = np.isclose(total_volume, conv_hull.volume, atol=1e-1)
    # if not is_convex and valid_iou: # debugging
    #     print(total_volume, conv_hull.volume, iou)
    #     if np.isnan(total_volume) or not is_convex:
    #         from IPython import embed; embed()

    return valid_iou and is_convex


def build_matrix_of_indices(height, width):
    """ Builds a [height, width, 2] numpy array containing coordinates.

        @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)


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

def saveable_depth_image(depth_img):
    """ Converts a depth image (z values) into a 16-bit saveable PNG

        @param depth_img: a [H x W] numpy array of z values in meters. dtype should be np.float32 or np.float64
    """
    return (depth_img * 1000).astype(np.uint16)



### These two functions were adatped from the DAVIS public dataset ###

def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation

def imwrite_indexed(filename,array):
    """ Save indexed png with palette."""

    palette_abspath = root_dir+'/data/tabletop_dataset_v5/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    aspect_ratio = camera_params['img_width'] / camera_params['img_height']
    e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
    t = camera_params['near'] / e; b = -t
    r = t * aspect_ratio; l = -r
    alpha = camera_params['img_width'] / (r-l) # pixels per meter
    focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
    fx = focal_length; fy = focal_length

    x_offset = camera_params['img_width']/2
    y_offset = camera_params['img_height']/2

    indices = build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    
    return xyz_img


############################## Related to ShapeNetSem ##############################

def load_shapenetsem_all_models_csv():

    csv_dir = root_dir+'/data/ShapeNetSem/ShapeNetSem_csv_files/'

    # All CSV files
    csv_files = [x for x in os.listdir(csv_dir) if x.endswith('.csv')]

    # Create empty dataframe
    shapenetsem_objects = pd.DataFrame(columns=['fullId', 'wnsynset', 'wnlemmas', 'up', 'front', 'name', 'tags'])

    for csv in csv_files:
        temp = pd.read_csv(csv_dir + csv)
        shapenetsem_objects = shapenetsem_objects.append(temp)

    return shapenetsem_objects

