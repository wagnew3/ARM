import numpy as np
import os, sys
import pybullet as p
import subprocess as sp
import time
import glob
import pandas
import json

from collections import namedtuple
from itertools import groupby
from pyquaternion import Quaternion
import trimesh

from geo.transform import Transform
import pybullet_suncg.house as suncg_house

# my libraries
sys.path.insert(0, os.path.abspath('..')) # Add parent path to the sys.path. should only be called ONCE
import simulation_util as sim_util

# handle to a simulated rigid body
Body = namedtuple('Body', ['id', 'bid', 'vid', 'cid', 'static'])

# a body-body contact record
Contact = namedtuple(
    'Contact', ['flags', 'idA', 'idB', 'linkIndexA', 'linkIndexB',
                'positionOnAInWS', 'positionOnBInWS',
                'contactNormalOnBInWS', 'distance', 'normalForce']
)

# ray intersection record
Intersection = namedtuple('Intersection', ['id', 'linkIndex', 'ray_fraction', 'position', 'normal'])


class Simulator:
    """
        This code is based on the public pybullet-based SUNCG simulator from 
            github.com/msavva/pybullet_suncg. 
        I have extended this code to randomly generate a SUNCG room with ShapeNet tables/objects.
        An instance of this class holds one PyBullet scene at a time. This scene is exported
            to a dictionary and the depth maps/labels/images are calculated and saved to disk.
    """
    def __init__(self, mode='direct', bullet_server_binary=None, suncg_data_dir_base=None, 
                 shapenet_data_dir_base=None, shapenetsem_data_dir_base=None, 
                 params=dict(), verbose=False):
        self._mode = mode
        self._verbose = verbose
        module_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize some path variables
        if suncg_data_dir_base:
            self._suncg_data_dir_base = suncg_data_dir_base
        else:
            self._suncg_data_dir_base = os.path.join(os.path.expanduser('~'), 'work', 'suncg')
        if shapenet_data_dir_base:
            self._shapenet_data_dir_base = shapenet_data_dir_base
        else:
            self._shapenet_data_dir_base = os.path.join(os.path.expanduser('~'), 'work', 'ShapeNetCore.v2')
        if shapenetsem_data_dir_base:
            self._shapenetsem_data_dir_base = shapenetsem_data_dir_base
        else:
            self._shapenetsem_data_dir_base = os.path.join(os.path.expanduser('~'), 'work', 'ShapeNetSem')
        if bullet_server_binary:
            self._bullet_server_binary = bullet_server_binary
        else:
            self._bullet_server_binary = os.path.join(module_dir, '..', 'bullet_shared_memory_server')

        # Load object class mapping. Assues ModelCategoryMapping.csv lives in suncg_data_dir_base
        self.object_class_mapping = pandas.read_csv(suncg_data_dir_base + 'ModelCategoryMapping.csv')

        # Simulation parameters
        self.params = params.copy()

        # Filtered objects
        self._filtered_objects = []

        # Initialize other stuff
        self._obj_id_to_body = {}
        self._bid_to_body = {}
        self._pid = None
        self._bullet_server = None
        self.connect()

    def connect(self):
        # disconnect and kill existing servers
        if self._pid is not None:
            p.disconnect(physicsClientId=self._pid)
            self._pid = None
        if self._bullet_server:
            print(f'Restarting by killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()
            time.sleep(1)  # seems necessary to prevent deadlock on re-connection attempt
            self._bullet_server = None

        # reconnect to appropriate server type
        if self._mode == 'gui':
            self._pid = p.connect(p.GUI)
        elif self._mode == 'direct':
            self._pid = p.connect(p.DIRECT)
        elif self._mode == 'shared_memory':
            print(f'Restarting bullet server process...')
            self._bullet_server = sp.Popen([self._bullet_server_binary])
            time.sleep(1)  # seems necessary to prevent deadlock on connection attempt
            self._pid = p.connect(p.SHARED_MEMORY)
        else:
            raise RuntimeError(f'Unknown simulator server mode={self._mode}')

        # reset and initialize gui if needed
        p.resetSimulation(physicsClientId=self._pid)
        if self._mode == 'gui':
            p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self._pid)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self._pid)

        # Set the gravity
        p.setGravity(0, -10, 0, physicsClientId=self._pid)

        # Reset, just in case this was called incorrectly
        self.reset()

    def disconnect(self):
        if self._pid is not None:
            p.disconnect(physicsClientId=self._pid)
            self._pid = None
        if self._bullet_server:
            print(f'Disconnecting. Killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()
            time.sleep(1)  # seems necessary to prevent deadlock on re-connection attempt
            self._bullet_server = None


    def __del__(self):
        if self._bullet_server:
            print(f'Process terminating. Killing bullet server pid={self._bullet_server.pid}...')
            self._bullet_server.kill()

    def set_gui_rendering(self, room=None):
        if not self._mode == 'gui':
            return False
        center = np.array([0.0, 0.0, 0.0])
        num_obj = 0

        objs_to_avoid = []
        if room is not None: # don't factor these rooms into camera target calculation
            objs_to_avoid = [body.id for body in room.body]
        for obj_id in self._obj_id_to_body.keys():
            if obj_id in objs_to_avoid:
                continue
            pos, _ = self.get_state(obj_id)
            if not np.allclose(pos, [0, 0, 0]):  # try to ignore room object 'identity' transform
                num_obj += 1
                center += pos
        center /= num_obj
        p.resetDebugVisualizerCamera(cameraDistance=5.0,
                                     cameraYaw=45.0,
                                     cameraPitch=-30.0,
                                     cameraTargetPosition=center,
                                     physicsClientId=self._pid)
        # return enabled

    def add_mesh(self, obj_id, obj_file, transform, vis_mesh_file=None, static=False):
        if static:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self._pid)
        else:
            cid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=transform.scale,
                                         physicsClientId=self._pid)
        vid = -1
        if vis_mesh_file:
            vid = p.createVisualShape(p.GEOM_MESH, fileName=vis_mesh_file, meshScale=transform.scale,
                                      physicsClientId=self._pid)
        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        mass = 0 if static else 1
        bid = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                baseVisualShapeIndex=vid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                                physicsClientId=self._pid)
        body = Body(id=obj_id, bid=bid, vid=vid, cid=cid, static=static)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    def add_box(self, obj_id, half_extents, transform, static=False):
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self._pid)
        rot_q = np.roll(transform.rotation.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        mass = 0 if static else 1
        bid = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                basePosition=transform.translation,
                                baseOrientation=rot_q,
                                physicsClientId=self._pid)
        body = Body(id=obj_id, bid=bid, vid=-1, cid=cid, static=static)
        self._obj_id_to_body[obj_id] = body
        self._bid_to_body[bid] = body
        return body

    # House-specific functions
    def add_object(self, node, create_vis_mesh=False, static=False):
        model_id = node.modelId.replace('_mirror', '')
        object_dir = os.path.join(self._suncg_data_dir_base, 'object')
        basename = f'{object_dir}/{model_id}/{model_id}'
        vis_obj_filename = f'{basename}.obj' if create_vis_mesh else None
        col_obj_filename = f'{basename}.vhacd.obj' # if you've used the VHACD algorithm to compute better collision meshes than a convex hull
        if not os.path.exists(col_obj_filename):
            # print('WARNING: collision mesh {col_obj_filename} unavailable, using visual mesh instead.')
            col_obj_filename = f'{basename}.obj'
        return self.add_mesh(obj_id=node.id, obj_file=col_obj_filename, transform=Transform.from_node(node),
                             vis_mesh_file=vis_obj_filename, static=static)

    def add_wall(self, node):
        h = node['height']
        p0 = np.transpose(np.matrix(node['points'][0]))
        p1 = np.transpose(np.matrix(node['points'][1]))
        c = (p0 + p1) * 0.5
        c[1] = h * 0.5
        dp = p1 - p0
        dp_l = np.linalg.norm(dp)
        dp = dp / dp_l
        angle = np.arccos(dp[0])
        rot_q = Quaternion(axis=[0, 1, 0], radians=angle)
        half_extents = np.array([dp_l, h, node['depth']]) * 0.5
        return self.add_box(obj_id=node['id'], half_extents=half_extents,
                            transform=Transform(translation=c, rotation=rot_q), static=True)

    def add_room(self, node, wall=True, floor=True, ceiling=False):
        def add_architecture(n, obj_file, suffix):
            return self.add_mesh(obj_id=n.id + suffix, obj_file=obj_file, transform=Transform(), 
                                 vis_mesh_file=obj_file, static=True)
        room_id = node.modelId
        room_dir = os.path.join(self._suncg_data_dir_base, 'room')
        basename = f'{room_dir}/{node.house_id}/{room_id}'
        body_ids = []
        if wall:
            body_wall = add_architecture(node, f'{basename}w.obj', '')  # treat walls as room (=room.id, no suffix)
            body_ids.append(body_wall)
        if floor:
            body_floor = add_architecture(node, f'{basename}f.obj', 'f')
            body_ids.append(body_floor)
        if ceiling:
            body_ceiling = add_architecture(node, f'{basename}c.obj', 'c')
            body_ids.append(body_ceiling)
        return body_ids


    def add_random_house_room(self, no_walls=False, no_ceil=True, no_floor=False, 
                              use_separate_walls=False, only_architecture=False, static=True):
        """ Select a random room from a random house and load it

            @param house_ids: List of house IDs
            @param valid_room_types: List of valid room types
        """
        room = None
        while room is None:
            house_id = np.random.choice(self.params['house_ids'])
            house = suncg_house.House(house_json=f'{self._suncg_data_dir_base}/house/{house_id}/house.json')

            for _room in house.rooms:
                valid_room_type = len(set(_room.roomTypes).intersection(self.params['valid_room_types'])) > 0
                room_xsize = _room.bbox['max'][0] - _room.bbox['min'][0]
                room_ysize = _room.bbox['max'][2] - _room.bbox['min'][2]
                valid_room_size = (room_xsize > self.params['min_xlength']) and \
                                  (room_ysize > self.params['min_ylength'])
                if valid_room_type and valid_room_size:
                    if self._verbose:
                        print(f"Using a {_room.roomTypes}")
                    room = _room
                    break

        # Print size of room
        room_xyz_center = np.array([room_xsize, 0, room_ysize]) / 2
        if self._verbose:
            print(f"Room xsize, zsize: {room_xsize}, {room_ysize}")
            print(f'Room xyz center: {room_xyz_center}')

        # Load the room
        self.add_house_room(house, room, no_walls=no_walls, no_ceil=no_ceil, no_floor=no_floor, 
                            use_separate_walls=use_separate_walls, only_architecture=only_architecture, 
                            static=static)

    def add_house_room(self, house, room, no_walls=False, no_ceil=True, no_floor=False, 
                       use_separate_walls=False, only_architecture=False, static=True):

        # Don't allow rendering. Speeds up the loading of the room
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._pid)

        room_node = [node for node in house.nodes if node.id == room.id]
        if len(room_node) < 1:
            raise Exception('Missing Room')
        if only_architecture:
            house.nodes = room_node
        else:
            house.nodes = [node for node in room.nodes]
            house.nodes.append(room_node[0])

        # First, filter out objects
        for node in house.nodes:
            if not node.valid:
                continue
            if node.type == 'Object':
                # Filtering out happens here
                classes = self.get_object_class(node)
                if classes['NYUv2'] in self.params['nyuv2_40_classes_filter_list'] or \
                   classes['Coarse'] in self.params['coarse_grained_classes_filter_list']:
                    if self._verbose:
                        print(f"Filtered a {classes['NYUv2']}, {classes['Coarse']}")
                    self._filtered_objects.append(node) # keep track of this so I can filter stuff that is on top

        # Now, add the meshes
        for node in house.nodes:
            if not node.valid:
                continue

            # Initiliaze the .body attribute
            if not hasattr(node, 'body'):
                node.body = None

            if node.type == 'Object':
                if node in self._filtered_objects:
                    continue
                if self.on_top_of_filtered_object(node):
                    classes = self.get_object_class(node)
                    if self._verbose:
                        print(f"Filtered a {classes['NYUv2']}, {classes['Coarse']} which was on top of a filtered object")
                    continue 
                # If not filtered, add the object to the scene
                node.body = self.add_object(node, create_vis_mesh=True, static=static)

            if node.type == 'Room':
                ceil = False if no_ceil else not (hasattr(node, 'hideCeiling') and node.hideCeiling == 1)
                wall = False if (no_walls or use_separate_walls) else not (hasattr(node, 'hideWalls') and node.hideWalls == 1)
                floor = False if no_floor else not (hasattr(node, 'hideFloor') and node.hideFloor == 1)
                node.body = self.add_room(node, wall=wall, floor=floor, ceiling=ceil)

            if node.type == 'Box':
                half_widths = list(map(lambda x: 0.5 * x, node.dimensions))
                node.body = self.add_box(obj_id=node.id, half_extents=half_widths, transform=Transform.from_node(node),
                                         static=static)

        if use_separate_walls and not no_walls:
            for wall in house.walls:
                wall['body'] = self.add_wall(wall)

        # Move room back to origin
        room_bbox = room.bbox
        for obj_id in self._obj_id_to_body.keys():
            pos, rot = self.get_state(obj_id)
            new_pos = list(np.array(pos) - np.array(room_bbox['min']))
            self.set_state(obj_id, new_pos, rot)
        self.set_gui_rendering(room=room)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._pid)

        self.loaded_room = room

    def on_top_of_filtered_object(self, node):
        """ Check if this object is on top of a filtered object
        """
        on_top = False
        for obj_node in self._filtered_objects:
            inside_xz = node.bbox['min'][0] >= obj_node.bbox['min'][0] and \
                        node.bbox['min'][2] >= obj_node.bbox['min'][2] and \
                        node.bbox['max'][0] <= obj_node.bbox['max'][0] and \
                        node.bbox['max'][2] <= obj_node.bbox['max'][2]
            higher_y = node.bbox['min'][1] > (obj_node.bbox['min'][1] + obj_node.bbox['max'][1])/2
            if inside_xz and higher_y:
                on_top = True
                break
        return on_top

    def get_object_class(self, node):
        """ Get class w.r.t. NYUv2 mappings and coarse grained mappings (provided ny the SUNCG dataset)
        """
        mID = node.modelId
        if '_mirror' in mID: # weird corner case of mirrored objects
            mID = mID.split('_mirror')[0]
        nyuv2_40class = self.object_class_mapping[self.object_class_mapping['model_id'] == mID]['nyuv2_40class'].item()
        coarse_grained_class = self.object_class_mapping[self.object_class_mapping['model_id'] == mID]['coarse_grained_class'].item()
        return {'NYUv2' : nyuv2_40class,
                'Coarse' : coarse_grained_class}


    def remove(self, obj_id):
        body = self._obj_id_to_body[obj_id]
        p.removeBody(bodyUniqueId=body.bid, physicsClientId=self._pid)
        del self._obj_id_to_body[obj_id]
        del self._bid_to_body[body.bid]

    def set_state(self, obj_id, position, rotation_q):
        body = self._obj_id_to_body[obj_id]
        rot_q = np.roll(rotation_q.elements, -1)  # w,x,y,z -> x,y,z,w (which pybullet expects)
        p.resetBasePositionAndOrientation(bodyUniqueId=body.bid, posObj=position, ornObj=rot_q,
                                          physicsClientId=self._pid)

    def get_state(self, obj_id):
        body = self._obj_id_to_body[obj_id]
        pos, q = p.getBasePositionAndOrientation(bodyUniqueId=body.bid, physicsClientId=self._pid)
        rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pos, rotation

    def step(self):
        p.stepSimulation(physicsClientId=self._pid)

    def reset(self):
        p.resetSimulation(physicsClientId=self._pid)
        self._obj_id_to_body = {}
        self._bid_to_body = {}
        self._filtered_objects = []
        p.setGravity(0, -10, 0, physicsClientId=self._pid)

    def get_closest_point(self, obj_id_a, obj_id_b, max_distance=np.inf):
        """
        Return record with distance between closest points between pair of nodes if within max_distance or None.
        """
        bid_a = self._obj_id_to_body[obj_id_a].bid
        bid_b = self._obj_id_to_body[obj_id_b].bid
        cps = p.getClosestPoints(bodyA=bid_a, bodyB=bid_b, distance=max_distance, physicsClientId=self._pid)
        cp = None
        if len(cps) > 0:
            closest_points = self._convert_contacts(cps)
            cp = min(closest_points, key=lambda x: x.distance)
        del cps  # NOTE force garbage collection of pybullet objects
        return cp

    def get_contacts(self, obj_id_a=None, obj_id_b=None, only_closest_contact_per_pair=True,
                     include_collision_with_static=True):
        """
        Return all current contacts. When include_collision_with_statics is true, include contacts with static bodies
        """
        bid_a = self._obj_id_to_body[obj_id_a].bid if obj_id_a else -1
        bid_b = self._obj_id_to_body[obj_id_b].bid if obj_id_b else -1
        cs = p.getContactPoints(bodyA=bid_a, bodyB=bid_b, physicsClientId=self._pid)
        contacts = self._convert_contacts(cs)
        del cs  # NOTE force garbage collection of pybullet objects

        if not include_collision_with_static:
            def not_contact_with_static(c):
                static_a = self._obj_id_to_body[c.idA].static
                static_b = self._obj_id_to_body[c.idB].static
                return not static_a and not static_b
            contacts = filter(not_contact_with_static, contacts)
            # print(f'#all_contacts={len(all_contacts)} to #non_static_contacts={len(non_static_contacts)}')

        if only_closest_contact_per_pair:
            def bid_pair_key(x):
                return str(x.idA) + '_' + str(x.idB)
            contacts = sorted(contacts, key=bid_pair_key)
            min_dist_contact_by_pair = {}
            for k, g in groupby(contacts, key=bid_pair_key):
                min_dist_contact = min(g, key=lambda x: x.distance)
                min_dist_contact_by_pair[k] = min_dist_contact
            contacts = min_dist_contact_by_pair.values()

        # convert into dictionary of form (id_a, id_b) -> Contact
        contacts_dict = {}
        for c in contacts:
            key = (c.idA, c.idB)
            contacts_dict[key] = c

        return contacts_dict

    def _convert_contacts(self, contacts):
        out = []
        for c in contacts:
            bid_a = c[1]
            bid_b = c[2]
            if bid_a not in self._bid_to_body or bid_b not in self._bid_to_body:
                continue
            id_a = self._bid_to_body[bid_a].id
            id_b = self._bid_to_body[bid_b].id
            o = Contact(flags=c[0], idA=id_a, idB=id_b, linkIndexA=c[3], linkIndexB=c[4],
                        positionOnAInWS=c[5], positionOnBInWS=c[6], contactNormalOnBInWS=c[7],
                        distance=c[8], normalForce=c[9])
            out.append(o)
        return out

    def ray_test(self, from_pos, to_pos):
        hit = p.rayTest(rayFromPosition=from_pos, rayToPosition=to_pos, physicsClientId=self._pid)
        intersection = Intersection._make(*hit)
        del hit  # NOTE force garbage collection of pybullet objects
        if intersection.id >= 0:  # if intersection, replace bid with id
            intersection = intersection._replace(id=self._bid_to_body[intersection.id].id)
        return intersection




    ########## FUNCTIONS FOR LOADING SHAPENET MODELS INTO SIMULATION ##########



    ##### UTILITIES #####
    def get_object_bbox_coordinates(self, obj_id):
        """ Return min/max coordinates of an encapsulating bounding box in x,y,z dims
        
            @param obj_id: ID of object
            @return: an np.array: [ [xmin, ymin, zmin],
                                    [xmax, ymax, zmax] ]
        """

        # Get max/min coordinates of table
        obj_min, obj_max = p.getAABB(self._obj_id_to_body[obj_id].bid, physicsClientId=self._pid)
          
        return {'xmin' : obj_min[0],
                'ymin' : obj_min[1],
                'zmin' : obj_min[2],
                'xmax' : obj_max[0],
                'ymax' : obj_max[1],
                'zmax' : obj_max[2],
                'xsize' : obj_max[0] - obj_min[0],
                'ysize' : obj_max[1] - obj_min[1],
                'zsize' : obj_max[2] - obj_min[2]
               }

    def model_has_texture(self, model_dir):
        """ Check if ShapeNetCore provides a texture with this mesh
        """
        directories_in_model_dir = glob.glob(model_dir + '*/')
        has_texture = False
        for directory in directories_in_model_dir:
            if 'images' in directory:
                has_texture = True
                break
        return has_texture

    def get_collision_list(self, obj_id, all_obj_ids):
        """ My own simple collision checking using axis-aligned bounding boxes

            @param obj_id: ID of query object
            @param all_obj_ids: list of IDs of objects you want to check collision with
        """
        obj_coords = self.get_object_bbox_coordinates(self._obj_id_to_body[obj_id].id)
        objects_in_collision = []

        for other_obj_id in all_obj_ids:
            if other_obj_id == obj_id:
                continue
            other_obj_coords = self.get_object_bbox_coordinates(self._obj_id_to_body[other_obj_id].id)
            
            collision = (min(obj_coords['xmax'], other_obj_coords['xmax']) - max(obj_coords['xmin'], other_obj_coords['xmin']) > 0) and \
                        (min(obj_coords['ymax'], other_obj_coords['ymax']) - max(obj_coords['ymin'], other_obj_coords['ymin']) > 0) and \
                        (min(obj_coords['zmax'], other_obj_coords['zmax']) - max(obj_coords['zmin'], other_obj_coords['zmin']) > 0)
            if collision:
                objects_in_collision.append(other_obj_id)

        return objects_in_collision

    def simulate(self, timesteps):
        """ Simulate dynamics. Don't allow rendering, which speeds up the process
        """
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._pid)
        for i in range(timesteps): 
            p.stepSimulation(physicsClientId=self._pid)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._pid)




    ##### CODE TO SIMULATE SCENES #####

    def generate_random_table(self):
        """ Randomly generate a shapenet table that is standing up in loaded SUNCG room
        """

        room_coords = self.get_object_bbox_coordinates(self.loaded_room.body[0].id) # Get bbox coordinates of room mesh

        is_up = False
        num_tried_tables = 0
        while not is_up:
            if num_tried_tables > self.params['max_initialization_tries']:
                self.table_stuff = None
                return
            num_tried_tables += 1

            ### Select random table ###
            model_dir = np.random.choice(self.params['valid_tables'])
            model_dir = self._shapenet_data_dir_base + self.params['taxonomy_dict']['table'] + '/' + model_dir + '/'
            table_mesh_filename = model_dir + 'models/model_normalized.obj'

            ### Create table object in pybullet ### 
            # Note: up direction is +Y axis in ShapeNet models
            table_obj_id = 'ShapeNet_table_0'
            table_transform = Transform()
            table_body = self.add_mesh(table_obj_id, table_mesh_filename, table_transform, table_mesh_filename)

            # Re-scale table to appropriate height and load it right above ground
            table_coords = self.get_object_bbox_coordinates(table_obj_id)
            table_height = table_coords['ysize']
            max_scale_factor = self.params['max_table_height'] / table_height
            min_scale_factor = self.params['min_table_height'] / table_height
            table_scale_factor = np.random.uniform(min_scale_factor, max_scale_factor)
            table_transform.rescale(table_scale_factor)

            # Reload the resacled table right above the ground
            self.remove(table_obj_id)
            table_body = self.add_mesh(table_obj_id, table_mesh_filename, table_transform, table_mesh_filename)
            table_coords = self.get_object_bbox_coordinates(table_obj_id) # scaled coordinates

            # List of pybullet object ids to check collsion
            room_obj_ids = [x.body.id for x in self.loaded_room.nodes if x.body is not None]

            # Walls id
            walls_id = self.loaded_room.body[0].id
            floor_coords = self.get_object_bbox_coordinates(self.loaded_room.body[1].id)

            # Sample xz_location until it's not in collision
            in_collision_w_objs = True; in_collision_w_walls = True
            num_tries = 0
            while in_collision_w_objs or in_collision_w_walls:

                xmin = (room_coords['xmin'] - table_coords['xmin']) / self.params['table_init_factor']
                xmax = (room_coords['xmax'] - table_coords['xmax']) * self.params['table_init_factor']
                random_start_xpos = np.random.uniform(xmin, xmax)
                ypos = floor_coords['ymax'] - table_coords['ymin'] + np.random.uniform(0, 0.1) # add a bit of space so no collision w/ carpets...
                zmin = (room_coords['zmin'] - table_coords['zmin']) / self.params['table_init_factor']
                zmax = (room_coords['zmax'] - table_coords['zmax']) * self.params['table_init_factor']
                random_start_zpos = np.random.uniform(zmin, zmax)
                if (xmax < xmin) or (zmax < zmin): # table is too large. pick a new table
                    break
                random_start_pos = np.array([random_start_xpos, ypos, random_start_zpos])
                self.set_state(table_obj_id, random_start_pos, Quaternion(x=0, y=0, z=0, w=1))

                if num_tries > self.params['max_initialization_tries']:
                    break # this will force code to pick a new table
                num_tries += 1

                # Check if it's in collision with anything
                in_collision_w_walls = self.get_closest_point(table_obj_id, walls_id).distance < 0
                in_collision_w_objs = len(self.get_collision_list(table_obj_id, room_obj_ids)) > 0 # Simpler coarse collision checking with these objects

            if in_collision_w_objs or in_collision_w_walls: # if still in collision, then it's because we exhausted number of tries
                self.remove(table_obj_id)
                continue

            # Let the table fall for about 1 second
            self.simulate(300)

            # Check if it fell down
            up_orientation = np.array([1,0,0,0]) # w,x,y,z
            is_up = np.allclose(self.get_state(table_obj_id)[1].elements, up_orientation, atol=1e-1)

            # Remove the table if it fell down or is in collision
            if not is_up:
                self.remove(table_obj_id)
                continue # yes, this is redundant
         
        # point debug camera at table
        if self._mode == 'gui':
            table_coords = self.get_object_bbox_coordinates(table_obj_id)
            table_pos = list(self.get_state(table_obj_id)[0])
            table_pos[1] = table_coords['ymax']
            p.resetDebugVisualizerCamera(cameraDistance=2.0,
                                         cameraYaw=45.0,
                                         cameraPitch=-30.0,
                                         cameraTargetPosition=table_pos,
                                         physicsClientId=self._pid)

        self.table_stuff = {'table_obj_id' : table_obj_id,
                            'table_mesh_filename' : table_mesh_filename,
                            'table_scale_factor' : table_scale_factor,
                           }

         

    def generate_random_shapenet_models(self):
        """ Sample random ShapeNet models

            NOTE: This is to be called AFTER self.generate_random_table() has been called
        """

        ##### Sample random ShapeNet models #####
        obj_mesh_filenames = []
        obj_ids = []
        obj_scale_factors = []

        # Get max/min coordinates of table
        table_coords = self.get_object_bbox_coordinates(self.table_stuff['table_obj_id'])
        
        i = 0
        num_objects_for_scene = np.random.randint(self.params['min_num_objects_per_scene'],
                                                  self.params['max_num_objects_per_scene']+1)
        if self._verbose:
            print(f"Number of objects chosen for scene: {num_objects_for_scene}")
        while len(obj_ids) < num_objects_for_scene:

            ### Sample the object ###
            if self.params['is_shapenetsem']:
                # self.params['object_ids'] is a Pandas DataFrame 
                selected_index = np.random.randint(0, self.params['object_ids'].shape[0])
                shapenetsem_obj_id = self.params['object_ids'].iloc[selected_index]
                obj_mesh_filename = self._shapenetsem_data_dir_base + shapenetsem_obj_id['fullId'].split('.')[1] + '.obj'
                obj_mesh_filenames.append(obj_mesh_filename)
            else:
                synsets = list(self.params['object_ids'].keys())
                synset_to_sample = np.random.choice(synsets)
                model_dir = np.random.choice(self.params['object_ids'][synset_to_sample])
                model_dir = self._shapenet_data_dir_base + self.params['taxonomy_dict'][synset_to_sample] + '/' + model_dir + '/'
                obj_mesh_filename = model_dir + 'models/model_normalized.obj'
                obj_mesh_filenames.append(model_dir + 'models/model_normalized.obj')

            ### Create an object in pybullet ###
            obj_id = f'ShapeNet_obj_{i}'
            obj_transform = Transform()
            obj_body = self.add_mesh(obj_id, obj_mesh_filename, obj_transform, obj_mesh_filename)

            ### Sample scale of object ###
            canonical_obj_coords = self.get_object_bbox_coordinates(obj_id)

            # Compute scale factor
            xscale_factor = 1; yscale_factor = 1; zscale_factor = 1
            if canonical_obj_coords['xsize'] / table_coords['xsize'] > self.params['max_xratio']:
                xscale_factor = self.params['max_xratio'] * table_coords['xsize'] / (canonical_obj_coords['xsize'])
            if canonical_obj_coords['ysize'] / table_coords['ysize'] > self.params['max_yratio']:
                yscale_factor = self.params['max_yratio'] * table_coords['ysize'] / (canonical_obj_coords['ysize'])
            if canonical_obj_coords['zsize'] / table_coords['zsize'] > self.params['max_zratio']:
                zscale_factor = self.params['max_zratio'] * table_coords['zsize'] / (canonical_obj_coords['zsize'])
            max_scale_factor = min(xscale_factor, yscale_factor, zscale_factor)
            obj_scale_factor = np.random.uniform(max_scale_factor * 0.75, max_scale_factor)
            obj_scale_factors.append(obj_scale_factor)
            obj_transform.rescale(obj_scale_factor)


            ##### Sample random location/orientation for object #####

            # Make sure the object is not in collision with any other object
            in_collision = True
            num_tries = 0
            while in_collision:

                # Get all objects that are straight (these are the ones that could be supporting objects)
                straight_obj_ids = [x for x in obj_ids if np.allclose(self.get_state(x)[1].elements, np.array([1,0,0,0]))]

                # Sample a random starting orientation
                sample = np.random.rand()
                if sample < 0.4: # Simulate straight up

                    q = np.array([0,0,0,1])
                    extra_y = 0.
                    x_range_min = table_coords['xmin'] - canonical_obj_coords['xmin']
                    x_range_max = table_coords['xmax'] - canonical_obj_coords['xmax']
                    z_range_min = table_coords['zmin'] - canonical_obj_coords['zmin']
                    z_range_max = table_coords['zmax'] - canonical_obj_coords['zmax']

                elif sample < 0.8 and len(straight_obj_ids) > 1: # put it one another object

                    q = np.array([0,0,0,1])
                    support_obj_id = np.random.choice(straight_obj_ids)
                    support_obj_coords = self.get_object_bbox_coordinates(support_obj_id)
                    extra_y = support_obj_coords['ysize'] + 1e-3 # 1mm for some extra wiggle room

                    # Select x,z coordinates to place it randomly on top
                    x_range_min = support_obj_coords['xmin'] - canonical_obj_coords['xmin']
                    x_range_max = support_obj_coords['xmax'] - canonical_obj_coords['xmax']
                    z_range_min = support_obj_coords['zmin'] - canonical_obj_coords['zmin']
                    z_range_max = support_obj_coords['zmax'] - canonical_obj_coords['zmax']

                    # If supporting object is too small, just place it in the middle
                    if x_range_min > x_range_max:
                        x_range_min = (support_obj_coords['xmin'] + support_obj_coords['xmax']) / 2.
                        x_range_max = x_range_min
                    if z_range_min > z_range_max:
                        z_range_min = (support_obj_coords['zmin'] + support_obj_coords['zmax']) / 2.
                        z_range_max = z_range_min

                else: # Simulate a random orientation

                    q = np.random.uniform(0, 2*np.pi, 3) # Euler angles
                    q = p.getQuaternionFromEuler(q)
                    extra_y = np.random.uniform(0, self.params['delta'])
                    x_range_min = table_coords['xmin'] - canonical_obj_coords['xmin']
                    x_range_max = table_coords['xmax'] - canonical_obj_coords['xmax']
                    z_range_min = table_coords['zmin'] - canonical_obj_coords['zmin']
                    z_range_max = table_coords['zmax'] - canonical_obj_coords['zmax']

                # Load this in and get axis-aligned bounding box
                self.remove(obj_id)
                obj_transform._r = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]) # HACK. Not good to access "private" attribute _r, but whatevs
                obj_body = self.add_mesh(obj_id, obj_mesh_filename, obj_transform, obj_mesh_filename)
                obj_coords = self.get_object_bbox_coordinates(obj_id) # scaled coordinates

                # Sample a random starting location
                random_start_xpos = np.random.uniform(x_range_min, x_range_max)
                ypos = table_coords['ymax'] - obj_coords['ymin'] + extra_y
                random_start_zpos = np.random.uniform(z_range_min, z_range_max)
                random_start_pos = np.array([random_start_xpos, ypos, random_start_zpos])

                # Set position/orientation
                self.set_state(obj_id, random_start_pos, obj_transform._r) # HACK. Not good to access "private" attribute _r, but whatevs

                if num_tries > self.params['max_initialization_tries']:
                    break
                num_tries += 1

                # Check for collision
                in_collision = len(self.get_collision_list(obj_id, obj_ids)) > 0

            if in_collision: # if still in collision, then it's because we exhausted number of tries
                self.remove(obj_id)
                obj_mesh_filenames.pop(-1) # remove this since we aren't using this object
                obj_scale_factors.pop(-1) # remove this since we aren't using this object
                continue

            # If we get here, then object has successfully by initialized
            obj_ids.append(obj_id)
            i += 1


        self.shapenet_obj_stuff = {'obj_ids' : obj_ids,
                                   'obj_mesh_filenames': obj_mesh_filenames,
                                   'obj_scale_factors' : obj_scale_factors,
                                  }

    def remove_fallen_objects(self):
        """ Remove any objects that have fallen (i.e. are lower than the table)
        """
        table_coords = self.get_object_bbox_coordinates(self.table_stuff['table_obj_id'])
        num_objects = len(self.shapenet_obj_stuff['obj_ids'])
        
        fallen_obj_ids = []
        for obj_id in self.shapenet_obj_stuff['obj_ids']:
            obj_coords = self.get_object_bbox_coordinates(obj_id)
            obj_ypos = (obj_coords['ymin'] + obj_coords['ymax']) / 2.
            if obj_ypos < table_coords['ymax']:
                fallen_obj_ids.append(obj_id)

        # This code actually removes the object from the scene
        for obj_id in fallen_obj_ids:
            self.remove(obj_id)

        # Update self.shapenet_obj_stuff dictionary
        valid_indices = []
        for i, obj_id in enumerate(self.shapenet_obj_stuff['obj_ids']):
            if obj_id not in fallen_obj_ids:
                valid_indices.append(i)
        self.shapenet_obj_stuff['obj_ids'] = [self.shapenet_obj_stuff['obj_ids'][i] 
                                              for i in range(num_objects) if i in valid_indices]
        self.shapenet_obj_stuff['obj_mesh_filenames'] = [self.shapenet_obj_stuff['obj_mesh_filenames'][i] 
                                                         for i in range(num_objects) if i in valid_indices]
        self.shapenet_obj_stuff['obj_scale_factors'] = [self.shapenet_obj_stuff['obj_scale_factors'][i] 
                                                        for i in range(num_objects) if i in valid_indices]

    def export_scene_to_dictionary(self):
        """ Exports the PyBullet scene to a dictionary
        """

        # Initialize empty scene description
        scene_description = {}

        # House/Room description
        room_description = {'house_id' : self.loaded_room.house_id,
                            'room_id' : self.loaded_room.id}
        scene_description['room'] = room_description

        # Table description
        temp = self.get_state(self.table_stuff['table_obj_id'])
        table_description = {'mesh_filename' : self.table_stuff['table_mesh_filename'],
                             'position' : list(temp[0]),
                             'orientation' : list(temp[1].elements), # w,x,y,z quaternion
                             'scale' : self.table_stuff['table_scale_factor']}
        scene_description['table'] = table_description

        # Get descriptions of objects on table
        object_descriptions = []
        for i, obj_id in enumerate(self.shapenet_obj_stuff['obj_ids']):

            mesh_filename = self.shapenet_obj_stuff['obj_mesh_filenames'][i]
            temp = self.get_state(obj_id)
            pos = list(temp[0])
            orientation = list(temp[1].elements) # w,x,y,z quaternion
            scale = self.shapenet_obj_stuff['obj_scale_factors'][i]

            description = {'mesh_filename' : mesh_filename,
                           'position' : pos,
                           'orientation' : orientation,
                           'scale' : scale}
            object_descriptions.append(description)
            
        scene_description['object_descriptions'] = object_descriptions

        return scene_description

    @sim_util.timeout(75) # Limit this function to max of _ seconds
    def generate_scenes(self, num_scenes):

        scenes = []
        times = []
        while len(scenes) < num_scenes:

            start_time = time.time()

            self.reset()
            self.add_random_house_room(no_walls=False, no_ceil=False, no_floor=False, 
                                       use_separate_walls=False, only_architecture=False, 
                                       static=True)          
            self.generate_random_table()
            if self.table_stuff is None: # This means we tried many tables and it didn't work
                continue # start over
            self.generate_random_shapenet_models()
            self.simulate(self.params['simulation_steps'])
            self.remove_fallen_objects()
            
            # Check some bad situations
            if len(self.shapenet_obj_stuff['obj_ids']) == 0: # everything fell off
                continue # start over
            if self.get_state(self.table_stuff['table_obj_id'])[0][1] < -0.1: # table fell way way way down
                continue # start over

            ### Export scene to dictionary ###
            scene_description = self.export_scene_to_dictionary()
            scenes.append(scene_description)

            # End of while loop. timing stuff
            end_time = time.time()
            times.append(round(end_time - start_time, 3))

        if self._verbose:
            print("Time taken to generate scene: {0} seconds".format(sum(times)))
            print("Average time taken to generate scene: {0} seconds".format(np.mean(times)))

        return scenes

    def load_house_room(self, scene_description):
        """ Takes a scene descrption dictionary (as exported by self.export_scene_to_dictionary())
            and loads the house only

            NOTE: This MUST be called before load_table() or load_objects

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """

        house = suncg_house.House(house_json=f"{self._suncg_data_dir_base}/house/{scene_description['room']['house_id']}/house.json")
        room = [r for r in house.rooms if r.id == scene_description['room']['room_id']][0]
        self.add_house_room(house, room, no_walls=False, no_ceil=True, no_floor=False, use_separate_walls=False,
                           only_architecture=False, static=True)

    def load_table(self, scene_description):
        """ Takes a scene descrption dictionary (as exported by self.export_scene_to_dictionary())
            and loads the table only

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """        
        table_description = scene_description['table']
        if not table_description['mesh_filename'].startswith(self._shapenet_data_dir_base):
            table_description['mesh_filename'] = self._shapenet_data_dir_base + table_description['mesh_filename']
        table_transform = Transform(translation=np.array(table_description['position']), 
                                    rotation=Quaternion(w=table_description['orientation'][0],
                                                        x=table_description['orientation'][1],
                                                        y=table_description['orientation'][2],
                                                        z=table_description['orientation'][3]),
                                    scale=np.ones(3) * 0.00001)#table_description['scale'])
        table_obj_id = 'ShapeNet_table_0'
        self.add_mesh(table_obj_id, 
                     table_description['mesh_filename'], 
                     table_transform, 
                     table_description['mesh_filename'])
        print('no table!')
        self.table_stuff = {'table_obj_id' : table_obj_id,
                           'table_mesh_filename' : table_description['mesh_filename'],
                           'table_scale_factor' : table_description['scale'],
                          }

    def load_objects(self, scene_description):
        """ Takes a scene descrption dictionary (as exported by self.export_scene_to_dictionary())
            and loads the objects only

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """                
        object_descriptions = scene_description['object_descriptions']
                                  
        obj_ids = []
        for i, obj in enumerate(object_descriptions):
            if not obj['mesh_filename'].startswith(self._shapenet_data_dir_base):
                obj['mesh_filename'] = self._shapenet_data_dir_base + obj['mesh_filename']
            obj_id = f'ShapeNet_obj_{i}'
            
            rotation=Quaternion(w=obj['orientation'][0], x=obj['orientation'][1], y=obj['orientation'][2], z=obj['orientation'][3])
            translation=np.array(obj['position'])
            
            #calculate center of gravity with transforms
            cog_path=os.path.dirname(obj['mesh_filename'])+'/cog.json'
            if os.path.exists(cog_path):
                com=np.array(json.load(open(cog_path))['cog'])
                u=0
            else:
                tmesh = trimesh.load(obj['mesh_filename'])
                com=tmesh.centroid
                json.dump({'cog': np.ndarray.tolist(com)}, open(cog_path, 'w')) 
            com=com*obj['scale']
            com=rotation.rotate(com)
            com_t=translation+com
            obj['cog']=np.ndarray.tolist(com_t)
            
            obj_transform = Transform(translation=translation,
                                      rotation=rotation,
                                      scale=np.ones(3) * obj['scale']
                                     )
            self.add_mesh(obj_id,
                         obj['mesh_filename'],
                         obj_transform,
                         obj['mesh_filename']
                        )
            obj_ids.append(obj_id)

        self.shapenet_obj_stuff = {}
        self.shapenet_obj_stuff['obj_ids'] = obj_ids
        self.shapenet_obj_stuff['obj_mesh_filenames'] = [x['mesh_filename'] for x in object_descriptions]
        self.shapenet_obj_stuff['obj_scale_factors'] = [x['scale'] for x in object_descriptions]
        self.shapenet_obj_stuff['cog'] = [x['cog'] for x in object_descriptions]

    def load_scene(self, scene_description):
        """ Takes a scene description dictionary (as exported by self.export_scene_to_dictionary())
            and loads it

            @param scene_description: a scene description dictionary (schema can be found in 
                                                          self.export_scene_to_dictionary())
        """
        # copy dictionary so we don't overwrite original
        import copy # use copy.deepcopy for nested dictionaries
        scene_description = copy.deepcopy(scene_description)

        # Reset the scene
        self.reset()

        # Load the room
        self.load_house_room(scene_description)
                                  
        # Load the table
        self.load_table(scene_description)
                                  
        # Load the objects
        self.load_objects(scene_description)

    def sample_room_view(self):
        """ Sample a view inside room
        """

        walls_coords = self.get_object_bbox_coordinates(self.loaded_room.body[0].id)

        # Sample anywhere inside room
        camera_pos = np.random.uniform(0,1, size=[3])
        camera_pos[0] = np.random.uniform(walls_coords['xmin'] + .25 * walls_coords['xsize'], 
                                          walls_coords['xmax'] - .25 * walls_coords['xsize'])
        camera_pos[1] = np.random.uniform(max(walls_coords['ymin'] + .25 * walls_coords['ysize'], 1.0), # minimum y height is 1.0 meters
                                          walls_coords['ymax'] - .25 * walls_coords['ysize'])
        camera_pos[2] = np.random.uniform(walls_coords['zmin'] + .25 * walls_coords['zsize'], 
                                          walls_coords['zmax'] - .25 * walls_coords['zsize'])

        # Sample a "lookat" position. Take the vector [0,0,1], rotate it on xz plane (about y-axis), then on yz plane (about x-axis)
        xz_plane_theta = np.random.uniform(0, 2*np.pi)   # horizontal rotation. rotate on xz plane, about y-axis
        yz_plane_theta = np.random.uniform(0, np.pi / 6) # up-down rotation. rotate on yz plane, about x-axis

        # Compose the two extrinsic rotations
        quat = p.multiplyTransforms(np.array([0,0,0]),
                                    p.getQuaternionFromEuler(([0,xz_plane_theta,0])), 
                                    np.array([0,0,0]),
                                    p.getQuaternionFromEuler(([yz_plane_theta,0,0]))
                                   )[1]

        direction = np.array([0,0,1])
        direction = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3,3).dot(direction)
        lookat_pos = camera_pos + direction

        if self._mode == 'gui':

            # Calculate yaw, pitch, direction for camera (Bullet needs this)
            # Equations for pitch/yaw is taken from:
            #     gamedev.stackexchange.com/questions/112565/finding-pitch-yaw-values-from-lookat-vector
            camera_direction = lookat_pos - camera_pos
            camera_distance = np.linalg.norm(camera_direction)
            camera_direction = camera_direction / camera_distance
            camera_pitch = np.arcsin(camera_direction[1]) * 180 / np.pi
            camera_yaw = np.arctan2(camera_direction[0], camera_direction[2]) * 180 / np.pi
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, lookat_pos)

        return self.get_camera_images(camera_pos, lookat_pos)

    def sample_table_view(self):
        """ Sample a view near table
        """

        # Sample position on xz bbox of table
        table_coords = self.get_object_bbox_coordinates(self.table_stuff['table_obj_id'])

        # First, select a side
        xz_bbox_side_probs = np.array([table_coords['xsize'], # x side 1
                                       table_coords['zsize'], # z side 1
                                       table_coords['xsize'], # x side 2
                                       table_coords['zsize']] # z side 2
                                     )
        xz_bbox_side_probs = xz_bbox_side_probs / np.sum(xz_bbox_side_probs)
        side = np.random.choice(range(4), p=xz_bbox_side_probs)
        if side == 0: # x side 1
            p1 = np.array([table_coords['xmin'], table_coords['zmin']])
            p2 = np.array([table_coords['xmax'], table_coords['zmin']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmax'], table_coords['zmax']]))
        elif side == 1: # z side 1
            p1 = np.array([table_coords['xmax'], table_coords['zmin']])
            p2 = np.array([table_coords['xmax'], table_coords['zmax']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmin'], table_coords['zmax']]))
        elif side == 2: # x side 2
            p1 = np.array([table_coords['xmax'], table_coords['zmax']])
            p2 = np.array([table_coords['xmin'], table_coords['zmax']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmin'], table_coords['zmin']]))
        elif side == 3: # z side 2
            p1 = np.array([table_coords['xmin'], table_coords['zmax']])
            p2 = np.array([table_coords['xmin'], table_coords['zmin']])
            side_length = np.linalg.norm(p2 - p1)
            other_side_length = np.linalg.norm(p2 - np.array([table_coords['xmax'], table_coords['zmin']]))

        # Select point on that side uniformly
        point = p1 + (p2 - p1) * np.random.uniform(0,1)

        # Sample xz distance from that point
        dist_from_table = np.random.uniform(0.0, 0.15)
        theta = np.radians(-90)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        away_from_table_direction = rot_matrix.dot ( (p2 - p1) / np.linalg.norm(p2 - p1) )
        camera_x, camera_z = point + dist_from_table * away_from_table_direction

        # Sample y distance
        # height_from_table = np.random.uniform(.5*table_coords['ysize'], 1.0*table_coords['ysize'])
        height_from_table = np.random.uniform(.5, 1.2) # anywhere from .5m to 1.2m above table
        camera_y = table_coords['ymax'] + height_from_table

        # Final camera position
        camera_pos = np.array([camera_x, camera_y, camera_z])

        if side in [0,2]:
            lookat_xmin = max(point[0] - side_length*0.2, table_coords['xmin'])
            lookat_xmax = min(point[0] + side_length*0.2, table_coords['xmax'])
            if side == 0:
                lookat_zmin = point[1] + other_side_length*0.1
                lookat_zmax = point[1] + other_side_length*0.5
            else: # side == 2
                lookat_zmin = point[1] - other_side_length*0.5
                lookat_zmax = point[1] - other_side_length*0.1
        else: # side in [1,3]
            lookat_zmin = max(point[1] - side_length*0.2, table_coords['zmin'])
            lookat_zmax = min(point[1] + side_length*0.2, table_coords['zmax'])
            if side == 1:
                lookat_xmin = point[0] - other_side_length*0.5
                lookat_xmax = point[0] - other_side_length*0.1
            else: # side == 3
                lookat_xmin = point[0] + other_side_length*0.1
                lookat_xmax = point[0] + other_side_length*0.5

        # Sample lookat position
        lookat_pos = np.array(self.get_state(self.table_stuff['table_obj_id'])[0])
        lookat_pos[0] = np.random.uniform(lookat_xmin, lookat_xmax)
        lookat_pos[1] = table_coords['ymax']
        lookat_pos[2] = np.random.uniform(lookat_zmin, lookat_zmax)

        print('using fixed camera info')
        #camera_pos=np.array([3,3,3])
        #lookat_pos=np.array([1.2179527697917631, 0.9330687595698151, 1.1673556013096767])

        if self._mode == 'gui':

            # Calculate yaw, pitch, direction for camera (Bullet needs this)
            # Equations for pitch/yaw is taken from:
            #     gamedev.stackexchange.com/questions/112565/finding-pitch-yaw-values-from-lookat-vector
            camera_direction = lookat_pos - camera_pos
            camera_distance = np.linalg.norm(camera_direction)
            camera_direction = camera_direction / camera_distance
            camera_pitch = np.arcsin(camera_direction[1]) * 180 / np.pi
            camera_yaw = np.arctan2(camera_direction[0], camera_direction[2]) * 180 / np.pi
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, lookat_pos)

        camera_images, bid_to_seglabel_mapping=self.get_camera_images(camera_pos, lookat_pos)
        return camera_images, bid_to_seglabel_mapping, camera_pos, lookat_pos

    def sample_camera_up_vector(self, camera_pos, lookat_pos):

        # To do this, I first generate the camera view with [0,1,0] as the up-vector, then sample a rotation from the camera x-y axis and apply it
        # truncated normal sampling. clip it to 2*sigma range, meaning ~5% of samples are maximally rotated
        theta = np.random.normal(0, self.params['max_camera_rotation'] / 2, size=[1])
        theta = theta.clip(-self.params['max_camera_rotation'], self.params['max_camera_rotation'])[0]
        my_rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        y_axis = np.array([0,1,0])
        camera_up_vector = my_rot_mat.dot(y_axis) # this is in camera coordinate frame. pybullet needs this in world coordinate frame
        camera_rotation_matrix = np.asarray(p.computeViewMatrix(camera_pos, lookat_pos, y_axis)).reshape(4,4, order='F')[:3,:3].T # Note that transpose is the inverse since it's orthogonal. this is camera rotation matrix
        camera_up_vector = camera_rotation_matrix.dot(camera_up_vector) # camera up vector in world coordinate frame

        print('using fixed camera up vector')
        #camera_up_vector=y_axis

        return camera_up_vector

    def get_tabletop_mask(self, depth_img, orig_seg_img, camera_pos, lookat_pos, camera_up_vector):
        # Filter out table labels to get tabletop ONLY
        H,W = self.params['img_height'], self.params['img_width']
        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
        view_matrix = np.array(view_matrix).reshape(4,4, order='F')
        cam_ext = np.linalg.inv(view_matrix) # Camera extrinsics matrix

        # negative depth because OpenGL camera z-axis faces behind. 
        xyz_img = sim_util.compute_xyz(depth_img, self.params) # Shape: [H x W x 3]
        xyz_img[..., 2] = -1 * xyz_img[..., 2] # negate the depth to get OpenGL camera frame

        # Multiply each homogenous xyz point by camera extrinsics matrix to bring it back to world coordinate frame
        world_frame_depth = cam_ext.dot(np.concatenate([xyz_img, np.ones((H,W,1))], axis=2).reshape(-1,4).T)
        world_frame_depth = world_frame_depth.T.reshape((H,W,4))[..., :3]

        # Get tabletop. Compute histogram of 1cm y-values and pick mode of histogram. 
        # It's kinda like RANSAC in 1 dimension, but using a discretization instead of randomness.
        table_mask = orig_seg_img == self._obj_id_to_body[self.table_stuff['table_obj_id']].bid
        highest_y_val = round(np.max(world_frame_depth[table_mask, 1]) + 0.05, 2)
        bin_count, bin_edges = np.histogram(world_frame_depth[table_mask, 1], 
                                            bins=int(highest_y_val / .01), 
                                            range=(0,highest_y_val))
        bin_index = np.argmax(bin_count)
        tabletop_y_low = bin_edges[bin_index-1] # a bit less than lower part
        tabletop_y_high = bin_edges[bin_index + 2] # a bit more than higher part
        tabletop_mask = np.logical_and(world_frame_depth[..., 1] >= tabletop_y_low, 
                                       world_frame_depth[..., 1] <= tabletop_y_high)
        tabletop_mask = np.logical_and(tabletop_mask, table_mask) # Make sure tabletop_mask is subset of table

        return tabletop_mask
    
    def transform_to_camera_vector(self, vector, camera_pos, lookat_pos, camera_up_vector):
        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
        view_matrix = np.array(view_matrix).reshape(4,4, order='F')
        vector=np.concatenate((vector, np.array([1])))
        transformed_vector=view_matrix.dot(vector)
        return transformed_vector[:3]
        

    def get_camera_images(self, camera_pos, lookat_pos, camera_up_vector=None):
        """ Get RGB/Depth/Segmentation images
        """

        if camera_up_vector is None:
            camera_up_vector = self.sample_camera_up_vector(camera_pos, lookat_pos)
        #camera_up_vector=np.array([0.7454185375920419, 0.6609673701716654, -0.08644937974558661])

        # Compute some stuff
        aspect_ratio = self.params['img_width']/self.params['img_height']
        e = 1/(np.tan(np.radians(self.params['fov']/2.)))
        t = self.params['near']/e; b = -t
        r = t*aspect_ratio; l = -r

        # Compute view/projection matrices and get images
        view_matrix = p.computeViewMatrix(camera_pos, lookat_pos, camera_up_vector)
        
        proj_matrix = p.computeProjectionMatrixFOV(self.params['fov'], aspect_ratio, self.params['near'], self.params['far'])
        intrensics_matrix=np.linalg.inv(np.reshape(np.array(proj_matrix), (4,4)))
        temp = p.getCameraImage(self.params['img_width'], self.params['img_height'], 
                                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL) # tuple of: width, height, rgbPixels, depthPixels, segmentation

        # RGB image
        rgb_img = np.reshape(temp[2], (self.params['img_height'], self.params['img_width'], 4))[..., :3]

        # Depth image
        depth_buffer = np.array(temp[3]).reshape(self.params['img_height'],self.params['img_width'])
        depth_img = self.params['far'] * self.params['near'] / \
                    (self.params['far'] - (self.params['far'] - self.params['near']) * depth_buffer)
        # Note: this gives positive z values. this equation multiplies the actual negative z values by -1
        #       Negative z values are because OpenGL camera +z axis points away from image


        # Segmentation image
        seg_img = np.array(temp[4]).reshape(self.params['img_height'],self.params['img_width'])

        # Set near/far clipped depth values to 0. This is indicated by seg_img == -1
        depth_img[seg_img == -1] = 0.

        # Convert seg_img to background (0), table (1), objects (2+). near/far clipped values get background label
        bid_to_seglabel_mapping = {-1 : 0} # Mapping from bid to segmentation label

        # Room bullet IDs
        room_bids = [v.bid for k, v in self._obj_id_to_body.items() 
                     if 'ShapeNet' not in k]
        for bid in room_bids:
            if bid in seg_img: # Make sure this object is visible
                bid_to_seglabel_mapping[bid] = 0

        # Table bullet ID
        if 'ShapeNet_table_0' in self._obj_id_to_body.keys():
            table_bid = self._obj_id_to_body['ShapeNet_table_0'].bid
            bid_to_seglabel_mapping[table_bid] = 1

        # Object bullet IDs
        object_bids = [v.bid for k, v in self._obj_id_to_body.items() 
                       if 'ShapeNet' in k and 'table' not in k]
        obj_id = 2
        for bid in object_bids:
            if bid in seg_img: # Make sure this object is visible
                bid_to_seglabel_mapping[bid] = obj_id
                obj_id += 1

        # Conversion happens here
        new_seg_img = seg_img.copy()
        for bid, seg_label in bid_to_seglabel_mapping.items():
            new_seg_img[seg_img == bid] = seg_label
            if seg_label == 1: # table
                table_mask = seg_img == bid
                if np.count_nonzero(table_mask) > 0: # if no table in this image, the calling function will throw away the image
                    tabletop_mask = self.get_tabletop_mask(depth_img, seg_img, camera_pos, lookat_pos, camera_up_vector)
                    new_seg_img[table_mask] = 0
                    new_seg_img[tabletop_mask] = 1

        return {'rgb' : rgb_img,
                'depth' : depth_img,
                'seg' : new_seg_img,
                'orig_seg_img' : seg_img,
                'view_params' : {
                                'camera_pos' : camera_pos.tolist() if type(camera_pos) == np.ndarray else camera_pos,
                                'lookat_pos' : lookat_pos.tolist() if type(lookat_pos) == np.ndarray else lookat_pos,
                                'camera_up_vector' : camera_up_vector.tolist() if type(camera_up_vector) == np.ndarray else camera_up_vector,
                                }
                }, bid_to_seglabel_mapping


            
