from xml.dom import minidom
import os
import trimesh
import dm_control.mujoco as mujoco
from pathlib import Path
import numpy as np
import shutil

#mujoc utilities to manipulate xml files

def add_object_to_mujoco(mj_file_path, obj_meshes, obj_pos, scratch_folder, obj_num, step, other_objects=[], color=None, joint=False, add_mesh_name=None, geom_args=[], include_collisions=False):
    contact_geom_list=[
        ("herb/wam_1/bhand//unnamed_geom_24", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_22", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_20", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_18", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_16", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_15", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_14", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_12", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_10", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_8", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_7", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_6", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_4", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_3", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_2", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1/bhand//unnamed_geom_1", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1//unnamed_geom_24", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1//unnamed_geom_22", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1//unnamed_geom_21", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1//unnamed_geom_20", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1//unnamed_geom_18", "1 1 0.05 0.01 0.01"),
        ("herb/wam_1//unnamed_geom_17", "1 1 0.05 0.01 0.01"),
        ("table_plane", "0.5 0.5 0.005 0.0001 0.0001")
    ]
    
    #num_files=len(os.listdir(scratch_folder))
    
    if not os.path.exists(scratch_folder):
        os.makedirs(scratch_folder)
    
    for obj_mesh_ind in range(len(obj_meshes)):
        mesh_file=scratch_folder+f'/mesh_{step}_{obj_num}_{obj_mesh_ind}.stl'
        #obj_meshes[obj_mesh_ind].show()
        obj_meshes[obj_mesh_ind].export(mesh_file)
    
    xmldoc = minidom.parse(mj_file_path)
    
    assets = xmldoc.getElementsByTagName('asset')[0]
    for obj_mesh_ind in range(len(obj_meshes)):
        new_mesh=xmldoc.createElement('mesh')
        if add_mesh_name is not None:
            new_mesh.setAttribute('name', f'gen_mesh_{obj_num}_{obj_mesh_ind}_{add_mesh_name}')
        else:
            new_mesh.setAttribute('name', f'gen_mesh_{obj_num}_{obj_mesh_ind}')
        new_mesh.setAttribute('class', 'geom0')
        new_mesh.setAttribute('file', scratch_folder+f'/mesh_{step}_{obj_num}_{obj_mesh_ind}.stl')
        assets.appendChild(new_mesh)
    
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('body')
    body_name=f'gen_body_{obj_num}'
    new_body.setAttribute('name', body_name)
    new_body.setAttribute('pos', f'{obj_pos[0]} {obj_pos[1]} {obj_pos[2]}')
    
    
    geom_names=[]
    for obj_mesh_ind in range(len(obj_meshes)):
        new_geom=xmldoc.createElement('geom')
        geom_name=f'gen_geom_{obj_num}_{obj_mesh_ind}'
        geom_names.append(geom_name)
        new_geom.setAttribute('name', geom_name)
        new_geom.setAttribute('class', '/')
        new_geom.setAttribute('type', 'mesh')
        new_geom.setAttribute('size', '1 1 1')
        if add_mesh_name is not None:
            new_geom.setAttribute('mesh', f'gen_mesh_{obj_num}_{obj_mesh_ind}_{add_mesh_name}')
        else:
            new_geom.setAttribute('mesh', f'gen_mesh_{obj_num}_{obj_mesh_ind}')
        for geom_arg in geom_args:
            new_geom.setAttribute(geom_arg[0], geom_arg[1])
        if color is not None:
            new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} {color[3]}')
        
        new_body.appendChild(new_geom)
    
    if joint:
        new_joint=xmldoc.createElement('joint')
        new_joint.setAttribute('name', f'gen_joint_{obj_num}')
        new_joint.setAttribute('class', '/')
        new_joint.setAttribute('type', 'free')
        #new_joint.setAttribute('damping', '0.001')
        new_body.appendChild(new_joint)
    
    world_body.appendChild(new_body)
  
    if include_collisions:
        contact = xmldoc.getElementsByTagName('contact')[0]
        for obj_mesh_ind in range(len(obj_meshes)):
            for contact_geom in contact_geom_list:
                new_contact=xmldoc.createElement('pair')
                geom_name=f'gen_geom_{obj_num}_{obj_mesh_ind}'
                new_contact.setAttribute('geom1', geom_name)
                new_contact.setAttribute('geom2', contact_geom[0])
                new_contact.setAttribute('friction', contact_geom[1])
                new_contact.setAttribute('solref', "0.01 1")
                new_contact.setAttribute('solimp', "0.999 0.999 0.01")
                new_contact.setAttribute('condim', "4")
                contact.appendChild(new_contact)
            for added_object_name in other_objects:
                new_contact=xmldoc.createElement('pair')
                geom_name=f'gen_geom_{obj_num}_{obj_mesh_ind}'
                geom2_name=added_object_name
                new_contact.setAttribute('geom1', geom_name)
                new_contact.setAttribute('geom2', geom2_name)
                new_contact.setAttribute('friction', "0.5 0.5 0.005 0.0001 0.0001")
                new_contact.setAttribute('solref', "0.01 1")
                new_contact.setAttribute('solimp', "0.999 0.999 0.01")
                new_contact.setAttribute('condim', "4")
                contact.appendChild(new_contact)
    
    with open(mj_file_path, "w") as f:
        xmldoc.writexml(f)
    
    return body_name, geom_names

def remove_objects_from_mujoco(mj_file_path, obj_num): #15 onwards: objects #easy manip: 15: 
    xmldoc = minidom.parse(mj_file_path)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    removed=True
    removed_objects=[]
    while(removed):
        removed=False
        passed_elements=0
        for ind in range(len(world_body.childNodes)):
            if isinstance(world_body.childNodes[ind], minidom.Element):
                passed_elements+=1
            if passed_elements>=obj_num:
                removed=True
                break
        
        if passed_elements>=obj_num:
            geom_name=world_body.childNodes[ind].childNodes[0]._attrs['mesh'].value
            geom_f_name=world_body.childNodes[ind].childNodes[0]._attrs['name'].value
            assets = xmldoc.getElementsByTagName('asset')[0]
            for assets_ind in range(len(assets.childNodes)):
                if isinstance(assets.childNodes[assets_ind], minidom.Element) and assets.childNodes[assets_ind]._attrs['name'].value==geom_name:
                    break
            
            removed_objects.append((world_body.childNodes[ind]._attrs['name'].value, 
                                    world_body.childNodes[ind].childNodes[0]._attrs['mesh'].value, 
                                    [float(i) for i in world_body.childNodes[ind]._attrs['pos'].value.split()], 
                                    [float(i) for i in world_body.childNodes[ind].childNodes[0]._attrs['rgba'].value.split()], 
                                    [float(i) for i in assets.childNodes[assets_ind]._attrs['scale'].value.split()],
                                    assets.childNodes[assets_ind]._attrs['file'].value,
                                    [float(i) for i in world_body.childNodes[ind]._attrs['euler'].value.split()]))
            world_body.removeChild(world_body.childNodes[ind])
            assets.removeChild(assets.childNodes[assets_ind])
            
            contact = xmldoc.getElementsByTagName('contact')[0]
            remove_contacts=[]
            for ind in range(len(contact.childNodes)):
                if isinstance(contact.childNodes[ind], minidom.Element):
                    if 'geom1' in contact.childNodes[ind]._attrs:
                        if contact.childNodes[ind]._attrs['geom1'].value==geom_f_name or contact.childNodes[ind]._attrs['geom2'].value==geom_f_name:
                            remove_contacts.append(ind)
            for remove_ind in reversed(remove_contacts):
                contact.removeChild(contact.childNodes[remove_ind])
                
            
              
    with open(mj_file_path, "w") as f:
        xmldoc.writexml(f)
    return removed_objects

def get_mesh_list(mj_scene_xml):
    xmldoc = minidom.parse(mj_scene_xml)
    mesh_root=os.path.join(str(Path(mj_scene_xml).parent), 'assets')
    name_to_file_dict={}
    name_to_scale_dict={}
    assets=xmldoc.getElementsByTagName('asset')[0]
    for child_node in assets.childNodes:
        if child_node.nodeName=='mesh':
            name_to_file_dict[child_node._attrs['name'].value]=os.path.join(mesh_root, child_node._attrs['file'].value)
            if 'scale' in child_node._attrs:
                name_to_scale_dict[child_node._attrs['name'].value]=float(child_node._attrs['scale'].value.split()[0])
    
    
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    mesh_list=[]
    
    get_mesh_list_recursive(world_body, mesh_list, name_to_file_dict)
    return mesh_list, name_to_file_dict, name_to_scale_dict

def get_mesh_list_recursive(body, mesh_list, name_to_file_dict):
    if body.nodeName=='geom':
        if 'mesh' in body._attrs:
            mesh_list+=[trimesh.load_mesh(name_to_file_dict[body._attrs['mesh'].value])]
        elif body._attrs['type'].value=='box':
            size=2.0*np.array([float(i) for i in body._attrs['size'].value.split()])
            mesh_list+=[trimesh.primitives.Box(extents=size)]
        else:
            mesh_list+=[None]
    for child_node in body.childNodes:
        get_mesh_list_recursive(child_node, mesh_list, name_to_file_dict)

#compute center of mesh when loaded into mujoco (seems to differ between 3D libraries)
def compute_mujoco_int_transform(mesh_file, save_id, size=1):
    ramdisk='/dev/shm'
    
    xml=("<mujoco model=\"scene\">\n"+
        "  <compiler coordinate=\"local\" angle=\"radian\" fusestatic=\"false\" meshdir=\".\" texturedir=\".\"/>\n"+
        "  <size njmax=\"1000\" nconmax=\"1000\"/>"+
        "  <visual>\n"+
        "    <global offwidth=\"800\" offheight=\"800\"/>\n"+
        "    <quality shadowsize=\"2048\"/>\n"+
        "    <headlight ambient=\"0 0 0\" diffuse=\"1.399999 1.399999 1.399999\" specular=\"2 2 2\"/>\n"+
        "    <map force=\"0.1\" zfar=\"30.0\"/>\n"+
        "    <rgba haze=\"0.1499999 0.25 0.3499998 1\"/>\n"+
        "  </visual>\n"+
        "  <default>\n"+
        "    <default class=\"/\"></default>\n"+
        "  </default>\n"
        "  <asset>\n"+
        "    <mesh name=\"test_mesh\" class=\"/\" scale=\""+str(size)+" "+str(size)+" "+str(size)+"\" file=\""+mesh_file+"\"/>\n"+
        "  </asset>\n"+
        "  <worldbody>\n"+
        "    <geom name=\"test_geom\" class=\"/\" type=\"mesh\" rgba=\"0 0 0 1\" pos=\"0 0 0\" quat=\"1 0 0 0\" mesh=\"test_mesh\"/>\n"+
        "  </worldbody>\n"+
        "</mujoco>")
        
    file_name=os.path.join(ramdisk, f"temp_scene_{save_id}.xml")
    f = open(file_name, "w")
    f.write(xml)
    f.close()
    
    physics=mujoco.Physics.from_xml_path(file_name)
    #print(mesh_file, physics.model.mesh_vert.shape[0])
    os.remove(file_name)
    
    return np.copy(physics.data.geom_xpos[0]), np.copy(np.reshape(physics.data.geom_xmat[0], (3,3)))

def make_global_contacts(scene_name):
    xmldoc = minidom.parse(scene_name)
    options = xmldoc.getElementsByTagName('option')[0]
    options.setAttribute('collision', 'dynamic')
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

def add_object(scene_name, object_name, mesh_name, xpos, y_pos, size, color, rot, other_objects, run_id, top_dir, target_objects, type='ycb', z_pos=None):
    geom_args=[]
    
    if mesh_name in target_objects[16:]:
        mesh_filename=os.path.join(top_dir, f'herb_reconf/assets/ycb_objects/{mesh_name}/google_16k/nontextured.stl')
        type='ycb'
    else:
        mesh_filename=os.path.join(top_dir, f'herb_reconf/cluttered_scenes/assets/downloaded_assets/{mesh_name}/scene.stl')
        type='downloaded'   
    
    if z_pos==None:
        mujoco_center, _=compute_mujoco_int_transform(mesh_filename, run_id, size=size)
        mic=mujoco_center[2]
        mesh=trimesh.load(mesh_filename)
        lower_z=-mic
        z_offset=0.3-mesh.bounds[0,2]
    else:
        z_offset=z_pos
    
    contact_geom_list=[
        ("herb/wam_1/bhand//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_16", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_15", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_14", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_12", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_10", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_8", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_7", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_6", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_4", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_3", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_2", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_1", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_21", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_17", "4 4 0.2 0.04 0.04"),
        ("table_plane", "0.5 0.5 0.005 0.0001 0.0001")
    ]

    xmldoc = minidom.parse(scene_name)
    
    assets = xmldoc.getElementsByTagName('asset')[0]
    new_mesh=xmldoc.createElement('mesh')
    new_mesh.setAttribute('name', f'gen_mesh_{object_name}')
    new_mesh.setAttribute('class', 'geom0')
    new_mesh.setAttribute('scale', f'{size} {size} {size}')
    if type=='ycb':
        new_mesh.setAttribute('file', f'ycb_objects/{mesh_name}/google_16k/nontextured.stl')
    elif type=='downloaded':
        new_mesh.setAttribute('file', f'downloaded_assets/{mesh_name}/scene.stl')
    assets.appendChild(new_mesh)
    
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('body')
    body_name=f'gen_body_{object_name}'
    new_body.setAttribute('name', body_name)
    new_body.setAttribute('pos', f'{xpos} {y_pos} {z_offset}')
    new_body.setAttribute('euler', f'0 0 {rot}')
    
    geom_names=[]
    new_geom=xmldoc.createElement('geom')
    geom_name=f'gen_geom_{object_name}'
    geom_names.append(geom_name)
    new_geom.setAttribute('name', geom_name)
    new_geom.setAttribute('class', '/')
    new_geom.setAttribute('type', 'mesh')
    new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
    new_geom.setAttribute('mesh', f'gen_mesh_{object_name}')
    for geom_arg in geom_args:
        new_geom.setAttribute(geom_arg[0], geom_arg[1])
    
    new_body.appendChild(new_geom)
    
    new_joint=xmldoc.createElement('joint')
    new_joint.setAttribute('name', f'gen_joint_{object_name}')
    new_joint.setAttribute('class', '/')
    new_joint.setAttribute('type', 'free')
    #new_joint.setAttribute('damping', '0.001')
    new_body.appendChild(new_joint)
    
    world_body.appendChild(new_body)
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)
    
    return body_name, geom_names, z_offset
    
def remove_object(scene_name, object_name):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    for ind in range(len(world_body.childNodes)):
        if isinstance(world_body.childNodes[ind], minidom.Element) and world_body.childNodes[ind]._attrs['name'].value==object_name:
            break
    geom_name=f'gen_geom_{object_name}'
    world_body.removeChild(world_body.childNodes[ind])
    
    world_body = xmldoc.getElementsByTagName('asset')[0]
    for ind in range(len(world_body.childNodes)):
        if isinstance(world_body.childNodes[ind], minidom.Element) and world_body.childNodes[ind]._attrs['name'].value==geom_name:
            break
    world_body.removeChild(world_body.childNodes[ind])
    
    contact = xmldoc.getElementsByTagName('contact')[0]
    remove_contacts=[]
    for ind in range(len(contact.childNodes)):
        if isinstance(contact.childNodes[ind], minidom.Element):
            if 'geom1' in contact.childNodes[ind]._attrs:
                if contact.childNodes[ind]._attrs['geom1'].value==geom_name or contact.childNodes[ind]._attrs['geom2'].value==geom_name:
                    remove_contacts.append(ind)
    for remove_ind in reversed(remove_contacts):
        contact.removeChild(contact.childNodes[remove_ind])
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

#produce a convex decomposition of the environment
def convex_decomp_target_object_env(scene_name, target_body_name, scene_save_dir, run_id, top_dir, new_scene_name=None, add_contacts=True):
    contact_geom_list=[
        ("herb/wam_1/bhand//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_16", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_15", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_14", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_12", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_10", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_8", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_7", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_6", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_4", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_3", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_2", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1/bhand//unnamed_geom_1", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_21", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
        ("herb/wam_1//unnamed_geom_17", "4 4 0.2 0.04 0.04"),
        ("table_plane", "0.5 0.5 0.005 0.0001 0.0001"),
    ]
    if new_scene_name is None:
        new_scene_name=os.path.join(scene_save_dir, str(run_id))+'.xml'
    
    body_poses=[]
    body_scales=[]
    body_orientatons=[]
    mesh_filenamess=[]
    body_names=[]
    
    target_ind=-1
    shutil.copyfile(scene_name, new_scene_name)
    removed_objects=remove_objects_from_mujoco(new_scene_name, 9)
    for removed_object_ind in range(len(removed_objects)):
        removed_object=removed_objects[removed_object_ind]
        mesh_filename=removed_object[5]
        mesh=trimesh.load_mesh(os.path.join(os.path.join(top_dir, 'herb_reconf/assets'), mesh_filename))
        mesh_name=mesh_filename.split('/')[1]
        
        decomps_folder=os.path.join(top_dir, f'herb_reconf/assets/decomps/{mesh_name}/')
        if not os.path.exists(decomps_folder):
            os.mkdir(decomps_folder)
            decomps=trimesh.decomposition.convex_decomposition(mesh, maxNumVerticesPerCH=1025, concavity=0.01, resolution=100000)
            if not isinstance(decomps, list):
                decomps=[decomps]
            c_decomps=[]
            for decmop in decomps:
                if decmop.faces.shape[0]>4 and decmop.mass>10e-8:
                    c_decomps.append(decmop)
            decomps=c_decomps
            mesh_filenames=[]
            mesh_masses=[]
            for decomp_ind in range(len(decomps)):
                mesh_filename=os.path.join(decomps_folder, f'{decomp_ind}.stl')
                mesh_filenames.append(mesh_filename)
                decomps[decomp_ind].export(mesh_filename)
                mesh_masses.append(decomps[decomp_ind].mass)
            if len(mesh_filenames)>25:
                heavy_inds=np.argsort(np.array(mesh_masses))
                new_mesh_names=[]
                for ind in range(25):
                    new_mesh_names.append(mesh_filenames[heavy_inds[-ind]])
                mesh_filenames=new_mesh_names
        else:
            mesh_filenames=os.listdir(decomps_folder)
            mesh_filenames=[os.path.join(decomps_folder, mesh_filename) for mesh_filename in mesh_filenames]
        
        if len(mesh_filenames)>0:
            body_poses.append(removed_object[2])
            body_scales.append(removed_object[4][0])
            body_orientatons.append(removed_object[6])
            mesh_filenamess.append(mesh_filenames)
            body_names.append(str(removed_object_ind))
            
            if removed_object[0]==target_body_name:
                target_ind=removed_object_ind
        
    body_name, geom_names=add_objectss(new_scene_name, body_names, mesh_filenamess, body_poses, body_scales, [0.5, 0.5, 0.5], body_orientatons, run_id, contact_geom_list=contact_geom_list, add_contacts=add_contacts)

    target_decomp_ind=target_ind
    target_num_decomps=len(mesh_filenamess[target_ind])
    
    return new_scene_name, target_decomp_ind, target_num_decomps, geom_names

def create_scene_with_mesh(mesh_filename, save_id, top_dir, scene_xml_file, task):
    mujoco_center, _=compute_mujoco_int_transform(mesh_filename, save_id)
    mic=mujoco_center[2]
    mesh=trimesh.load(mesh_filename)
    lower_z=-mic
    z_offset=0.3-mesh.bounds[0,2]
    
    temp_scene_xml_file=os.path.join(top_dir, f'herb_reconf/temp_scene_{save_id}.xml')
    shutil.copyfile(scene_xml_file, temp_scene_xml_file)
    remove_objects_from_mujoco(temp_scene_xml_file, [9])#,10
    if task=='hard_pushing' or task=="grasping":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([0,0,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 0, 0, joint=True, add_mesh_name='init', include_collisions=True) #
    elif task=='easy_pushing':
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([-0.05,-0.35,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 0, 0, joint=True, add_mesh_name='init', include_collisions=True)
    
    if task=="hard_pushing":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([-0.05,-0.35,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 1, 0, joint=False, geom_args=[['contype', '0'], ['conaffinity', '0'], ['group', '1'], ['rgba', '0 0 0 0.0']])
    elif task=="grasping":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([0,0,z_offset+0.2]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 1, 0, joint=False, geom_args=[['contype', '0'], ['conaffinity', '0'], ['group', '1'], ['rgba', '0 0 0 0.0']])
    elif task=="easy_pushing":
        add_object_to_mujoco(temp_scene_xml_file, [mesh], np.array([0,0,z_offset]), os.path.join(top_dir, f'herb_reconf/temp_{save_id}/'), 1, 0, joint=False, geom_args=[['contype', '0'], ['conaffinity', '0'], ['group', '1'], ['rgba', '0 0 0 0.1666']])
    return temp_scene_xml_file   

#enable/disable gravity in xml
def set_gravity(scene_name, set_unset):
    xmldoc = minidom.parse(scene_name)
    options = xmldoc.getElementsByTagName('option')[0]
    if set_unset:
        options.setAttribute('gravity', "0 0 -9.81")
    else:
        options.setAttribute('gravity', "0 0 0")
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

def add_camera(scene_name, cam_name, cam_pos, cam_target, cam_id):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('camera')
    new_body.setAttribute('name', cam_name)
    new_body.setAttribute('mode', 'targetbody')
    new_body.setAttribute('pos', f'{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}')
    new_body.setAttribute('target', f'added_cam_target_{cam_id}')
    world_body.appendChild(new_body)
    
    new_body=xmldoc.createElement('body')
    new_body.setAttribute('name', f'added_cam_target_{cam_id}')
    new_body.setAttribute('pos', f'{cam_target[0]} {cam_target[1]} {cam_target[2]}')
    new_geom=xmldoc.createElement('geom')
    geom_name=f'added_cam_target_geom_{cam_id}'
    new_geom.setAttribute('name', geom_name)
    new_geom.setAttribute('class', '/')
    new_geom.setAttribute('type', 'box')
    new_geom.setAttribute('contype', '0')
    new_geom.setAttribute('conaffinity', '0')
    new_geom.setAttribute('group', '1')
    new_geom.setAttribute('size', "1 1 1")
    new_geom.setAttribute('rgba', f'0 0 0 0')
    new_body.appendChild(new_geom)
    world_body.appendChild(new_body)
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)

def add_objects(scene_name, object_name, mesh_names, pos, size, color, rot, run_id, contact_geom_list=None, add_ind=-1, add_contacts=True):
    
    if contact_geom_list is None:
        contact_geom_list=[
            ("herb/wam_1/bhand//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_16", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_15", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_14", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_12", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_10", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_8", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_7", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_6", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_4", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_3", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_2", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_1", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_21", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_17", "4 4 0.2 0.04 0.04"),
        ]

    xmldoc = minidom.parse(scene_name)
    
    assets = xmldoc.getElementsByTagName('asset')[0]
    for mesh_ind in range(len(mesh_names)):
        new_mesh=xmldoc.createElement('mesh')
        new_mesh.setAttribute('name', f'gen_mesh_{object_name}_{mesh_ind}')
        new_mesh.setAttribute('class', 'geom0')
        new_mesh.setAttribute('scale', f'{size} {size} {size}')
        new_mesh.setAttribute('file', mesh_names[mesh_ind])
        assets.appendChild(new_mesh)
    
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    
    new_body=xmldoc.createElement('body')
    body_name=f'gen_body_{object_name}'
    new_body.setAttribute('name', body_name)
    new_body.setAttribute('pos', f'{pos[0]} {pos[1]} {pos[2]}')
    new_body.setAttribute('euler', f'{rot[0]} {rot[1]} {rot[2]}')
    
    geom_names=[]
    for geom_ind in range(len(mesh_names)):
        new_geom=xmldoc.createElement('geom')
        geom_name=f'gen_geom_{object_name}_{geom_ind}'
        geom_names.append(geom_name)
        new_geom.setAttribute('name', geom_name)
        new_geom.setAttribute('class', '/')
        new_geom.setAttribute('type', 'mesh')
        new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
        new_geom.setAttribute('mesh', f'gen_mesh_{object_name}_{geom_ind}')
        new_body.appendChild(new_geom)
    
    new_joint=xmldoc.createElement('joint')
    new_joint.setAttribute('name', f'gen_joint_{object_name}')
    new_joint.setAttribute('class', '/')
    new_joint.setAttribute('type', 'free')
    #new_joint.setAttribute('damping', '0.001')
    new_body.appendChild(new_joint)
    
    if add_ind>-1 and add_ind<len(world_body.childNodes):
        world_body.insertBefore(new_body, world_body.childNodes[add_ind])
    else:
        world_body.appendChild(new_body)
  
    if add_contacts:
        contact = xmldoc.getElementsByTagName('contact')[0]
        for contact_geom in contact_geom_list:
            for geom_name in geom_names:
                new_contact=xmldoc.createElement('pair')
                new_contact.setAttribute('geom1', geom_name)
                new_contact.setAttribute('geom2', contact_geom[0])
                new_contact.setAttribute('friction', contact_geom[1])
                new_contact.setAttribute('solref', "0.01 1")
                new_contact.setAttribute('solimp', "0.999 0.999 0.01")
                new_contact.setAttribute('condim', "4")
                contact.appendChild(new_contact)
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)
    
    return body_name, geom_names

def add_objectss(scene_name, object_names, mesh_namess, poses, sizes, color, rots, run_id, contact_geom_list=None, add_ind=-1, add_contacts=True):
    
    if contact_geom_list is None:
        contact_geom_list=[
            ("herb/wam_1/bhand//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_16", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_15", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_14", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_12", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_10", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_8", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_7", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_6", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_4", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_3", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_2", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1/bhand//unnamed_geom_1", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_24", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_22", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_21", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_20", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_18", "4 4 0.2 0.04 0.04"),
            ("herb/wam_1//unnamed_geom_17", "4 4 0.2 0.04 0.04"),
        ]

    xmldoc = minidom.parse(scene_name)
    
    for object_ind in range(len(object_names)):
        object_name=object_names[object_ind]
        mesh_names=mesh_namess[object_ind]
        pos=poses[object_ind]
        size=sizes[object_ind]
        #color=colors[object_ind]
        rot=rots[object_ind]
    
        assets = xmldoc.getElementsByTagName('asset')[0]
        for mesh_ind in range(len(mesh_names)):
            new_mesh=xmldoc.createElement('mesh')
            new_mesh.setAttribute('name', f'gen_mesh_{object_name}_{mesh_ind}')
            new_mesh.setAttribute('class', 'geom0')
            new_mesh.setAttribute('scale', f'{size} {size} {size}')
            new_mesh.setAttribute('file', mesh_names[mesh_ind])
            assets.appendChild(new_mesh)
        
        world_body = xmldoc.getElementsByTagName('worldbody')[0]
        
        new_body=xmldoc.createElement('body')
        body_name=f'gen_body_{object_name}'
        new_body.setAttribute('name', body_name)
        new_body.setAttribute('pos', f'{pos[0]} {pos[1]} {pos[2]}')
        new_body.setAttribute('euler', f'{rot[0]} {rot[1]} {rot[2]}')
        
        geom_names=[]
        for geom_ind in range(len(mesh_names)):
            new_geom=xmldoc.createElement('geom')
            geom_name=f'gen_geom_{object_name}_{geom_ind}'
            geom_names.append(geom_name)
            new_geom.setAttribute('name', geom_name)
            new_geom.setAttribute('class', '/')
            new_geom.setAttribute('type', 'mesh')
            new_geom.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} 1')
            new_geom.setAttribute('mesh', f'gen_mesh_{object_name}_{geom_ind}')
            new_body.appendChild(new_geom)
        
        new_joint=xmldoc.createElement('joint')
        new_joint.setAttribute('name', f'gen_joint_{object_name}')
        new_joint.setAttribute('class', '/')
        new_joint.setAttribute('type', 'free')
        #new_joint.setAttribute('damping', '0.001')
        new_body.appendChild(new_joint)
        
        if add_ind>-1 and add_ind<len(world_body.childNodes):
            world_body.insertBefore(new_body, world_body.childNodes[add_ind])
        else:
            world_body.appendChild(new_body)
      
        if add_contacts:
            contact = xmldoc.getElementsByTagName('contact')[0]
            for contact_geom in contact_geom_list:
                for geom_name in geom_names:
                    new_contact=xmldoc.createElement('pair')
                    new_contact.setAttribute('geom1', geom_name)
                    new_contact.setAttribute('geom2', contact_geom[0])
                    new_contact.setAttribute('friction', contact_geom[1])
                    new_contact.setAttribute('solref', "0.01 1")
                    new_contact.setAttribute('solimp', "0.999 0.999 0.01")
                    new_contact.setAttribute('condim', "4")
                    contact.appendChild(new_contact)
            
            contact_geom_list+=[(geom_name, "4 4 0.2 0.04 0.04") for geom_name in geom_names]
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)
    
    return body_name, geom_names

def move_object_in_xml(scene_name, object_name, object_pos, object_rot):
    xmldoc = minidom.parse(scene_name)
    world_body = xmldoc.getElementsByTagName('worldbody')[0]
    for ind in range(len(world_body.childNodes)):
        if isinstance(world_body.childNodes[ind], minidom.Element) and world_body.childNodes[ind]._attrs['name'].nodeValue==object_name:
            break
    world_body.childNodes[ind].setAttribute('pos', f'{object_pos[0]} {object_pos[1]} {object_pos[2]}')
    world_body.childNodes[ind].setAttribute('quat', f'{object_rot[0]} {object_rot[1]} {object_rot[2]} {object_rot[3]}')
    
    with open(scene_name, "w") as f:
        xmldoc.writexml(f)
