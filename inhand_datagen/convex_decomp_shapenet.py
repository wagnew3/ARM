import os
os.environ["OMP_NUM_THREADS"]="1"
import trimesh
import json
import multiprocessing as mp
import time
from optparse import OptionParser
import shutil

parser = OptionParser()
parser.add_option("--shapenet_dir", dest="shapenet_dir", default='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/ShapeNetCore.v2/')
parser.add_option("--tabletop_dataset_dir", dest="tabletop_dataset_dir", default="/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/tabletop_dataset_v5")
(options, args) = parser.parse_args()

shapenet_filepath=options.shapenet_dir
training_instances_filename = os.path.join(options.tabletop_dataset_dir, 'training_shapenet_objects.json')
test_instances_filename = os.path.join(options.tabletop_dataset_dir, 'test_shapenet_objects.json')

def package_decomps(object_ids, taxonomy_dict, save_dir):
    if not os.path.exists(save_dir+'/shapenet_conv_decmops'):
        os.mkdir(save_dir+'/shapenet_conv_decmops')
    mesh_name_to_decomp_names={}
    num_decomped=0
    for obj_id_ind in range(len(object_ids)):
        obj_id=object_ids[obj_id_ind]
        obj_cat=taxonomy_dict[obj_id[0]]
        obj_mesh_filename = shapenet_filepath + f'/{obj_cat}/{obj_id[1]}/models/model_normalized.obj'
        if not os.path.exists(save_dir+'/shapenet_conv_decmops'+f'/{obj_cat}'):
            os.mkdir(save_dir+'/shapenet_conv_decmops'+f'/{obj_cat}')
        if not os.path.exists(save_dir+'/shapenet_conv_decmops'+f'/{obj_cat}/{obj_id[1]}'):
            os.mkdir(save_dir+'/shapenet_conv_decmops'+f'/{obj_cat}/{obj_id[1]}')
        
        for file in os.listdir(shapenet_filepath + f'/{obj_cat}/{obj_id[1]}/models/'):
            if 'appr_convex_decomp_' in file:
                new_file=save_dir+'/shapenet_conv_decmops'+f'/{obj_cat}/{obj_id[1]}/'+file[file.index('appr_convex_decomp_'):]
                shutil.copyfile(file, new_file)


def compute_appr_conv_decomp(object_ids, taxonomy_dict, process_num, num_processes):
    start_time=time.time()
    num_decomped=0
    for obj_id_ind in range(process_num, len(object_ids), num_processes):
        obj_id=object_ids[obj_id_ind]
        obj_cat=taxonomy_dict[obj_id[0]]
        obj_mesh_filename = shapenet_filepath + f'/{obj_cat}/{obj_id[1]}/models/model_normalized.obj'
        mesh=trimesh.load(obj_mesh_filename)
        decomp=trimesh.decomposition.convex_decomposition(mesh, maxNumVerticesPerCH=1024, concavity=0.0, resolution=1000000)
        if not isinstance(decomp, list):
            decomp=[decomp]
#         combined=None
        
                
        for decomp_num in range(len(decomp)):
            decomp[decomp_num].export(shapenet_filepath + f'/{obj_cat}/{obj_id[1]}/models/appr_convex_decomp_{decomp_num}.obj')
#             if combined==None:
#                 combined=decomp[decomp_num]
#             else:
#                 combined+=decomp[decomp_num]
#         mesh.show()
#         combined.show()
        num_decomped+=1
        if num_decomped%1==0:
            print(f'num_decomped {num_decomped} in {time.time()-start_time} s')





train_models = json.load(open(training_instances_filename))
test_models = json.load(open(test_instances_filename))

new_object_ids=[]
for cat in train_models:
    for obj_id in train_models[cat]:
        new_object_ids.append((cat, obj_id))
for cat in test_models:
    for obj_id in test_models[cat]:
        new_object_ids.append((cat, obj_id))
object_ids=new_object_ids

temp = json.load(open(shapenet_filepath + 'taxonomy.json'))
taxonomy_dict = {x['name'] : x['synsetId'] for x in temp}

# weirdly, the synsets in the taxonomy file are not the same as what's in the ShapeNetCore.v2 directory. Filter this out
synsets_in_dir = os.listdir(shapenet_filepath)
synsets_in_dir.remove('taxonomy.json')

taxonomy_dict = {k:v for (k,v) in taxonomy_dict.items() if v in synsets_in_dir}

# selected_index = np.random.randint(0, object_ids.shape[0])

# useful synsets for simulation
useful_named_synsets = [
    'ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin',
    'bag,traveling bag,travelling bag,grip,suitcase',
    'birdhouse',
    'bottle',
    'bowl',
    'camera,photographic camera',
    'can,tin,tin can',
    'cap',
    'clock',
    'computer keyboard,keypad',
    'dishwasher,dish washer,dishwashing machine',
    'display,video display',
    'helmet',
    'jar',
    'knife',
    'laptop,laptop computer',
    'loudspeaker,speaker,speaker unit,loudspeaker system,speaker system',
    'microwave,microwave oven',
    'mug',
    'pillow',
    'printer,printing machine',
    'remote control,remote',
    'telephone,phone,telephone set',
    'cellular telephone,cellular phone,cellphone,cell,mobile phone',
    'washer,automatic washer,washing machine'
]

package_decomps(object_ids, taxonomy_dict, '/scratch/datasets/')

# num_processes=3
# pool = mp.Pool(processes=num_processes, maxtasksperchild=1)
# parallel_runs = [pool.apply_async(compute_appr_conv_decomp, args=(object_ids, taxonomy_dict, i, num_processes)) for i in range(num_processes)]   
# results = [p.get() for p in parallel_runs]