import os
import json
import cv2
import numpy as np
from shutil import copyfile
import yaml
import pickle

dataset_fir='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/tabletop_dataset_with_poses/'

def print_missing():
    for ind in range(4000):
        a=dataset_fir+f'training/scene_{ind:05d}'
        if not os.path.exists(dataset_fir+f'test_set/scene_{ind:05d}'):
            print(ind)

def fix_pose_filenames():
    for dir_1 in os.listdir(dataset_fir):
        for dir_2 in os.listdir(dataset_fir+dir_1):
            for file_2 in os.listdir(dataset_fir+dir_1+'/'+dir_2):
                if file_2.startswith('pose_info') and not file_2.endswith('json'):
                    os.rename(dataset_fir+dir_1+'/'+dir_2+'/'+file_2, dataset_fir+dir_1+'/'+dir_2+'/'+file_2[:-3]+'json')
                    
def fix_bounding_boxes(t_ind, num_ts):
    
    for dir_1 in os.listdir(dataset_fir):
        ind=0
        for dir_2 in os.listdir(dataset_fir+dir_1):
            for file_2 in os.listdir(dataset_fir+dir_1+'/'+dir_2):
                if file_2.startswith('pose_info'):
                    object_pose_info=json.load(open(dataset_fir+dir_1+'/'+dir_2+'/'+file_2, 'r'))
                    image_file=dataset_fir+dir_1+'/'+dir_2+'/'+'segmentation_'+file_2[10:-4]+'png'
                    single_object_seg=cv2.imread(image_file, 0)
                    
                    upper_left=[int(np.min(np.argwhere(single_object_seg==255)[:, 1])), int(np.min(np.argwhere(single_object_seg==255)[:, 0]))]
                    lower_right=[np.max(np.argwhere(single_object_seg==255)[:, 1]), np.max(np.argwhere(single_object_seg==255)[:, 0])]
                    object_pose_info['obj_bb']=upper_left+[int(lower_right[0]-upper_left[0]), int(lower_right[1]-upper_left[1])]
                    
                    json.dump(object_pose_info, open(dataset_fir+dir_1+'/'+dir_2+'/'+file_2, 'w'))
            ind+=1
            if ind%1==0:
                print(ind)

def chris_to_linemod_format(t_ind, num_ts):
    linemod_format_dir='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/tabletop_dataset_poses_linemod_format/data/0'+str(t_ind+1)+'/'
    
    gt_yml={}
    train_files=[]
    test_files=[]
    ind=0
    dir_ind=0
    for dir_1 in os.listdir(dataset_fir):
        for dir_2 in os.listdir(dataset_fir+dir_1)[t_ind::num_ts]:
            dir_ind+=1
            for file_2 in os.listdir(dataset_fir+dir_1+'/'+dir_2):
                if file_2.startswith('pose_info'):
                    object_pose_info=json.load(open(dataset_fir+dir_1+'/'+dir_2+'/'+file_2, 'r'))
                    for key in object_pose_info:
                        object_pose_info[key]=[object_pose_info[key]]
                    #Create GT
                    gt_yml[ind]=object_pose_info
                    gt_yml[ind]['obj_id']=1
                    #Create trrain/test split
                    if dir_1=='training_set':
                        train_files+=[f'{ind:07d}']
                    else:
                        test_files+=[f'{ind:07d}']
                    #Move images to folder
                    seg_file_src=dataset_fir+dir_1+'/'+dir_2+'/'+'segmentation_'+file_2[10:17]+'.png'
                    seg_file_dst=linemod_format_dir+'mask/'+f'{ind:07d}'+'.png'
                    copyfile(seg_file_src, seg_file_dst)
                    rbg_file_src=dataset_fir+dir_1+'/'+dir_2+'/'+'rgb_'+file_2[10:15]+'.jpeg'
                    rgb_file_dst=linemod_format_dir+'rgb/'+f'{ind:07d}'+'.png'
                    rgb_image=cv2.imread(rbg_file_src)
                    cv2.imwrite(rgb_file_dst, rgb_image)
                    depth_file_src=dataset_fir+dir_1+'/'+dir_2+'/'+'depth_'+file_2[10:15]+'.png'
                    depth_file_dst=linemod_format_dir+'depth/'+f'{ind:07d}'+'.png'
                    copyfile(depth_file_src, depth_file_dst)
                    
                    ind+=1
                    if ind%10==0:
                        print(ind)
            if dir_ind%10==0:
                print('dir_ind', dir_ind)
    
    #dump info and train test splits to files
    yaml.dump(gt_yml, open(linemod_format_dir+'gt.yml', 'w'))
    with open(linemod_format_dir+'train.txt', 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)
    with open(linemod_format_dir+'test.txt', 'w') as f:
        for item in test_files:
            f.write("%s\n" % item)
    pass

def pickle_com():
    shapenet_dir='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/ShapeNetCore.v2/'
    file_com_info_dict={}
    for dir_1 in os.listdir(shapenet_dir):
        if os.path.isdir(shapenet_dir+dir_1):
            for dir_2 in os.listdir(shapenet_dir+dir_1):
                cog_file_name=shapenet_dir+dir_1+'/'+dir_2+'/models/cog.json'
                if os.path.isfile(cog_file_name):
                    file_com_info_dict[cog_file_name]=json.load(open(cog_file_name, 'rb'))
    pickle.dump(file_com_info_dict, open('/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/shapenet_cog.p', 'wb')) 

def pickle_decom():
    shapenet_dir='/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/ShapeNetCore.v2/'
    new_shapenet_dir='/media/ssda/data/ShapeNetCore.v2/'
    file_com_info_dict=pickle.load(open('/home/william/shapenet_cog.p', 'rb')) 
    for com_info_file in file_com_info_dict:
        new_file_name=new_shapenet_dir+com_info_file[len(shapenet_dir):]
        json.dump(file_com_info_dict[com_info_file], open(new_file_name, 'w'))
                
    
  
fix_bounding_boxes(0, 40000)    
#chris_to_linemod_format(7, 8)
#pickle_decom()
#fix_pose_filenames()
#print_missing()