import os
from os import path as osp
import mmengine
from pyquaternion import Quaternion
import json
import numpy as np
import open3d as o3d
import math
import sys

pcdClass_names = ['pedestrian', 'vehicle']
pcdClass_order = [0, 1]
pcdCategories = dict(zip(pcdClass_names, pcdClass_order))

def create_custom_infos(root_path, info_prefix):

    train_infos, val_infos = _fill_trainval_infos(root_path)
    metainfo = {
        'categories': pcdCategories,
        'dataset': 'custom',
        'info_version': 1.0,
    }
   
    if train_infos is not None:
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)

    if val_infos is not None:
        data['data_list'] = val_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def _fill_trainval_infos(root_path):
    train_infos = []
    val_infos = []
    use_camera = False

    trainSet = root_path + '/ImageSets/train.txt'
    valSet = root_path + '/ImageSets/val.txt'
    train_dict  , val_dict = set(), set()
    with open(trainSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            train_dict.add(ann)
    with open(valSet, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            val_dict.add(ann)

    totalPoints = sorted(os.listdir(root_path + '/points'))
    pcdLabels = sorted(os.listdir(root_path + '/labels'))

    for i in range(len(totalPoints)):
       
        file_name = totalPoints[i][:-4]

        lidar_path = root_path + '/points/' + file_name + '.bin'
       
        pcdLabel_path = root_path + '/labels/' + file_name + '.txt'
       
        mmengine.check_file_exist(lidar_path)
        mmengine.check_file_exist(pcdLabel_path)

        with open(lidar_path, 'rb') as f:
            # Read the data into a NumPy array
            point_cloud_array = np.fromfile(f, dtype=np.float64).reshape(-1, 6)
        
        time_stamp = point_cloud_array[0, 5]

        # print("time_stamp: ", time_stamp)
        
        info = {
            'sample_idx': i,
            'timestamp': time_stamp,
            'lidar_points': dict(),
            'instances': [],
        }
       
       
        info['lidar_points']['lidar_path'] = lidar_path
        info['lidar_points']['num_pts_feats'] = 5

        with open(pcdLabel_path, 'r', encoding='utf-8') as f:
            # i = 0
            for ann in f.readlines(): # ann right here is string not dict
                annotation = eval(ann)
                # print(annotation)
                # print(annotation['sample_idx'])
                # print(annotation['labels'])
                # print(annotation['boxes'])
                # print(annotation['confidence'])
                if len(annotation):
                    for index in range(len(annotation['boxes'])):
                        info['instances'].append({'bbox': [annotation['boxes'][index][0][0], annotation['boxes'][index][0][1], annotation['boxes'][index][1][0], annotation['boxes'][index][1][1]], 'bbox_label': annotation['labels'][index]})
                        # info['instances'].append(['bbox'] = [0, 0, 0, 0])
                        # info['instances'][-1]['bbox_label'] = 2
                        # info['instances'][-1]['bbox_3d'] = [1, 1, 1, 1]
                        # info['instances'][-1]['bbox_label_3d'] = 3
      

        if file_name in train_dict:
            train_infos.append(info)
        else:
            val_infos.append(info)     
    
    # print(train_infos)
    # print(val_infos)
    
    return train_infos, val_infos

if __name__ == '__main__':
    train_infos, val_infos = _fill_trainval_infos('/data/mmdetection3d-main/projects/custom/data')
    print(len(train_infos))
    print(len(val_infos))
