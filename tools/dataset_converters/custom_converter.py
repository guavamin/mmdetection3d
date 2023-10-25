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

    totalPoints = os.listdir(root_path + '/points')
    pcdLabels = os.listdir(root_path + '/labels')
    for i in range(len(totalPoints)):
       
        file_name = totalPoints[i][:-4]
        print("file_name: ", file_name)
        lidar_path = root_path + '/points/' + file_name + '.bin'
       
        pcdLabel_path = root_path + '/labels/' + file_name + '.txt'
       
        mmengine.check_file_exist(lidar_path)
        mmengine.check_file_exist(pcdLabel_path)

        time_stamp = 
        
        print('time_stamp: ', time_stamp)
        
        info = {
            'sample_idx': i,
            'timestamp': time_stamp,
            'lidar_points': dict(),
            'instances': [],
        }
       
       
        info['lidar_points']['lidar_path'] = lidar_path
        info['lidar_points']['num_pts_feats'] = 5

        sys.exit("Debug stop right here")

        with open(pcdLabel_path, 'r', encoding='utf-8') as f:
            # i = 0
            for ann in f.readlines():
                ann = ann.strip('\n')
                ann = ann.split()
                if len(ann):
                    info['instances'].append(dict())
                    info['instances'][-1]['bbox'] = [0, 0, 0, 0]
                    info['instances'][-1]['bbox_label'] = pcdCategories[ann[1]]
                    info['instances'][-1]['bbox_3d'] = [float(ann[2])/100, float(ann[3])/100, float(ann[4])/100, float(ann[7])/100, float(ann[9])/100, float(ann[8])/100, float(ann[6])/180*math.pi]
                    info['instances'][-1]['bbox_label_3d'] = pcdCategories[ann[1]]
                    info['instances'][-1]['alpha'] = 0.0
                    info['instances'][-1]['occluded'] = int(float(ann[11]))
                    info['instances'][-1]['truncated'] = int(float(ann[10]))
                    info['instances'][-1]['difficulty'] = int(float(ann[12]))-2
                    info['instances'][-1]['score'] = 0.0
       
        if file_name in train_dict:
            train_infos.append(info)
        else:
            val_infos.append(info)
               

    return train_infos, val_infos

if __name__ == '__main__':
    train_infos, val_infos = _fill_trainval_infos('/data/mmdetection3d-main/projects/custom/data')
    print(len(train_infos))
    print(len(val_infos))
