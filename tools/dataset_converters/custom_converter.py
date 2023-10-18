import os
from os import path as osp
import mmengine
from pyquaternion import Quaternion
import json
import numpy as np
import open3d as o3d
import math

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
    use_camera = True

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
    pcdLabels = os.listdir(root_path + '/pcd_labels')
    imageLabels = os.listdir(root_path + '/img_labels')
    for i in range(len(totalPoints)):
       
        file_name = totalPoints[i][:-4]
        # print(file_name)
        lidar_path = root_path + '/points/' + file_name + '.bin'
       
        # add_camera_path_later
        pcdLabel_path = root_path + '/pcd_labels/' + file_name + '.txt'
        imgLabel_path = root_path + '/img_labels/' + file_name + '.txt'
       
        mmengine.check_file_exist(lidar_path)
        mmengine.check_file_exist(pcdLabel_path)
        mmengine.check_file_exist(imgLabel_path)
       
        time_stamp_list = file_name.split('_')
        time_stamp = int(time_stamp_list[0][-4:]) + int(time_stamp_list[1]) / (10 * len(time_stamp_list[1]))
        # print(time_stamp)
        info = {
            'sample_idx': i,
            'timestamp': time_stamp,
            'lidar_points': dict(),
            'images': dict(),
            'instances': [],
            'cam_instances': dict(),
        }
       
       
        info['lidar_points']['lidar_path'] = lidar_path
        info['lidar_points']['num_pts_feats'] = 4
        info['lidar_points']['Tr_velo_cam'] = np.array([
                                                        [-7.77146017e-01, 6.29320313e-01, -1.05090637e-04, -202.343],
                                                        [1.94470387e-01, 2.39992150e-01, -9.51096755e-01, 262.929],
                                                        [-5.98519287e-01, -7.39161492e-01, -3.08893108e-01, -257.586],
                                                        [0, 0, 0, 1]
                                                    ])
        info['lidar_points']['Tr_imu_to_velo'] = None

        cameras = [
            'cam62',
            # 'cam63',
            # 'cam64',
        ]

        for cam_name in cameras:
            if cam_name not in info['images']:
                info['images'][cam_name] = dict()
            cam_path = root_path + '/images/' + cam_name + '/'+ file_name + '.jpg'
            info['images'][cam_name]['img_path'] = cam_path
            info['images'][cam_name]['height'] = 1080
            info['images'][cam_name]['width'] = 1920
            info['images'][cam_name]['cam2img'] = np.array([
                                                            [1167.2, 0, 972.8049, 0],
                                                            [0, 1162.3, 554.0905, 0],
                                                            [0, 0, 1.0, 0],
                                                            [0, 0, 0, 1]
                                                        ])
            info['images'][cam_name]['lidar2cam'] = np.array([
                                                        [-7.77146017e-01, 6.29320313e-01, -1.05090637e-04, -202.343],
                                                        [1.94470387e-01, 2.39992150e-01, -9.51096755e-01, 262.929],
                                                        [-5.98519287e-01, -7.39161492e-01, -3.08893108e-01, -257.586],
                                                        [0, 0, 0, 1]
                                                    ])
            info['images'][cam_name]['lidar2img'] = np.array([
                                                        [-1.48932733e+03,  1.54827480e+01, -3.00615391e+02, -4.86755673e+05],
                                                        [-1.05600920e+02, -1.30619485e+02, -1.27661449e+03,  1.62876421e+05],
                                                        [-5.98519287e-01, -7.39161492e-01, -3.08893108e-01, -2.57586000e+02],
                                                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
                                                    ])
            # print(info['images'][cam_name])
        info['images']['R0_rect'] = np.array([
                                            [1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]
                                        ])


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


        for cam_name in cameras:
            if cam_name not in info['cam_instances']:
                info['cam_instances'][cam_name] = []
            with open(imgLabel_path, 'r', encoding='utf-8') as f:
                # i = 0
                for ann in f.readlines():
                    ann = ann.strip('\n')
                    ann = ann.split()
                    if len(ann):
                        info['cam_instances'][cam_name].append(dict())
                        info['cam_instances'][cam_name][-1]['bbox'] = [(float(ann[1])-float(ann[3])/2)*1920, (float(ann[2])-float(ann[4])/2)*1080, (float(ann[1])+float(ann[3])/2)*1920, (float(ann[2])+float(ann[4])/2)*1080]
                        info['cam_instances'][cam_name][-1]['bbox_label'] = int(ann[0])
                    # i += 1
       
        if file_name in train_dict:
            train_infos.append(info)
        else:
            val_infos.append(info)
               



    return train_infos, val_infos

if __name__ == '__main__':
    train_infos, val_infos = _fill_trainval_infos('/data/mmdetection3d-main/projects/custom/data')
    print(len(train_infos))
    print(len(val_infos))
