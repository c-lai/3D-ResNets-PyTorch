import imp
import json
from pathlib import Path

import torch
import torch.utils.data as data

import numpy as np

from heart_volume.heart_utils import remove_zero_margin
from .loader import HeartVolumeLoader
from config import target_image_size



def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, key))

    return video_ids, video_paths, annotations


class HeartDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 hv_loader=None,
                 hv_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 target_type='label'):
        self.data, self.class_names, self.class_sample_num = self.__make_dataset(
            root_path, annotation_path, subset, hv_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if hv_loader is None:
            self.loader = HeartVolumeLoader()
        else:
            self.loader = hv_loader

        self.target_type = target_type
        self.pos_weight = torch.tensor(self.class_sample_num[0]/self.class_sample_num[1])

    def __make_dataset(self, root_path, annotation_path, subset,
                       hv_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, hv_path_formatter)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        class_sample_num = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name
            class_sample_num[label] = 0

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            sample = {
                'video': video_path,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)
            class_sample_num[label_id] += 1

        return dataset, idx_to_class, class_sample_num

    def __loading(self, path):
        hv = self.loader(path)
        hv_img = hv.pixel_array

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            slices = [self.spatial_transform(np.clip(hv_img[:,:,slice,None],0,1)) for slice in range(hv_img.shape[2])]
        hv_img = torch.stack(slices, 0).permute(1, 0, 2, 3)

        return hv_img
    
    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        img = self.__loading(path)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
