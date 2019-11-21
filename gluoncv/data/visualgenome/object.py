"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import warnings
import json
import numpy as np
import mxnet as mx
from ..base import VisionDataset
from collections import Counter
from ...data.transforms.pose import crop_resize_normalize

class VGObject(VisionDataset):
    """Pascal VOC detection Dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'visualgenome'),
                 top_frequent_obj=150,):
        super(VGObject, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._dict_path = os.path.join(self._root, 'objects.json')
        self._img_path = os.path.join(self._root, 'VG_100K', '{}.jpg')
        with open(self._dict_path) as f:
            tmp = f.read()
            self._ori_dict = json.loads(tmp)
        obj_ctr = {}
        for it in self._ori_dict:
            for r in it['objects']:
                if len(r['synsets']) > 0:
                    k = r['synsets'][0].split('.')[0]
                    if k in obj_ctr:
                        obj_ctr[k] += 1
                    else:
                        obj_ctr[k] = 1
        obj_ctr_sorted = sorted(obj_ctr, key=obj_ctr.get, reverse=True)[0:top_frequent_obj]
        obj_set = set(obj_ctr_sorted)

        self._obj_classes = sorted(list(obj_set))
        self._obj_classes_dict = {}
        for i, obj in enumerate(self._obj_classes):
            self._obj_classes_dict[obj] = i

        _dict = []
        for it in self._ori_dict:
            label = []
            for objects in it['objects']:
                if len(objects['synsets']) <= 0:
                    continue
                obj_cls = objects['synsets'][0].split('.')[0]
                if obj_cls not in self._obj_classes_dict:
                    continue
                cls = self._obj_classes_dict[obj_cls]
                xmin = objects['x']
                ymin = objects['y']
                xmax = objects['w'] + xmin
                ymax = objects['h'] + ymin
                label.append([xmin, ymin, xmax, ymax, cls])
            if len(label) <= 0:
                continue
            _dict.append({'image_id': it['image_id'],
                          'label': label})
        self._dict = _dict

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, idx):
        item = self._dict[idx]

        img_id = item['image_id']
        img_path = self._img_path.format(img_id)
        img = mx.image.imread(img_path)

        label = np.array(item['label'])

        return img, label
