"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import warnings
import json
import dgl
import numpy as np
import mxnet as mx
from ..base import VisionDataset
from collections import Counter
from ...data.transforms.pose import crop_resize_normalize

class VGRelation(VisionDataset):
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
                 top_frequent=50):
        super(VGRelation, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._dict_path = os.path.join(self._root, 'relationships.json')
        self._synsets_path = os.path.join(self._root, 'relationship_synsets.json')
        self._img_path = os.path.join(self._root, 'VG_100K', '{}.jpg')
        with open(self._dict_path) as f:
            tmp = f.read()
            self._dict = json.loads(tmp)
        with open(self._synsets_path) as f:
            tmp = f.read()
            self._synsets = json.loads(tmp)
        synsets_list = [v.split('.')[0] for k,v in self._synsets.items()]
        ctr = Counter(synsets_list)
        ctr = ctr.most_common()[0:top_frequent]
        labels = ['none'] + [x[0] for x in ctr]
        labels_dict = {}
        for i, label in enumerate(labels):
            labels_dict[label] = i
        self._labels = labels
        self._labels_dict = labels_dict

    def __len__(self):
        return len(self._dict)

    def _extract_label(self, rel):
        n = len(rel)
        # extract global ids first, and map into local ids
        object_ids = []
        for rl in rel:
            sub = rl['subject']
            ob = rl['object']
            object_ids.append(sub['object_id'])
            object_ids.append(ob['object_id'])
        object_ids = list(set(object_ids))

        ids_dict = {}
        for i, obj in enumerate(object_ids):
            ids_dict[str(obj)] = i
        m = len(object_ids)
        bbox = mx.nd.zeros((m, 4))
        visit_ind = set()
        edges = {'src': [], 
                 'dst': [],
                 'rel': []}
        for rl in rel:
            # extract xyhw and remap object id
            sub = rl['subject']
            ob = rl['object']
            sub_ind = ids_dict[str(sub['object_id'])]
            ob_ind = ids_dict[str(ob['object_id'])]
            if sub_ind == ob_ind:
                continue
            if sub_ind not in visit_ind:
                visit_ind.add(sub_ind)
                bbox[sub_ind,] = mx.nd.array([sub['x'], sub['y'], sub['h'], sub['w']])
            if ob_ind not in visit_ind:
                visit_ind.add(ob_ind)
                bbox[ob_ind,] = mx.nd.array([ob['x'], ob['y'], ob['h'], ob['w']])
            sub['object_id'] = sub_ind
            ob['object_id'] = ob_ind

            # relational label id of the edge
            if len(rl['synsets']) == 0:
                label_idx = 0
            else:
                synset = rl['synsets'][0].split('.')[0]
                if synset in self._labels_dict:
                    label_idx = self._labels_dict[synset]
                else:
                    label_idx = 0
            edges['src'].append(sub_ind)
            edges['dst'].append(ob_ind)
            edges['rel'].append(label_idx)
        return edges, bbox

    def _build_complete_graph(self, edges, bbox, img):
        N = bbox.shape[0]
        g = dgl.DGLGraph()
        g.add_nodes(N)

        # complete graph
        edge_list = []
        for i in range(N-1):
            for j in range(i+1, N):
                edge_list.append((i, j))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)

        # node features
        g.ndata['bbox'] = bbox

        # assign class label to edges
        eids = g.edge_ids(edges['src'], edges['dst'])
        g.edata['class'] = mx.nd.zeros((g.number_of_edges(), 1))
        # import pdb; pdb.set_trace()
        g.edata['class'][eids,1] = mx.nd.array([edges['rel']])

        # cut img to each node
        bbox_list = []
        for i in range(bbox.shape[0]):
            bbox_list.append(bbox[i].asnumpy())
        img_list = crop_resize_normalize(img, bbox_list, (224, 224))
        imgs = mx.nd.stack(*img_list)
        g.ndata['images'] = imgs
        return g

    def __getitem__(self, idx):
        item = self._dict[idx]

        img_id = item['image_id']
        rel = item['relationships']

        img_path = self._img_path.format(img_id)
        img = mx.image.imread(img_path)

        edges, bbox = self._extract_label(rel)
        g = self._build_complete_graph(edges, bbox, img)

        return img, g
