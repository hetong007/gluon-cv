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
# from mxnet import npx
# npx.set_np()
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
                 top_frequent_rel=50, top_frequent_obj=100, top_frequent_pair=500, balancing='sample'):
        super(VGRelation, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._dict_path = os.path.join(self._root, 'relationships.json')
        self._synsets_path = os.path.join(self._root, 'relationship_synsets.json')
        self._img_path = os.path.join(self._root, 'VG_100K', '{}.jpg')
        with open(self._dict_path) as f:
            tmp = f.read()
            self._dict = json.loads(tmp)
        rel_ctr = {}
        obj_ctr = {}
        obj_pair_ctr = {}
        for it in self._dict:
            for r in it['relationships']:
                if len(r['synsets']) > 0:
                    k = r['synsets'][0].split('.')[0]
                    if k in rel_ctr:
                        rel_ctr[k] += 1
                    else:
                        rel_ctr[k] = 1
                if len(r['subject']['synsets']) > 0:
                    k = r['subject']['synsets'][0].split('.')[0]
                    if k in obj_ctr:
                        obj_ctr[k] += 1
                    else:
                        obj_ctr[k] = 1
                if len(r['object']['synsets']) > 0:
                    k = r['object']['synsets'][0].split('.')[0]
                    if k in obj_ctr:
                        obj_ctr[k] += 1
                    else:
                        obj_ctr[k] = 1
                if len(r['subject']['synsets']) > 0 and len(r['object']['synsets']) > 0:
                    k = r['subject']['synsets'][0].split('.')[0] + '_' +\
                        r['object']['synsets'][0].split('.')[0]
                    if k in obj_pair_ctr:
                        obj_pair_ctr[k] += 1
                    else:
                        obj_pair_ctr[k] = 1
        rel_ctr_sorted = sorted(rel_ctr, key=rel_ctr.get, reverse=True)[0:top_frequent_rel]
        obj_ctr_sorted = sorted(obj_ctr, key=obj_ctr.get, reverse=True)[0:top_frequent_obj]
        obj_pair_ctr_sorted = sorted(obj_pair_ctr, key=obj_pair_ctr.get, reverse=True)[0:top_frequent_pair]
        rel_set = set(rel_ctr_sorted)
        obj_set = set(obj_ctr_sorted)
        ''' 
        # using pair
        obj_set = []
        for obj in obj_pair_ctr_sorted:
            obj_set += obj.split('_')
        obj_set = set(obj_set)
        '''
        self._relations = ['none'] + list(rel_set)
        self._relations_dict = {}
        for i, rel in enumerate(self._relations):
            self._relations_dict[rel] = i

        self._obj_classes = ['others'] + list(obj_set)
        self._obj_classes_dict = {}
        for i, obj in enumerate(self._obj_classes):
            self._obj_classes_dict[obj] = i

        self._balancing = balancing

    def __len__(self):
        return len(self._dict)

    def _extract_label(self, rel):
        n = len(rel)
        # extract global ids first, and map into local ids
        object_ids = [0]
        for rl in rel:
            sub = rl['subject']
            ob = rl['object']
            if sub['object_id'] == ob['object_id']:
                continue
            object_ids.append(sub['object_id'])
            object_ids.append(ob['object_id'])
        object_ids = list(set(object_ids))

        ids_dict = {}
        for i, obj in enumerate(object_ids):
            ids_dict[str(obj)] = i
        m = len(object_ids)
        bbox = mx.nd.zeros((m, 4))
        node_class = mx.nd.zeros((m))
        visit_ind = set()
        edges = {'src': [],
                 'dst': [],
                 'rel': [],
                 'link': []}
        for rl in rel:
            # extract xyhw and remap object id
            sub = rl['subject']
            ob = rl['object']
            sub_key = str(sub['object_id'])
            ob_key = str(ob['object_id'])
            if sub_key not in ids_dict or ob_key not in ids_dict:
                continue
            sub_ind = ids_dict[sub_key]
            ob_ind = ids_dict[ob_key]
            if sub_ind == ob_ind:
                continue
            if sub_ind not in visit_ind:
                visit_ind.add(sub_ind)
                bbox[sub_ind,] = mx.nd.array([sub['x'], sub['y'],
                                              sub['w'] + sub['x'], sub['h'] + sub['y']])
                if len(sub['synsets']) == 0:
                    node_class[sub_ind] = 0
                else:
                    k = sub['synsets'][0].split('.')[0]
                    if k in self._obj_classes_dict:
                        node_class[sub_ind] = self._obj_classes_dict[k]
                    else:
                        node_class[sub_ind] = 0
            if ob_ind not in visit_ind:
                visit_ind.add(ob_ind)
                bbox[ob_ind,] = mx.nd.array([ob['x'], ob['y'],
                                             ob['w'] + ob['x'], ob['h'] + ob['y']])
                if len(ob['synsets']) == 0:
                    node_class[ob_ind] = 0
                else:
                    k = ob['synsets'][0].split('.')[0]
                    if k in self._obj_classes_dict:
                        node_class[ob_ind] = self._obj_classes_dict[k]
                    else:
                        node_class[ob_ind] = 0
            sub['object_id'] = sub_ind
            ob['object_id'] = ob_ind

            # relational label id of the edge
            if len(rl['synsets']) == 0:
                rel_idx = 0
            else:
                synset = rl['synsets'][0].split('.')[0]
                if synset in self._relations_dict:
                    rel_idx = self._relations_dict[synset]
                else:
                    rel_idx = 0
            link = 0 if rel_idx == 0 else 1
            edges['src'].append(sub_ind)
            edges['dst'].append(ob_ind)
            edges['rel'].append(rel_idx)
            edges['link'].append(link)
        n_classes = len(self._obj_classes_dict)
        eta = 0.1
        node_class = node_class.one_hot(n_classes, on_value = 1 - eta + eta/n_classes, off_value = eta / n_classes)
        return edges, bbox, node_class

    def _build_complete_graph(self, edges, bbox, node_class, img):
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
        bbox[:,0] /= bbox[0, 2]
        bbox[:,1] /= bbox[0, 3]
        bbox[:,2] /= bbox[0, 2]
        bbox[:,3] /= bbox[0, 3]
        g.ndata['bbox'] = bbox
        g.ndata['node_class'] = node_class

        # assign class label to edges
        eids = g.edge_ids(edges['src'], edges['dst'])
        n = g.number_of_edges()
        k = eids.shape[0]

        classes = np.zeros((n))
        classes[eids.asnumpy()] = edges['rel']
        g.edata['classes'] = mx.nd.array(classes)
        links = np.zeros((n))
        links[eids.asnumpy()] = edges['link']
        if links.sum() == 0:
            return None
        g.edata['link'] = mx.nd.array(links)

        if self._balancing == 'weight':
            if n == k:
                w0 = 0
            else:
                w0 = 1.0 / (2 * (n - k))
            if k == 0:
                wn = 0
            else:
                wn = 1.0 / (2 * k)
            weights = np.zeros((n)) + w0
            weights[eids.asnumpy()] = wn
        elif self._balancing == 'sample':
            sample_ind = np.random.randint(0, n, 2*k)
            weights = np.zeros((n))
            weights[sample_ind] = 1
            weights[eids.asnumpy()] = 1
        else:
            raise NotImplementedError
        g.edata['weights'] = mx.nd.array(weights)

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

        edges, bbox, node_class = self._extract_label(rel)
        if bbox.shape[0] < 2:
            return None
        bbox[0] = mx.nd.array([0, 0, img.shape[1], img.shape[0]])
        g = self._build_complete_graph(edges, bbox, node_class, img)

        return g
