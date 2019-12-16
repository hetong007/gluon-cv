import dgl
import gluoncv as gcv
import mxnet as mx
import numpy as np
import logging, time
from operator import itemgetter
from mxnet import nd, gluon
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Pad

class SoftmaxHD(nn.HybridBlock):
    """Softmax on multiple dimensions
    Parameters
    ----------
    axis : the axis for softmax normalization
    """
    def __init__(self, axis=(2, 3), **kwargs):
        super(SoftmaxHD, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x):
        x_max = F.max(x, axis=self.axis, keepdims=True)
        x_exp = F.exp(F.broadcast_minus(x, x_max))
        norm = F.sum(x_exp, axis=self.axis, keepdims=True)
        res = F.broadcast_div(x_exp, norm)
        return res

class EdgeLinkMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeLinkMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu1 = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_hidden)
        self.relu2 = nn.Activation('relu')
        self.mlp3 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_prob'], edges.src['pred_bbox'],
                         edges.dst['node_class_prob'], edges.dst['pred_bbox'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'link_preds': out}

class EdgeMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu1 = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_hidden)
        self.relu2 = nn.Activation('relu')
        self.mlp3 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_prob'], edges.src['emb'], edges.src['pred_bbox'],
                         edges.dst['node_class_prob'], edges.dst['emb'], edges.dst['pred_bbox'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'preds': out}

class EdgeGCN(nn.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 box_feat_ext,
                 pretrained_base=True,
                 ctx=mx.cpu()):
        super(EdgeGCN, self).__init__()
        self.layers = nn.Sequential()
        # input layer
        self.layers.add(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.edge_link_mlp = EdgeLinkMLP(50, 2)
        self.edge_mlp = EdgeMLP(100, n_classes)
        # self._box_feat_ext = get_model(box_feat_ext, pretrained=pretrained_base, ctx=ctx).features[:-2]
        self._softmax = SoftmaxHD(axis=(1))

    def forward(self, g):
        if g is None or g.number_of_nodes() == 0:
            return g
        cls = g.ndata['node_class_pred']
        g.ndata['node_class_prob'] = self._softmax(cls)
        # link pred
        g.apply_edges(self.edge_link_mlp)
        eids = np.where(nd.softmax(g.edata['link_preds'])[:,1].asnumpy() > 0.1)[0]
        if len(eids) == 0:
            return None
        sub_g = g.edge_subgraph(toindex(eids.tolist()))
        sub_g.copy_from_parent()
        x = sub_g.ndata['node_feat']
        for i, layer in enumerate(self.layers):
            x = layer(sub_g, x)
        sub_g.ndata['emb'] = x
        sub_g.apply_edges(self.edge_mlp)
        sub_g.copy_to_parent()
        '''
        # graph conv
        x = g.ndata['node_feat']
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
        g.ndata['emb'] = x
        # link classification
        g.apply_edges(self.edge_mlp)
        '''
        return g

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Hyperparams
# ctx = mx.gpu() 
num_gpus = 1
batch_size = num_gpus * 16
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
N_relations = 50
N_objects = 150

net = EdgeGCN(in_feats=49, n_hidden=32, n_classes=N_relations,
              n_layers=2, activation=nd.relu,
              box_feat_ext='mobilenet1.0', pretrained_base=False, ctx=ctx)
net.load_parameters('params/model-9.params', ctx=ctx)

vg_val = gcv.data.VGRelation(top_frequent_rel=N_relations, top_frequent_obj=N_objects,
                             balancing='weight', split='val')
logger.info('data loaded!')
val_data = gluon.data.DataLoader(vg_val, batch_size=1, shuffle=False, num_workers=60,
                                   batchify_fn=gcv.data.dataloader.dgl_mp_batchify_fn)

'''
detector = get_model('faster_rcnn_resnet50_v1b_custom', classes=vg_val._obj_classes,
                     pretrained_base=False, pretrained=False, additional_output=True)
params_path = 'faster_rcnn_resnet50_v1b_custom_0007_0.2398.params'
'''
detector = get_model('faster_rcnn_resnet101_v1d_custom', classes=vg_val._obj_classes,
                     pretrained_base=False, pretrained=False, additional_output=True)
params_path = 'faster_rcnn_resnet101_v1d_custom_0005_0.2528.params'
detector.load_parameters(params_path, ctx=ctx, ignore_extra=True, allow_missing=True)
detector.class_predictor.load_parameters('params/class_predictor-9.params', ctx=ctx)

def get_data_batch(g_list, img_list, ctx_list):
    if g_list is None or len(g_list) == 0:
        return None, None
    n_gpu = len(ctx_list)
    size = len(g_list)
    if size < n_gpu:
        raise Exception("too small batch")
    step = size // n_gpu
    G_list = [g_list[i*step:(i+1)*step] if i < n_gpu - 1 else g_list[i*step:size] for i in range(n_gpu)]
    img_list = [img_list[i*step:(i+1)*step] if i < n_gpu - 1 else img_list[i*step:size] for i in range(n_gpu)]

    for G_slice, ctx in zip(G_list, ctx_list):
        for G in G_slice:
            G.ndata['bbox'] = G.ndata['bbox'].as_in_context(ctx)
            G.ndata['node_class_ids'] = G.ndata['node_class_ids'].as_in_context(ctx)
            G.ndata['node_class_vec'] = G.ndata['node_class_vec'].as_in_context(ctx)
            G.edata['classes'] = G.edata['classes'].as_in_context(ctx)
            G.edata['link'] = G.edata['link'].as_in_context(ctx)
            G.edata['weights'] = G.edata['weights'].expand_dims(1).as_in_context(ctx)
    img_list = [img.as_in_context(ctx) for img in img_list]
    return G_list, img_list

@mx.metric.register
@mx.metric.alias('predcls')
class PredCls(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(PredCls, self).__init__('predcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        from operator import itemgetter
        preds = sorted(preds, key=itemgetter(0), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i in range(m):
            score, pred_sub, pred_rel, pred_ob = preds[i]
            for gt_sub, gt_rel, gt_ob in labels:
                if gt_sub == pred_sub and \
                   gt_rel == pred_rel and \
                   gt_ob == pred_ob:
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

@mx.metric.register
@mx.metric.alias('phrcls')
class PhrCls(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(PhrCls, self).__init__('phrcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        from operator import itemgetter
        preds = sorted(preds, key=itemgetter(0), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i in range(m):
            score, pred_sub, pred_rel, pred_ob, pred_sub_cls, pred_ob_cls = preds[i]
            for gt_sub, gt_rel, gt_ob, gt_sub_cls, gt_ob_cls in labels:
                if gt_sub_cls == pred_sub_cls and \
                   gt_ob_cls == pred_ob_cls and \
                   gt_sub == pred_sub and \
                   gt_rel == pred_rel and \
                   gt_ob == pred_ob:
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

def get_triplet_predcls(g, link_pred_topk=100):
    # gt triplet
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel = g.edata['classes'][gt_eids].asnumpy()
    gt_triplet = []
    for sub, rel, ob in zip(gt_node_sub, gt_rel, gt_node_ob):
        gt_triplet.append((int(sub), int(rel), int(ob)))

    # pred triplet
    tmp = nd.softmax(g.edata['link_preds'][:,1]).asnumpy()
    # eids = np.where(tmp > link_pred_thresh)[0]
    # eids = tmp.argsort()[::-1][0:link_pred_topk]
    eids = tmp.argsort()[::-1]
    if len(eids) == 0:
        return gt_triplet, []
    pred_node_ids = g.find_edges(eids)
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()
    pred_link = nd.softmax(g.edata['link_preds'][eids]).asnumpy()
    pred_rel = nd.softmax(g.edata['preds'][eids]).asnumpy()
    pred_triplet = []

    n = len(eids)
    # for rel in range(pred_rel.shape[1]):
    rel_ind = pred_rel.argmax(axis=1)
    for i in range(n):
        rel = rel_ind[i]
        scores = (pred_link[i,1] * pred_rel[i, rel])
        sub = pred_node_sub[i]
        ob = pred_node_ob[i]
        pred_triplet.append((scores, int(sub), int(rel), int(ob)))
    return gt_triplet, pred_triplet

def get_triplet_phrcls(g, link_pred_topk=100):
    # gt triplet
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel = g.edata['classes'][gt_eids].asnumpy()
    gt_sub_class = g.ndata['node_class_ids'][gt_node_sub][:,0].asnumpy()
    gt_ob_class = g.ndata['node_class_ids'][gt_node_ob][:,0].asnumpy()
    gt_triplet = []
    for sub, rel, ob, sub_class, ob_class in zip(gt_node_sub, gt_rel, gt_node_ob, gt_sub_class, gt_ob_class):
        gt_triplet.append((int(sub), int(rel), int(ob), int(sub_class), int(ob_class)))

    # pred triplet
    tmp = nd.softmax(g.edata['link_preds'][:,1]).asnumpy()
    # eids = np.where(tmp > link_pred_thresh)[0]
    eids = tmp.argsort()[::-1][0:link_pred_topk]
    if len(eids) == 0:
        return gt_triplet, []
    pred_node_ids = g.find_edges(eids)
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()
    pred_link = nd.softmax(g.edata['link_preds'][eids]).asnumpy()
    pred_rel = nd.softmax(g.edata['preds'][eids]).asnumpy()
    pred_sub_prob = g.ndata['node_class_prob'][pred_node_sub, 1:].asnumpy()
    pred_ob_prob = g.ndata['node_class_prob'][pred_node_ob, 1:].asnumpy()
    pred_sub_class = pred_sub_prob.argmax(axis=1)
    pred_ob_class = pred_ob_prob.argmax(axis=1)
    pred_triplet = []

    n = len(eids)
    rel_ind = pred_rel.argmax(axis=1)
    # for rel in range(pred_rel.shape[1]):
    # import pdb; pdb.set_trace()
    for i in range(n):
        rel = rel_ind[i]
        scores = (pred_link[i,1] * pred_rel[i, rel])
        scores *= pred_sub_prob[i, rel]
        scores *= pred_ob_prob[i, rel]
        sub = pred_node_sub[i]
        ob = pred_node_ob[i]
        sub_class = pred_sub_class[i]
        ob_class = pred_ob_class[i]
        pred_triplet.append((scores, int(sub), int(rel), int(ob),
                             int(sub_class), int(ob_class)))
    return gt_triplet, pred_triplet

def get_triplet_sgdet(g):
    # gt triplet
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel = g.edata['classes'][gt_eids].asnumpy()
    gt_sub_class = g.ndata['node_class_ids'][gt_node_sub][:,0].asnumpy()
    gt_ob_class = g.ndata['node_class_ids'][gt_node_ob][:,0].asnumpy()
    gt_sub_bbox = g.ndata['bbox'][gt_node_sub].asnumpy()
    gt_ob_bbox = g.ndata['bbox'][gt_node_ob].asnumpy()
    gt_triplet = []
    for sub, rel, ob, sub_class, ob_class, sub_bbox, ob_bbox in zip(gt_node_sub, gt_rel, gt_node_ob, gt_sub_class, gt_ob_class, gt_sub_bbox, gt_ob_bbox):
        gt_triplet.append((int(sub), int(rel), int(ob), int(sub_class), int(ob_class),
                           sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                           ob_bbox[0], ob_bbox[1], ob_bbox[2], ob_bbox[3]))

    # pred triplet
    tmp = nd.softmax(g.edata['link_preds'][:,1]).asnumpy()
    # eids = np.where(tmp > link_pred_thresh)[0]
    eids = tmp.argsort()[::-1][0:link_pred_topk]
    if len(eids) == 0:
        return gt_triplet, []
    pred_node_ids = g.find_edges(eids)
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()
    pred_link = nd.softmax(g.edata['link_preds'][eids]).asnumpy()
    pred_rel = nd.softmax(g.edata['preds'][eids]).asnumpy()
    pred_sub_prob = g.ndata['node_class_prob'][pred_node_sub, 1:].asnumpy()
    pred_ob_prob = g.ndata['node_class_prob'][pred_node_ob, 1:].asnumpy()
    pred_sub_class = pred_sub_prob.argmax(axis=1)
    pred_ob_class = pred_ob_prob.argmax(axis=1)
    pred_sub_bbox = g.ndata['pred_bbox'][gt_node_sub].asnumpy()
    pred_ob_bbox = g.ndata['pred_bbox'][gt_node_ob].asnumpy()
    pred_triplet = []

    n = len(eids)
    rel_ind = pred_rel.argmax(axis=1)
    # for rel in range(pred_rel.shape[1]):
    # import pdb; pdb.set_trace()
    for i in range(n):
        rel = rel_ind[i]
        scores = (pred_link[i,1] * pred_rel[i, rel])
        scores *= pred_sub_prob[i, rel]
        scores *= pred_ob_prob[i, rel]
        sub = pred_node_sub[i]
        ob = pred_node_ob[i]
        sub_class = pred_sub_class[i]
        ob_class = pred_ob_class[i]
        sub_bbox = pred_sub_bbox[i]
        ob_bbox = pred_ob_bbox[i]
        pred_triplet.append((scores, int(sub), int(rel), int(ob), int(sub_class), int(ob_class), 
                             sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                             ob_bbox[0], ob_bbox[1], ob_bbox[2], ob_bbox[3]))
    return gt_triplet, pred_triplet

@mx.metric.register
@mx.metric.alias('sgdet')
class SGDet(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(SGDet, self).__init__('predcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        from operator import itemgetter
        preds = sorted(preds, key=itemgetter(0), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i in range(m):
            score, pred_sub, pred_sub_bbox, pred_rel, pred_ob, pred_ob_bbox = preds[i]
            for gt_sub, gt_rel, gt_ob in labels:
                if gt_sub == pred_sub and \
                   gt_rel == pred_rel and \
                   gt_ob == pred_ob:
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

def merge_res_iou(g_slice, img_batch, scores, bbox_list, feat_ind, spatial_feat, cls_pred):
    import pdb; pdb.set_trace()
    for i, g in enumerate(g_slice):
        img = img_batch[i]
        gt_bbox = g.ndata['bbox']
        img_size = img.shape[1:3]
        gt_bbox[:, 0] /= img_size[1]
        gt_bbox[:, 1] /= img_size[0]
        gt_bbox[:, 2] /= img_size[1]
        gt_bbox[:, 3] /= img_size[0]
        inds = np.where(scores[i,:,0].asnumpy() > 0)[0].tolist()
        if len(inds) == 0:
            return None
        bbox = bbox_list[i,inds,:]
        bbox[:, 0] /= img_size[1]
        bbox[:, 1] /= img_size[0]
        bbox[:, 2] /= img_size[1]
        bbox[:, 3] /= img_size[0]
        ious = nd.contrib.box_iou(gt_bbox, bbox[inds])
        # assignment
        H, W = ious.shape
        h = H
        w = W
        assign_ind = [-1 for i in range(H)]
        while h > 0 and w > 0:
            ind = int(ious.argmax().asscalar())
            row_ind = ind // W
            col_ind = ind % W
            if ious[row_ind, col_ind].asscalar() < iou_thresh:
                break
            assign_ind[row_ind] = col_ind
            ious[row_ind, :] = -1
            ious[:, col_ind] = -1
            h -= 1
            w -= 1

        remove_inds = [i for i in range(H) if assign_ind[i] == -1]
        assign_ind = [ind for ind in assign_ind if ind > -1]
        g.remove_nodes(remove_inds)
        box_ind = [inds[i] for i in assign_ind]
        roi_ind = feat_ind[box_ind].squeeze(1)
        g.ndata['pred_bbox'] = bbox[box_ind]
        g.ndata['node_feat'] = spatial_feat[roi_ind]
        g.ndata['node_class_pred'] = cls_pred[roi_ind]
    return g

def merge_res(g_slice, img, bbox, spatial_feat, cls_pred):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]
    for i, g in enumerate(g_slice):
        n_node = g.number_of_nodes()
        g.ndata['pred_bbox'] = bbox[i, 0:n_node]
        g.ndata['node_class_pred'] = cls_pred[i, 0:n_node]
        g.ndata['node_feat'] = spatial_feat[i, 0:n_node]
    return dgl.batch(g_slice)

def validate(net, val_data, ctx, mode=['predcls']):
    if 'predcls' in mode:
        predcls_20 = PredCls(topk=20)
        predcls_50 = PredCls(topk=50)
        predcls_100 = PredCls(topk=100)
        predcls_20.reset()
        predcls_50.reset()
        predcls_100.reset()
    if 'phrcls' in mode:
        phrcls_20 = PhrCls(topk=20)
        phrcls_50 = PhrCls(topk=50)
        phrcls_100 = PhrCls(topk=100)
        phrcls_20.reset()
        phrcls_50.reset()
        phrcls_100.reset()
    if 'sgdet' in mode:
        sgdet_20 = SGDet(topk=20)
        sgdet_50 = SGDet(topk=50)
        sgdet_100 = SGDet(topk=100)
        sgdet_20.reset()
        sgdet_50.reset()
        sgdet_100.reset()
    for i, (g_list, img_list) in enumerate(val_data):
        G_list, img_list = get_data_batch(g_list, img_list, ctx)
        if G_list is None or img_list is None:
            continue

        detector_res_list = []
        G_batch = []
        bbox_pad = Pad(axis=(0))
        for G_slice, img in zip(G_list, img_list):
            if 'predcls' in mode or 'phrcls' in mode:
                cur_ctx = img.context
                bbox_list = [G.ndata['bbox'] for G in G_slice]
                bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
                bbox, spatial_feat, cls_pred = detector((img, bbox_stack))
                G_batch.append(merge_res(G_slice, img, bbox, spatial_feat, cls_pred))
            if 'sgdet' in mode or 'sgdet+' in mode:
                box_ids, scores, bboxes, feat, feat_ind, spatial_feat, cls_pred = detector(img)
                G_batch.append(merge_res_iou(G_slice, img, scores, bboxes, feat_ind, spatial_feat, cls_pred))

        if len(G_batch) > 0:
            G_batch = [net(G) for G in G_batch]

        for G in G_batch:
            if G is not None and G.number_of_nodes() > 0:
                if 'predcls' in mode:
                    gt_triplet, pred_triplet = get_triplet_predcls(G)
                    predcls_20.update(gt_triplet, pred_triplet)
                    predcls_50.update(gt_triplet, pred_triplet)
                    predcls_100.update(gt_triplet, pred_triplet)
                if 'phrcls' in mode:
                    gt_triplet, pred_triplet = get_triplet_phrcls(G)
                    phrcls_20.update(gt_triplet, pred_triplet)
                    phrcls_50.update(gt_triplet, pred_triplet)
                    phrcls_100.update(gt_triplet, pred_triplet)
                if 'sgdet' in mode:
                    gt_triplet, pred_triplet = get_triplet_sgdet(G)
                    sgdet_20.update(gt_triplet, pred_triplet)
                    sgdet_50.update(gt_triplet, pred_triplet)
                    sgdet_100.update(gt_triplet, pred_triplet)
            else:
                if 'predcls' in mode:
                    predcls_20.update(None, None)
                    predcls_50.update(None, None)
                    predcls_100.update(None, None)
                if 'phrcls' in mode:
                    phrcls_20.update(None, None)
                    phrcls_50.update(None, None)
                    phrcls_100.update(None, None)
                if 'sgdet' in mode:
                    sgdet_20.update(None, None)
                    sgdet_50.update(None, None)
                    sgdet_100.update(None, None)
        if (i+1) % 100 == 0:
            print_txt = ''
            if 'predcls' in mode:
                name20, pred20 = predcls_20.get()
                name50, pred50 = predcls_50.get()
                name100, pred100 = predcls_100.get()
                print_txt += '%s=%.4f,%s=%.4f,%s=%.4f\t'%(name20, pred20, name50, pred50, name100, pred100)
            if 'phrcls' in mode:
                name20, pred20 = phrcls_20.get()
                name50, pred50 = phrcls_50.get()
                name100, pred100 = phrcls_100.get()
                print_txt += '%s=%.4f,%s=%.4f,%s=%.4f\t'%(name20, pred20, name50, pred50, name100, pred100)
            if 'sgdet' in mode:
                name20, pred20 = sgdet_20.get()
                name50, pred50 = sgdet_50.get()
                name100, pred100 = sgdet_100.get()
                print_txt += '%s=%.4f,%s=%.4f,%s=%.4f\t'%(name20, pred20, name50, pred50, name100, pred100)
            logger.info(print_txt)
    print_txt = ''
    if 'predcls' in mode:
        name20, pred20 = predcls_20.get()
        name50, pred50 = predcls_50.get()
        name100, pred100 = predcls_100.get()
        print_txt += '%s=%.4f,%s=%.4f,%s=%.4f\t'%(name20, pred20, name50, pred50, name100, pred100)
    if 'phrcls' in mode:
        name20, pred20 = phrcls_20.get()
        name50, pred50 = phrcls_50.get()
        name100, pred100 = phrcls_100.get()
        print_txt += '%s=%.4f,%s=%.4f,%s=%.4f\t'%(name20, pred20, name50, pred50, name100, pred100)
    if 'sgdet' in mode:
        name20, pred20 = sgdet_20.get()
        name50, pred50 = sgdet_50.get()
        name100, pred100 = sgdet_100.get()
        print_txt += '%s=%.4f,%s=%.4f,%s=%.4f\t'%(name20, pred20, name50, pred50, name100, pred100)
    logger.info(print_txt)

validate(net, val_data, ctx, mode=['predcls', 'phrcls'])
'''
validate(net, val_data, ctx, mode='sgdet')
'''
