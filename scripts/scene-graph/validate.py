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
        self.relu = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_vec'], edges.src['bbox'],
                         edges.dst['node_class_vec'], edges.dst['bbox'])
        out = self.relu(self.mlp1(feat))
        out = self.mlp2(out)
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
        feat = nd.concat(edges.src['node_class_vec'], edges.src['emb'], edges.src['bbox'],
                         edges.dst['node_class_vec'], edges.src['emb'], edges.dst['bbox'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'preds': out}

class EdgeGCN(nn.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_obj_classes,
                 n_layers,
                 activation,
                 box_feat_ext,
                 pretrained_base,
                 ctx):
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
        self._box_feat_ext = get_model(box_feat_ext, pretrained=pretrained_base, ctx=ctx).features[:-2]
        self._softmax = SoftmaxHD(axis=(1))

    def forward(self, g):
        if g is None or g.number_of_nodes() == 0:
            return g
        # link pred
        g.apply_edges(self.edge_link_mlp)
        # graph conv
        x = self._box_feat_ext(g.ndata['images'])
        x = x.sum(axis=1).reshape((0, -1))
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
        g.ndata['emb'] = x
        # link classification
        g.apply_edges(self.edge_mlp)
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
batch_size = num_gpus * 4
ctx = [mx.gpu(i) for i in range(num_gpus)]
N_relations = 50
N_objects = 150

net = EdgeGCN(in_feats=49, n_hidden=32, n_classes=N_relations, n_obj_classes=N_objects+1,
              n_layers=3, activation=nd.relu,
              box_feat_ext='mobilenet1.0', pretrained_base=False, ctx=ctx)
net.load_parameters('params/model-9.params', ctx=ctx)

vg_val = gcv.data.VGRelation(top_frequent_rel=N_relations, top_frequent_obj=N_objects,
                             balancing='weight', split='val', randomness=False)
logger.info('data loaded!')
val_data = gluon.data.DataLoader(vg_val, batch_size=1, shuffle=False, num_workers=60,
                                   batchify_fn=gcv.data.dataloader.dgl_mp_batchify_fn)

detector = get_model('faster_rcnn_resnet50_v1b_custom', classes=vg_val._obj_classes,
                     pretrained_base=False, pretrained=False, additional_output=True)
params_path = 'faster_rcnn_resnet50_v1b_custom_0007_0.2398.params'
detector.load_parameters(params_path, ctx=ctx, ignore_extra=True, allow_missing=True)

def get_data_batch(g_list, ctx_list):
    n_gpu = len(ctx_list)
    size = len(g_list)
    if size < n_gpu:
        raise ValueError("Too many slices for data.")
    step = size // n_gpu
    slices = [g_list[i*step:(i+1)*step] if i < n_gpu - 1 else g_list[i*step:size] for i in range(n_gpu)]
    G_list = [dgl.batch(slc) for slc in slices]
    for G, ctx in zip(G_list, ctx_list):
        G.ndata['images'] = G.ndata['images'].as_in_context(ctx)
        G.ndata['bbox'] = G.ndata['bbox'].as_in_context(ctx)
        G.ndata['node_class_ids'] = G.ndata['node_class_ids'].as_in_context(ctx)
        G.ndata['node_class_vec'] = G.ndata['node_class_vec'].as_in_context(ctx)
        G.edata['classes'] = G.edata['classes'].as_in_context(ctx)
        G.edata['link'] = G.edata['link'].as_in_context(ctx)
        G.edata['weights'] = G.edata['weights'].expand_dims(1).as_in_context(ctx)
    return G_list

@mx.metric.register
@mx.metric.alias('predcls')
class PredCls(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(PredCls, self).__init__('predcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
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

def get_triplet_predcls(g, link_pred_topk=100):
    # gt triplet
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel = g.edata['classes'][gt_eids]
    gt_triplet = [(sub, rel, ob) for sub, rel, ob in zip(gt_node_sub, gt_rel, gt_node_ob)]

    # pred triplet
    tmp = nd.softmax(g.edata['link_preds'][:,1]).asnumpy()
    # eids = np.where(tmp > link_pred_thresh)[0]
    eids = tmp.argsort()[0:link_pred_topk]
    if len(eids) == 0:
        return gt_triplet, []
    pred_node_ids = g.find_edges(eids)
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()
    pred_link = nd.softmax(g.edata['link_preds'][eids])
    pred_rel = nd.softmax(g.edata['preds'][eids])
    pred_triplet = []

    n = len(eids)
    for rel in range(pred_rel.shape[1]):
        scores = (pred_link[:,1] * pred_rel[:, rel]).asnumpy()
        for i in range(n):
            sub = pred_node_sub[i]
            ob = pred_node_ob[i]
            pred_triplet.append((scores[i], sub, rel, ob))
    return gt_triplet, pred_triplet

@mx.metric.register
@mx.metric.alias('sgdet')
class SGDet(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(SGDet, self).__init__('predcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
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

def get_triplet_sgdet(g, link_pred_topk=100):
    # gt triplet
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    import pdb; pdb.set_trace()
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel = g.edata['classes'][gt_eids].asnumpy().astype(gt_node_sub.dtype)
    gt_tuple = [(sub, rel, ob) for sub, rel, ob in zip(gt_node_sub, gt_rel, gt_node_ob)]

    gt_sub_class = g.ndata['node_class_ids'][gt_node_sub].asnumpy()
    gt_ob_class = g.ndata['node_class_ids'][gt_node_ob].asnumpy()
    gt_sub_bbox = g.ndata['bbox'][gt_node_sub].asnumpy()
    gt_ob_bbox = g.ndata['bbox'][gt_node_ob].asnumpy()

    # pred triplet
    tmp = nd.softmax(g.edata['link_preds'][:,1]).asnumpy()
    eids = tmp.argsort()[0:link_pred_topk]
    pred_node_ids = g.find_edges(eids)
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()
    pred_link = nd.softmax(g.edata['link_preds'][eids])
    pred_rel = nd.softmax(g.edata['preds'][eids])
    pred_triplet = []

    n = len(eids)
    for rel in range(pred_rel.shape[1]):
        scores = (pred_link[:,1] * pred_rel[:, rel]).asnumpy()
        for i in range(n):
            sub = pred_node_sub[i]
            ob = pred_node_ob[i]
            pred_triplet.append((scores[i], sub, rel, ob))
    return gt_triplet, pred_triplet

def merge_res(g, scores, bbox, feat_ind, spatial_feat, cls_pred, iou_thresh=0.5):
    img = g.ndata['images'][0]
    gt_bbox = g.ndata['bbox']
    gt_bbox[0,:] = [0, 0, 0, 0]
    img_size = img.shape[1:3]
    bbox[:, 0] /= img_size[1]
    bbox[:, 1] /= img_size[0]
    bbox[:, 2] /= img_size[1]
    bbox[:, 3] /= img_size[0]
    inds = np.where(scores[:,0].asnumpy() > 0)[0].tolist()
    if len(inds) == 0:
        return None
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

def validate(net, val_data, ctx, mode='predcls'):
    if mode == 'predcls':
        metric_20 = PredCls(topk=20)
        metric_50 = PredCls(topk=50)
        metric_100 = PredCls(topk=100)
    elif mode == 'sgdet':
        metric_20 = SGDet(topk=20)
        metric_50 = SGDet(topk=50)
        metric_100 = SGDet(topk=100)
    metric_20.reset()
    metric_50.reset()
    metric_100.reset()
    for i, g_list in enumerate(val_data):
        if (i+1) % 100 == 0:
            _, pred20 = metric_20.get()
            _, pred50 = metric_50.get()
            _, pred100 = metric_100.get()
            print("%d, %f, %f, %f"%(i, pred20, pred50, pred100))
        if len(g_list) == 0:
            continue
        if isinstance(g_list, dgl.DGLGraph):
            G_list = get_data_batch([g_list], ctx)
        elif len(g_list) < len(ctx):
            continue
        else:
            G_list = get_data_batch(g_list, ctx)
        detect_res_list = [detector(G.ndata['images'][0].expand_dims(axis=0)) for G in G_list]
        G_list = [merge_res(G, scores[0], bounding_boxs[0], feat_ind[0], spatial_feat[0], cls_pred[0]) \
                      for G, (class_ids, scores, bounding_boxs, feat, feat_ind, spatial_feat, cls_pred) in \
                          zip(G_list, detector_res_list)]
        G_list = [net(G) for G in G_list]
        for G in G_list:
            if mode == 'predcls':
                gt_triplet, pred_triplet = get_triplet_predcls(G)
            elif mode == 'sgdet':
                gt_triplet, pred_triplet = get_triplet_sgdet(G)
            metric_20.update(gt_triplet, pred_triplet)
            metric_50.update(gt_triplet, pred_triplet)
            metric_100.update(gt_triplet, pred_triplet)
            '''
            sgdet_20.update(gt_triplet, pred_triplet)
            sgdet_50.update(gt_triplet, pred_triplet)
            sgdet_100.update(gt_triplet, pred_triplet)
            '''
    _, pred20 = metric_20.get()
    _, pred50 = metric_50.get()
    _, pred100 = metric_100.get()
    return pred20, pred50, pred100

predcls20, predcls50, predcls100 = validate(net, val_data, ctx, mode='predcls')
print("%f, %f, %f"%(predcls20, predcls50, predcls100))

