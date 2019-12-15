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
        # self._box_feat_ext = get_model(box_feat_ext, pretrained=pretrained_base, ctx=ctx).features[:-2]
        self._softmax = SoftmaxHD(axis=(1))

    def forward(self, g):
        if g is None or g.number_of_nodes() == 0:
            return g
        cls = g.ndata['node_class_pred']
        g.ndata['node_class_prob'] = self._softmax(cls)
        # link pred
        g.apply_edges(self.edge_link_mlp)
        # subgraph for gconv
        eids = np.where(g.edata['link'].asnumpy() > 0)
        sub_g = g.edge_subgraph(toindex(eids[0].tolist()))
        sub_g.copy_from_parent()
        # graph conv
        x = sub_g.ndata['node_feat']
        for i, layer in enumerate(self.layers):
            x = layer(sub_g, x)
        sub_g.ndata['emb'] = x
        # link classification
        sub_g.apply_edges(self.edge_mlp)
        sub_g.copy_to_parent()
        n_relations = g.edata['preds'].shape[1]
        g.edata['triplet_label'] = g.edata['classes'].one_hot(n_relations)
        return g

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Hyperparams
num_gpus = 1
batch_size = num_gpus * 8
ctx = [mx.gpu(i) for i in range(num_gpus)]
nepoch = 10
N_relations = 50
N_objects = 150
save_dir = 'params'
batch_verbose_freq = 100

net = EdgeGCN(in_feats=49, n_hidden=32, n_classes=N_relations,
              n_layers=2, activation=nd.relu,
              box_feat_ext='mobilenet1.0', pretrained_base=False, ctx=ctx)
# net.initialize(ctx=ctx)
net.edge_mlp.initialize(ctx=ctx)
net.edge_link_mlp.initialize(ctx=ctx)
net.layers.initialize(ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'adam', 
                        {'learning_rate': 0.01, 'wd': 0.00001})

@mx.metric.register
@mx.metric.alias('auc')
class AUCMetric(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12):
        super(AUCMetric, self).__init__(
              'auc')
        self.eps = eps

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        label_weight = labels[0].asnumpy()
        preds = preds[0].asnumpy()
        tmp = []
        for i in range(preds.shape[0]):
            tmp.append((label_weight[i], preds[i][1]))
        tmp = sorted(tmp, key=itemgetter(1), reverse=True)
        label_sum = label_weight.sum()
        if label_sum == 0 or label_sum == label_weight.size:
            return

        label_one_num = np.count_nonzero(label_weight)
        label_zero_num = len(label_weight) - label_one_num
        total_area = label_zero_num * label_one_num
        height = 0
        width = 0
        area = 0
        for a, _ in tmp:
            if a == 1.0:
                height += 1.0
            else:
                width += 1.0
                area += height

        self.sum_metric += area / total_area
        self.num_inst += 1

L_link = gluon.loss.SoftmaxCELoss()
L_rel = gluon.loss.SoftmaxCELoss()
L_cls = gluon.loss.SoftmaxCELoss()
# L_rel = gcv.loss.FocalLoss(num_class=N_relations)

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

def get_triplet_predcls(g):
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
    rel_ind = pred_rel.argmax(axis=1)
    for i in range(n):
        rel = rel_ind[i]
        scores = (pred_link[i,1] * pred_rel[i, rel])
        sub = pred_node_sub[i]
        ob = pred_node_ob[i]
        pred_triplet.append((scores, int(sub), int(rel), int(ob)))
    return gt_triplet, pred_triplet

train_metric = mx.metric.Accuracy()
train_metric_top5 = mx.metric.TopKAccuracy(5)
train_metric_node = mx.metric.Accuracy()
train_metric_node_top5 = mx.metric.TopKAccuracy(5)
train_metric_f1 = mx.metric.F1()
train_metric_auc = AUCMetric()
train_metric_r100 = PredCls(100)

# dataset and dataloader
vg_train = gcv.data.VGRelation(top_frequent_rel=N_relations, top_frequent_obj=N_objects,
                               balancing='weight', split='train')
'''
                               balancing='weight', split='custom',
                               rel_json_path='/home/ubuntu/visualgenome/relationships_small.json')
'''
logger.info('data loaded!')
train_data = gluon.data.DataLoader(vg_train, batch_size=batch_size, shuffle=True, num_workers=8*num_gpus,
                                   batchify_fn=gcv.data.dataloader.dgl_mp_batchify_fn)

'''
detector = get_model('faster_rcnn_resnet50_v1b_custom', classes=vg_train._obj_classes,
                     pretrained_base=False, pretrained=False, additional_output=True)
params_path = 'faster_rcnn_resnet50_v1b_custom_0007_0.2398.params'
'''
detector = get_model('faster_rcnn_resnet101_v1d_custom', classes=vg_train._obj_classes,
                     pretrained_base=False, pretrained=False, additional_output=True)
params_path = 'faster_rcnn_resnet101_v1d_custom_0005_0.2528.params'
detector.load_parameters(params_path, ctx=ctx, ignore_extra=True, allow_missing=True)
for k, v in detector.collect_params().items():
    v.grad_req = 'null'
for k, v in detector.class_predictor.collect_params().items():
    v.grad_req = 'write'

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

def merge_res_iou(g, scores, bbox, feat_ind, spatial_feat, cls_pred, iou_thresh=0.5):
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
    assign_scores = [-1 for i in range(H)]
    while h > 0 and w > 0:
        ind = int(ious.argmax().asscalar())
        row_ind = ind // W
        col_ind = ind % W
        if ious[row_ind, col_ind].asscalar() < iou_thresh:
            break
        assign_ind[row_ind] = col_ind
        assign_scores[row_ind] = ious[row_ind, col_ind].asscalar()
        ious[row_ind, :] = -1
        ious[:, col_ind] = -1
        h -= 1
        w -= 1

    remove_inds = [i for i in range(H) if assign_ind[i] == -1]
    assign_ind = [ind for ind in assign_ind if ind > -1]
    if len(remove_inds) >= g.number_of_nodes() - 1:
        return None
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

for epoch in range(nepoch):
    loss_val = 0
    tic = time.time()
    btic = time.time()
    train_metric.reset()
    train_metric_top5.reset()
    train_metric_node.reset()
    train_metric_node_top5.reset()
    train_metric_f1.reset()
    train_metric_auc.reset()
    train_metric_r100.reset()
    if epoch == 8:
        trainer.set_learning_rate(trainer.learning_rate*0.1)
    n_batches = len(train_data)
    for i, (G_list, img_list) in enumerate(train_data):
        G_list, img_list = get_data_batch(G_list, img_list, ctx)
        if G_list is None or img_list is None:
            continue

        loss = []
        detector_res_list = []
        G_batch = []
        bbox_pad = Pad(axis=(0))
        with mx.autograd.record():
            for G_slice, img in zip(G_list, img_list):
                cur_ctx = img.context
                bbox_list = [G.ndata['bbox'] for G in G_slice]
                bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
                bbox, spatial_feat, cls_pred = detector((img, bbox_stack))
                G_batch.append(merge_res(G_slice, img, bbox, spatial_feat, cls_pred))

            G_batch = [net(G) for G in G_batch]

            for G in G_batch:
                if G is not None and G.number_of_nodes() > 0:
                    # loss_rel = L_rel(G.edata['preds'], G.edata['classes'], G.edata['link'].expand_dims(1))
                    loss_rel = L_rel(G.edata['preds'], G.edata['classes'], G.edata['link'])
                    loss_link = L_link(G.edata['link_preds'], G.edata['link'], G.edata['weights'])
                    loss_cls = L_link(G.ndata['node_class_pred'], G.ndata['node_class_ids'] + 1)
                    loss.append(loss_rel.sum() + loss_link.sum() + loss_cls.sum())

        for l in loss:
            l.backward()
        trainer.step(batch_size)
        loss_val += sum([l.mean().asscalar() for l in loss]) / num_gpus
        for G in G_batch:
            if G is None or G.number_of_nodes() == 0:
                continue
            link_ind = np.where(G.edata['link'].asnumpy() == 1)[0]
            if len(link_ind) == 0:
                continue
            train_metric.update([G.edata['classes'][link_ind]], [G.edata['preds'][link_ind]])
            train_metric_top5.update([G.edata['classes'][link_ind]], [G.edata['preds'][link_ind]])
            train_metric_node.update([G.ndata['node_class_ids'] + 1], [G.ndata['node_class_pred']])
            train_metric_node_top5.update([G.ndata['node_class_ids'] + 1], [G.ndata['node_class_pred']])
            train_metric_f1.update([G.edata['link']], [G.edata['link_preds']])
            train_metric_auc.update([G.edata['link']], [G.edata['link_preds']])
            gt_triplet, pred_triplet = get_triplet_predcls(G)
            train_metric_r100.update(gt_triplet, pred_triplet)
        if (i+1) % batch_verbose_freq == 0:
            _, acc = train_metric.get()
            _, acc_top5 = train_metric_top5.get()
            _, node_acc = train_metric_node.get()
            _, node_acc_top5 = train_metric_node_top5.get()
            _, f1 = train_metric_f1.get()
            _, auc = train_metric_auc.get()
            _, r100 = train_metric_r100.get()
            logger.info('Epoch[%d] Batch [%d/%d] \ttime: %d\tloss=%.4f\tacc=%.4f,acc-top5=%.4f\tnode-acc=%.4f,node-acc-top5=%.4f\tf1=%.4f,auc=%.4f\tpredcls@100=%.4f'%(
                        epoch, i, n_batches, int(time.time() - btic), loss_val / (i+1), acc, acc_top5, node_acc, node_acc_top5, f1, auc, r100))
            btic = time.time()
    _, acc = train_metric.get()
    _, acc_top5 = train_metric_top5.get()
    _, node_acc = train_metric_node.get()
    _, node_acc_top5 = train_metric_node_top5.get()
    _, f1 = train_metric_f1.get()
    _, auc = train_metric_auc.get()
    _, r100 = train_metric_r100.get()
    logger.info('Epoch[%d] \ttime: %d\tloss=%.4f\tacc=%.4f,acc-top5=%.4f\tnode-acc=%.4f,node-acc-top5=%.4f\tf1=%.4f,auc=%.4f\tpredcls@100=%.4f\n'%(
                epoch, int(time.time() - tic), loss_val / (i+1), acc, acc_top5, node_acc, node_acc_top5, f1, auc, r100))
    # detector.save_parameters('%s/detector-%d.params'%(save_dir, epoch))
    net.save_parameters('%s/model-%d.params'%(save_dir, epoch))
    detector.class_predictor.save_parameters('%s/class_predictor-%d.params'%(save_dir, epoch))

