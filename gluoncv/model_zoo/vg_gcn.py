import dgl
import gluoncv as gcv
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn
from dgl.nn.mxnet import GraphConv

from gluoncv.model_zoo import get_model

class EdgeMLP(nn.Block):
    def __init__(self, n_classes):
        super(EdgeMLP, self).__init__()
        self.mlp = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['emb'], edges.dst['emb'])
        out = self.mlp(feat)
        return {'rel': out}

class EdgeGCN(nn.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 global_feat_ext,
                 box_feat_ext):
        super(EdgeGCN, self).__init__()
        self.layers = nn.Sequential()
        # input layer
        self.layers.add(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.edge_mlp = EdgeMLP(n_classes)
        self.global_feat_ext = get_model(global_feat_ext)
        self.box_feat_ext = get_model(box_feat_ext)

    def forward(self, g, img):
        h = self.box_feat_ext(g.ndata['images'])
        gh = self.global_feat_ext(img)
        x = nd.concat(h, gh)
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
        g.ndata['emb'] = x
        g.apply_edges(self.edge_mlp)
        return g

