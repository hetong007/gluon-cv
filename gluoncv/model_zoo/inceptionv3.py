# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,missing-docstring
"""ResNext, implemented in Gluon."""
from __future__ import division

__all__ = ['InceptionV3', 'inceptionv3']

import os
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

class BasicConv2D(HybridBlock):
    r"""BasicConv2D
    """
    def __init__(self, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2D, self).__init__()
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(out_planes, kernel_size=kernel_size, strides=stride,
                                padding=padding, use_bias=False))
        self.body.add(nn.BatchNorm(momentum=0.1, eposilon=0.001))
        self.body.relu(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return self.body(x)

class InceptionA(HybridBlock):
    def __init__(self, pool_features):
        super(InceptionA, self).__init__()

        self.branch1x1 = BasicConv2D(64, 1, 1)

        self.branch5x5 = nn.HybridSequential(prefix='')
        self.branch5x5.add(BasicConv2D(48, 1, 1))
        self.branch5x5.add(BasicConv2D(64, 5, 1, 2))

        self.branch3x3_dbl = nn.HybridSequential(prefix='')
        self.branch3x3_dbl.add(BasicConv2D(64, 1, 1))
        self.branch3x3_dbl.add(BasicConv2D(96, 3, 1, 1))
        self.branch3x3_dbl.add(BasicConv2D(96, 3, 1, 1))

        self.branch_pool = nn.HybridSequential(prefix='')
        self.branch_pool.add(nn.AvgPool2D(3, 1, 1))
        self.branch_pool.add(BasicConv2D(pool_features, 1, 1))

    def hybrid_forward(self, F, x):
        x_1x1 = self.branch1x1(x)
        x_5x5 = self.branch5x5(x)
        x_3x3_dbl = self.branch3x3_dbl(x)
        x_pool = self.branch_pool(x)

        x_out = F.concat(x_1x1, x_5x5, x_3x3_dbl, x_pool, dim=1)

        return x_out

class InceptionB(HybridBlock):
    def __init__(self, scale=1.0):
        super(InceptionB, self).__init__()

        self.branch3x3 = BasicConv2D(384, 3, 2)

        self.branch3x3_dbl = nn.HybridSequential(prefix='')
        self.branch3x3_dbl.add(BasicConv2D(64, 1, 1))
        self.branch3x3_dbl.add(BasicConv2D(96, 3, 1, 1))
        self.branch3x3_dbl.add(BasicConv2D(96, 3, 2))

        self.branch_pool = nn.MaxPool2D(3, 2)

    def hybrid_forward(self, F, x):
        x_3x3 = self.branch3x3(x)
        x_3x3_dbl = self.branch3x3_dbl(x)
        x_pool = self.branch_pool(x)

        x_out = F.concat(x_3x3, x_3x3_dbl, x_pool, dim=1)

        return x_out

class InceptionC(HybridBlock):
    def __init__(self, channels_7x7):
        super(InceptionC, self).__init__()

        self.branch1x1 = BasicConv2D(192, 1, 1)

        self.branch7x7 = nn.HybridSequential(prefix='')
        self.branch7x7.add(BasicConv2D(channels_7x7, 1, 1))
        self.branch7x7.add(BasicConv2D(channels_7x7, (1, 7), 1, (0, 3)))
        self.branch7x7.add(BasicConv2D(192, (7, 1), 1, (3, 0)))

        self.branch7x7_dbl = nn.HybridSequential(prefix='')
        self.branch7x7_dbl.add(BasicConv2D(channels_7x7, 1, 1))
        self.branch7x7_dbl.add(BasicConv2D(channels_7x7, (7, 1), 1, (3, 0)))
        self.branch7x7_dbl.add(BasicConv2D(channels_7x7, (1, 7), 1, (0, 3)))
        self.branch7x7_dbl.add(BasicConv2D(channels_7x7, (7, 1), 1, (3, 0)))
        self.branch7x7_dbl.add(BasicConv2D(192, (1, 7), 1, (0, 3)))

        self.branch_pool = nn.HybridSequential(prefix='')
        self.branch_pool.add(AvgPool2D(3, 1, 1))
        self.branch_pool.add(BasicConv2D(192, 1, 1))

    def hybrid_forward(self, F, x):
        x_1x1 = self.branch1x1(x)
        x_7x7 = self.branch7x7(x)
        x_7x7_dbl = self.branch7x7_dbl(x)
        x_pool = self.branch_pool(x)

        x_out = F.concat(x_1x1, x_7x7, x_7x7_dbl, x_pool, dim=1)

        return x_out

class InceptionD(HybridBlock):
    def __init__(self):
        super(InceptionD, self).__init__()

        self.branch3x3 = nn.HybridSequential(prefix='')
        self.branch3x3.add(BasicConv2D(192, 1, 1))
        self.branch3x3.add(BasicConv2D(320, 3, 2))

        self.branch7x7x3 = nn.HybridSequential(prefix='')
        self.branch7x7x3.add(BasicConv2D(192, 1, 1))
        self.branch7x7x3.add(BasicConv2D(192, (1, 7), 1, (0, 3)))
        self.branch7x7x3.add(BasicConv2D(192, (7, 1), 1, (3, 0)))
        self.branch7x7x3.add(BasicConv2D(192, 3, 2))

        self.branch_pool = MaxPool2D(3, 2)

    def hybrid_forward(self, F, x):
        x_3x3 = self.branch3x3(x)
        x_7x7x3 = self.branch7x7x3(x)
        x_pool = self.branch_pool(x)

        x_out = F.concat(x_3x3, x_7x7x3, x_pool, dim=1)

        return x_out

class InceptionE(HybridBlock):
    def __init__(self):
        super(InceptionE, self).__init__()

        self.branch1x1 = BasicConv2D(320, 1, 1)

        self.branch3x3 = nn.HybridSequential(prefix='')
        self.branch3x3.add(BasicConv2D(384, 1, 1))
        self.branch3x3_a = BasicConv2D(384, (1, 3), 1, (0, 1))
        self.branch3x3_b = BasicConv2D(384, (3, 1), 1, (1, 0))

        self.branch3x3_dbl = nn.HybridSequential(prefix='')
        self.branch3x3_dbl.add(BasicConv2D(448, 1, 1))
        self.branch3x3_dbl.add(BasicConv2D(384, 3, 1, 1))
        self.branch3x3_dbl_a = BasicConv2D(384, (1, 3), 1, (0, 1))
        self.branch3x3_dbl_b = BasicConv2D(384, (3, 1), 1, (1, 0))

        self.branch_pool = nn.HybridSequential(prefix='')
        self.branch_pool.add(AvgPool2D(3, 1, 1))
        self.branch_pool.add(BasicConv2D(192, 1, 1))

    def hybrid_forward(self, F, x):
        x_1x1 = self.branch1x1(x)

        x_3x3 = self.branch3x3
        x_3x3_a = self.branch3x3_a(x_3x3)
        x_3x3_b = self.branch3x3_b(x_3x3)

        x_3x3_dbl = self.branch3x3_dbl
        x_3x3_dbl_a = self.branch3x3_dbl_a(x_3x3_dbl)
        x_3x3_dbl_b = self.branch3x3_dbl_b(x_3x3_dbl)

        x_pool = self.branch_pool(x)

        x_out = F.concat(x_1x1, x_3x3_a, x_3x3_b, x_3x3_dbl_a, x_3x3_dbl_b, x_pool, dim=1)

        return x_out

class InceptionAux(HybridBlock):
    def __init__(self, classes):
        super(InceptionAux, self).__init__()
        self.out_aux = nn.HybridSequential(prefix='')
        self.out_aux.add(BasicConv2D(128, 1, 1))
        self.out_aux.add(BasicConv2D(768, 5, 1))
        self.out_aux.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x = F.contrib.AdaptiveAvgPooling2D(x, output_size=5)
        x_out = self.out_aux(x)

        return x_out

class InceptionV3(HybridBlock):

    def __init__(self, classes=1000, use_aux=True, **kwargs):
        super(InceptionV3, self).__init__(**kwargs)

        self.use_aux = use_aux

        self.features_1 = nn.HybridSequential(prefix='')
        self.features_1.add(BasicConv2D(32, 3, 2))
        self.features_1.add(BasicConv2D(32, 3, 1))
        self.features_1.add(BasicConv2D(64, 3, 1, 1))
        self.features_1.add(MaxPool2D(3, 2))
        self.features_1.add(BasicConv2D(80, 1, 1))
        self.features_1.add(BasicConv2D(192, 3, 1))
        self.features_1.add(MaxPool2D(3, 2))

        self.features_1.add(InceptionA(32))
        self.features_1.add(InceptionA(64))
        self.features_1.add(InceptionA(64))
        self.features_1.add(InceptionB())
        self.features_1.add(InceptionC(128))
        self.features_1.add(InceptionC(160))
        self.features_1.add(InceptionC(160))
        self.features_1.add(InceptionC(192))
        
        if use_aux:
            self.out_aux = InceptionAux(classes)
        else:
            self.out_aux = None

        self.features_2 = nn.HybridSequential(prefix='')
        self.features_2.add(InceptionD())
        self.features_2.add(InceptionE())
        self.features_2.add(InceptionE())

        self.out = nn.HybridSequential(prefix='')
        self.out.add(nn.GlobalAvgPool2D())
        self.out.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x = self.features_1(x)
        if self.use_aux:
            x_aux = self.out_aux(x)
        x = self.features_2(x)
        x = self.out(x)
        if self.use_aux:
            return x, x_aux
        else:
            return x

def inceptionv3(pretrained=False, ctx=cpu(),
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""
    """

    net = InceptionV3(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('inceptionv3', root=root), ctx=ctx)
    return net

