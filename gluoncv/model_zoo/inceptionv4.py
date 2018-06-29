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

__all__ = ['InceptionV4', 'inceptionv4']

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

class Mixed_3a(HybridBlock):
    def __init__(self):
        super(Mixed_3a, self).__init__()

        self.maxpool = nn.MaxPool2D(3, 2))
        self.conv = BasicConv2D(96, 3, 2))

    def hybrid_forward(self, F, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)

        x_out = F.concat(x0, x1, dim=1)

        return x_out

class Mixed_4a(HybridBlock):
    def __init__(self):
        super(Mixed_4a, self).__init__()
        
        self.branch0 = nn.HybridSequential(prefix='')
        self.branch0.add(BasicConv2D(64, 1, 1))
        self.branch0.add(BasicConv2D(96, 3, 1))

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(64, 1, 1))
        self.branch1.add(BasicConv2D(64, (1, 7), 1, (0, 3)))
        self.branch1.add(BasicConv2D(64, (7, 1), 1, (3, 0)))
        self.branch1.add(BasicConv2D(96, 3, 1))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        x_out = F.concat(x0, x1, dim=1)

        return x_out

class Mixed_5a(HybridBlock):
    def __init__(self):
        super(Mixed_5a, self).__init__()

        self.conv = BasicConv2D(192, 3, 2))
        self.maxpool = nn.MaxPool2D(3, 2))

    def hybrid_forward(self, F, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)

        x_out = F.concat(x0, x1, dim=1)

        return x_out

class InceptionA(HybridBlock):
    def __init__(self, scale=1.0):
        super(InceptionA, self).__init__()

        self.branch0 = BasicConv2D(96, 1, 1)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(64, 1, 1))
        self.branch1.add(BasicConv2D(96, 3, 1, 1))

        self.branch2 = nn.HybridSequential(prefix='')
        self.branch2.add(BasicConv2D(64, 1, 1))
        self.branch2.add(BasicConv2D(96, 3, 1, 1))
        self.branch2.add(BasicConv2D(96, 3, 1, 1))

        self.branch3 = nn.HybridSequential(prefix='')
        self.branch3.add(AvgPool2D(3, 1, 1, count_include_pad=False))
        self.branch3.add(BasicConv2D(96, 1, 1))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_out = F.concat(x0, x1, x2, x3, dim=1)

        return x_out

class ReductionA(HybridBlock):
    def __init__(self):
        super(ReductionA, self).__init__()

        self.branch0 = BasicConv2D(384, 3, 2)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(192, 1, 1))
        self.branch1.add(BasicConv2D(224, 3, 1, 1))
        self.branch1.add(BasicConv2D(256, 3, 2))

        self.branch2 = nn.MaxPool2D(3, 2))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_out = F.concat(x0, x1, x2, dim=1)

        return x_out

class InceptionB(HybridBlock):
    def __init__(self, scale=1.0):
        super(InceptionB, self).__init__()

        self.branch0 = BasicConv2D(384, 1, 1)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(192, 1, 1))
        self.branch1.add(BasicConv2D(224, (1, 7), 1, (0, 3)))
        self.branch1.add(BasicConv2D(256, (7, 1), 1, (3, 0)))

        self.branch2 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(192, 1, 1))
        self.branch1.add(BasicConv2D(192, (7, 1), 1, (3, 0)))
        self.branch1.add(BasicConv2D(224, (1, 7), 1, (0, 3)))
        self.branch1.add(BasicConv2D(224, (7, 1), 1, (3, 0)))
        self.branch1.add(BasicConv2D(256, (1, 7), 1, (0, 3)))

        self.branch3 = nn.HybridSequential(prefix='')
        self.branch3.add(AvgPool2D(3, 1, 1, count_include_pad=False))
        self.branch3.add(BasicConv2D(128, 1, 1))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_out = F.concat(x0, x1, x2, x3, dim=1)

        return x_out


class ReductionB(HybridBlock):
    def __init__(self):
        super(ReductionB, self).__init__()

        self.branch0 = nn.HybridSequential(prefix='')
        self.branch0.add(BasicConv2D(192, 1, 1))
        self.branch0.add(BasicConv2D(192, 3, 2))

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(256, 1, 1))
        self.branch1.add(BasicConv2D(256, (1, 7), 1, (0, 3)))
        self.branch1.add(BasicConv2D(320, (7, 1), 1, (3, 0)))
        self.branch1.add(BasicConv2D(320, 3, 2))

        self.branch2 = nn.MaxPool2D(3, 2))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_out = F.concat(x0, x1, x2, dim=1)

        return x_out

class InceptionC(HybridBlock):
    def __init__(self, scale=1.0):
        super(InceptionC, self).__init__()

        self.branch0 = BasicConv2D(256, 1, 1)

        self.branch1_0 = BasicConv2D(384, 1, 1)
        self.branch1_1a = BasicConv2D(256, (1, 3), 1, (0, 1))
        self.branch1_1b = BasicConv2D(256, (3, 1), 1, (1, 0))

        self.branch2_0 = nn.HybridSequential(prefix='')
        self.branch2_0.add(BasicConv2D(384, 1, 1)
        self.branch2_0.add(BasicConv2D(448, (3, 1), 1, (1, 0)))
        self.branch2_0.add(BasicConv2D(512, (1, 3), 1, (0, 1)))
        self.branch2_1a = BasicConv2D(256, (1, 3), 1, (0, 1))
        self.branch2_1b = BasicConv2D(256, (3, 1), 1, (1, 0))

        self.branch3 = nn.HybridSequential(prefix='')
        self.branch3.add(AvgPool2D(3, 1, 1, count_include_pad=False))
        self.branch3.add(BasicConv2D(256, 1, 1))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = F.concat(x1_1a, x1_1b, dim=1)

        x2_0 = self.branch2_0(x)
        x2_1a = self.branch2_1a(x1_0)
        x2_1b = self.branch2_1b(x1_0)
        x2 = F.concat(x2_1a, x2_1b, dim=1)

        x3 = self.branch3(x)

        x_out = F.concat(x0, x1, x2, x3, dim=1)

        return x_out

class InceptionV4(HybridBlock):

    def __init__(self, classes=1000, **kwargs):
        super(InceptionV4, self).__init__(**kwargs)

        self.features = nn.HybridSequential(prefix='')
        self.features.add(BasicConv2D(32, 3, 2))
        self.features.add(BasicConv2D(32, 3, 1))
        self.features.add(BasicConv2D(64, 3, 1, 1))

        self.features.add(Mixed_3a())
        self.features.add(Mixed_4a())
        self.features.add(Mixed_5a())

        for i in range(4):
            self.features.add(InceptionA())
        self.features.add(ReductionA())

        for i in range(7):
            self.features.add(InceptionB())
        self.features.add(ReductionB())

        for i in range(7):
            self.features.add(InceptionC())

        self.out = nn.HybridSequential(prefix='')
        self.out.add(nn.GlobalAvgPool2D())
        self.out.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.out(x)
        return x

def inceptionv4(pretrained=False, ctx=cpu(),
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""
    """

    net = InceptionV4(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('inceptionv4', root=root), ctx=ctx)
    return net

