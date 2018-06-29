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

__all__ = ['InceptionResnetV2', 'inceptionresnetv2']

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

class Mixed_5b(HybridBlock):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2D(96, 1, 1)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(48, 1, 1))
        self.branch1.add(BasicConv2D(64, 5, 1, 2))

        self.branch2 = nn.HybridSequential(prefix='')
        self.branch2.add(BasicConv2D(64, 1, 1))
        self.branch2.add(BasicConv2D(96, 3, 1, 1))
        self.branch2.add(BasicConv2D(96, 3, 1, 1))

        self.branch3 = nn.HybridSequential(prefix='')
        self.branch3.add(nn.AvgPool2D(3, 1, 1, count_include_pad=False))
        self.branch3.add(BasicConv2D(64, 1, 1))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_out = F.concat(x0, x1, x2, x3, dim=1)

        return x_out

class Block35(HybridBlock):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2D(32, 1, 1)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(32, 1, 1))
        self.branch1.add(BasicConv2D(32, 3, 1, 1))

        self.branch2 = nn.HybridSequential(prefix='')
        self.branch2.add(BasicConv2D(32, 1, 1))
        self.branch2.add(BasicConv2D(48, 3, 1, 1))
        self.branch2.add(BasicConv2D(64, 3, 1, 1))

        self.conv2d = nn.Conv2D(320, 1, 1)
        self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_out = F.concat(x0, x1, x2, dim=1)
        x_out = self.conv2d(x_out)

        x_out = F.Activation(x_out*self.scale + x, act_type='relu')
        return x_out

class Mixed_6a(HybridBlock):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2D(384, 3, 2)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(256, 1, 1))
        self.branch1.add(BasicConv2D(256, 3, 1, 1))
        self.branch1.add(BasicConv2D(384, 3, 2))

        self.branch2 = nn.MaxPool2D(3, 2))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_out = F.concat(x0, x1, x2, dim=1)

        return x_out

class Block17(HybridBlock):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2D(192, 1, 1)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(128, 1, 1))
        self.branch1.add(BasicConv2D(160, (1, 7), 1, (0, 3)))
        self.branch1.add(BasicConv2D(192, (7, 1), 1, (3, 0)))

        self.conv2d = nn.Conv2D(1088, 1, 1)
        self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        x_out = F.concat(x0, x1, dim=1)
        x_out = self.conv2d(x_out)

        x_out = F.Activation(x_out*self.scale + x, act_type='relu')
        return x_out

class Mixed_7a(HybridBlock):
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.HybridSequential(prefix='')
        self.branch0.add(BasicConv2D(256, 1, 1))
        self.branch0.add(BasicConv2D(384, 3, 2))

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(256, 1, 1))
        self.branch1.add(BasicConv2D(288, 3, 2))

        self.branch2 = nn.HybridSequential(prefix='')
        self.branch2.add(BasicConv2D(256, 1, 1))
        self.branch2.add(BasicConv2D(288, 3, 1, 1))
        self.branch2.add(BasicConv2D(320, 3, 2))

        self.branch3 = nn.MaxPool2D(3, 2))

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_out = F.concat(x0, x1, x2, x3, dim=1)

        return x_out

class Block8(HybridBlock):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2D(192, 1, 1)

        self.branch1 = nn.HybridSequential(prefix='')
        self.branch1.add(BasicConv2D(192, 1, 1))
        self.branch1.add(BasicConv2D(224, (1, 3), 1, (0, 1)))
        self.branch1.add(BasicConv2D(256, (3, 1), 1, (1, 0)))

        self.conv2d = nn.Conv2D(2080, 1, 1)
        if not self.noReLU:
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        x_out = F.concat(x0, x1, dim=1)
        x_out = self.conv2d(x_out)

        if not self.noReLU:
            x_out = F.Activation(x_out*self.scale + x, act_type='relu')
        else:
            x_out = x_out*self.scale + x
        return x_out

class InceptionResNetV2(HybridBlock):

    def __init__(self, classes=1000, **kwargs):
        super(InceptionResNetV2, self).__init__(**kwargs)

        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(32, 3, 2)
        self.conv2d_2a = BasicConv2d(32, 3, 1)
        self.conv2d_2b = BasicConv2d(64, 3, 1, 1)
        self.maxpool_3a = nn.MaxPool2d(3, 2)
        self.conv2d_3b = BasicConv2d(80, 1, 1)
        self.conv2d_4a = BasicConv2d(192, 3, 1)
        self.maxpool_5a = nn.MaxPool2d(3, 2)
        self.mixed_5b = Mixed_5b()

        self.repeat = nn.HybridSequential(prefix='')
        for i in range(10):
            self.repeat.add(Block35(0.17))

        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.HybridSequential(prefix='')
        for i in range(20):
            self.repeat_1.add(Block17(0.10))

        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.HybridSequential(prefix='')
        for i in range(9):
            self.repeat_2.add(Block8(0.20))

        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(1536, 1, 1)
        self.avgpool_1a = nn.GlobalAvgPool2D()
        self.last_linear = nn.Dense(classes)

    def features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x) 
        return x

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def inceptionresnetv2(pretrained=False, ctx=cpu(),
                          root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""
    """

    net = InceptionResNetV2(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('inceptionresnetv2', root=root), ctx=ctx)
    return net

