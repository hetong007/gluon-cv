# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# coding: utf-8
# pylint: disable=missing-docstring,unused-argument,arguments-differ

from __future__ import division

__all__ = ['get_simple_pose_mobilenet', 'SimplePoseMobileNet',
           'simple_pose_mobilenet1_0',
           'simple_pose_mobilenet_v2_1_0',
           'simple_pose_mobilenet_v3_large']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import initializer
import gluoncv as gcv

class SimplePoseMobileNet(HybridBlock):

    def __init__(self, base_name='mobilenet1_0', features_ind=81,
                 pretrained_base=False, pretrained_ctx=cpu(),
                 num_joints=17,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1, deconv_with_bias=False, **kwargs):
        super(SimplePoseMobileNet, self).__init__(**kwargs)

        from ..model_zoo import get_model
        base_network = get_model(base_name, pretrained=pretrained_base, ctx=pretrained_ctx,
                                 norm_layer=gcv.nn.BatchNormCudnnOff)

        self.mobilenet = base_network.features[:features_ind]
        self.deconv_with_bias = deconv_with_bias

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        self.final_layer = nn.Conv2D(
            channels=num_joints,
            kernel_size=final_conv_kernel,
            strides=1,
            padding=1 if final_conv_kernel == 3 else 0,
            weight_initializer=initializer.Normal(0.001),
            bias_initializer=initializer.Zero()
        )

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'

        layer = nn.HybridSequential(prefix='')
        with layer.name_scope():
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i])

                planes = num_filters[i]
                layer.add(
                    nn.Conv2DTranspose(
                        channels=planes,
                        kernel_size=kernel,
                        strides=2,
                        padding=padding,
                        output_padding=output_padding,
                        use_bias=self.deconv_with_bias,
                        weight_initializer=initializer.Normal(0.001),
                        bias_initializer=initializer.Zero()))
                layer.add(gcv.nn.BatchNormCudnnOff(gamma_initializer=initializer.One(),
                                                   beta_initializer=initializer.Zero()))
                layer.add(nn.Activation('relu'))
                self.inplanes = planes

        return layer

    def hybrid_forward(self, F, x):
        x = self.mobilenet(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

def get_simple_pose_mobilenet(base_name, feature_ind, pretrained=False, ctx=cpu(),
                              root='~/.mxnet/models', **kwargs):

    net = SimplePoseMobileNet(base_name, feature_ind, **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file('simple_pose_%s'%(base_name),
                                           tag=pretrained, root=root), ctx=ctx)

    return net

def simple_pose_mobilenet1_0(**kwargs):
    r"""MobileNet 1.0 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_mobilenet('mobilenet1.0', 81, **kwargs)

def simple_pose_mobilenet_v2_1_0(**kwargs):
    r"""MobileNet v2 1.0 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_mobilenet('mobilenetv2_1.0', 23, **kwargs)

def simple_pose_mobilenet_v3_large(**kwargs):
    r"""MobileNet v3 large backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_mobilenet('mobilenetv3_large', 21, **kwargs)
