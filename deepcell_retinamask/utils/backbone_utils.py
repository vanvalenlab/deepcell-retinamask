# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-retinamask/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for creating model backbones"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy

import keras_applications as applications
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import is_keras_tensor
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, TimeDistributed
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.utils.layer_utils import get_source_inputs


def get_backbone(backbone, input_tensor=None, input_shape=None,
                 use_imagenet=False, return_dict=True,
                 frames_per_batch=1, **kwargs):
    """Retrieve backbones for the construction of feature pyramid networks.

    Args:
        backbone (str): Name of the backbone to be retrieved.
        input_tensor (tensor): The input tensor for the backbone.
            Should have channel dimension of size 3
        use_imagenet (bool): Load pre-trained weights for the backbone
        return_dict (bool): Whether to return a dictionary of backbone layers,
            e.g. {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}.
            If false, the whole model is returned instead
        kwargs (dict): Keyword dictionary for backbone constructions.
            Relevant keys include 'include_top', 'weights' (should be set to None),
            'input_shape', and 'pooling'

    Returns:
        tensorflow.keras.Model: An instantiated backbone

    Raises:
        ValueError: bad backbone name
        ValueError: featurenet backbone with pre-trained imagenet
    """
    _backbone = str(backbone).lower()

    # set up general Utils class to deal with different tf versions
    class Utils(object):  # pylint: disable=useless-object-inheritance
        pass

    utils = Utils()
    utils.get_file = get_file
    utils.get_source_inputs = get_source_inputs

    K.is_keras_tensor = is_keras_tensor

    kwargs['backend'] = K
    kwargs['layers'] = tf.keras.layers
    kwargs['models'] = tf.keras.models
    kwargs['utils'] = utils

    vgg_backbones = {
        'vgg16': applications.vgg16.VGG16,
        'vgg19': applications.vgg19.VGG19,
    }
    densenet_backbones = {
        'densenet121': applications.densenet.DenseNet121,
        'densenet169': applications.densenet.DenseNet169,
        'densenet201': applications.densenet.DenseNet201,
    }
    mobilenet_backbones = {
        'mobilenet': applications.mobilenet.MobileNet,
        'mobilenetv2': applications.mobilenet_v2.MobileNetV2,
        'mobilenet_v2': applications.mobilenet_v2.MobileNetV2
    }
    resnet_backbones = {
        'resnet50': applications.resnet.ResNet50,
        'resnet101': applications.resnet.ResNet101,
        'resnet152': applications.resnet.ResNet152,
    }
    resnet_v2_backbones = {
        'resnet50v2': applications.resnet_v2.ResNet50V2,
        'resnet101v2': applications.resnet_v2.ResNet101V2,
        'resnet152v2': applications.resnet_v2.ResNet152V2,
    }
    resnext_backbones = {
        'resnext50': applications.resnext.ResNeXt50,
        'resnext101': applications.resnext.ResNeXt101,
    }
    nasnet_backbones = {
        'nasnet_large': applications.nasnet.NASNetLarge,
        'nasnet_mobile': applications.nasnet.NASNetMobile,
    }
    efficientnet_backbones = {
        # 'efficientnetb0': applications.efficientnet.EfficientNetB0,
        # 'efficientnetb1': applications.efficientnet.EfficientNetB1,
        # 'efficientnetb2': applications.efficientnet.EfficientNetB2,
        # 'efficientnetb3': applications.efficientnet.EfficientNetB3,
        # 'efficientnetb4': applications.efficientnet.EfficientNetB4,
        # 'efficientnetb5': applications.efficientnet.EfficientNetB5,
        # 'efficientnetb6': applications.efficientnet.EfficientNetB6,
        # 'efficientnetb7': applications.efficientnet.EfficientNetB7,
    }

    # TODO: Check and make sure **kwargs is in the right format.
    # 'weights' flag should be None, and 'input_shape' must have size 3 on the channel axis
    if frames_per_batch == 1:
        if input_tensor is not None:
            img_input = input_tensor
        else:
            if input_shape:
                img_input = Input(shape=input_shape)
            else:
                img_input = Input(shape=(None, None, 3))
    else:
        # using 3D data but a 2D backbone.
        # TODO: why ignore input_tensor
        if input_shape:
            img_input = Input(shape=input_shape)
        else:
            img_input = Input(shape=(None, None, 3))

    if use_imagenet:
        kwargs_with_weights = copy.copy(kwargs)
        kwargs_with_weights['weights'] = 'imagenet'
    else:
        kwargs['weights'] = None

    if _backbone in vgg_backbones:
        model_cls = vgg_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in densenet_backbones:
        model_cls = densenet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)
        if _backbone == 'densenet121':
            blocks = [6, 12, 24, 16]
        elif _backbone == 'densenet169':
            blocks = [6, 12, 32, 32]
        elif _backbone == 'densenet201':
            blocks = [6, 12, 48, 32]

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['conv1/relu'] + ['conv{}_block{}_concat'.format(idx + 2, block_num)
                                        for idx, block_num in enumerate(blocks)]
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in resnet_backbones:
        model_cls = resnet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        if _backbone == 'resnet50':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block6_out', 'conv5_block3_out']
        elif _backbone == 'resnet101':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block23_out', 'conv5_block3_out']
        elif _backbone == 'resnet152':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block8_out',
                           'conv4_block36_out', 'conv5_block3_out']

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in resnet_v2_backbones:
        model_cls = resnet_v2_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        if _backbone == 'resnet50v2':
            layer_names = ['post_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block6_out', 'conv5_block3_out']
        elif _backbone == 'resnet101v2':
            layer_names = ['post_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block23_out', 'conv5_block3_out']
        elif _backbone == 'resnet152v2':
            layer_names = ['post_relu', 'conv2_block3_out', 'conv3_block8_out',
                           'conv4_block36_out', 'conv5_block3_out']

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in resnext_backbones:
        model_cls = resnext_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        if _backbone == 'resnext50':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block6_out', 'conv5_block3_out']
        elif _backbone == 'resnext101':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block23_out', 'conv5_block3_out']

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in mobilenet_backbones:
        model_cls = mobilenet_backbones[_backbone]
        alpha = kwargs.pop('alpha', 1.0)
        model = model_cls(alpha=alpha, input_tensor=img_input, **kwargs)
        if _backbone.endswith('v2'):
            block_ids = (2, 5, 12)
            layer_names = ['expanded_conv_project_BN'] + \
                          ['block_%s_add' % i for i in block_ids] + \
                          ['block_16_project_BN']
        else:
            block_ids = (1, 3, 5, 11, 13)
            layer_names = ['conv_pw_%s_relu' % i for i in block_ids]

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(alpha=alpha, **kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in nasnet_backbones:
        model_cls = nasnet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)
        if _backbone.endswith('large'):
            block_ids = [5, 12, 18]
        else:
            block_ids = [3, 8, 12]

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['stem_bn1', 'reduction_concat_stem_1']
        layer_names.extend(['normal_concat_%s' % i for i in block_ids])
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in efficientnet_backbones:
        model_cls = efficientnet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['block2a_expand_activation', 'block3a_expand_activation',
                       'block4a_expand_activation', 'block6a_expand_activation',
                       'top_activation']
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    else:
        join = lambda x: [v for y in x for v in list(y.keys())]
        backbones = join([densenet_backbones, resnet_backbones,
                          resnext_backbones, resnet_v2_backbones,
                          vgg_backbones, nasnet_backbones,
                          mobilenet_backbones, efficientnet_backbones])
        raise ValueError('Invalid value for `backbone`. Must be one of: %s' %
                         ', '.join(backbones))

    if frames_per_batch > 1:

        time_distributed_outputs = []
        for i, out in enumerate(layer_outputs):
            time_distributed_outputs.append(
                TimeDistributed(Model(model.input, out))(input_tensor))

        if time_distributed_outputs:
            layer_outputs = time_distributed_outputs

    output_dict = {'C{}'.format(i + 1): j for i, j in enumerate(layer_outputs)}
    return (model, output_dict) if return_dict else model