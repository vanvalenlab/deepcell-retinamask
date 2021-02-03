# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
"""Upsampling layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils


class Upsample(Layer):
    """Upsample layer adapted from https://github.com/fizyr/keras-maskrcnn.

    Args:
        target_size (tuple): 2D tuple for target size ``(x, y)``.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """

    def __init__(self, target_size, data_format=None, *args, **kwargs):
        self.target_size = target_size
        self.data_format = conv_utils.normalize_data_format(data_format)
        super(Upsample, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        new_shape = (self.target_size[0], self.target_size[1])
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, (0, 2, 3, 1))
        outputs = tf.image.resize(
            inputs, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, (0, 3, 1, 2))
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            output_shape = (
                input_shape[0],
                input_shape[1],
                self.target_size[0],
                self.target_size[1])
        else:
            output_shape = (
                input_shape[0],
                self.target_size[0],
                self.target_size[1],
                input_shape[-1])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            'target_size': self.target_size,
            'data_format': self.data_format
        }
        base_config = super(Upsample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
