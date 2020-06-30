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
"""Test the RetinaMask models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import keras_parameterized

from deepcell_retinamask.model_zoo import RetinaMask


class RetinaMaskTest(keras_parameterized.TestCase):

    # @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters([
        {
            'testcase_name': 'retinamask_basic',
            'pooling': None,
            'panoptic': False,
            'nms': True,
            'class_specific_filter': True,
            'frames': 1,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinamask_basic_td',
            'pooling': None,
            'panoptic': False,
            'nms': True,
            'class_specific_filter': True,
            'frames': 32,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinamask_avgnorm',
            'pooling': 'avg',
            'panoptic': False,
            'nms': False,
            'class_specific_filter': False,
            'frames': 1,
            'pyramid_levels': ['P3', 'P4', 'P5'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinamask_panoptic_maxnorm',
            'pooling': 'max',
            'panoptic': True,
            'nms': True,
            'class_specific_filter': True,
            'frames': 1,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinamask_panoptic_maxnorm_td',
            'pooling': 'max',
            'panoptic': True,
            'nms': True,
            'class_specific_filter': True,
            'frames': 32,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinamask_basic_cf',
            'pooling': None,
            'panoptic': False,
            'nms': True,
            'class_specific_filter': True,
            'frames': 1,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinamask_basic_td_cf',
            'pooling': None,
            'panoptic': False,
            'nms': True,
            'class_specific_filter': True,
            'frames': 32,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinamask_avgnorm_cf',
            'pooling': 'avg',
            'panoptic': False,
            'nms': False,
            'class_specific_filter': False,
            'frames': 1,
            'pyramid_levels': ['P3', 'P4', 'P5'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinamask_panoptic_maxnorm_cf',
            'pooling': 'max',
            'panoptic': True,
            'nms': True,
            'class_specific_filter': True,
            'frames': 1,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinamask_panoptic_maxnorm_td_cf',
            'pooling': 'max',
            'panoptic': True,
            'nms': True,
            'class_specific_filter': True,
            'frames': 32,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_first',
        }
    ])
    def test_retinamask(self, pooling, panoptic, frames, pyramid_levels,
                        nms, class_specific_filter, data_format):
        num_classes = 3
        crop_size = (14, 14)
        mask_size = (28, 28)

        max_detections = 10
        norm_method = None

        # not all backbones work with channels_first
        backbone = 'mobilenetv2'

        # TODO: RetinaMask fails with channels_first and frames_per_batch > 1
        if frames > 1 and data_format == 'channels_first':
            return

        with self.cached_session():
            K.set_image_data_format(data_format)
            if data_format == 'channels_first':
                axis = 1
                input_shape = (1, 32, 32)
            else:
                axis = -1
                input_shape = (32, 32, 1)

            num_semantic_classes = [3, 4]

            model = RetinaMask(
                backbone=backbone,
                num_classes=num_classes,
                input_shape=input_shape,
                norm_method=norm_method,
                pooling=pooling,
                nms=nms,
                class_specific_filter=class_specific_filter,
                frames_per_batch=frames,
                panoptic=panoptic,
                crop_size=crop_size,
                mask_size=mask_size,
                max_detections=max_detections,
                num_semantic_heads=len(num_semantic_classes),
                num_semantic_classes=num_semantic_classes,
                backbone_levels=['C3', 'C4', 'C5'],
                pyramid_levels=pyramid_levels,
            )

            # TODO: What are the extra 2 for panoptic models?
            expected_size = 7 + panoptic * (len(num_semantic_classes) + 2)

            # TODO: What are these new outputs?
            if frames > 1:
                expected_size += 2

            self.assertIsInstance(model.output_shape, list)
            self.assertEqual(len(model.output_shape), expected_size)

            self.assertEqual(model.output_shape[0][-1], 4)
            self.assertEqual(model.output_shape[1][-1], num_classes)

            delta = (frames > 1)  # TODO: New output?
            self.assertEqual(model.output_shape[3 + delta][-1], 4)
            self.assertEqual(model.output_shape[4 + delta][-1], max_detections)
            self.assertEqual(model.output_shape[5 + delta][-1], max_detections)
            # max_detections is in axis == 1
            _axis = axis + int(K.image_data_format() == 'channels_first')
            self.assertEqual(model.output_shape[6 + delta][_axis], num_classes)

            if panoptic:
                for i, n in enumerate(num_semantic_classes):
                    index = expected_size - len(num_semantic_classes) + i
                    self.assertEqual(model.output_shape[index][axis], n)