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
"""Tests for retinanet_anchor_utils"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test

from deepcell_retinamask.utils import anchor_utils


class TestRetinaNetAnchorUtils(test.TestCase):

    def test_anchors_for_shape_dimensions(self):
        sizes = [32, 64, 128]
        strides = [8, 16, 32]
        ratios = np.array([0.5, 1, 2, 3], K.floatx())
        scales = np.array([1, 1.2, 1.6], K.floatx())
        anchor_params = anchor_utils.AnchorParameters(
            sizes, strides, ratios, scales)

        pyramid_levels = [3, 4, 5]
        image_shape = tensor_shape.TensorShape((64, 64))
        all_anchors = anchor_utils.anchors_for_shape(
            image_shape,
            pyramid_levels=pyramid_levels,
            anchor_params=anchor_params)

        self.assertTupleEqual(all_anchors.shape, (1008, 4))
        self.assertEqual(anchor_params.num_anchors(), 12)

    def test_anchors_for_shape_values(self):
        sizes = [12]
        strides = [8]
        ratios = np.array([1, 2], K.floatx())
        scales = np.array([1, 2], K.floatx())
        anchor_params = anchor_utils.AnchorParameters(
            sizes, strides, ratios, scales)

        pyramid_levels = [3]
        image_shape = (16, 16)
        all_anchors = anchor_utils.anchors_for_shape(
            image_shape,
            pyramid_levels=pyramid_levels,
            anchor_params=anchor_params)

        # using almost_equal for floating point imprecisions
        self.assertAllClose(all_anchors[0, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[1, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[2, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[3, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[4, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[5, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[6, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[7, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[8, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[9, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[10, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[11, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[12, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[13, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[14, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[15, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])

    def test_anchor_targets_bbox(self):
        # TODO: test correct-ness
        sizes = [12]
        strides = [8]
        ratios = np.array([1, 2], K.floatx())
        scales = np.array([1, 2], K.floatx())
        anchor_params = anchor_utils.AnchorParameters(
            sizes, strides, ratios, scales)

        pyramid_levels = [3]
        image_shape = (16, 16)
        anchors = anchor_utils.anchors_for_shape(
            image_shape,
            pyramid_levels=pyramid_levels,
            anchor_params=anchor_params)

        # test image / annotation size mismatch
        with self.assertRaises(ValueError):
            anchor_utils.anchor_targets_bbox(anchors, [1], [1, 2, 3], 1)
        # test image / annotation not empty
        with self.assertRaises(ValueError):
            anchor_utils.anchor_targets_bbox(anchors, [], [], 1)
        # test annotation structure
        with self.assertRaises(ValueError):
            anchor_utils.anchor_targets_bbox(anchors, [1], [{'labels': 1}], 1)
        with self.assertRaises(ValueError):
            anchor_utils.anchor_targets_bbox(anchors, [1], [{'bboxes': 1}], 1)

    def test_bbox_transform(self):
        # TODO: test correct-ness
        sizes = [12]
        strides = [8]
        ratios = np.array([1, 2], K.floatx())
        scales = np.array([1, 2], K.floatx())
        anchor_params = anchor_utils.AnchorParameters(
            sizes, strides, ratios, scales)

        pyramid_levels = [3]
        image_shape = (16, 16)
        anchors = anchor_utils.anchors_for_shape(
            image_shape,
            pyramid_levels=pyramid_levels,
            anchor_params=anchor_params)

        # test custom std/mean
        targets = anchor_utils.bbox_transform(
            anchors, np.random.random((1, 4)), mean=[0], std=[0.2])

        self.assertTupleEqual(targets.shape, (16, 4))

        # test bad `mean` value
        with self.assertRaises(ValueError):
            anchor_utils.bbox_transform(anchors, [1], mean='invalid', std=None)
        # test image / annotation not empty
        with self.assertRaises(ValueError):
            anchor_utils.bbox_transform(anchors, [1], mean=None, std='invalid')

    def test_layer_shapes(self):
        image_shape = (16, 16, 1)
        model = Sequential()

        model.add(Conv2D(3, (3, 3), input_shape=image_shape, name='P1'))
        model.add(Conv2D(3, (3, 3), name='P2'))
        shapes = anchor_utils.layer_shapes(image_shape, model)
        self.assertIsInstance(shapes, dict)
        self.assertTrue('P1' in shapes)
        self.assertEqual(shapes['P1'], tuple([None] + list(image_shape)))
        self.assertEqual(shapes['P2'], (None, 14, 14, 3))

        # test TensorShape compatibility
        image_shape = tensor_shape.TensorShape(image_shape)
        shapes = anchor_utils.layer_shapes(image_shape, model)
        self.assertIsInstance(shapes, dict)
        self.assertTrue('P1' in shapes)
        self.assertEqual(shapes['P1'], tuple([None] + list(image_shape)))
        self.assertEqual(shapes['P2'], (None, 14, 14, 3))

    def test_make_shapes_callback(self):
        image_shape = (16, 16, 1)
        level = np.random.randint(1, 3)
        pyramid_levels = [level]
        name = 'P{}'.format(level)
        model = Sequential()

        model.add(Conv2D(3, (3, 3), input_shape=image_shape, name=name))
        cb = anchor_utils.make_shapes_callback(model)
        shape = cb(image_shape, pyramid_levels)
        # TODO: should the output be a TensorShape?
        self.assertEqual(shape, [tensor_shape.TensorShape((16, 16))])

    def test_get_sorted_keys(self):
        d = {'C1': 1, 'C3': 2, 'C2': 3}
        self.assertListEqual(
            anchor_utils.get_sorted_keys(d),
            ['C1', 'C2', 'C3']
        )

if __name__ == '__main__':
    test.main()
