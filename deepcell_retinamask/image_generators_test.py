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
"""Tests for RetinaMask data generators"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.platform import test

from deepcell_retinamask import image_generators


def _generate_test_images(img_w=21, img_h=21):
    rgb_images = []
    gray_images = []
    for _ in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = array_to_img(imarray, scale=False)
        rgb_images.append(im)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = array_to_img(imarray, scale=False)
        gray_images.append(im)

    return [rgb_images, gray_images]


class TestRetinaNetDataGenerator(test.TestCase):

    def test_retinanet_data_generator(self):
        for test_images in _generate_test_images(21, 21):
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.RetinaNetGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            num_classes = np.random.randint(1, 3)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 10, 10, 3)),
                'y': np.random.random((8, 10, 10, 1)),
            }
            generator.flow(train_dict, num_classes=num_classes)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            for x, y in generator.flow(
                    train_dict,
                    num_classes=num_classes,
                    include_bbox=True,
                    include_masks=True,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertIsInstance(x, dict)
                self.assertEqual('input' in x, True)
                self.assertEqual('boxes_input' in x, True)
                self.assertEqual(x['input'].shape[1:], images.shape[1:])
                self.assertIsInstance(y, dict)
                self.assertEqual('regression' in y, True)
                self.assertEqual('classification' in y, True)
                r = y['regression']
                l = y['classification']
                self.assertEqual(r.shape[:-1], l.shape[:-1])
                self.assertEqual(r.shape[-1], 5)
                self.assertEqual(l.shape[-1], num_classes + 1)
                break

    def test_retinanet_data_generator_channels_first(self):
        for test_images in _generate_test_images(21, 21):
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.RetinaNetGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            num_classes = np.random.randint(1, 3)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 10, 10)),
                'y': np.random.random((8, 1, 10, 10)),
            }
            generator.flow(train_dict, num_classes=num_classes)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)

            for x, y in generator.flow(
                    train_dict,
                    num_classes=num_classes,
                    include_bbox=True,
                    include_masks=True,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertIsInstance(x, dict)
                self.assertEqual('input' in x, True)
                self.assertEqual('boxes_input' in x, True)
                self.assertEqual(x['input'].shape[1:], images.shape[1:])
                self.assertIsInstance(y, dict)
                self.assertEqual('regression' in y, True)
                self.assertEqual('classification' in y, True)
                r = y['regression']
                l = y['classification']
                self.assertEqual(r.shape[:-1], l.shape[:-1])
                self.assertEqual(r.shape[-1], 5)
                self.assertEqual(l.shape[-1], num_classes + 1)
                break

    def test_retinanet_data_generator_invalid_data(self):
        generator = image_generators.RetinaNetGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((8, 10, 10)),
                'y': np.random.random((8, 10, 10))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((7, 10, 10, 1))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 10, 10, 5)))

        with self.assertRaises(ValueError):
            generator = image_generators.RetinaNetGenerator(
                data_format='unknown')

        generator = image_generators.RetinaNetGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.RetinaNetGenerator(
                zoom_range=(2, 2, 2))


class TestRetinaMovieDataGenerator(test.TestCase):

    def test_retinamovie_data_generator(self):
        frames = 7
        frames_per_batch = 5
        for test_images in _generate_test_images(21, 21):
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batches = images.shape[0] // frames
            images = np.reshape(images, tuple([batches, frames] +
                                              list(images.shape[1:])))
            generator = image_generators.RetinaMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            num_classes = np.random.randint(1, 3)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 3)),
                'y': np.random.random((8, 11, 10, 10, 1)),
            }
            generator.flow(train_dict, num_classes=num_classes)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            # generator.fit(images, augment=True, seed=1)

            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)

            for x, y in generator.flow(
                    train_dict,
                    frames_per_batch=frames_per_batch,
                    num_classes=num_classes,
                    include_bbox=True,
                    include_masks=True,
                    save_to_dir=temp_dir,
                    shuffle=True):
                expected = list(images.shape)
                expected[1] = frames_per_batch
                expected = tuple(expected)
                self.assertIsInstance(x, dict)
                self.assertEqual('input' in x, True)
                self.assertEqual('boxes_input' in x, True)
                self.assertEqual(x['input'].shape[1:], expected[1:])
                self.assertIsInstance(y, dict)
                self.assertEqual('regression' in y, True)
                self.assertEqual('classification' in y, True)
                r = y['regression']
                l = y['classification']
                self.assertEqual(r.shape[:-1], l.shape[:-1])
                self.assertEqual(r.shape[-1], 5)
                self.assertEqual(l.shape[-1], num_classes + 1)
                break

    def test_retinamovie_data_generator_channels_first(self):
        frames = 7
        frames_per_batch = 5
        for test_images in _generate_test_images(21, 21):
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, tuple([batch_count, frames] +
                                              list(images.shape[1:])))
            images = np.rollaxis(images, 4, 1)
            generator = image_generators.RetinaMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            num_classes = np.random.randint(1, 3)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 11, 10, 10)),
                'y': np.random.random((8, 1, 11, 10, 10)),
            }
            generator.flow(train_dict, num_classes=num_classes)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            # generator.fit(images, augment=True, seed=1)

            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)

            for x, y in generator.flow(
                    train_dict,
                    num_classes=num_classes,
                    include_bbox=True,
                    include_masks=True,
                    frames_per_batch=frames_per_batch,
                    save_to_dir=temp_dir,
                    shuffle=True):
                expected = list(images.shape)
                expected[2] = frames_per_batch
                expected = tuple(expected)
                self.assertIsInstance(x, dict)
                self.assertEqual('input' in x, True)
                self.assertEqual('boxes_input' in x, True)
                self.assertEqual(x['input'].shape[1:], expected[1:])

                self.assertIsInstance(y, dict)
                self.assertEqual('regression' in y, True)
                self.assertEqual('classification' in y, True)
                r = y['regression']
                l = y['classification']
                self.assertEqual(r.shape[:-1], l.shape[:-1])
                self.assertEqual(r.shape[-1], 5)
                self.assertEqual(l.shape[-1], num_classes + 1)
                break

    def test_retinamovie_data_generator_invalid_data(self):
        generator = image_generators.RetinaMovieDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((8, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((7, 11, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with bigger frames_per_batch than frames
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((8, 11, 10, 10, 1))
            }
            generator.flow(train_dict, frames_per_batch=31)

        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 3, 10, 10, 5)))

        with self.assertRaises(ValueError):
            generator = image_generators.MovieDataGenerator(
                data_format='unknown')

        generator = image_generators.MovieDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.MovieDataGenerator(
                zoom_range=(2, 2, 2))
