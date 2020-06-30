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
"""Image generators for training RetinaNet based models.

The RetinaNetGenerator and RetinaMovieDataGenerator take in a raw image as well
as a label mask, and extract bounding boxes for all objects in the label mask.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import warnings

import numpy as np

from skimage.measure import regionprops
from skimage.segmentation import clear_border

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.platform import tf_logging as logging

from deepcell_retinamask.utils.retinanet_anchor_utils import anchor_targets_bbox
from deepcell_retinamask.utils.retinanet_anchor_utils import anchors_for_shape
from deepcell_retinamask.utils.retinanet_anchor_utils import guess_shapes

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None


def _transform_masks(y, transform, data_format=None, **kwargs):
    """Based on the transform key, apply a transform function to the masks.

    Refer to :mod:`deepcell.utils.transform_utils` for more information about
    available transforms. Caution for unknown transform keys.

    Args:
        y (numpy.array): Labels of ndim 4 or 5
        transform (str): Name of the transform, one of
            {"deepcell", "disc", "watershed", None}
        data_format (str): One of 'channels_first', 'channels_last'.
        kwargs (dict): Optional transform keyword arguments.

    Returns:
        numpy.array: the output of the given transform function on y

    Raises:
        ValueError: Rank of y is not 4 or 5.
        ValueError: Channel dimension of y is not 1.
        ValueError: Transform is invalid value.
    """
    valid_transforms = {
        'deepcell',  # deprecated for "pixelwise"
        'pixelwise',
        'disc',
        'watershed',  # deprecated for "outer-distance"
        'watershed-cont',  # deprecated for "outer-distance"
        'inner-distance',
        'outer-distance',
        'centroid',  # deprecated for "inner-distance"
        'fgbg'
    }
    if data_format is None:
        data_format = K.image_data_format()

    if y.ndim not in {4, 5}:
        raise ValueError('`labels` data must be of ndim 4 or 5.  Got', y.ndim)

    channel_axis = 1 if data_format == 'channels_first' else -1

    if y.shape[channel_axis] != 1:
        raise ValueError('Expected channel axis to be 1 dimension. Got',
                         y.shape[1 if data_format == 'channels_first' else -1])

    if isinstance(transform, str):
        transform = transform.lower()

    if transform not in valid_transforms and transform is not None:
        raise ValueError('`{}` is not a valid transform'.format(transform))

    if transform in {'pixelwise', 'deepcell'}:
        if transform == 'deepcell':
            warnings.warn('The `{}` transform is deprecated. Please use the '
                          '`pixelwise` transform instead.'.format(transform),
                          DeprecationWarning)
        dilation_radius = kwargs.pop('dilation_radius', None)
        separate_edge_classes = kwargs.pop('separate_edge_classes', False)

        edge_class_shape = 4 if separate_edge_classes else 3

        if data_format == 'channels_first':
            y_transform = np.zeros(tuple([y.shape[0]] + [edge_class_shape] + list(y.shape[2:])))
        else:
            y_transform = np.zeros(tuple(list(y.shape[0:-1]) + [edge_class_shape]))

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = transform_utils.pixelwise_transform(
                mask, dilation_radius, data_format=data_format,
                separate_edge_classes=separate_edge_classes)

    elif transform in {'outer-distance', 'watershed', 'watershed-cont'}:
        if transform in {'watershed', 'watershed-cont'}:
            warnings.warn('The `{}` transform is deprecated. Please use the '
                          '`outer-distance` transform instead.'.format(transform),
                          DeprecationWarning)

        bins = kwargs.pop('distance_bins', None)
        erosion = kwargs.pop('erosion_width', 0)
        by_frame = kwargs.pop('by_frame', True)

        if data_format == 'channels_first':
            y_transform = np.zeros(tuple([y.shape[0]] + list(y.shape[2:])))
        else:
            y_transform = np.zeros(y.shape[0:-1])

        if y.ndim == 5:
            if by_frame:
                _distance_transform = transform_utils.outer_distance_transform_movie
            else:
                _distance_transform = transform_utils.outer_distance_transform_3d
        else:
            _distance_transform = transform_utils.outer_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = _distance_transform(
                mask, bins=bins, erosion_width=erosion)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if bins is None:
            pass
        else:
            # convert to one hot notation
            y_transform = to_categorical(y_transform, num_classes=bins)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform in {'inner-distance', 'centroid'}:
        if transform == 'centroid':
            warnings.warn('The `{}` transform is deprecated. Please use the '
                          '`inner-distance` transform instead.'.format(transform),
                          DeprecationWarning)

        bins = kwargs.pop('distance_bins', None)
        erosion = kwargs.pop('erosion_width', 0)
        by_frame = kwargs.pop('by_frame', True)
        alpha = kwargs.pop('alpha', 0.1)
        beta = kwargs.pop('beta', 1)

        if data_format == 'channels_first':
            y_transform = np.zeros(tuple([y.shape[0]] + list(y.shape[2:])))
        else:
            y_transform = np.zeros(y.shape[0:-1])

        if y.ndim == 5:
            if by_frame:
                _distance_transform = transform_utils.inner_distance_transform_movie
            else:
                _distance_transform = transform_utils.inner_distance_transform_3d
        else:
            _distance_transform = transform_utils.inner_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = _distance_transform(mask, bins=bins,
                                                     erosion_width=erosion,
                                                     alpha=alpha, beta=beta)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if bins is None:
            pass
        else:
            # convert to one hot notation
            y_transform = to_categorical(y_transform, num_classes=bins)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'disc':
        y_transform = to_categorical(y.squeeze(channel_axis))
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'fgbg':
        y_transform = np.where(y > 1, 1, y)
        # convert to one hot notation
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, 1, y.ndim)
        y_transform = to_categorical(y_transform)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform is None:
        y_transform = to_categorical(y.squeeze(channel_axis))
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    return y_transform


class RetinaNetGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.

    The data will be looped over (in batches).

    Args:
        featurewise_center (bool): Set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center (bool): Set each sample mean to 0.
        featurewise_std_normalization (bool): Divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization (bool): Divide each input by its std.
        zca_epsilon (float): Epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening (bool): Apply ZCA whitening.
        rotation_range (int): Degree range for random rotations.
        width_shift_range (float): 1-D array-like or int

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1], while with
              width_shift_range=1.0 possible values are floats in the interval
              [-1.0, +1.0).

        shear_range (float): Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range (float): float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range (float): range for random channel shifts.
        fill_mode (str): One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:

                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval (float): Value used for points outside the boundaries
            when fill_mode = "constant".
        horizontal_flip (bool): Randomly flip inputs horizontally.
        vertical_flip (bool): Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format (str): One of {"channels_first", "channels_last"}.

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

        validation_split (float): Fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def flow(self,
             train_dict,
             compute_shapes=guess_shapes,
             min_objects=3,
             num_classes=1,
             clear_borders=False,
             include_masks=False,
             panoptic=False,
             transforms=['watershed'],
             transforms_kwargs={},
             anchor_params=None,
             pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
             batch_size=32,
             shuffle=False,
             semantic_only=False,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict (dict): Consists of numpy arrays for X and y.
            compute_shapes: Function to determine the shapes of the anchors.
            min_objects (int): images with fewer than 'min_objects' are ignored.
            num_classes (int): Number of classes to predict.
            clear_borders (bool): Whether to use clear_border on y.
            include_masks (bool): Train on mask data (MaskRCNN).
            batch_size (int): Size of a batch.
            shuffle (bool): Whether to shuffle the data between epochs.
            seed (int): Random seed for data shuffling.
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if save_to_dir is set).
            save_format (str): Format to use for saving sample images
                (if save_to_dir is set).

        Returns:
            RetinaNetIterator: An Iterator yielding tuples of (x, y),
                where x is a numpy array of image data and y is a numpy array
                of labels of the same shape.
        """
        return RetinaNetIterator(
            train_dict,
            self,
            compute_shapes=compute_shapes,
            min_objects=min_objects,
            num_classes=num_classes,
            clear_borders=clear_borders,
            include_masks=include_masks,
            panoptic=panoptic,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            anchor_params=anchor_params,
            pyramid_levels=pyramid_levels,
            batch_size=batch_size,
            shuffle=shuffle,
            semantic_only=semantic_only,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x: 3D tensor or list of 3D tensors,
                single image.
            y: 3D tensor or list of 3D tensors,
                label mask(s) for x, optional.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
            If y is passed, it is transformed if necessary and returned.
        """
        params = self.get_random_transform(x.shape, seed)

        if isinstance(x, list):
            x = [self.apply_transform(x_i, params) for x_i in x]
        else:
            x = self.apply_transform(x, params)

        if y is None:
            return x

        # Nullify the transforms that don't affect `y`
        params['brightness'] = None
        params['channel_shift_intensity'] = None
        _interpolation_order = self.interpolation_order
        self.interpolation_order = 0

        if isinstance(y, list):
            y_new = []
            for y_i in y:
                y_t = self.apply_transform(y_i, params)
                y_new.append(y_t)
            y = y_new
        else:
            y = self.apply_transform(y, params)

        self.interpolation_order = _interpolation_order
        return x, y


class RetinaNetIterator(Iterator):
    """Iterator yielding data from Numpy arrayss (X and y).

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        train_dict (dict): Consists of numpy arrays for X and y.
        image_data_generator (RetinaNetGenerator): For random
            transformations and normalization.
        compute_shapes: Function to determine the shapes of the anchors.
        min_objects (int): Images with fewer than 'min_objects' are ignored.
        num_classes (int): Number of classes to predict.
        clear_borders (bool): Whether to use clear_border on y.
        include_masks (bool): Train on mask data (MaskRCNN).
        batch_size (int): Size of a batch.
        shuffle (bool): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        data_format (str): One of 'channels_first', 'channels_last'.
        save_to_dir (str): Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix (str): Prefix to use for saving sample
            images (if save_to_dir is set).
        save_format (str): Format to use for saving sample images
            (if save_to_dir is set).
    """

    def __init__(self,
                 train_dict,
                 image_data_generator,
                 compute_shapes=guess_shapes,
                 anchor_params=None,
                 pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                 min_objects=3,
                 num_classes=1,
                 clear_borders=False,
                 include_masks=False,
                 panoptic=False,
                 transforms=['watershed'],
                 transforms_kwargs={},
                 semantic_only=False,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']

        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))

        if X.ndim != 4:
            raise ValueError('Input data in `RetinaNetIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', X.shape)

        self.x = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype='int32')

        # `compute_shapes` changes based on the model backbone.
        self.compute_shapes = compute_shapes
        self.anchor_params = anchor_params
        self.pyramid_levels = [int(l[1:]) for l in pyramid_levels]
        self.min_objects = min_objects
        self.num_classes = num_classes
        self.include_masks = include_masks
        self.panoptic = panoptic
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.semantic_only = semantic_only

        self.y_semantic_list = []  # optional semantic segmentation targets

        # Add semantic segmentation targets if panoptic segmentation
        # flag is True
        if panoptic:
            # Create a list of all the semantic targets. We need to be able
            # to have multiple semantic heads
            # Add all the keys that contain y_semantic
            for key in train_dict:
                if 'y_semantic' in key:
                    self.y_semantic_list.append(train_dict[key])

            # Add transformed masks
            for transform in transforms:
                transform_kwargs = transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y, transform,
                                               data_format=data_format,
                                               **transform_kwargs)
                if y_transform.shape[self.channel_axis] > 1:
                    y_transform = np.asarray(y_transform, dtype='int32')
                else:
                    y_transform = np.asarray(y_transform, dtype=K.floatx())
                self.y_semantic_list.append(y_transform)

        invalid_batches = []
        # Remove images with small numbers of cells
        for b in range(self.x.shape[0]):
            y_batch = np.squeeze(self.y[b], axis=self.channel_axis - 1)
            y_batch = clear_border(y_batch) if clear_borders else y_batch
            y_batch = np.expand_dims(y_batch, axis=self.channel_axis - 1)

            self.y[b] = y_batch

            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.y = np.delete(self.y, invalid_batches, axis=0)
        self.x = np.delete(self.x, invalid_batches, axis=0)

        self.y_semantic_list = [np.delete(y, invalid_batches, axis=0)
                                for y in self.y_semantic_list]

        super(RetinaNetIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def filter_annotations(self, image, annotations):
        """Filter annotations by removing those that are outside of the
        image bounds or whose width/height < 0.

        Args:
            image (numpy.array): The raw image data.
            annotations (dict): Annotations including labels and bboxes

        Returns:
            dict: filtered annotations.
        """
        row_axis = 1 if self.data_format == 'channels_first' else 0
        invalid_indices = np.where(
            (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
            (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
            (annotations['bboxes'][:, 0] < 0) |
            (annotations['bboxes'][:, 1] < 0) |
            (annotations['bboxes'][:, 2] > image.shape[row_axis + 1]) |
            (annotations['bboxes'][:, 3] > image.shape[row_axis])
        )[0]

        # delete invalid indices
        if invalid_indices.size > 0:
            logging.warn('Image with shape {} contains the following invalid '
                         'boxes: {}.'.format(
                             image.shape,
                             annotations['bboxes'][invalid_indices, :]))

            for k in annotations.keys():
                filtered = np.delete(annotations[k], invalid_indices, axis=0)
                annotations[k] = filtered
        return annotations

    def load_annotations(self, y):
        """Generate bounding box and label annotations for a tensor

        Args:
            y (tensor): Tensor to annotate

        Returns:
            dict: Annotations of bboxes and labels
        """
        labels, bboxes, masks = [], [], []
        for prop in regionprops(np.squeeze(y.astype('int'))):
            y1, x1, y2, x2 = prop.bbox
            bboxes.append([x1, y1, x2, y2])
            labels.append(0)  # boolean object detection
            masks.append(np.where(y == prop.label, 1, 0))

        labels = np.array(labels)
        bboxes = np.array(bboxes)
        masks = np.array(masks).astype('uint8')

        # reshape bboxes in case it is empty.
        bboxes = np.reshape(bboxes, (bboxes.shape[0], 4))

        annotations = {'labels': labels, 'bboxes': bboxes}

        if self.include_masks:
            annotations['masks'] = masks

        annotations = self.filter_annotations(y, annotations)
        return annotations

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))

        batch_y_semantic_list = []
        for y_sem in self.y_semantic_list:
            shape = tuple([len(index_array)] + list(y_sem.shape[1:]))
            batch_y_semantic_list.append(np.zeros(shape, dtype=y_sem.dtype))

        annotations_list = []

        max_shape = []

        for i, j in enumerate(index_array):
            x = self.x[j]
            y = self.y[j]

            y_semantic_list = [y_sem[j] for y_sem in self.y_semantic_list]

            # Apply transformation
            x, y_list = self.image_data_generator.random_transform(
                x, [y] + y_semantic_list)

            y = y_list[0]
            y_semantic_list = y_list[1:]

            # Find max shape of image data.  Used for masking.
            if not max_shape:
                max_shape = list(x.shape)
            else:
                for k in range(len(x.shape)):
                    if x.shape[k] > max_shape[k]:
                        max_shape[k] = x.shape[k]

            # Get the bounding boxes from the transformed masks!
            annotations = self.load_annotations(y)
            annotations_list.append(annotations)

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

            for k, y_sem in enumerate(y_semantic_list):
                batch_y_semantic_list[k][i] = y_sem

        anchors = anchors_for_shape(
            batch_x.shape[1:],
            pyramid_levels=self.pyramid_levels,
            anchor_params=self.anchor_params,
            shapes_callback=self.compute_shapes)

        regressions, labels = anchor_targets_bbox(
            anchors,
            batch_x,
            annotations_list,
            self.num_classes)

        max_shape = tuple(max_shape)  # was a list for max shape indexing

        if self.include_masks:
            # masks_batch has shape: (batch size, max_annotations,
            #     bbox_x1 + bbox_y1 + bbox_x2 + bbox_y2 + label +
            #     width + height + max_image_dimension)
            max_annotations = max(len(a['masks']) for a in annotations_list)
            masks_batch_shape = (len(index_array), max_annotations,
                                 5 + 2 + max_shape[0] * max_shape[1])
            masks_batch = np.zeros(masks_batch_shape, dtype=K.floatx())

            for i, ann in enumerate(annotations_list):
                masks_batch[i, :ann['bboxes'].shape[0], :4] = ann['bboxes']
                masks_batch[i, :ann['labels'].shape[0], 4] = ann['labels']
                masks_batch[i, :, 5] = max_shape[1]  # width
                masks_batch[i, :, 6] = max_shape[0]  # height

                # add flattened mask
                for j, mask in enumerate(ann['masks']):
                    masks_batch[i, j, 7:] = mask.flatten()

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                if self.data_format == 'channels_first':
                    img_x = np.expand_dims(batch_x[i, 0, ...], 0)
                else:
                    img_x = np.expand_dims(batch_x[i, ..., 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        batch_outputs = [regressions, labels]

        if self.include_masks:
            batch_outputs.append(masks_batch)

        batch_outputs.extend(batch_y_semantic_list)

        if self.semantic_only:
            batch_outputs = batch_y_semantic_list

        return batch_x, batch_outputs

    def next(self):
        """For python 2.x. Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class RetinaMovieIterator(Iterator):
    """Iterator yielding data from Numpy arrayss (X and y).

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        train_dict (dict): Consists of numpy arrays for X and y.
        movie_data_generator (RetinaMovieDataGenerator): For random
            transformations and normalization.
        batch_size (int): Size of a batch.
        shuffle (bool): Whether to shuffle the data between epochs.
        compute_shapes: functor for generating shapes, based on the model.
        min_objects (int): Image with fewer than min_objects are ignored.
        num_classes (int): Number of classes for classification.
        frames_per_batch (int): Size of z axis in generated batches.
        clear_borders (bool): Whether to call clear_border on y.
        include_masks (bool): Whether to yield mask data.
        seed (int): Random seed for data shuffling.
        data_format (str): One of 'channels_first', 'channels_last'.
        save_to_dir (str): Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix (str): Prefix to use for saving sample
            images (if save_to_dir is set).
        save_format (str): Format to use for saving sample images
            (if save_to_dir is set).
    """

    def __init__(self,
                 train_dict,
                 movie_data_generator,
                 compute_shapes=guess_shapes,
                 anchor_params=None,
                 pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                 min_objects=3,
                 num_classes=1,
                 frames_per_batch=2,
                 clear_borders=False,
                 include_masks=False,
                 include_final_detection_layer=False,
                 panoptic=False,
                 transforms=['watershed'],
                 transforms_kwargs={},
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']

        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))

        if X.ndim != 5:
            raise ValueError('Input data in `RetinaNetIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', X.shape)

        self.x = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype='int32')

        # `compute_shapes` changes based on the model backbone.
        self.compute_shapes = compute_shapes
        self.anchor_params = anchor_params
        self.pyramid_levels = [int(l[1:]) for l in pyramid_levels]
        self.min_objects = min_objects
        self.num_classes = num_classes
        self.frames_per_batch = frames_per_batch
        self.include_masks = include_masks
        self.include_final_detection_layer = include_final_detection_layer
        self.panoptic = panoptic
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.row_axis = 2 if data_format == 'channels_last' else 3
        self.col_axis = 3 if data_format == 'channels_last' else 4
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.y_semantic_list = []  # optional semantic segmentation targets

        if X.shape[self.time_axis] - frames_per_batch < 0:
            raise ValueError(
                'The number of frames used in each training batch should '
                'be less than the number of frames in the training data!')

        # Add semantic segmentation targets if panoptic segmentation
        # flag is True
        if panoptic:
            # Create a list of all the semantic targets. We need to be able
            # to have multiple semantic heads
            # Add all the keys that contain y_semantic
            for key in train_dict:
                if 'y_semantic' in key:
                    self.y_semantic_list.append(train_dict[key])

            # Add transformed masks
            for transform in transforms:
                transform_kwargs = transforms_kwargs.get(transform, dict())
                y_transforms = []
                for time in range(y.shape[self.time_axis]):
                    if data_format == 'channels_first':
                        y_temp = y[:, :, time, ...]
                    else:
                        y_temp = y[:, time, ...]
                    y_temp_transform = _transform_masks(
                        y_temp, transform,
                        data_format=data_format,
                        **transform_kwargs)
                    y_temp_transform = np.asarray(y_temp_transform, dtype='int32')
                    y_transforms.append(y_temp_transform)

                y_transform = np.stack(y_transforms, axis=self.time_axis)
                self.y_semantic_list.append(y_transform)

        invalid_batches = []
        # Remove images with small numbers of cells
        for b in range(self.x.shape[0]):
            y_batch = np.squeeze(self.y[b], axis=self.channel_axis - 1)
            y_batch = clear_border(y_batch) if clear_borders else y_batch
            y_batch = np.expand_dims(y_batch, axis=self.channel_axis - 1)

            self.y[b] = y_batch

            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.y = np.delete(self.y, invalid_batches, axis=0)
        self.x = np.delete(self.x, invalid_batches, axis=0)

        self.y_semantic_list = [np.delete(y, invalid_batches, axis=0)
                                for y in self.y_semantic_list]

        super(RetinaMovieIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def filter_annotations(self, image, annotations):
        """Filter annotations by removing those that are outside of the
        image bounds or whose width/height < 0.

        Args:
            image (numpy.array): The raw image data.
            annotations (dict): Annotations including labels and bboxes.

        Returns:
            dict: filtered annotations.
        """
        row_axis = 1 if self.data_format == 'channels_first' else 0
        invalid_indices = np.where(
            (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
            (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
            (annotations['bboxes'][:, 0] < 0) |
            (annotations['bboxes'][:, 1] < 0) |
            (annotations['bboxes'][:, 2] > image.shape[row_axis + 1]) |
            (annotations['bboxes'][:, 3] > image.shape[row_axis])
        )[0]

        # delete invalid indices
        if invalid_indices.size > 0:
            logging.warn('Image with shape {} contains the following invalid '
                         'boxes: {}.'.format(
                             image.shape,
                             annotations['bboxes'][invalid_indices, :]))

            for k in annotations.keys():
                filtered = np.delete(annotations[k], invalid_indices, axis=0)
                annotations[k] = filtered
        return annotations

    def load_annotations(self, y):
        """Generate bounding box and label annotations for a tensor

        Args:
            y (tensor): tensor to annotate.

        Returns:
            dict: Annotations of bboxes and labels
        """
        labels, bboxes, masks = [], [], []
        for prop in regionprops(np.squeeze(y.astype('int'))):
            y1, x1, y2, x2 = prop.bbox
            bboxes.append([x1, y1, x2, y2])
            labels.append(0)  # boolean object detection
            masks.append(np.where(y == prop.label, 1, 0))

        labels = np.array(labels)
        bboxes = np.array(bboxes)
        masks = np.array(masks).astype('uint8')

        # reshape bboxes in case it is empty.
        bboxes = np.reshape(bboxes, (bboxes.shape[0], 4))

        annotations = {'labels': labels, 'bboxes': bboxes}

        if self.include_masks:
            annotations['masks'] = masks

        annotations = self.filter_annotations(y, annotations)
        return annotations

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            batch_x = np.zeros((len(index_array),
                                self.x.shape[1],
                                self.frames_per_batch,
                                self.x.shape[3],
                                self.x.shape[4]))
        else:
            batch_x = np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                     list(self.x.shape)[2:]))

        if self.panoptic:
            if self.data_format == 'channels_first':
                batch_y_semantic_list = [np.zeros(tuple([len(index_array),
                                                         y_semantic.shape[1],
                                                         self.frames_per_batch,
                                                         y_semantic.shape[3],
                                                         y_semantic.shape[4]]))
                                         for y_semantic in self.y_semantic_list]
            else:
                batch_y_semantic_list = [
                    np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                   list(y_semantic.shape[2:])))
                    for y_semantic in self.y_semantic_list
                ]

        annotations_list = [[] for _ in range(self.frames_per_batch)]

        max_shape = []

        for i, j in enumerate(index_array):
            last_frame = self.x.shape[self.time_axis] - self.frames_per_batch
            time_start = np.random.randint(0, high=last_frame)
            time_end = time_start + self.frames_per_batch
            times = list(np.arange(time_start, time_end))

            if self.time_axis == 1:
                x = self.x[j, time_start:time_end, ...]
                y = self.y[j, time_start:time_end, ...]
            elif self.time_axis == 2:
                x = self.x[j, :, time_start:time_end, ...]
                y = self.y[j, :, time_start:time_end, ...]

            if self.panoptic:
                if self.time_axis == 1:
                    y_semantic_list = [y_semantic[j, time_start:time_end, ...]
                                       for y_semantic in self.y_semantic_list]
                elif self.time_axis == 2:
                    y_semantic_list = [y_semantic[j, :, time_start:time_end, ...]
                                       for y_semantic in self.y_semantic_list]

            # Apply transformation
            if self.panoptic:
                x, y_list = self.movie_data_generator.random_transform(x, [y] + y_semantic_list)
                y = y_list[0]
                y_semantic_list = y_list[1:]
            else:
                x, y = self.movie_data_generator.random_transform(x, y)

            x = self.movie_data_generator.standardize(x)

            # Find max shape of image data.  Used for masking.
            if not max_shape:
                max_shape = list(x.shape)
            else:
                for k in range(len(x.shape)):
                    if x.shape[k] > max_shape[k]:
                        max_shape[k] = x.shape[k]

            # Get the bounding boxes from the transformed masks!
            for idx_time, time in enumerate(times):
                if self.time_axis == 1:
                    annotations = self.load_annotations(y[idx_time])
                elif self.time_axis == 2:
                    annotations = self.load_annotations(y[:, idx_time, ...])
                annotations_list[idx_time].append(annotations)

            batch_x[i] = x

            if self.panoptic:
                for k in range(len(y_semantic_list)):
                    batch_y_semantic_list[k][i] = y_semantic_list[k]

        if self.data_format == 'channels_first':
            batch_x_shape = [batch_x.shape[1], batch_x.shape[3], batch_x.shape[4]]
        else:
            batch_x_shape = batch_x.shape[2:]

        anchors = anchors_for_shape(
            batch_x_shape,
            pyramid_levels=self.pyramid_levels,
            anchor_params=self.anchor_params,
            shapes_callback=self.compute_shapes)

        regressions_list = []
        labels_list = []

        if self.data_format == 'channels_first':
            batch_x_frame = batch_x[:, :, 0, ...]
        else:
            batch_x_frame = batch_x[:, 0, ...]
        for idx, time in enumerate(times):
            regressions, labels = anchor_targets_bbox(
                anchors,
                batch_x_frame,
                annotations_list[idx],
                self.num_classes)
            regressions_list.append(regressions)
            labels_list.append(labels)

        regressions = np.stack(regressions_list, axis=self.time_axis)
        labels = np.stack(labels_list, axis=self.time_axis)

        # was a list for max shape indexing
        max_shape = tuple([max_shape[self.row_axis - 1],
                           max_shape[self.col_axis - 1]])

        if self.include_masks:
            # masks_batch has shape: (batch size, max_annotations,
            #     bbox_x1 + bbox_y1 + bbox_x2 + bbox_y2 + label +
            #     width + height + max_image_dimension)

            flatten = lambda l: [item for sublist in l for item in sublist]
            annotations_list_flatten = flatten(annotations_list)
            max_annotations = max(len(a['masks']) for a in annotations_list_flatten)
            masks_batch_shape = (len(index_array), self.frames_per_batch, max_annotations,
                                 5 + 2 + max_shape[0] * max_shape[1])
            masks_batch = np.zeros(masks_batch_shape, dtype=K.floatx())

            for idx_time, time in enumerate(times):
                annotations_frame = annotations_list[idx_time]
                for idx_batch, ann in enumerate(annotations_frame):
                    masks_batch[idx_batch, idx_time, :ann['bboxes'].shape[0], :4] = ann['bboxes']
                    masks_batch[idx_batch, idx_time, :ann['labels'].shape[0], 4] = ann['labels']
                    masks_batch[idx_batch, idx_time, :, 5] = max_shape[1]  # width
                    masks_batch[idx_batch, idx_time, :, 6] = max_shape[0]  # height

                    # add flattened mask
                    for idx_mask, mask in enumerate(ann['masks']):
                        masks_batch[idx_batch, idx_time, idx_mask, 7:] = mask.flatten()

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                for frame in range(batch_x.shape[self.time_axis]):
                    if self.time_axis == 2:
                        img = array_to_img(batch_x[i, :, frame], self.data_format, scale=True)
                    else:
                        img = array_to_img(batch_x[i, frame], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        batch_outputs = [regressions, labels]
        if self.include_masks:
            batch_outputs.append(masks_batch)
        if self.include_final_detection_layer:
            batch_outputs.append(masks_batch)
        if self.panoptic:
            batch_outputs += batch_y_semantic_list

        return batch_x, batch_outputs

    def next(self):
        """For python 2.x. Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class RetinaMovieDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.

    The data will be looped over (in batches).

    Args:
        featurewise_center (bool): Set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center (bool): Set each sample mean to 0.
        featurewise_std_normalization (bool): Divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization (bool): Divide each input by its std.
        zca_epsilon (float): Epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening (bool): Apply ZCA whitening.
        rotation_range (int): Degree range for random rotations.
        width_shift_range (float): 1-D array-like or int

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1], while with
              width_shift_range=1.0 possible values are floats in the interval
              [-1.0, +1.0).

        shear_range (float): Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range (float): float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range (float): range for random channel shifts.
        fill_mode (str): One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:

                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval (float): Value used for points outside the boundaries
            when fill_mode = "constant".
        horizontal_flip (bool): Randomly flip inputs horizontally.
        vertical_flip (bool): Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format (str): One of {"channels_first", "channels_last"}.

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

        validation_split (float): Fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def __init__(self, **kwargs):
        super(RetinaMovieDataGenerator, self).__init__(**kwargs)
        # Change the axes for 5D data
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 3
            self.col_axis = 4
            self.time_axis = 2
        if self.data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.time_axis = 1

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        Args:
            x (tensor): batch of inputs to be normalized.

        Returns:
            tensor: The normalized inputs.
        """
        # TODO: standardize each image, not all frames at once
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it '
                                'hasn\'t been fit on any training data. '
                                'Fit it first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`zca_whitening`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x (tensor): 4D stack of images.
            y (tensor): 4D label mask for x, optional.
            seed (int): Random seed.

        Returns:
            tensor: A randomly transformed version of the input (same shape).
                If y is passed, it is transformed if necessary and returned.
        """
        # Note: Workaround to use self.apply_transform on our 4D tensor
        self.row_axis -= 1
        self.col_axis -= 1
        self.time_axis -= 1
        self.channel_axis -= 1
        x_new = np.empty(x.shape)
        if y is not None:
            y_new = np.empty(y.shape)
        # apply_transform expects ndim=3, but we are ndim=4
        for frame in range(x.shape[self.time_axis]):
            if self.data_format == 'channels_first':
                params = self.get_random_transform(x[:, frame].shape, seed)
                x_trans = self.apply_transform(x[:, frame], params)
                x_new[:, frame] = np.rollaxis(x_trans, -1, 0)
            else:
                params = self.get_random_transform(x[frame].shape, seed)
                x_new[frame] = self.apply_transform(x[frame], params)

            if y is not None:
                params['brightness'] = None
                params['channel_shift_intensity'] = None
                _interpolation_order = self.interpolation_order
                self.interpolation_order = 0
                if self.data_format == 'channels_first':
                    y_trans = self.apply_transform(y[:, frame], params)
                    y_new[:, frame] = np.rollaxis(y_trans, 1, 0)
                else:
                    y_new[frame] = self.apply_transform(y[frame], params)
                self.interpolation_order = _interpolation_order
        # Note: Undo workaround
        self.row_axis += 1
        self.col_axis += 1
        self.time_axis += 1
        self.channel_axis += 1
        if y is None:
            return x_new
        return x_new, y_new

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        Args:
            x (numpy.array): The data to fit on. Should have rank 5.
            augment (bool): Whether to fit on randomly augmented samples.
            rounds (bool): If augment,
                how many augmentation passes to do over the data.
            seed (int): Random seed for data shuffling.

        Raises:
            ValueError: If input rank is not 5.
            ImportError: If zca_whitening is used and scipy is not available.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            logging.warning(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' +
                self.data_format + '" (channels on axis ' +
                str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' +
                str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' +
                str(x.shape) + ' (' + str(x.shape[self.channel_axis]) +
                ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype)
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            axis = (0, self.time_axis, self.row_axis, self.col_axis)
            self.mean = np.mean(x, axis=axis)
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            axis = (0, self.time_axis, self.row_axis, self.col_axis)
            self.std = np.std(x, axis=axis)
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            if scipy is None:
                raise ImportError('Using zca_whitening requires SciPy. '
                                  'Install SciPy.')
            flat_x = np.reshape(
                x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = scipy.linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)

    def flow(self,
             train_dict,
             batch_size=1,
             frames_per_batch=5,
             compute_shapes=guess_shapes,
             num_classes=1,
             clear_borders=False,
             include_masks=False,
             include_final_detection_layer=False,
             panoptic=False,
             transforms=['watershed'],
             transforms_kwargs={},
             anchor_params=None,
             pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
             shuffle=False,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict (dict): Consists of numpy arrays for X and y.
            batch_size (int): Size of a batch.
            frames_per_batch (int): Size of z axis in generated batches.
            shuffle (bool): Whether to shuffle the data between epochs.
            compute_shapes: functor for generating shapes, based on the model.
            num_classes (int): Number of classes for classification.
            clear_borders (bool): Whether to call clear_border on y.
            include_masks (bool): Whether to yield mask data.
            seed (int): Random seed for data shuffling.
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if save_to_dir is set).
            save_format (str): Format to use for saving sample images
                (if save_to_dir is set).

        Returns:
            RetinaMovieIterator: An Iterator yielding tuples of (x, y),
                where x is a numpy array of image data and y is a numpy array
                of labels of the same shape.
        """
        return RetinaMovieIterator(
            train_dict,
            self,
            compute_shapes=compute_shapes,
            num_classes=num_classes,
            clear_borders=clear_borders,
            include_masks=include_masks,
            include_final_detection_layer=include_final_detection_layer,
            panoptic=panoptic,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            anchor_params=anchor_params,
            pyramid_levels=pyramid_levels,
            batch_size=batch_size,
            frames_per_batch=frames_per_batch,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)
