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
"""Custom loss functions for DeepCell-RetinaMask"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras import backend as K

from deepcell_retinamask.utils.anchor_utils import overlap


def smooth_l1(y_true, y_pred, sigma=3.0, axis=None):
    """Compute the smooth L1 loss of y_pred w.r.t. y_true.

    Args:
        y_true: Tensor from the generator of shape (B, N, 5).
            The last value for each box is the state of the anchor
            (ignore, negative, positive).
        y_pred: Tensor from the network of shape (B, N, 4).
        sigma: The point where the loss changes from L2 to L1.

    Returns:
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1

    sigma_squared = sigma ** 2

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = K.abs(y_true - y_pred)  # |y - f(x)|

    regression_loss = tf.where(
        K.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * K.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared)
    return K.sum(regression_loss, axis=axis)


def focal(y_true, y_pred, alpha=0.25, gamma=2.0, axis=None):
    """Compute the focal loss given the target tensor and the predicted tensor.

    As defined in https://arxiv.org/abs/1708.02002

    Args:
        y_true: Tensor of target data with shape (B, N, num_classes).
        y_pred: Tensor of predicted data with shape (B, N, num_classes).
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns:
        The focal loss of y_pred w.r.t. y_true.
    """
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1

    # compute the focal loss
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)

    return K.sum(cls_loss, axis=axis)


class RetinaNetLosses(object):
    def __init__(self, sigma=3.0, alpha=0.25, gamma=2.0,
                 iou_threshold=0.5, fdl_iou_threshold=0.5,
                 mask_size=(28, 28),
                 parallel_iterations=32):
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.iou_threshold = iou_threshold
        self.fdl_iou_threshold = fdl_iou_threshold
        self.mask_size = mask_size
        self.parallel_iterations = parallel_iterations

    def regress_loss(self, y_true, y_pred):
        # separate target and state
        regression = y_pred
        regression_target = y_true[..., :-1]
        anchor_state = y_true[..., -1]

        # filter out "ignore" anchors
        indices = tf.where(K.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute the loss
        loss = smooth_l1(regression_target, regression, sigma=self.sigma)

        # compute the normalizer: the number of positive anchors
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())

        return K.sum(loss) / normalizer

    def classification_loss(self, y_true, y_pred):
        # TODO: try weighted_categorical_crossentropy
        labels = y_true[..., :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[..., -1]

        classification = y_pred
        # filter out "ignore" anchors
        indices = tf.where(K.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the loss
        loss = focal(labels, classification, alpha=self.alpha, gamma=self.gamma)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return K.sum(loss) / normalizer

    def mask_loss(self, y_true, y_pred):
        def _mask_conditional(y_true, y_pred):
            # if there are no masks annotations, return 0; else, compute the masks loss
            return tf.cond(
                K.any(K.equal(K.shape(y_true), 0)),
                lambda: K.cast_to_floatx(0.0),
                lambda: _mask_batch(y_true, y_pred,
                                    iou_threshold=self.iou_threshold,
                                    mask_size=self.mask_size,
                                    parallel_iterations=self.parallel_iterations)
            )

        def _mask_batch(y_true, y_pred,
                        iou_threshold=0.5,
                        mask_size=(28, 28),
                        parallel_iterations=32):
            if K.ndim(y_pred) == 4:
                y_pred_shape = tf.shape(y_pred)
                new_y_pred_shape = [y_pred_shape[0] * y_pred_shape[1],
                                    y_pred_shape[2], y_pred_shape[3]]
                y_pred = tf.reshape(y_pred, new_y_pred_shape)

                y_true_shape = tf.shape(y_true)
                new_y_true_shape = [y_true_shape[0] * y_true_shape[1],
                                    y_true_shape[2], y_true_shape[3]]
                y_true = tf.reshape(y_true, new_y_true_shape)

            # split up the different predicted blobs
            boxes = y_pred[:, :, :4]
            masks = y_pred[:, :, 4:]

            # split up the different blobs
            annotations = y_true[:, :, :5]
            width = K.cast(y_true[0, 0, 5], dtype='int32')
            height = K.cast(y_true[0, 0, 6], dtype='int32')
            masks_target = y_true[:, :, 7:]

            # reshape the masks back to their original size
            masks_target = K.reshape(masks_target, (K.shape(masks_target)[0],
                                                    K.shape(masks_target)[1],
                                                    height, width))
            masks = K.reshape(masks, (K.shape(masks)[0], K.shape(masks)[1],
                                      mask_size[0], mask_size[1], -1))

            def _mask(args):
                boxes = args[0]
                masks = args[1]
                annotations = args[2]
                masks_target = args[3]

                return compute_mask_loss(
                    boxes,
                    masks,
                    annotations,
                    masks_target,
                    width,
                    height,
                    iou_threshold=iou_threshold,
                    mask_size=mask_size,
                )

            mask_batch_loss = tf.map_fn(
                _mask,
                elems=[boxes, masks, annotations, masks_target],
                dtype=K.floatx(),
                parallel_iterations=parallel_iterations
            )

            return K.mean(mask_batch_loss)

        return _mask_conditional(y_true, y_pred)

    def final_detection_loss(self, y_true, y_pred):
        def _fd_conditional(y_true, y_pred):
            # if there are no masks annotations, return 0; else, compute fdl loss
            return tf.cond(
                K.any(K.equal(K.shape(y_true), 0)),
                lambda: K.cast_to_floatx(0.0),
                lambda: _fd_batch(y_true, y_pred,
                                  iou_threshold=self.fdl_iou_threshold,
                                  parallel_iterations=self.parallel_iterations))

        def _fd_batch(y_true, y_pred, iou_threshold=0.75, parallel_iterations=32):
            if K.ndim(y_pred) == 4:
                y_pred_shape = tf.shape(y_pred)
                new_y_pred_shape = [y_pred_shape[0] * y_pred_shape[1],
                                    y_pred_shape[2], y_pred_shape[3]]
                y_pred = tf.reshape(y_pred, new_y_pred_shape)

                y_true_shape = tf.shape(y_true)
                new_y_true_shape = [y_true_shape[0] * y_true_shape[1],
                                    y_true_shape[2], y_true_shape[3]]
                y_true = tf.reshape(y_true, new_y_true_shape)

            # split up the different predicted blobs
            boxes = y_pred[:, :, :4]
            scores = y_pred[:, :, 4:5]

            # split up the different blobs
            annotations = y_true[:, :, :5]

            def _fd(args):
                boxes = args[0]
                scores = args[1]
                annotations = args[2]

                return compute_fd_loss(
                    boxes,
                    scores,
                    annotations,
                    iou_threshold=iou_threshold)

            fd_batch_loss = tf.map_fn(
                _fd,
                elems=[boxes, scores, annotations],
                dtype=K.floatx(),
                parallel_iterations=parallel_iterations)

            return K.mean(fd_batch_loss)

        return _fd_conditional(y_true, y_pred)


def compute_mask_loss(boxes,
                      masks,
                      annotations,
                      masks_target,
                      width,
                      height,
                      iou_threshold=0.5,
                      mask_size=(28, 28)):
    """compute overlap of boxes with annotations"""
    iou = overlap(boxes, annotations)
    argmax_overlaps_inds = K.argmax(iou, axis=1)
    max_iou = K.max(iou, axis=1)

    # filter those with IoU > 0.5
    indices = tf.where(K.greater_equal(max_iou, iou_threshold))
    boxes = tf.gather_nd(boxes, indices)
    masks = tf.gather_nd(masks, indices)
    argmax_overlaps_inds = K.cast(tf.gather_nd(argmax_overlaps_inds, indices), 'int32')
    labels = K.cast(K.gather(annotations[:, 4], argmax_overlaps_inds), 'int32')

    # make normalized boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    boxes = K.stack([
        y1 / (K.cast(height, dtype=K.floatx()) - 1),
        x1 / (K.cast(width, dtype=K.floatx()) - 1),
        (y2 - 1) / (K.cast(height, dtype=K.floatx()) - 1),
        (x2 - 1) / (K.cast(width, dtype=K.floatx()) - 1),
    ], axis=1)

    # crop and resize masks_target
    # append a fake channel dimension
    masks_target = K.expand_dims(masks_target, axis=3)
    masks_target = tf.image.crop_and_resize(
        masks_target,
        boxes,
        argmax_overlaps_inds,
        mask_size
    )
    masks_target = masks_target[:, :, :, 0]  # remove fake channel dimension

    # gather the predicted masks using the annotation label
    masks = tf.transpose(masks, (0, 3, 1, 2))
    label_indices = K.stack([tf.range(K.shape(labels)[0]), labels], axis=1)

    masks = tf.gather_nd(masks, label_indices)

    # compute mask loss
    mask_loss = K.binary_crossentropy(masks_target, masks)
    normalizer = K.shape(masks)[0] * K.shape(masks)[1] * K.shape(masks)[2]
    normalizer = K.maximum(K.cast(normalizer, K.floatx()), 1)
    mask_loss = K.sum(mask_loss) / normalizer

    return mask_loss


def compute_fd_loss(boxes, scores, annotations, iou_threshold=0.75):
    """compute the overlap of boxes with annotations"""
    iou = overlap(boxes, annotations)

    max_iou = K.max(iou, axis=1, keepdims=True)
    targets = K.cast(K.greater_equal(max_iou, iou_threshold), K.floatx())

    # compute the loss
    loss = focal(targets, scores)  # alpha=self.alpha, gamma=self.gamma)

    # compute the normalizer: the number of cells present in the image
    normalizer = K.cast(K.shape(annotations)[0], K.floatx())
    normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

    return K.sum(loss) / normalizer
