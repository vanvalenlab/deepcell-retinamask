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
"""Utilities plotting data"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import numpy as np


def draw_box(image, box, color, thickness=2):
    """Draws a box on an image with a given color.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image (numpy.array): The image to draw on.
        box (int[]): A list of 4 elements ``(x1, y1, x2, y2)``.
        color (int[]): The color of the box.
        thickness (int): The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """Draws a caption above the box in an image.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image (numpy.array): The image to draw on.
        box (int[]): A list of 4 elements ``(x1, y1, x2, y2)``.
        caption (str): String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_mask(image,
              box,
              mask,
              color=[31, 0, 255],
              binarize_threshold=0.5):
    """Draws a mask in a given box.

    Args:
        image (numpy.array): Three dimensional image to draw on.
        box (tuple): Vector of at least 4 values ``(x1, y1, x2, y2)``
            representing a box in the image.
        mask (numpy.array): A 2D float mask which will be reshaped to the size
            of the box, binarized and drawn over the image.
        color (tuple): Color to draw the mask with. If the box has 5 values,
            the last value is assumed to be the label and used to
            construct a default color.
        binarize_threshold (float): Threshold used for binarizing the mask.
    """
    # resize to fit the box
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

    # binarize the mask
    mask = (mask > binarize_threshold).astype('uint8')

    # draw the mask in the image
    mask_image = np.zeros((image.shape[0], image.shape[1]), 'uint8')
    mask_image[box[1]:box[3], box[0]:box[2]] = mask
    mask = mask_image

    # compute a nice border around the mask
    border = mask - cv2.erode(mask, np.ones((5, 5), 'uint8'), iterations=1)

    # apply color to the mask and border
    mask = (np.stack([mask] * 3, axis=2) * color).astype('uint8')
    border = (np.stack([border] * 3, axis=2) * (255, 255, 255)).astype('uint8')

    # draw the mask
    indices = np.where(mask != [0, 0, 0])
    _mask = 0.5 * image[indices[0], indices[1], :] + \
        0.5 * mask[indices[0], indices[1], :]
    image[indices[0], indices[1], :] = _mask

    # draw the border
    indices = np.where(border != [0, 0, 0])
    _border = 0.2 * image[indices[0], indices[1], :] + \
        0.8 * border[indices[0], indices[1], :]
    image[indices[0], indices[1], :] = _border


def draw_masks(image, boxes, scores, masks,
               color=[31, 0, 255],
               score_threshold=0.5,
               binarize_threshold=0.5):
    """Draws a list of masks given a list of boxes.

    Args:
        image (numpy.array): Three dimensional image to draw on.
        boxes (list): Matrix of shape ``(N, >=4)``
            (at least 4 values: ``(x1, y1, x2, y2)``) representing boxes
            in the image.
        scores (list): A list of N classification scores.
        masks (numpy.array): Matrix of shape ``(N, H, W)`` of N masks of shape
            ``(H, W)`` which will be reshaped to the size of the corresponding
            box, binarized and drawn over the image.
        color (list): Color or to draw the masks with.
        score_threshold (float): Threshold used for determining
            the masks to draw.
        binarize_threshold (float): Threshold used for binarizing the masks.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        if not any(b == -1 for b in boxes[i]):
            draw_mask(image, boxes[i].astype(int), masks[i], color=color,
                      binarize_threshold=binarize_threshold)


def draw_detections(image,
                    boxes,
                    scores,
                    labels,
                    color=[31, 0, 255],
                    label_to_name=None,
                    score_threshold=0.5):
    """Draws detections in an image.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image (numpy.array): The image to draw on.
        boxes (list): A [N, 4] matrix ``(x1, y1, x2, y2)``.
        scores (list): A list of N classification scores.
        labels (list): A list of N labels.
        color (list): The color of the boxes.
        label_to_name (function): (optional) Functor for mapping a
            label to a name.
        score_threshold (float): Threshold used for determining
            the detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        draw_box(image, boxes[i, :], color=color)

        # draw labels
        name = label_to_name(labels[i]) if label_to_name else labels[i]
        caption = '{0}: {1:.2f}'.format(name, scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image,
                     annotations,
                     color=[31, 0, 255],
                     label_to_name=None):
    """Draws annotations in an image.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        image (numpy.array): The image to draw on.
        annotations (numpy.array): A [N, 5] matrix ``(x1, y1, x2, y2, label)``
            or dictionary containing bboxes (shaped ``[N, 4]``)
            and labels (shaped ``[N]``).
        color (list): The color of the boxes.
        label_to_name (function): (optional) Functor for mapping a
            label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert 'bboxes' in annotations
    assert 'labels' in annotations
    assert annotations['bboxes'].shape[0] == annotations['labels'].shape[0]

    for i in range(annotations['bboxes'].shape[0]):
        label = annotations['labels'][i]
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=color)
