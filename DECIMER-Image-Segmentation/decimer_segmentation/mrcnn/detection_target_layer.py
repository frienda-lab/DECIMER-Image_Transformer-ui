"""
Detection Target Layer for Mask R-CNN Training

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

This layer generates detection targets for training.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL

from . import utils


def overlaps_graph(boxes1, boxes2):
    """
    Compute IoU overlaps between two sets of boxes.

    Args:
        boxes1: [N, (y1, x1, y2, x2)]
        boxes2: [M, (y1, x1, y2, x2)]

    Returns:
        IoU matrix [N, M]
    """
    b1 = tf.reshape(
        tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4]
    )
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)

    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)

    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """
    Generate detection targets for training.

    Args:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] normalized
        gt_class_ids: [MAX_GT_INSTANCES] class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] normalized
        gt_masks: [height, width, MAX_GT_INSTANCES] masks
        config: Configuration object

    Returns:
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        class_ids: [TRAIN_ROIS_PER_IMAGE]
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        masks: [TRAIN_ROIS_PER_IMAGE, MASK_SHAPE[0], MASK_SHAPE[1]]
    """
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals]),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    gt_masks = tf.gather(
        gt_masks, tf.compat.v1.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks"
    )

    # Handle COCO crowd annotations
    crowd_ix = tf.compat.v1.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.compat.v1.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps
    overlaps = overlaps_graph(proposals, gt_boxes)
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = crowd_iou_max < 0.001

    # Determine positive/negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)

    positive_roi_bool = roi_iou_max >= 0.5
    positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]

    negative_indices = tf.compat.v1.where(
        tf.logical_and(roi_iou_max < 0.5, no_crowd_bool)
    )[:, 0]

    # Subsample ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]

    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = (
        tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    )
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64),
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)

    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(
        tf.cast(roi_masks, tf.float32), boxes, box_ids, config.MASK_SHAPE
    )
    masks = tf.squeeze(masks, axis=3)
    masks = tf.round(masks)

    # Append negative ROIs and pad
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)

    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], [0, 0], [0, 0]])

    return rois, roi_gt_class_ids, deltas, masks


def trim_zeros_graph(boxes, name="trim_zeros"):
    """
    Remove zero-padded boxes.
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


class DetectionTargetLayer(KL.Layer):
    """
    Generate detection targets for training.

    Subsamples proposals and generates target class IDs, bounding box
    deltas, and masks for each.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice batch
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU,
            names=names,
        )
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (
                None,
                self.config.TRAIN_ROIS_PER_IMAGE,
                self.config.MASK_SHAPE[0],
                self.config.MASK_SHAPE[1],
            ),  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]
