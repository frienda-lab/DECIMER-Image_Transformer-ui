"""
Mask R-CNN - Optimized Utility Functions

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified and optimized for DECIMER Segmentation 2024
"""

from __future__ import annotations

import logging
import random
import urllib.request
import shutil
import warnings
from typing import List, Tuple, Optional, Callable, Any

import numpy as np
import tensorflow as tf
import cv2
import scipy.ndimage

# URL for model weights
COCO_MODEL_URL = "https://zenodo.org/records/18412030/files/mask_rcnn_molecule.h5?download=1"

logger = logging.getLogger(__name__)


# =============================================================================
# Bounding Box Operations
# =============================================================================


def extract_bboxes(mask: np.ndarray) -> np.ndarray:
    """
    Compute bounding boxes from masks using vectorized operations.

    Args:
        mask: [height, width, num_instances] mask array

    Returns:
        bbox array [num_instances, (y1, x1, y2, x2)]
    """
    num_instances = mask.shape[-1]
    boxes = np.zeros([num_instances, 4], dtype=np.int32)

    for i in range(num_instances):
        m = mask[:, :, i]

        # Find bounding box using vectorized operations
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)

        if rows.any() and cols.any():
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y1, y2 = y_indices[0], y_indices[-1] + 1
            x1, x2 = x_indices[0], x_indices[-1] + 1
            boxes[i] = [y1, x1, y2, x2]

    return boxes


def compute_iou(
    box: np.ndarray, boxes: np.ndarray, box_area: float, boxes_area: np.ndarray
) -> np.ndarray:
    """
    Calculate IoU of given box with array of boxes.

    Args:
        box: [y1, x1, y2, x2]
        boxes: [N, (y1, x1, y2, x2)]
        box_area: Area of the box
        boxes_area: Array of areas for boxes

    Returns:
        IoU values for each box
    """
    # Calculate intersection
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area - intersection

    return intersection / union


def compute_overlaps(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU overlaps between two sets of boxes.

    Args:
        boxes1: [N, (y1, x1, y2, x2)]
        boxes2: [M, (y1, x1, y2, x2)]

    Returns:
        Overlap matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    for i in range(boxes2.shape[0]):
        overlaps[:, i] = compute_iou(boxes2[i], boxes1, area2[i], area1)

    return overlaps


def compute_overlaps_masks(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    """
    Compute IoU overlaps between two sets of masks.

    Args:
        masks1: [Height, Width, instances1]
        masks2: [Height, Width, instances2]

    Returns:
        Overlap matrix [instances1, instances2]
    """
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))

    # Flatten and compute
    masks1_flat = masks1.reshape(-1, masks1.shape[-1]).astype(np.float32) > 0.5
    masks2_flat = masks2.reshape(-1, masks2.shape[-1]).astype(np.float32) > 0.5

    area1 = np.sum(masks1_flat, axis=0)
    area2 = np.sum(masks2_flat, axis=0)

    intersections = np.dot(masks1_flat.T, masks2_flat)
    union = area1[:, None] + area2[None, :] - intersections

    return intersections / (union + 1e-10)


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Perform non-maximum suppression.

    Args:
        boxes: [N, (y1, x1, y2, x2)]
        scores: [N] confidence scores
        threshold: IoU threshold

    Returns:
        Indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)

    boxes = boxes.astype(np.float32)

    # Compute areas
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Sort by score
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)

        if len(indices) == 1:
            break

        # Compute IoU with remaining boxes
        iou = compute_iou(boxes[i], boxes[indices[1:]], area[i], area[indices[1:]])

        # Remove overlapping boxes
        remove_mask = iou > threshold
        indices = indices[1:][~remove_mask]

    return np.array(keep, dtype=np.int32)


def apply_box_deltas(boxes: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """
    Apply bounding box deltas.

    Args:
        boxes: [N, (y1, x1, y2, x2)]
        deltas: [N, (dy, dx, log(dh), log(dw))]

    Returns:
        Refined boxes [N, (y1, x1, y2, x2)]
    """
    boxes = boxes.astype(np.float32)

    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width

    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])

    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width

    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box: tf.Tensor, gt_box: tf.Tensor) -> tf.Tensor:
    """
    Compute refinement needed to transform box to gt_box (TensorFlow graph).
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    return tf.stack([dy, dx, dh, dw], axis=1)


def box_refinement(box: np.ndarray, gt_box: np.ndarray) -> np.ndarray:
    """
    Compute refinement needed to transform box to gt_box (NumPy).
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


# =============================================================================
# Dataset Base Class
# =============================================================================


class Dataset:
    """Base class for datasets."""

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source: str, class_id: int, class_name: str) -> None:
        """Add a class to the dataset."""
        assert "." not in source, "Source name cannot contain a dot"

        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                return

        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
            }
        )

    def add_image(self, source: str, image_id: int, path: str, **kwargs) -> None:
        """Add an image to the dataset."""
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id: int) -> str:
        """Return a reference to the image."""
        return ""

    def prepare(self, class_map=None) -> None:
        """Prepare the dataset for use."""

        def clean_name(name):
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.class_info, self.class_ids)
        }

        self.image_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        self.sources = list(set([i["source"] for i in self.class_info]))
        self.source_class_ids = {}

        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id: str) -> int:
        """Map source class ID to internal class ID."""
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id: int, source: str) -> int:
        """Get source class ID from internal class ID."""
        info = self.class_info[class_id]
        assert info["source"] == source
        return info["id"]

    @property
    def image_ids(self) -> np.ndarray:
        return self._image_ids

    def source_image_link(self, image_id: int) -> str:
        """Return path to image."""
        return self.image_info[image_id]["path"]

    def load_image(self, image_id: int) -> np.ndarray:
        """Load image using OpenCV (faster than skimage)."""
        path = self.image_info[image_id]["path"]
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not load image: {path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def load_mask(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load instance masks. Override in subclass."""
        logger.warning("Using default load_mask(), define your own.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], dtype=np.int32)
        return mask, class_ids


# =============================================================================
# Image Processing
# =============================================================================


def resize_image(
    image: np.ndarray,
    min_dim: Optional[int] = None,
    max_dim: Optional[int] = None,
    min_scale: Optional[float] = None,
    mode: str = "square",
) -> Tuple[np.ndarray, Tuple[int, int, int, int], float, List, Optional[Tuple]]:
    """
    Resize image using OpenCV (faster than skimage).

    Args:
        image: Input image
        min_dim: Minimum dimension
        max_dim: Maximum dimension
        min_scale: Minimum scale factor
        mode: Resize mode ('none', 'square', 'pad64', 'crop')

    Returns:
        Tuple of (resized_image, window, scale, padding, crop)
    """
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Calculate scale
    if min_dim:
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize using OpenCV (much faster than skimage)
    if scale != 1:
        new_h, new_w = round(h * scale), round(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Handle padding/cropping
    if mode == "square":
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

    elif mode == "pad64":
        h, w = image.shape[:2]
        assert min_dim % 64 == 0, "min_dim must be divisible by 64"

        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0

        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0

        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

    elif mode == "crop":
        h, w = image.shape[:2]
        y = random.randint(0, h - min_dim)
        x = random.randint(0, w - min_dim)
        crop = (y, x, min_dim, min_dim)
        image = image[y : y + min_dim, x : x + min_dim]
        window = (0, 0, min_dim, min_dim)

    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(
    mask: np.ndarray,
    scale: float,
    padding: List[Tuple[int, int]],
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Resize mask to match image resizing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)

    if crop is not None:
        y, x, h, w = crop
        mask = mask[y : y + h, x : x + w]
    else:
        mask = np.pad(mask, padding, mode="constant", constant_values=0)

    return mask


def resize(
    image: np.ndarray,
    output_shape: Tuple[int, int],
    order: int = 1,
    mode: str = "constant",
    cval: float = 0,
    clip: bool = True,
    preserve_range: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Resize image using OpenCV (faster than skimage).
    """
    # Map order to interpolation method
    if order == 0:
        interpolation = cv2.INTER_NEAREST
    elif order == 1:
        interpolation = cv2.INTER_LINEAR
    elif order == 3:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LANCZOS4

    # Handle output shape (OpenCV uses width, height)
    output_size = (output_shape[1], output_shape[0])

    resized = cv2.resize(image, output_size, interpolation=interpolation)

    if clip:
        if image.dtype == bool:
            resized = resized > 0.5
        elif np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            resized = np.clip(resized, info.min, info.max)

    return resized


def minimize_mask(
    bbox: np.ndarray, mask: np.ndarray, mini_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize masks to smaller size to reduce memory.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)

    for i in range(mask.shape[-1]):
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]

        if m.size == 0:
            raise ValueError("Invalid bounding box with zero area")

        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(bool)

    return mini_mask


def expand_mask(
    bbox: np.ndarray, mini_mask: np.ndarray, image_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Expand mini masks back to image size.
    """
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)

    for i in range(mask.shape[-1]):
        m = mini_mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        h, w = y2 - y1, x2 - x1
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(bool)

    return mask


def unmold_mask(
    mask: np.ndarray, bbox: np.ndarray, image_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Convert neural network mask to full size.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox

    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = (mask >= threshold).astype(bool)

    full_mask = np.zeros(image_shape[:2], dtype=bool)
    full_mask[y1:y2, x1:x2] = mask

    return full_mask


# =============================================================================
# Anchor Generation
# =============================================================================


def generate_anchors(
    scales: np.ndarray,
    ratios: np.ndarray,
    shape: Tuple[int, int],
    feature_stride: int,
    anchor_stride: int,
) -> np.ndarray:
    """
    Generate anchor boxes.
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape(-1, 2)
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape(-1, 2)

    boxes = np.concatenate(
        [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1
    )

    return boxes


def generate_pyramid_anchors(
    scales: List[int],
    ratios: List[float],
    feature_shapes: np.ndarray,
    feature_strides: List[int],
    anchor_stride: int,
) -> np.ndarray:
    """
    Generate anchors for feature pyramid.
    """
    anchors = []
    for i in range(len(scales)):
        anchors.append(
            generate_anchors(
                scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride
            )
        )
    return np.concatenate(anchors, axis=0)


# =============================================================================
# Evaluation Utilities
# =============================================================================


def trim_zeros(x: np.ndarray) -> np.ndarray:
    """Remove zero-padded rows."""
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(
    gt_boxes: np.ndarray,
    gt_class_ids: np.ndarray,
    gt_masks: np.ndarray,
    pred_boxes: np.ndarray,
    pred_class_ids: np.ndarray,
    pred_scores: np.ndarray,
    pred_masks: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find matches between predictions and ground truth.
    """
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[:, :, : gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[: pred_boxes.shape[0]]

    # Sort by score
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[:, :, indices]

    # Compute overlaps
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Match
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])

    for i in range(len(pred_boxes)):
        sorted_ixs = np.argsort(overlaps[i])[::-1]

        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[: low_score_idx[0]]

        for j in sorted_ixs:
            if gt_match[j] > -1:
                continue
            if overlaps[i, j] < iou_threshold:
                break
            if pred_class_ids[i] == gt_class_ids[j]:
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(
    gt_boxes: np.ndarray,
    gt_class_ids: np.ndarray,
    gt_masks: np.ndarray,
    pred_boxes: np.ndarray,
    pred_class_ids: np.ndarray,
    pred_scores: np.ndarray,
    pred_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Average Precision.
    """
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes,
        gt_class_ids,
        gt_masks,
        pred_boxes,
        pred_class_ids,
        pred_scores,
        pred_masks,
        iou_threshold,
    )

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return mAP, precisions, recalls, overlaps


# =============================================================================
# Batch Processing
# =============================================================================


def batch_slice(
    inputs: List, graph_fn: Callable, batch_size: int, names: Optional[List[str]] = None
) -> Any:
    """
    Process inputs in batches.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n) for o, n in zip(outputs, names)]

    if len(result) == 1:
        result = result[0]

    return result


# =============================================================================
# Model Utilities
# =============================================================================


def download_trained_weights(coco_model_path: str, verbose: int = 1) -> None:
    """
    Download pretrained model weights.
    """
    if verbose > 0:
        logger.info(f"Downloading pretrained model to {coco_model_path}...")

    with urllib.request.urlopen(COCO_MODEL_URL) as resp:
        with open(coco_model_path, "wb") as out:
            shutil.copyfileobj(resp, out)

    if verbose > 0:
        logger.info("Download complete!")


def norm_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert boxes from pixel to normalized coordinates.
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return ((boxes - shift) / scale).astype(np.float32)


def denorm_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert boxes from normalized to pixel coordinates.
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(boxes * scale + shift).astype(np.int32)
