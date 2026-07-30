"""
Mask R-CNN Configuration

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified for DECIMER Segmentation 2024
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Callable


class Config:
    """
    Base configuration class for Mask R-CNN.

    Subclass this and override settings as needed.
    """

    # Configuration name
    NAME: Optional[str] = None

    # GPU settings
    GPU_COUNT: int = 1
    IMAGES_PER_GPU: int = 2

    # Training settings
    STEPS_PER_EPOCH: int = 1000
    VALIDATION_STEPS: int = 50

    # Backbone architecture: "resnet50" or "resnet101"
    BACKBONE: str = "resnet101"
    COMPUTE_BACKBONE_SHAPE: Optional[Callable] = None
    BACKBONE_STRIDES: List[int] = [4, 8, 16, 32, 64]

    # FPN settings
    FPN_CLASSIF_FC_LAYERS_SIZE: int = 1024
    TOP_DOWN_PYRAMID_SIZE: int = 256

    # Number of classes (including background)
    NUM_CLASSES: int = 1

    # RPN settings
    RPN_ANCHOR_SCALES: tuple = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS: List[float] = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE: int = 1
    RPN_NMS_THRESHOLD: float = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE: int = 256

    # Proposal settings
    PRE_NMS_LIMIT: int = 6000
    POST_NMS_ROIS_TRAINING: int = 2000
    POST_NMS_ROIS_INFERENCE: int = 1000

    # Mask settings
    USE_MINI_MASK: bool = True
    MINI_MASK_SHAPE: tuple = (56, 56)

    # Image settings
    IMAGE_RESIZE_MODE: str = "square"
    IMAGE_MIN_DIM: int = 800
    IMAGE_MAX_DIM: int = 1024
    IMAGE_MIN_SCALE: float = 0
    IMAGE_CHANNEL_COUNT: int = 3
    MEAN_PIXEL: np.ndarray = np.array([123.7, 116.8, 103.9])

    # Training ROI settings
    TRAIN_ROIS_PER_IMAGE: int = 200
    ROI_POSITIVE_RATIO: float = 0.33

    # Pooling settings
    POOL_SIZE: int = 7
    MASK_POOL_SIZE: int = 14
    MASK_SHAPE: List[int] = [28, 28]

    # Instance limits
    MAX_GT_INSTANCES: int = 100

    # Bounding box refinement
    RPN_BBOX_STD_DEV: np.ndarray = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV: np.ndarray = np.array([0.1, 0.1, 0.2, 0.2])

    # Detection settings
    DETECTION_MAX_INSTANCES: int = 100
    DETECTION_MIN_CONFIDENCE: float = 0.7
    DETECTION_NMS_THRESHOLD: float = 0.3

    # Learning settings
    LEARNING_RATE: float = 0.001
    LEARNING_MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0001

    # Loss weights
    LOSS_WEIGHTS: Dict[str, float] = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "mrcnn_class_loss": 1.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.0,
    }

    # Training mode settings
    USE_RPN_ROIS: bool = True
    TRAIN_BN: bool = False
    GRADIENT_CLIP_NORM: float = 5.0

    def __init__(self):
        """Compute derived settings."""
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array(
                [self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.IMAGE_CHANNEL_COUNT]
            )
        else:
            self.IMAGE_SHAPE = np.array(
                [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT]
            )

        # Image metadata size
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in sorted(dir(self))
            if not attr.startswith("__") and not callable(getattr(self, attr))
        }

    def display(self) -> None:
        """Print configuration settings."""
        print("\nConfiguration:")
        print("-" * 50)
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        print("-" * 50)
        print()
