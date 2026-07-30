"""
Molecule Detection Configuration for DECIMER Segmentation

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Modified for DECIMER Segmentation by Kohulan Rajan 2020
"""

from __future__ import annotations

from .config import Config


class MolDetectConfig(Config):
    """
    Configuration for chemical structure detection.

    Derives from base Config and customizes for molecule detection.
    """

    # Configuration name
    NAME = "Molecule"

    # GPU settings - adjust based on available hardware
    IMAGES_PER_GPU = 2

    # Number of classes: background + molecule
    NUM_CLASSES = 1 + 1

    # Training settings
    STEPS_PER_EPOCH = 100

    # Detection confidence threshold
    DETECTION_MIN_CONFIDENCE = 0.9

    def __init__(self):
        super().__init__()
