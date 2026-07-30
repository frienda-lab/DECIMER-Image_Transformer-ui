"""
Mask R-CNN Visualization Functions

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Modified for DECIMER Segmentation 2024
- Optimized imports (lazy loading)
- Streamlined for production use
"""

from __future__ import annotations

import random
import colorsys
from typing import List, Tuple, Optional, Any

import numpy as np


def random_colors(N: int, bright: bool = True) -> List[Tuple[float, float, float]]:
    """
    Generate random visually distinct colors.

    Args:
        N: Number of colors to generate
        bright: Whether to use bright colors

    Returns:
        List of RGB color tuples (0-1 range)
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    random.shuffle(colors)
    return colors


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Apply colored mask to image.

    Args:
        image: Input image (will be modified in place)
        mask: Binary mask
        color: RGB color tuple (0-1 range)
        alpha: Transparency (0-1)

    Returns:
        Image with mask applied
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return image


def display_instances(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    class_ids: np.ndarray,
    class_names: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = "",
    figsize: Tuple[int, int] = (16, 16),
    ax: Optional[Any] = None,
    show_mask: bool = True,
    show_bbox: bool = True,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    captions: Optional[List[str]] = None,
) -> None:
    """
    Display detected instances on image.

    Args:
        image: Input image
        boxes: [N, (y1, x1, y2, x2)] bounding boxes
        masks: [height, width, N] instance masks
        class_ids: [N] class IDs
        class_names: List of class names
        scores: [N] confidence scores (optional)
        title: Figure title
        figsize: Figure size
        ax: Matplotlib axis (optional)
        show_mask: Whether to show masks
        show_bbox: Whether to show bounding boxes
        colors: Custom colors (optional)
        captions: Custom captions (optional)
    """
    # Lazy import matplotlib
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.patches import Polygon
        from skimage.measure import find_contours
    except ImportError:
        print("Visualization requires matplotlib and scikit-image")
        return

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display ***\n")
        return

    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    auto_show = False
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    colors = colors or random_colors(N)

    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]

        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]

        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        # Caption
        if captions:
            caption = captions[i]
        else:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = f"{label} {score:.3f}" if score else label

        ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask contour
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))

    if auto_show:
        plt.show()


def display_images(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    cmap: Optional[str] = None,
    norm: Optional[Any] = None,
    interpolation: Optional[str] = None,
) -> None:
    """
    Display multiple images in a grid.

    Args:
        images: List of images
        titles: List of titles (optional)
        cols: Number of columns
        cmap: Colormap (optional)
        norm: Normalization (optional)
        interpolation: Interpolation method (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Visualization requires matplotlib")
        return

    titles = titles or [""] * len(images)
    rows = (len(images) + cols - 1) // cols

    plt.figure(figsize=(14, 14 * rows // cols))

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.title(title, fontsize=9)
        plt.axis("off")
        plt.imshow(
            image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation
        )

    plt.show()
