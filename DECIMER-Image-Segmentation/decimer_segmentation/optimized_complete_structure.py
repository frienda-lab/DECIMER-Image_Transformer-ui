"""
DECIMER Segmentation - Mask Expansion Module

Optimized implementation for expanding chemical structure masks to capture
complete molecular structures from scientific literature images.

This module replaces both complete_structure.py and optimized_complete_structure.py
with a unified, production-optimized implementation.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def complete_structure_mask(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    max_depiction_size: Tuple[int, int],
    debug: bool = False,
) -> np.ndarray:
    """
    Expand masks to capture complete chemical structures.

    This function takes initial detection masks and expands them to include
    any parts of the chemical structure that may have been missed, while
    avoiding inclusion of tables, lines, or other non-structure elements.

    Args:
        image_array: Input image as numpy array (BGR format)
        mask_array: Initial masks from MRCNN, shape (height, width, num_masks)
        max_depiction_size: Tuple of (max_height, max_width) for structure sizing
        debug: If True, display intermediate results (requires matplotlib)

    Returns:
        Expanded masks array of shape (height, width, num_masks)
    """
    if mask_array.size == 0 or mask_array.shape[2] == 0:
        return mask_array

    # Step 1: Binarize the image
    binarized = _binarize_image_fast(image_array, threshold=0.72)

    if debug:
        _debug_plot(binarized, "Binarized Image")

    # Step 2: Apply erosion to clean up the image
    blur_factor = max(2, image_array.shape[1] // 185)
    kernel = np.ones((blur_factor, blur_factor), dtype=np.uint8)

    # Use OpenCV for faster morphological operations
    eroded = cv2.erode(binarized.astype(np.uint8) * 255, kernel, iterations=1) > 127

    if debug:
        _debug_plot(eroded, "Eroded Image")

    # Step 3: Detect exclusion regions (lines, tables, etc.)
    exclusion_mask = _create_exclusion_mask(
        binarized=binarized,
        eroded=eroded,
        mask_array=mask_array,
        max_depiction_size=max_depiction_size,
        kernel=kernel,
        debug=debug,
    )

    # Step 4: Create the working image with exclusions applied
    working_image = eroded.copy()
    working_image[exclusion_mask] = True  # Set exclusion regions to white (background)

    if debug:
        _debug_plot(working_image, "Working Image with Exclusions")

    # Step 5: Expand each mask
    num_masks = mask_array.shape[2]

    if num_masks <= 3:
        # Sequential processing for small numbers
        expanded_masks = [
            _expand_single_mask(mask_array[:, :, i], working_image, exclusion_mask)
            for i in range(num_masks)
        ]
    else:
        # Parallel processing for larger numbers
        expanded_masks = _expand_masks_parallel(
            mask_array, working_image, exclusion_mask
        )

    # Step 6: Filter duplicates efficiently
    unique_masks = _filter_duplicate_masks(expanded_masks)

    if not unique_masks:
        return np.empty((image_array.shape[0], image_array.shape[1], 0), dtype=bool)

    return np.stack(unique_masks, axis=-1)


def _binarize_image_fast(image: np.ndarray, threshold: float = 0.72) -> np.ndarray:
    """
    Fast image binarization using OpenCV.

    Args:
        image: Input BGR image
        threshold: Binarization threshold (0-1)

    Returns:
        Binary image (True = white/background, False = dark/foreground)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Normalize to 0-1 range
    normalized = gray.astype(np.float32) / 255.0

    return normalized > threshold


def _create_exclusion_mask(
    binarized: np.ndarray,
    eroded: np.ndarray,
    mask_array: np.ndarray,
    max_depiction_size: Tuple[int, int],
    kernel: np.ndarray,
    debug: bool = False,
) -> np.ndarray:
    """
    Create a mask of regions to exclude from expansion (lines, tables, etc.).

    Args:
        binarized: Binarized image
        eroded: Eroded image
        mask_array: Original detection masks
        max_depiction_size: Max structure size for line detection thresholds
        kernel: Morphological kernel
        debug: Whether to display debug plots

    Returns:
        Boolean mask where True indicates exclusion regions
    """
    # Detect horizontal and vertical lines
    hv_lines = _detect_horizontal_vertical_lines(eroded, max_depiction_size)

    if debug:
        _debug_plot(hv_lines, "Horizontal/Vertical Lines")

    # Detect arbitrary lines using Hough transform
    segmentation_mask = np.any(mask_array, axis=2)
    hough_lines = _detect_hough_lines(binarized, max_depiction_size, segmentation_mask)

    # Dilate Hough lines to create buffer zone
    if hough_lines.any():
        hough_lines = (
            cv2.dilate(hough_lines.astype(np.uint8) * 255, kernel, iterations=1) > 127
        )

    if debug:
        _debug_plot(hough_lines, "Hough Lines (dilated)")

    # Combine exclusion masks
    exclusion_mask = hv_lines | hough_lines

    if debug:
        _debug_plot(exclusion_mask, "Combined Exclusion Mask")

    return exclusion_mask


def _detect_horizontal_vertical_lines(
    image: np.ndarray, max_depiction_size: Tuple[int, int]
) -> np.ndarray:
    """
    Detect long horizontal and vertical lines (table borders, separators).

    Args:
        image: Binarized/eroded image (True = white)
        max_depiction_size: (height, width) for line length thresholds

    Returns:
        Boolean mask of detected lines
    """
    # Convert to uint8 for OpenCV (invert so lines are white)
    img_uint8 = (~image).astype(np.uint8) * 255

    structure_height, structure_width = max_depiction_size

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (structure_width, 1))
    horizontal_mask = cv2.morphologyEx(
        img_uint8, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, structure_height))
    vertical_mask = cv2.morphologyEx(
        img_uint8, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Combine and return as boolean
    return (horizontal_mask > 127) | (vertical_mask > 127)


def _detect_hough_lines(
    binarized: np.ndarray,
    max_depiction_size: Tuple[int, int],
    segmentation_mask: np.ndarray,
) -> np.ndarray:
    """
    Detect arbitrary lines using probabilistic Hough transform.

    Args:
        binarized: Binarized image
        max_depiction_size: For minimum line length threshold
        segmentation_mask: Mask indicating detected structure regions

    Returns:
        Boolean mask of detected lines
    """
    # Convert to uint8 (invert so lines are white)
    img_uint8 = (~binarized).astype(np.uint8) * 255

    # Detect lines
    min_line_length = max(max_depiction_size) // 4
    lines = cv2.HoughLinesP(
        img_uint8,
        rho=1,
        theta=np.pi / 180,
        threshold=10,  # Slightly higher threshold to reduce noise
        minLineLength=min_line_length,
        maxLineGap=10,
    )

    if lines is None:
        return np.zeros_like(binarized, dtype=bool)

    # Create exclusion mask
    exclusion = np.zeros_like(img_uint8, dtype=np.uint8)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if line passes through structure regions
        if _line_intersects_structure(x1, y1, x2, y2, segmentation_mask):
            continue

        # Draw line on exclusion mask
        cv2.line(exclusion, (x1, y1), (x2, y2), 255, 2)

    return exclusion > 127


def _line_intersects_structure(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    segmentation_mask: np.ndarray,
    num_samples: int = 7,
) -> bool:
    """
    Check if a line intersects with structure regions.

    Args:
        x1, y1: Start point
        x2, y2: End point
        segmentation_mask: Mask of structure regions
        num_samples: Number of points to sample along line

    Returns:
        True if line intersects structure regions
    """
    h, w = segmentation_mask.shape

    # Sample points along the line (excluding endpoints)
    for i in range(1, num_samples - 1):
        t = i / (num_samples - 1)
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))

        # Bounds check
        if 0 <= x < w and 0 <= y < h:
            if segmentation_mask[y, x]:
                return True

    return False


def _expand_single_mask(
    mask: np.ndarray, working_image: np.ndarray, exclusion_mask: np.ndarray
) -> np.ndarray:
    """
    Expand a single mask to capture complete structure.

    Args:
        mask: Binary mask for a single structure
        working_image: Processed image (True = background)
        exclusion_mask: Regions to exclude from expansion

    Returns:
        Expanded binary mask
    """
    # Find seed pixels within the mask
    seeds = _get_seed_pixels(mask, working_image, exclusion_mask)

    if not seeds:
        return mask

    # Use connected components to expand from seeds
    return _flood_fill_from_seeds(working_image, seeds)


def _get_seed_pixels(
    mask: np.ndarray, image: np.ndarray, exclusion_mask: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Find seed pixels for flood fill expansion.

    Seeds are pixels that are:
    - Within the inner 80% of the mask
    - On dark (foreground) pixels in the image
    - Not in exclusion regions

    Args:
        mask: Binary mask
        image: Working image (True = background)
        exclusion_mask: Exclusion regions

    Returns:
        List of (x, y) seed coordinates
    """
    # Find mask bounds
    mask_coords = np.where(mask)
    if len(mask_coords[0]) == 0:
        return []

    y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
    x_min, x_max = mask_coords[1].min(), mask_coords[1].max()

    # Calculate inner 80% bounds
    y_margin = (y_max - y_min) * 0.1
    x_margin = (x_max - x_min) * 0.1

    inner_y_min = int(y_min + y_margin)
    inner_y_max = int(y_max - y_margin)
    inner_x_min = int(x_min + x_margin)
    inner_x_max = int(x_max - x_margin)

    # Find valid seed pixels using vectorized operations
    # Create a combined mask for valid seed regions
    valid_region = np.zeros_like(mask, dtype=bool)
    valid_region[inner_y_min : inner_y_max + 1, inner_x_min : inner_x_max + 1] = True

    # Combine conditions: in mask, in valid region, on dark pixel, not excluded
    seed_mask = (
        mask & valid_region & (~image) & (~exclusion_mask)  # Dark pixels (foreground)
    )

    # Extract coordinates
    seed_coords = np.where(seed_mask)

    # Convert to list of (x, y) tuples
    # Limit to reasonable number of seeds for performance
    max_seeds = 1000
    step = max(1, len(seed_coords[0]) // max_seeds)

    seeds = [
        (seed_coords[1][i], seed_coords[0][i])
        for i in range(0, len(seed_coords[0]), step)
    ]

    return seeds


def _flood_fill_from_seeds(
    image: np.ndarray, seeds: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Perform flood fill from seed pixels using connected components.

    Args:
        image: Working image (True = background)
        seeds: List of (x, y) seed coordinates

    Returns:
        Expanded binary mask
    """
    # Create inverted image for connected components (foreground = 255)
    foreground = (~image).astype(np.uint8)

    # Find connected components
    num_labels, labels = cv2.connectedComponents(foreground, connectivity=8)

    # Find labels at seed positions
    seed_labels = set()
    for x, y in seeds:
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            label = labels[y, x]
            if label > 0:  # Ignore background (label 0)
                seed_labels.add(label)

    # Create expanded mask from seed labels
    expanded_mask = np.zeros_like(image, dtype=bool)
    for label in seed_labels:
        expanded_mask |= labels == label

    return expanded_mask


def _expand_masks_parallel(
    mask_array: np.ndarray, working_image: np.ndarray, exclusion_mask: np.ndarray
) -> List[np.ndarray]:
    """
    Expand multiple masks in parallel.

    Args:
        mask_array: Array of masks, shape (h, w, num_masks)
        working_image: Processed image
        exclusion_mask: Exclusion regions

    Returns:
        List of expanded masks
    """
    num_masks = mask_array.shape[2]

    def expand_mask_wrapper(i: int) -> Tuple[int, np.ndarray]:
        expanded = _expand_single_mask(
            mask_array[:, :, i], working_image, exclusion_mask
        )
        return i, expanded

    expanded_masks = [None] * num_masks

    with ThreadPoolExecutor(max_workers=min(4, num_masks)) as executor:
        futures = [executor.submit(expand_mask_wrapper, i) for i in range(num_masks)]
        for future in futures:
            i, expanded = future.result()
            expanded_masks[i] = expanded

    return expanded_masks


def _filter_duplicate_masks(masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Remove duplicate masks efficiently.

    Uses a hash-based approach with collision detection for accuracy.

    Args:
        masks: List of binary masks

    Returns:
        List of unique masks
    """
    if not masks:
        return []

    unique_masks = []
    seen_hashes = {}

    for mask in masks:
        if mask is None or mask.size == 0:
            continue

        # Compute hash
        mask_bytes = mask.tobytes()
        mask_hash = hash(mask_bytes)

        # Check for collision
        if mask_hash in seen_hashes:
            # Verify it's actually a duplicate (handle hash collisions)
            is_duplicate = False
            for existing_mask in seen_hashes[mask_hash]:
                if np.array_equal(mask, existing_mask):
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_hashes[mask_hash].append(mask)
                unique_masks.append(mask)
        else:
            seen_hashes[mask_hash] = [mask]
            unique_masks.append(mask)

    return unique_masks


def _debug_plot(image: np.ndarray, title: str = "") -> None:
    """
    Display an image for debugging purposes.

    Args:
        image: Image to display
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()
    except ImportError:
        pass  # Matplotlib not available


# Backward compatibility aliases
def binarize_image(image_array: np.ndarray, threshold="otsu") -> np.ndarray:
    """Backward compatible binarize function."""
    if threshold == "otsu":
        threshold = 0.72  # Approximate Otsu for typical document images
    return _binarize_image_fast(image_array, float(threshold))


def expand_masks(
    image_array: np.ndarray, seed_pixels: List[Tuple[int, int]], mask_array: np.ndarray
) -> np.ndarray:
    """Backward compatible expand_masks function."""
    return _flood_fill_from_seeds(~_binarize_image_fast(image_array), seed_pixels)
