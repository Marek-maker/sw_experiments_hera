import cv2
import numpy as np
from typing import List, Tuple


def _largest_rectangle_histogram(heights: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Finds the maximum area rectangle under a 1D histogram using a stack approach.
    Returns: (max_area, left_col, right_col, height)
    """
    stack = []
    max_area = 0
    best_left, best_right, best_h = 0, 0, 0

    # Append 0 to flush remaining elements from stack at the end
    extended = np.append(heights, 0)

    for i, h in enumerate(extended):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            area = height * (i - idx)
            if area > max_area:
                max_area = area
                best_left = idx
                best_right = i
                best_h = height
            start = idx
        stack.append((start, h))

    return max_area, best_left, best_right, best_h


def _find_largest_rectangle_in_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Finds the largest contiguous sub-rectangle of True values in a 2D boolean grid.
    Returns: (row, col, height, width) in grid coordinates.
    """
    grid_h, grid_w = mask.shape
    heights = np.zeros(grid_w, dtype=int)

    best_area = 0
    best_rect = (0, 0, 0, 0)  # r, c, h, w

    for r in range(grid_h):
        for c in range(grid_w):
            heights[c] = heights[c] + 1 if mask[r, c] else 0

        area, left, right, h = _largest_rectangle_histogram(heights)
        if area > best_area:
            best_area = area
            w = right - left
            top_r = r - h + 1
            best_rect = (top_r, left, h, w)

    return best_rect


def merge_tiles(
    score_grid: np.ndarray,
    tile_size: int = 16,
    min_score_threshold: float = 0.5,
    bridge_gaps: bool = True
) -> List[Tuple[int, int, int, int, float]]:
    """
    Merges high-interest tiles into optimal, non-overlapping rectangles.

    Args:
        score_grid: 2D numpy array of tile scores in range [0.0, 1.0]
        tile_size: Pixel dimension per atomic tile (e.g., 16x16)
        min_score_threshold: Minimum tile score to qualify for transmission
        bridge_gaps: If True, bridges 1-tile gaps between clusters to save metadata headers

    Returns:
        List of priority-sorted tuples: [(pixel_x, pixel_y, width, height, avg_score), ...]
    """
    grid_h, grid_w = score_grid.shape
    
    # 1. Create binary mask of candidate tiles
    target_mask = (score_grid >= min_score_threshold).astype(np.uint8)

    # 2. Morphological Closing (Bridge isolated 1-tile gaps between clusters)
    if bridge_gaps:
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        # Dilate then Erode to close internal holes/gaps
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)

    working_mask = target_mask.astype(bool)
    rectangles = []

    # 3. Iteratively extract largest valid sub-rectangles until no target tiles remain
    while np.any(working_mask):
        r, c, h, w = _find_largest_rectangle_in_mask(working_mask)

        # Safety break if no valid rectangle remains
        if h == 0 or w == 0:
            break

        # Zero out extracted rectangle from mask to prevent overlaps
        working_mask[r:r + h, c:c + w] = False

        # Calculate average priority score of original tile regions inside rectangle
        region_scores = score_grid[r:r + h, c:c + w]
        avg_score = float(np.mean(region_scores))

        # Convert tile grid space -> pixel space
        pixel_x = c * tile_size
        pixel_y = r * tile_size
        pixel_w = w * tile_size
        pixel_h = h * tile_size

        rectangles.append((pixel_x, pixel_y, pixel_w, pixel_h, avg_score))

    # 4. Sort rectangles by average score (highest interest first)
    rectangles.sort(key=lambda item: item[4], reverse=True)
    return rectangles


import numpy as np
from typing import List, Tuple


import numpy as np
from typing import List, Tuple

def merge_tiles_q(
    score_grid: np.ndarray,
    tile_size: int = 16,
    min_score_threshold: float = 0.4,
    max_score_std: float = 0.15,     # Linear scale: 0.15 = 15% allowed deviation (much more stable)
    max_block_tiles: int = 8          # Maximum block size in tiles (e.g., 8x8 tiles = 128x128 pixels)
) -> List[Tuple[int, int, int, int, float]]:
    """
    Robust Quadtree merging with linear standard deviation thresholding 
    and bandwidth-aware pruning splits.
    """
    rectangles = []

    def quadtree_split(r: int, c: int, h: int, w: int):
        region = score_grid[r : r + h, c : c + w]
        max_val = np.max(region)
        min_val = np.min(region)
        avg_val = float(np.mean(region))
        std_dev = float(np.std(region))

        # 1. PRUNE: Entire block is below threshold -> Drop completely (0 bytes sent)
        if max_val < min_score_threshold:
            return

        is_atomic = (h == 1 and w == 1)
        exceeds_max_size = (h > max_block_tiles or w > max_block_tiles)

        # 2. BANDWIDTH CHECK: Are some sub-quadrants worth dropping?
        # If min_val < threshold, splitting allows us to discard the low-scoring sub-blocks!
        has_prunable_subregions = (min_val < min_score_threshold)

        # 3. LEAF DECISION: Merge into single block if uniform enough AND non-prunable AND under max size
        if not is_atomic and not exceeds_max_size:
            if not has_prunable_subregions and std_dev <= max_score_std:
                # Region is homogeneous and fully valuable -> Keep merged
                pixel_x, pixel_y = c * tile_size, r * tile_size
                pixel_w, pixel_h = w * tile_size, h * tile_size
                rectangles.append((pixel_x, pixel_y, pixel_w, pixel_h, avg_val))
                return
        elif is_atomic:
            # Single tile leaf node
            pixel_x, pixel_y = c * tile_size, r * tile_size
            rectangles.append((pixel_x, pixel_y, tile_size, tile_size, avg_val))
            return

        # 4. SPLIT: Region is too diverse, too large, or contains background worth pruning
        mid_h = max(1, h // 2)
        mid_w = max(1, w // 2)

        h_slices = [(r, mid_h), (r + mid_h, h - mid_h)] if h > 1 else [(r, 1)]
        w_slices = [(c, mid_w), (c + mid_w, w - mid_w)] if w > 1 else [(c, 1)]

        for sub_r, sub_h in h_slices:
            for sub_c, sub_w in w_slices:
                if sub_h > 0 and sub_w > 0:
                    quadtree_split(sub_r, sub_c, sub_h, sub_w)

    grid_h, grid_w = score_grid.shape
    quadtree_split(0, 0, grid_h, grid_w)

    # Sort descending by priority score
    rectangles.sort(key=lambda item: item[4], reverse=True)
    return rectangles