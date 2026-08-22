import numpy
import cv2
from sklearn.ensemble import IsolationForest


def score_tiles(features: numpy.ndarray, blur_kernel: int = 3) -> numpy.ndarray:
    """
    Computes absolute anomaly scores in range [0.0, 1.0].
    
    - Flat/Black areas strictly output ~0.0 (Pruned by Quadtree).
    - Spatial smoothing blends neighboring feature tiles into unified 
      high-score blocks (Prevents Quadtree over-splitting).
    """
   
    # 1. Compute raw interest score based on texture and gradient mean
    # Flat black regions have std=0, grad=0 -> raw_score = 0
    raw_scores = features.mean(axis=-1).astype(numpy.float32)
    
    # 2. Absolute Min-Max Normalization (0.0 for dead space, 1.0 for highest activity)
    max_val = numpy.max(raw_scores)
    min_val = numpy.min(raw_scores)
    
    if max_val > min_val:
        scores = (raw_scores - min_val) / (max_val - min_val)
    else:
        scores = numpy.zeros_like(raw_scores)
        
    # 3. Suppress low-level background noise directly to absolute 0
    #scores[scores < 0.15] = 0.0

    # 4. Spatial Smoothing (Crucial for Quadtree!):
    # Expands peak feature scores to neighboring tiles so that a rock/crater
    # forms a smooth high-value region instead of noisy tile-to-tile jumps.
    if blur_kernel > 1:
        scores = cv2.dilate(scores, numpy.ones((blur_kernel, blur_kernel), numpy.float32))
        scores = cv2.blur(scores, (blur_kernel, blur_kernel))
        
    return numpy.clip(scores, 0.0, 1.0)




def score_tiles_iforest(
    features: numpy.ndarray, blur_kernel: int = 3
) -> numpy.ndarray:
    """Computes unsupervised anomaly scores [0.0, 1.0] per tile using an Isolation Forest.

    - Input features shape: (grid_h, grid_w, 7)
    - Automatically suppresses flat/dead background tiles.
    - Applies spatial dilation + blur for smooth quadtree merging.
    """
    grid_h, grid_w, n_features = features.shape
    flat_features = features.reshape(-1, n_features)

    # 1. Mask out flat black / dead space tiles (f_std is feature index 1)
    # Isolation Forest scores 'rarity' - without this mask, pitch-black space
    # will be scored as a rare, highly anomalous feature!
    std_features = flat_features[:, 1]
    valid_mask = std_features > 1e-5

    scores_flat = numpy.zeros(flat_features.shape[0], dtype=numpy.float32)

    if numpy.any(valid_mask):
        valid_features = flat_features[valid_mask]

        # 2. Fit Isolation Forest on valid surface tiles
        # 32 trees and max_samples=256 keeps execution blazingly fast
        clf = IsolationForest(n_estimators=64, max_samples=512)
        clf.fit(valid_features)

        # 3. Compute decision scores
        # score_samples returns negative values (more negative = more anomalous)
        raw_anomalies = -clf.score_samples(valid_features)

        # 4. Normalize valid scores to [0.0, 1.0]
        min_val, max_val = raw_anomalies.min(), raw_anomalies.max()
        if max_val > min_val:
            norm_scores = (raw_anomalies - min_val) / (max_val - min_val)
        else:
            norm_scores = numpy.zeros_like(raw_anomalies)

        scores_flat[valid_mask] = norm_scores

    # 5. Reshape back to 2D grid (64, 64)
    scores = scores_flat.reshape(grid_h, grid_w)

    # 6. Suppress low-level background noise
    scores[scores < 0.15] = 0.0

    # 7. Spatial Smoothing (Dilation + Blur) for Quadtree aggregation
    if blur_kernel > 1:
        kernel = numpy.ones((blur_kernel, blur_kernel), numpy.float32)
        scores = cv2.dilate(scores, kernel)
        scores = cv2.blur(scores, (blur_kernel, blur_kernel))

    return numpy.clip(scores, 0.0, 1.0)