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

    - Input features shape: (grid_h, grid_w, N_FEATURES) where N_FEATURES = 4
      (mean, std, gradient, outlier fraction - see features.extract_tile_features_advanced)
    - The forest is FITTED PER FRAME. This is convenient for analysis but is NOT the
      flight-realistic design: fitting needs an RNG, dynamic structures and O(n log n)
      tree building, which the Hera Core 1 sandbox forbids (no dynamic allocation).
      For the flight design use `score_tiles_iforest_model()` with a model fitted
      on the ground (frozen static table).
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


def _valid_tile_features(features: numpy.ndarray):
    """Flat feature rows of non-flat tiles (f_std > 1e-5), plus the mask.

    Flat / dead-space tiles must be excluded from forest training and scoring:
    Isolation Forest scores 'rarity', so pitch-black space would otherwise be
    reported as the most anomalous thing in the frame.
    """
    grid_h, grid_w, n_features = features.shape
    flat = features.reshape(-1, n_features)
    mask = flat[:, 1] > 1e-5
    return flat[mask], mask, (grid_h, grid_w)


def train_iforest_ground(features_list, n_estimators: int = 64,
                         max_samples: int = 512, seed: int = 42):
    """Fit ONE Isolation Forest on pooled features from several frames.

    This is the FLIGHT-REALISTIC direction: the model is trained on the ground
    (offline, on a reference set), then frozen and shipped as a static table.
    On board only inference runs - no RNG, no tree building, no allocation.

    Returns the fitted sklearn estimator (the ground-side artefact that would be
    converted to a static C table via the call's `bin2c` tool).
    """
    pooled = numpy.vstack([_valid_tile_features(f)[0] for f in features_list])
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=int(min(max_samples, len(pooled))),
        random_state=seed,
    )
    clf.fit(pooled)
    return clf


def score_tiles_iforest_model(
    features: numpy.ndarray, model, blur_kernel: int = 3,
    normalize: bool = True,
) -> numpy.ndarray:
    """Inference with a FROZEN, ground-trained model -> scores per tile.

    `normalize=True` applies the same per-frame min-max normalisation and noise
    floor as `score_tiles_iforest`, so the two variants are directly comparable
    at the same threshold. `normalize=False` returns the raw (absolute) anomaly
    scores, which is what an on-board fixed threshold would use.
    """
    valid_features, mask, (grid_h, grid_w) = _valid_tile_features(features)

    scores_flat = numpy.zeros(features.shape[0] * features.shape[1], dtype=numpy.float32)

    if numpy.any(mask):
        raw = -model.score_samples(valid_features)  # higher = more anomalous
        if normalize:
            lo, hi = raw.min(), raw.max()
            vals = (raw - lo) / (hi - lo) if hi > lo else numpy.zeros_like(raw)
        else:
            vals = raw
        scores_flat[mask] = vals

    scores = scores_flat.reshape(grid_h, grid_w)

    if normalize:
        scores[scores < 0.15] = 0.0

    if blur_kernel > 1:
        kernel = numpy.ones((blur_kernel, blur_kernel), numpy.float32)
        scores = cv2.dilate(scores, kernel)
        scores = cv2.blur(scores, (blur_kernel, blur_kernel))

    if normalize:
        scores = numpy.clip(scores, 0.0, 1.0)
    return scores


def raw_iforest_scores(features: numpy.ndarray, model) -> numpy.ndarray:
    """Raw absolute anomaly scores of the valid tiles (for threshold calibration)."""
    valid_features, _, _ = _valid_tile_features(features)
    if len(valid_features) == 0:
        return numpy.array([], dtype=numpy.float32)
    return -model.score_samples(valid_features)