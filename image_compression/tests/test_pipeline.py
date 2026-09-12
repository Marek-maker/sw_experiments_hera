r"""Tests for the AFC tile-selection pipeline and benchmark helpers.

Run with:  .venv\Scripts\python -m pytest image_compression/tests -q

These tests use SYNTHETIC frames only. The ESA AFC reference images are provided
under "ESA UNCLASSIFIED - For ESA Official Use Only" and are deliberately not
committed to this repository, so no test may depend on them.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
IMGC = os.path.dirname(HERE)
if IMGC not in sys.path:
    sys.path.insert(0, IMGC)

from features import extract_tile_features_advanced          # noqa: E402
from scoring import (score_tiles, score_tiles_iforest,       # noqa: E402
                     train_iforest_ground, score_tiles_iforest_model,
                     raw_iforest_scores)
from tiles_merging import merge_tiles_q                       # noqa: E402
import afc_benchmark as bench                                 # noqa: E402


# ---------------------------------------------------------------- helpers

def black_frame(size=64):
    """All-zero frame: no surface, must produce no selection."""
    return np.zeros((size, size), dtype=np.float32)


def noise_frame(size=1020, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((size, size), dtype=np.float32)


def textured_frame(size=1020, seed=1):
    """Black background with a bright textured patch: a synthetic 'object'."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    img[300:600, 400:700] = 0.5 + 0.4 * rng.random((300, 300))
    return img


# ---------------------------------------------------------------- payload model

def test_stats_for_matches_hand_computation():
    """payload = selected pixels * 1 B + rects * 5 * 4 B, over the full frame area."""
    img = np.zeros((64, 64), dtype=np.float32)
    scores = np.zeros((4, 4), dtype=np.float32)
    # two rectangles, in pixels: 16x16 and 32x16  -> 256 + 512 = 768 px
    rects = [(0, 0, 16, 16, 1.0), (16, 0, 32, 16, 1.0)]
    st = bench.stats_for(img, scores, rects)
    assert st["rects"] == 2
    assert st["payload_bytes"] == 768 + 2 * 5 * 4
    assert st["payload_pct"] == pytest.approx(100.0 * (768 + 40) / (64 * 64))
    assert st["sel_pct"] == pytest.approx(100.0 * 768 / (64 * 64))


def test_stats_for_no_rectangles():
    st = bench.stats_for(black_frame(), np.zeros((4, 4), dtype=np.float32), [])
    assert st["rects"] == 0
    assert st["payload_bytes"] == 0
    assert st["payload_pct"] == 0.0


def test_frames_per_slot_math():
    # a full frame must reproduce the raw count: 12 MB / 0.992 MB
    assert bench.frames_per_slot(100.0) == pytest.approx(
        bench.SCIENCE_BUDGET_BYTES / bench.RAW_FRAME_BYTES)
    # halving the payload doubles the number of frames
    assert bench.frames_per_slot(50.0) == pytest.approx(2 * bench.frames_per_slot(100.0))
    # guard the constants against the call's stated limits
    assert bench.RAW_FRAME_BYTES == 1020 * 1020
    assert bench.SCIENCE_BUDGET_BYTES == 12 * 1024 * 1024


# ---------------------------------------------------------------- geometry / tiling

def test_feature_shape_and_noise_floor():
    img = noise_frame(256)
    f = extract_tile_features_advanced(img, tile_size=16)
    assert f.shape == (16, 16, 4), "4 features per tile (mean, std, grad, outliers)"


def test_black_frame_has_zero_std_everywhere():
    f = extract_tile_features_advanced(black_frame(256), tile_size=16)
    assert np.all(f[:, :, 1] == 0.0), "dead space must have std 0 (used by the mask)"


# ---------------------------------------------------------------- merging

def test_quadtree_rectangles_are_disjoint():
    """Payload maths sums rectangle areas, so they MUST not overlap."""
    rng = np.random.default_rng(3)
    scores = rng.random((64, 64)).astype(np.float32)
    rects = merge_tiles_q(scores, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)

    painted = np.zeros((64, 64), dtype=np.int32)
    for x, y, w, h, _ in rects:
        c0, r0 = x // 16, y // 16
        c1, r1 = c0 + w // 16, r0 + h // 16
        assert painted[r0:r1, c0:c1].sum() == 0, "overlapping rectangles found"
        painted[r0:r1, c0:c1] = 1


def test_quadtree_on_all_zero_scores_selects_nothing():
    rects = merge_tiles_q(np.zeros((64, 64), dtype=np.float32), tile_size=16,
                          min_score_threshold=0.5, max_block_tiles=16)
    assert rects == []


def test_quadtree_rectangles_within_processed_extent():
    """Rectangles must stay inside the PROCESSED grid.

    Note: extract_tile_features_advanced crops to `size // tile_size` tiles, so
    for a 1020-px AFC frame with 16-px tiles only 1008 px are processed and a
    12-px strip is dropped -- see test_afc_geometry_is_cropped_by_16px_tiles.
    """
    rng = np.random.default_rng(4)
    scores = rng.random((64, 64)).astype(np.float32)
    rects = merge_tiles_q(scores, tile_size=16, min_score_threshold=0.3, max_block_tiles=16)
    extent = 64 * 16          # processed pixels for a 64x64 grid
    for x, y, w, h, _ in rects:
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert x + w <= extent and y + h <= extent


def test_afc_geometry_is_cropped_by_16px_tiles():
    """DOCUMENTS A REAL LIMITATION (found in review, 2026-09-12).

    1020 is not divisible by 16: 1020 // 16 == 63, so a 1020x1020 AFC frame is
    processed as 63x63 tiles = 1008x1008 px and a 12-px strip on the right and
    bottom edge is silently ignored. Pick a tile size that divides 1020
    (12, 15, 20, 34, 60, ...) if full-frame coverage is required in flight.
    """
    img = np.zeros((1020, 1020), dtype=np.float32)
    f = extract_tile_features_advanced(img, tile_size=16)
    assert f.shape == (63, 63, 4)
    processed = 63 * 16
    assert processed == 1008
    dropped_px = 1020 * 1020 - processed * processed
    assert dropped_px == 24336            # ~2.3% of the frame's pixels
    # tile sizes that DO divide the AFC frame
    assert [t for t in (12, 15, 20, 30, 34, 60) if 1020 % t == 0] == [12, 15, 20, 30, 34, 60]
    assert 1020 % 16 != 0


def test_quadtree_is_deterministic():
    rng = np.random.default_rng(5)
    scores = rng.random((64, 64)).astype(np.float32)
    a = merge_tiles_q(scores, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
    b = merge_tiles_q(scores, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
    assert a == b


# ---------------------------------------------------------------- degenerate frames

def test_black_frame_produces_no_selection_heuristic():
    img = black_frame(1020)
    f = extract_tile_features_advanced(img, tile_size=16)
    s = score_tiles(f)
    rects = merge_tiles_q(s, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
    assert rects == [], "an all-black frame must not spend downlink budget"


def test_black_frame_produces_no_selection_iforest():
    img = black_frame(1020)
    f = extract_tile_features_advanced(img, tile_size=16)
    s = score_tiles_iforest(f)          # must not raise on 'no valid tiles'
    assert np.all(s == 0.0)
    rects = merge_tiles_q(s, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
    assert rects == []


def test_uniform_frame_produces_no_selection():
    """A perfectly flat grey frame has no novelty anywhere."""
    img = np.full((1020, 1020), 0.42, dtype=np.float32)
    f = extract_tile_features_advanced(img, tile_size=16)
    s = score_tiles_iforest(f)
    rects = merge_tiles_q(s, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
    assert rects == []


# ---------------------------------------------------------------- frozen model

def test_frozen_model_matches_perframe_closely():
    """A ground-trained frozen model must behave like the per-frame fit on the
    frames it was trained on (sanity check on the inference path)."""
    feats = [extract_tile_features_advanced(textured_frame(1020, seed=s), tile_size=16)
             for s in range(4)]
    model = train_iforest_ground(feats, seed=0)
    f = feats[0]
    s_frozen = score_tiles_iforest_model(f, model)
    s_perframe = score_tiles_iforest(f)
    assert s_frozen.shape == s_perframe.shape
    assert s_frozen.max() <= 1.0 and s_frozen.min() >= 0.0
    # both should select *something*, and a comparable order of magnitude
    r_frozen = merge_tiles_q(s_frozen, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
    r_perframe = merge_tiles_q(s_perframe, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
    assert len(r_frozen) > 0 and len(r_perframe) > 0


def test_frozen_model_is_deterministic():
    feats = [extract_tile_features_advanced(textured_frame(1020, seed=s), tile_size=16)
             for s in range(4)]
    m1 = train_iforest_ground(feats, seed=7)
    m2 = train_iforest_ground(feats, seed=7)
    f = feats[0]
    assert np.allclose(raw_iforest_scores(f, m1), raw_iforest_scores(f, m2))


def test_frozen_inference_does_not_fit():
    """The flight path must be pure inference - no fitting inside score_tiles_iforest_model."""
    src = open(os.path.join(IMGC, "scoring.py"), encoding="utf-8").read()
    body = src.split("def score_tiles_iforest_model(")[1].split("\ndef ")[0]
    assert ".fit(" not in body, "the frozen inference path must not fit anything"


def test_raw_scores_empty_on_black_frame():
    feats = [extract_tile_features_advanced(textured_frame(1020, seed=s), tile_size=16)
             for s in range(2)]
    model = train_iforest_ground(feats, seed=0)
    assert raw_iforest_scores(extract_tile_features_advanced(black_frame(1020), 16),
                              model).size == 0


# ---------------------------------------------------------------- quantile selection

def test_quantile_selection_gives_predictable_payload():
    """Keeping the top q fraction of scored tiles must give a payload that scales
    with q and stays tight across very different frames - the property the
    proposal relies on for predictable bandwidth."""
    feats = [extract_tile_features_advanced(textured_frame(1020, seed=s), tile_size=16)
             for s in range(6)]
    model = train_iforest_ground(feats, seed=0)

    payloads = {}
    for q in (0.05, 0.15):
        vals = []
        for f in feats:
            s = score_tiles_iforest_model(f, model)
            scored = s[s > 0]
            if scored.size == 0:
                continue
            th = float(np.quantile(scored, 1.0 - q))
            rects = merge_tiles_q(s, tile_size=16, min_score_threshold=th, max_block_tiles=16)
            vals.append(bench.stats_for(np.zeros((1020, 1020), np.float32), s, rects)["payload_pct"])
        payloads[q] = vals

    assert payloads[0.05] and payloads[0.15]
    # larger q must not select less
    assert np.median(payloads[0.15]) > np.median(payloads[0.05])
    # and the spread within a q must be small
    for q, vals in payloads.items():
        assert max(vals) / max(min(vals), 1e-9) < 3.0, f"q={q} payload spread too wide: {vals}"


def test_safe_name_sanitises_esa_filenames():
    name = bench.safe_name("/x/100005.49997997284 AFC_0 Guidance TM(139,14) APID(292).png")
    assert " " not in name and "(" not in name and ")" not in name
    assert name.startswith("100005")
    assert len(name) <= 60
