"""
Benchmark of the tile-selection pipeline on REAL Hera AFC images.

Input: a directory of 1020x1020 8-bit grayscale AFC frames (ESA "AFC_images.tar.gz",
e.g. "* AFC_0 Guidance TM(139,14) APID(292).png").

Three scoring paths are measured and reported side by side, because they are NOT
the same thing and the difference matters for flight:

  heuristic      - deterministic weighted-feature score, no stored model.
                   A BASELINE, not the product.
  iforest-perframe - Isolation Forest FITTED PER FRAME. What the analysis code
                   does today. NOT flight-realistic: fitting needs an RNG,
                   dynamic structures and O(n log n) tree building, all of which
                   the Core 1 sandbox forbids (no dynamic allocation).
  iforest-frozen - ONE forest trained on the ground on `--train-frames` frames,
                   then applied to every frame (inference only). THIS is the
                   design described in the OSIP proposal, so this is the number
                   that must be quoted.

Payload model (identical for every path, and deliberately conservative - it
charges 1 byte per selected pixel and gives NO credit for the lossless
compression that would follow in flight):

    payload_bytes = selected_pixels * 1 B  +  n_rectangles * 5 * 4 B
    (each ROI rectangle = 5 int32 fields: x, y, w, h, score)

Quadtree output rectangles are disjoint by construction, so summing their areas
does not double-count.

Outputs:
    results/afc/afc_stats.csv      one row per frame, all paths
    results/afc/afc_summary.txt    aggregate statistics + bandwidth framing
    results/afc/sweep.csv          (with --sweep) payload vs threshold
    results/afc/vis/<name>__*.jpg  separate full-size images (--vis N frames)

Usage:
    python afc_benchmark.py --dir <AFC_IMAGES>
    python afc_benchmark.py --dir <AFC_IMAGES> --limit 40 --vis 3
    python afc_benchmark.py --dir <AFC_IMAGES> --sweep 0.3,0.4,0.5,0.6,0.7
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

from utils         import load_image
from features      import extract_tile_features_advanced
from scoring       import (score_tiles, score_tiles_iforest,
                           train_iforest_ground, score_tiles_iforest_model,
                           raw_iforest_scores)
from tiles_merging import merge_tiles_q

TILE_SIZE = 16
RAW_FRAME_BYTES = 1020 * 1020          # AFC geometry, 8-bit
SCIENCE_BUDGET_BYTES = 12 * 1024 * 1024  # 12 MB per 3 h slot (call requirement)


def stats_for(image, scores, rectangles, tile_size=TILE_SIZE):
    """Payload model (see module docstring).

    Reports the payload relative to the FULL frame - that is the honest
    bandwidth metric (bytes downlinked vs. bytes of a raw frame).

    It also reports `cropped_pct`: the fraction of the frame the tiler never
    examined. `extract_tile_features_advanced` crops to `size // tile_size`
    tiles, so for a 1020-px AFC frame with 16-px tiles only 1008 px are
    processed and ~2.3 % of the pixels are silently dropped. Without this
    column a larger crop would look like a better payload while actually
    covering less of the scene.
    """
    h, w = image.shape
    grid_h, grid_w = scores.shape
    sel_px = sum(bw * bh for _, _, bw, bh, _ in rectangles)
    full_px = h * w
    meta_bytes = len(rectangles) * 5 * 4
    payload = sel_px + meta_bytes
    processed_px = (grid_h * tile_size) * (grid_w * tile_size)
    return {
        "rects": len(rectangles),
        "sel_pct": 100.0 * sel_px / full_px,
        "payload_pct": 100.0 * payload / full_px,
        "payload_bytes": payload,
        "cropped_pct": 100.0 * (1.0 - processed_px / full_px) if full_px else 0.0,
    }


def safe_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem[:60]


def save_separate_visuals(out_dir, name, image, scores, rectangles, mode):
    """Saves SEPARATE full-size images (no 2x2 grid)."""
    h, w = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    scores_up = cv2.resize(scores, (w, h), interpolation=cv2.INTER_LINEAR)
    scores_up = np.expand_dims(scores_up, 2)
    blue = np.zeros((1, 1, 3)); blue[0, 0, 0] = 1.0
    red = np.zeros((1, 1, 3)); red[0, 0, 2] = 1.0
    heat = (1.0 - scores_up) * blue + scores_up * red

    img_rect = np.array(img_rgb)
    img_sel = np.zeros_like(img_rgb)
    for x, y, bw, bh, _ in rectangles:
        img_rect = cv2.rectangle(img_rect, (x, y), (x + bw, y + bh), (1, 0, 0), 2)
        img_sel[y:y + bh, x:x + bw, :] = img_rgb[y:y + bh, x:x + bw, :]

    for tag, arr in (("1_original", img_rgb), ("2_heatmap", heat),
                     ("3_rectangles", img_rect), ("4_selected", img_sel)):
        cv2.imwrite(os.path.join(out_dir, f"{name}__{mode}__{tag}.jpg"),
                    np.array(255 * arr, dtype=np.uint8))


def pctl(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def summarize(rows, key):
    """Aggregate payload statistics for one path (`key` = CSV column prefix)."""
    pay = [r[f"{key}_payload_pct"] for r in rows if r.get(f"{key}_payload_pct") is not None]
    if not pay:
        return None
    sel = [r[f"{key}_sel_pct"] for r in rows if r.get(f"{key}_payload_pct") is not None]
    rec = [r[f"{key}_rects"] for r in rows if r.get(f"{key}_payload_pct") is not None]
    return {
        "n": len(pay),
        "mean": float(np.mean(pay)), "median": float(np.median(pay)),
        "min": float(np.min(pay)), "max": float(np.max(pay)),
        "p25": pctl(pay, 25), "p75": pctl(pay, 75),
        "sel_mean": float(np.mean(sel)), "rects_mean": float(np.mean(rec)),
    }


def frames_per_slot(payload_pct):
    """How many frames of this payload size fit the 12 MB / 3 h science budget."""
    b = RAW_FRAME_BYTES * payload_pct / 100.0
    return SCIENCE_BUDGET_BYTES / b if b > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser(description="Tile-selection benchmark on real Hera AFC images")
    ap.add_argument("--dir", required=True, help="directory with AFC PNG frames")
    ap.add_argument("--limit", type=int, default=0, help="max frames (0 = all)")
    ap.add_argument("--tile-size", type=int, default=TILE_SIZE)
    ap.add_argument("--threshold", type=float, default=0.5, help="min_score_threshold")
    ap.add_argument("--model", choices=["perframe", "frozen", "both"], default="both",
                    help="which iForest path(s) to measure")
    ap.add_argument("--train-frames", type=int, default=40,
                    help="frames used to train the ground/frozen model")
    ap.add_argument("--sweep", default="",
                    help="comma-separated thresholds, e.g. 0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--quantile", default="",
                    help="comma-separated top-fractions, e.g. 0.05,0.10,0.15,0.20. "
                         "Selects the top q fraction of SCORED tiles per frame instead of "
                         "using an absolute threshold - OOD-robust and gives a predictable "
                         "per-frame bandwidth. Applied to the frozen path.")
    ap.add_argument("--vis", type=int, default=0, help="save separate visuals for first N frames")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "results", "afc"))
    args = ap.parse_args()

    tile = args.tile_size
    sweep = [float(x) for x in args.sweep.split(",") if x.strip()] if args.sweep else []
    quantiles = [float(x) for x in args.quantile.split(",") if x.strip()] if args.quantile else []

    files = sorted(glob.glob(os.path.join(args.dir, "*.png")))
    if not files:
        print(f"CHYBA: žiadne .png v {args.dir}", file=sys.stderr)
        return 1
    if args.limit:
        files = files[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    vis_dir = os.path.join(args.out, "vis")
    if args.vis:
        os.makedirs(vis_dir, exist_ok=True)

    print("=== Hera AFC tile-selection benchmark ===")
    print(f"frames: {len(files)} | tile {tile}px | threshold {args.threshold} | model: {args.model}")
    print(f"output: {args.out}\n")

    # ---- pass 1: load + features (kept in memory: 64x64x4 float32 ~= 64 kB per frame) ----
    t0 = time.time()
    feats, shapes, names, skipped = [], [], [], 0
    for i, path in enumerate(files, 1):
        try:
            img = load_image(path)
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}")
            skipped += 1
            continue
        feats.append(extract_tile_features_advanced(img, tile_size=tile))
        shapes.append(img.shape)
        names.append(path)
        if i % 100 == 0:
            print(f"  features {i}/{len(files)}  ({time.time()-t0:.1f}s)")
    print(f"features ready: {len(feats)} frames in {time.time()-t0:.1f}s "
          f"(skipped {skipped})\n")

    # ---- frozen model: ONE forest, trained on the ground ----
    frozen = None
    if args.model in ("frozen", "both"):
        n_train = max(1, min(args.train_frames, len(feats)))
        t0 = time.time()
        frozen = train_iforest_ground(feats[:n_train])
        print(f"ground model trained on {n_train} frames in {time.time()-t0:.1f}s "
              f"(frozen -> static table in flight)\n")

    rows = []
    t_start = time.time()

    for i, (feat, shape, path) in enumerate(zip(feats, shapes, names), 1):
        img = load_image(path)
        name = safe_name(path)
        row = {"file": os.path.basename(path), "size": f"{shape[1]}x{shape[0]}"}

        # ---- heuristic (baseline) ----
        t0 = time.time()
        s_h = score_tiles(feat)
        r_h = merge_tiles_q(s_h, tile_size=tile, min_score_threshold=args.threshold,
                            max_block_tiles=16)
        st = stats_for(img, s_h, r_h, tile)
        row.update({"heuristic_rects": st["rects"],
                    "heuristic_sel_pct": round(st["sel_pct"], 2),
                    "heuristic_payload_pct": round(st["payload_pct"], 2),
                    "cropped_pct": round(st["cropped_pct"], 2),
                    "heuristic_ms": int((time.time() - t0) * 1000)})
        if i <= args.vis:
            save_separate_visuals(vis_dir, name, img, s_h, r_h, "heuristic")

        # ---- iForest, per-frame fit (current analysis code) ----
        if args.model in ("perframe", "both"):
            t0 = time.time()
            s_pf = score_tiles_iforest(feat)
            r_pf = merge_tiles_q(s_pf, tile_size=tile, min_score_threshold=args.threshold,
                                 max_block_tiles=16)
            st_pf = stats_for(img, s_pf, r_pf, tile)
            row.update({"perframe_rects": st_pf["rects"],
                        "perframe_sel_pct": round(st_pf["sel_pct"], 2),
                        "perframe_payload_pct": round(st_pf["payload_pct"], 2),
                        "perframe_ms": int((time.time() - t0) * 1000)})
        else:
            row.update({"perframe_rects": None, "perframe_sel_pct": None,
                        "perframe_payload_pct": None, "perframe_ms": None})

        # ---- iForest, frozen ground-trained model (flight design) ----
        if frozen is not None:
            t0 = time.time()
            s_fr = score_tiles_iforest_model(feat, frozen)
            r_fr = merge_tiles_q(s_fr, tile_size=tile, min_score_threshold=args.threshold,
                                 max_block_tiles=16)
            st_fr = stats_for(img, s_fr, r_fr, tile)
            row.update({"frozen_rects": st_fr["rects"],
                        "frozen_sel_pct": round(st_fr["sel_pct"], 2),
                        "frozen_payload_pct": round(st_fr["payload_pct"], 2),
                        "frozen_ms": int((time.time() - t0) * 1000)})
            if i <= args.vis:
                save_separate_visuals(vis_dir, name, img, s_fr, r_fr, "frozen")
            # absolute-score spread (for on-board fixed-threshold calibration)
            raw = raw_iforest_scores(feat, frozen)
            row["frozen_raw_median"] = round(float(np.median(raw)), 5) if len(raw) else None
        else:
            row.update({"frozen_rects": None, "frozen_sel_pct": None,
                        "frozen_payload_pct": None, "frozen_ms": None,
                        "frozen_raw_median": None})

        # ---- threshold sweep (cheap: scores already computed) ----
        if sweep:
            for th in sweep:
                for key, s in (("heuristic", s_h), ("frozen", s_fr if frozen is not None else None)):
                    if s is None:
                        row[f"sweep_{key}_{th}"] = None
                        continue
                    rr = merge_tiles_q(s, tile_size=tile, min_score_threshold=th,
                                       max_block_tiles=16)
                    row[f"sweep_{key}_{th}"] = round(stats_for(img, s, rr, tile)["payload_pct"], 2)

        # ---- quantile-based selection: keep the top q fraction of SCORED tiles ----
        # Robust to out-of-distribution frames (a fixed absolute threshold is not:
        # an unfamiliar scene makes every tile look anomalous) and it makes the
        # per-frame bandwidth predictable, which is operationally valuable.
        if quantiles:
            if frozen is not None:
                scored = s_fr[s_fr > 0]
                for q in quantiles:
                    if scored.size == 0:
                        row[f"quantile_{q}"] = None
                        continue
                    th = float(np.quantile(scored, 1.0 - q))
                    rr = merge_tiles_q(s_fr, tile_size=tile, min_score_threshold=th,
                                       max_block_tiles=16)
                    row[f"quantile_{q}"] = round(stats_for(img, s_fr, rr, tile)["payload_pct"], 2)
            else:
                for q in quantiles:
                    row[f"quantile_{q}"] = None

        rows.append(row)
        if i % 25 == 0 or i == len(feats):
            print(f"[{i}/{len(feats)}] {time.time()-t_start:6.1f}s  "
                  f"heur {row['heuristic_payload_pct']:5.1f}% | "
                  f"perframe {(row['perframe_payload_pct'] if row['perframe_payload_pct'] is not None else float('nan')):5.1f}% | "
                  f"frozen {(row['frozen_payload_pct'] if row['frozen_payload_pct'] is not None else float('nan')):5.1f}%")

    # ---- CSV ----
    base_cols = ["file", "size", "cropped_pct",
                 "heuristic_rects", "heuristic_sel_pct", "heuristic_payload_pct", "heuristic_ms",
                 "perframe_rects", "perframe_sel_pct", "perframe_payload_pct", "perframe_ms",
                 "frozen_rects", "frozen_sel_pct", "frozen_payload_pct", "frozen_ms",
                 "frozen_raw_median"]
    sweep_cols = [f"sweep_{k}_{t}" for t in sweep for k in ("heuristic", "frozen")]
    quantile_cols = [f"quantile_{q}" for q in quantiles]
    cols = base_cols + sweep_cols + quantile_cols
    csv_path = os.path.join(args.out, "afc_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    if sweep:
        with open(os.path.join(args.out, "sweep.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["threshold"] + [f"{k}_median_payload_pct" for k in ("heuristic", "frozen")])
            for t in sweep:
                vals = {}
                for k in ("heuristic", "frozen"):
                    col = [r[f"sweep_{k}_{t}"] for r in rows if r.get(f"sweep_{k}_{t}") is not None]
                    vals[k] = f"{np.median(col):.2f}" if col else "n/a"
                wr.writerow([t, vals["heuristic"], vals["frozen"]])

    # ---- summary ----
    L = []
    L.append("Hera AFC tile-selection benchmark")
    L.append(f"frames processed: {len(rows)}  (skipped {skipped})")
    L.append(f"tile size: {tile}px | min_score_threshold: {args.threshold}")
    if frozen is not None:
        L.append(f"ground-trained (frozen) model: trained on {min(args.train_frames, len(feats))} frames")
    L.append(f"total wall time: {time.time() - t_start:.1f}s")
    if rows and rows[0].get("cropped_pct") is not None:
        L.append(f"WARNING: {tile}px tiles do not divide the 1020px AFC frame, so "
                 f"{rows[0]['cropped_pct']:.2f}% of every frame is NEVER EXAMINED "
                 f"(edge strip). Payload % is still reported against the FULL frame. "
                 f"Use a tile size that divides 1020 (12/15/20/30/34/60) for full coverage.")
    L.append("")
    for key, label in (("heuristic", "heuristic (BASELINE, no stored model)"),
                       ("perframe", "iForest fitted PER FRAME (analysis code; not flight-realistic)"),
                       ("frozen", "iForest FROZEN ground-trained model (FLIGHT DESIGN)")):
        s = summarize(rows, key)
        if not s:
            continue
        L.append(f"[{label}]")
        L.append(f"  payload vs full frame: mean {s['mean']:.2f}%  median {s['median']:.2f}%")
        L.append(f"  range: min {s['min']:.2f}%  p25 {s['p25']:.2f}%  p75 {s['p75']:.2f}%  max {s['max']:.2f}%")
        L.append(f"  selected area: mean {s['sel_mean']:.2f}% of frame | rectangles: mean {s['rects_mean']:.1f}")
        L.append(f"  frames per 12 MB / 3 h slot: {frames_per_slot(s['median']):.1f} "
                 f"(raw would be {frames_per_slot(100.0):.1f})")
        L.append("")

    if frozen is not None:
        raw_med = [r["frozen_raw_median"] for r in rows if r.get("frozen_raw_median") is not None]
        if raw_med:
            L.append("Frozen-model absolute anomaly score (median per frame) - for a fixed on-board threshold:")
            L.append(f"  min {min(raw_med):.4f}  median {np.median(raw_med):.4f}  max {max(raw_med):.4f}")
            spread = max(raw_med) - min(raw_med)
            L.append(f"  spread across frames: {spread:.4f} -> a single fixed threshold is "
                     f"{'plausible' if spread < 0.05 else 'NOT safe; per-frame normalisation needed'}")
            L.append("")

    if sweep:
        L.append(f"Threshold sweep (median payload %, {len(rows)} frames):")
        L.append("  threshold | heuristic | frozen")
        for t in sweep:
            vals = {}
            for k in ("heuristic", "frozen"):
                col = [r[f"sweep_{k}_{t}"] for r in rows if r.get(f"sweep_{k}_{t}") is not None]
                vals[k] = f"{np.median(col):7.2f}" if col else "    n/a"
            L.append(f"  {t:9.2f} | {vals['heuristic']} | {vals['frozen']}")
        L.append("")

    if quantiles:
        L.append("Quantile selection on the frozen model (keep top q of scored tiles per frame):")
        L.append("  q     | median payload % | min     | max     | frames/slot | predictable?")
        for q in quantiles:
            col = [r[f"quantile_{q}"] for r in rows if r.get(f"quantile_{q}") is not None]
            if not col:
                L.append(f"  {q:5.2f} | n/a")
                continue
            med, lo, hi = float(np.median(col)), float(np.min(col)), float(np.max(col))
            L.append(f"  {q:5.2f} | {med:16.2f} | {lo:7.2f} | {hi:7.2f} | "
                     f"{frames_per_slot(med):11.1f} | max/min = {hi/max(med,1e-9):.2f}x")
        L.append("")

    L.append("Bandwidth framing (AFC 1020x1020, 8-bit, science budget 12 MB / 3 h slot)")
    L.append(f"  raw frame: {RAW_FRAME_BYTES/1024/1024:.3f} MB -> {frames_per_slot(100.0):.1f} raw frames per slot")
    s = summarize(rows, "frozen")
    if s:
        L.append(f"  frozen median  {s['median']:.2f}% -> {frames_per_slot(s['median']):.1f} frames/slot "
                 f"({frames_per_slot(s['median'])/frames_per_slot(100.0):.1f}x raw)")
        L.append(f"  frozen best    {s['min']:.2f}% -> {frames_per_slot(s['min']):.1f} frames/slot "
                 f"({frames_per_slot(s['min'])/frames_per_slot(100.0):.1f}x raw)")
        L.append(f"  frozen worst   {s['max']:.2f}% -> {frames_per_slot(s['max']):.1f} frames/slot "
                 f"({frames_per_slot(s['max'])/frames_per_slot(100.0):.1f}x raw)")
        L.append("  NOTE: these are PRIORITISED SUBSETS, not complete frames - see the proposal caveat.")

    summary = "\n".join(L)
    with open(os.path.join(args.out, "afc_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print("\n" + summary)
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
