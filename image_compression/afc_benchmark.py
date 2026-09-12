"""
Benchmark of the tile-selection pipeline on REAL Hera AFC images.

Input: a directory of 1020x1020 8-bit grayscale AFC frames
(ESA "AFC_images.tar.gz", e.g. "* AFC_0 Guidance TM(139,14) APID(292).png").

For each frame it runs the full pipeline in both scoring modes
(heuristic + Isolation Forest) and records:

    - number of rectangles
    - selected area  (% of the frame)
    - payload vs full frame (%)

Payload model (kept identical to demo.py so numbers are comparable):
    payload_bytes = selected_pixels * 1 B  +  n_rectangles * 5 * 4 B
    (each ROI rectangle = 5 int32 fields: x, y, w, h, score)

Outputs:
    results/afc/afc_stats.csv      one row per frame, both modes
    results/afc/afc_summary.txt    aggregate statistics
    results/afc/vis/<name>__*.jpg  separate full-size images (only for --vis N frames)

Usage:
    python afc_benchmark.py --dir <AFC_IMAGES> --limit 20
    python afc_benchmark.py --dir <AFC_IMAGES>            # all frames
    python afc_benchmark.py --dir <AFC_IMAGES> --vis 3    # + 3 sample visualisations
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
from scoring       import score_tiles, score_tiles_iforest
from tiles_merging import merge_tiles_q

TILE_SIZE = 16


def stats_for(image, scores, rectangles):
    """Same payload model as demo.py (comparable numbers)."""
    h, w = image.shape
    grid_h, grid_w = scores.shape
    sel_px = sum(bw * bh for _, _, bw, bh, _ in rectangles)
    full_px = h * w
    meta_bytes = len(rectangles) * 5 * 4
    payload = sel_px + meta_bytes
    return {
        "tiles": grid_h * grid_w,
        "rects": len(rectangles),
        "sel_px": sel_px,
        "sel_pct": 100.0 * sel_px / full_px,
        "payload_pct": 100.0 * payload / full_px,
        "payload_bytes": payload,
    }


def safe_name(path):
    """Filename -> safe stem (AFC names contain spaces, parens, dots)."""
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


def pct(vals, q):
    return float(np.percentile(vals, q)) if vals else float("nan")


def summarize(rows, mode):
    """Aggregate stats for one mode."""
    sel = [r[f"{mode}_sel_pct"] for r in rows if r[f"{mode}_payload_pct"] is not None]
    pay = [r[f"{mode}_payload_pct"] for r in rows if r[f"{mode}_payload_pct"] is not None]
    rec = [r[f"{mode}_rects"] for r in rows if r[f"{mode}_rects"] is not None]
    if not pay:
        return None
    return {
        "n": len(pay),
        "payload_mean": float(np.mean(pay)),
        "payload_median": float(np.median(pay)),
        "payload_min": float(np.min(pay)),
        "payload_max": float(np.max(pay)),
        "payload_p25": pct(pay, 25),
        "payload_p75": pct(pay, 75),
        "sel_mean": float(np.mean(sel)),
        "rects_mean": float(np.mean(rec)),
    }


def main():
    ap = argparse.ArgumentParser(description="Tile-selection benchmark on real Hera AFC images")
    ap.add_argument("--dir", required=True, help="directory with AFC PNG frames")
    ap.add_argument("--limit", type=int, default=0, help="max frames (0 = all)")
    ap.add_argument("--tile-size", type=int, default=TILE_SIZE)
    ap.add_argument("--threshold", type=float, default=0.5, help="min_score_threshold")
    ap.add_argument("--vis", type=int, default=0, help="save separate visuals for first N frames")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "results", "afc"))
    args = ap.parse_args()

    tile = args.tile_size

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

    print(f"=== Hera AFC tile-selection benchmark ===")
    print(f"frames: {len(files)} | tile {tile}px | threshold {args.threshold}")
    print(f"output: {args.out}\n")

    rows = []
    t_start = time.time()

    for i, path in enumerate(files, 1):
        name = safe_name(path)
        try:
            image = load_image(path)
        except Exception as e:
            print(f"[{i}/{len(files)}] SKIP {name}: {e}")
            continue
        h, w = image.shape

        row = {"file": os.path.basename(path), "size": f"{w}x{h}"}

        # --- heuristic ---
        t0 = time.time()
        features = extract_tile_features_advanced(image, tile_size=tile)
        scores = score_tiles(features)
        rects = merge_tiles_q(scores, tile_size=tile,
                              min_score_threshold=args.threshold, max_block_tiles=16)
        st = stats_for(image, scores, rects)
        row.update({
            "heuristic_rects": st["rects"],
            "heuristic_sel_pct": round(st["sel_pct"], 2),
            "heuristic_payload_pct": round(st["payload_pct"], 2),
            "heuristic_ms": int((time.time() - t0) * 1000),
        })

        # --- isolation forest ---
        t0 = time.time()
        try:
            scores_if = score_tiles_iforest(features)
            rects_if = merge_tiles_q(scores_if, tile_size=tile,
                                     min_score_threshold=args.threshold, max_block_tiles=16)
            st_if = stats_for(image, scores_if, rects_if)
            row.update({
                "iforest_rects": st_if["rects"],
                "iforest_sel_pct": round(st_if["sel_pct"], 2),
                "iforest_payload_pct": round(st_if["payload_pct"], 2),
                "iforest_ms": int((time.time() - t0) * 1000),
            })
            if i <= args.vis:
                save_separate_visuals(vis_dir, name, image, scores_if, rects_if, "iforest")
        except Exception as e:
            row.update({"iforest_rects": None, "iforest_sel_pct": None,
                        "iforest_payload_pct": None, "iforest_ms": None})
            print(f"    iForest zlyhal: {e}")

        if i <= args.vis:
            save_separate_visuals(vis_dir, name, image, scores, rects, "heuristic")

        rows.append(row)
        if i % 25 == 0 or i == len(files):
            el = time.time() - t_start
            print(f"[{i}/{len(files)}] {el:6.1f}s  "
                  f"(posledná: heur {row['heuristic_payload_pct']:5.1f}% | "
                  f"iforest {row['iforest_payload_pct'] if row['iforest_payload_pct'] is not None else float('nan'):5.1f}%)")

    # --- CSV ---
    csv_path = os.path.join(args.out, "afc_stats.csv")
    cols = ["file", "size", "heuristic_rects", "heuristic_sel_pct", "heuristic_payload_pct",
            "heuristic_ms", "iforest_rects", "iforest_sel_pct", "iforest_payload_pct", "iforest_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    # --- summary ---
    lines = []
    lines.append(f"Hera AFC tile-selection benchmark")
    lines.append(f"frames processed: {len(rows)}")
    lines.append(f"tile size: {tile}px | min_score_threshold: {args.threshold}")
    lines.append(f"total wall time: {time.time() - t_start:.1f}s")
    lines.append("")
    for mode in ("heuristic", "iforest"):
        s = summarize(rows, mode)
        if not s:
            lines.append(f"[{mode}] no data")
            continue
        lines.append(f"[{mode}] n={s['n']}")
        lines.append(f"  payload vs full frame: mean {s['payload_mean']:.2f}%  "
                     f"median {s['payload_median']:.2f}%")
        lines.append(f"  range: min {s['payload_min']:.2f}%  "
                     f"p25 {s['payload_p25']:.2f}%  p75 {s['payload_p75']:.2f}%  max {s['payload_max']:.2f}%")
        lines.append(f"  selected area: mean {s['sel_mean']:.2f}% of the frame")
        lines.append(f"  rectangles: mean {s['rects_mean']:.1f}")
        lines.append("")

    # bandwidth framing against the call's science-telemetry budget
    raw = 1020 * 1020
    budget = 12 * 1024 * 1024
    lines.append("Bandwidth framing (AFC 1020x1020, 8-bit, science budget 12 MB / 3 h slot)")
    lines.append(f"  raw frame: {raw/1024/1024:.3f} MB -> {budget/raw:.1f} raw frames per slot")
    s_if = summarize(rows, "iforest")
    if s_if:
        for label, p in (("median", s_if["payload_median"]), ("best (min)", s_if["payload_min"])):
            b = raw * p / 100.0
            lines.append(f"  iForest {label} payload {p:.2f}% = {b/1024/1024:.3f} MB -> "
                         f"{budget/b:.1f} frames per slot ({budget/b/(budget/raw):.1f}x raw)")
    s_h = summarize(rows, "heuristic")
    if s_h:
        lines.append(f"  (heuristic median payload {s_h['payload_median']:.2f}% - baseline only)")

    summary = "\n".join(lines)
    with open(os.path.join(args.out, "afc_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print("\n" + summary)
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
