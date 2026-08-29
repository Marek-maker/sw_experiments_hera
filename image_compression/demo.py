"""
Headless demo of the Hera tile-selection pipeline.
Runs the full pipeline on every image in data/, saves visualizations
to results/demo/ and prints bandwidth-saving statistics.
"""
import os
import glob
import cv2
import numpy as np

from utils          import load_image
from features       import extract_tile_features_advanced
from scoring        import score_tiles, score_tiles_iforest
from tiles_merging  import merge_tiles_q, merge_tiles


def build_visualization(image, scores, rectangles, tile_size=16):
    """Replicates main.show_process but headless (no imshow/waitKey).

    Returns (composite 2x2 grid, dict of 4 individual quadrant arrays).
    """
    h, w = image.shape

    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    scores_up = cv2.resize(scores, (w, h), interpolation=cv2.INTER_LINEAR)
    scores_up = np.expand_dims(scores_up, 2)

    blue = np.zeros((1, 1, 3)); blue[0, 0, 0] = 1.0
    red  = np.zeros((1, 1, 3)); red[0, 0, 2] = 1.0
    heat = (1.0 - scores_up) * blue + scores_up * red

    img_tiles = np.array(img_rgb)
    img_sel   = np.zeros_like(img_rgb)

    for x, y, bw, bh, score in rectangles:
        img_tiles = cv2.rectangle(img_tiles, (x, y), (x + bw, y + bh), (1, 0, 0), 2)
        img_sel[y:y + bh, x:x + bw, :] = img_rgb[y:y + bh, x:x + bw, :]

    top    = np.hstack((img_rgb, heat))
    bottom = np.hstack((img_tiles, img_sel))
    composite = np.vstack((top, bottom))

    quadrants = {
        "1_original":   img_rgb,
        "2_heatmap":    heat,
        "3_rectangles": img_tiles,
        "4_selected":   img_sel,
    }
    return composite, quadrants


def save_visualization(out_dir, name, mode, composite, quadrants):
    """Saves the composite grid AND each quadrant as a separate image."""
    cv2.imwrite(os.path.join(out_dir, f"{name}__{mode}.jpg"),
                np.array(255 * composite, dtype=np.uint8))
    for tag, quad in quadrants.items():
        cv2.imwrite(os.path.join(out_dir, f"{name}__{mode}__{tag}.jpg"),
                    np.array(255 * quad, dtype=np.uint8))


def stats_for(image, scores, rectangles, tile_size=16):
    h, w = image.shape
    grid_h, grid_w = scores.shape
    total_tiles = grid_h * grid_w

    sel_px = sum(bw * bh for _, _, bw, bh, _ in rectangles)
    full_px = h * w
    n_rect = len(rectangles)

    # Estimate metadata cost: each rectangle = 5 ints (x, y, w, h, score) * 4 bytes
    meta_bytes = n_rect * 5 * 4
    # Payload: selected pixels at 8 bits (compressed, would be less) + metadata
    payload_bytes = sel_px + meta_bytes
    full_bytes = full_px
    ratio = payload_bytes / full_bytes if full_bytes else 0

    return {
        "size": f"{h}x{w}",
        "tiles": total_tiles,
        "rects": n_rect,
        "sel_px": sel_px,
        "sel_pct": 100.0 * sel_px / full_px,
        "payload_ratio_pct": 100.0 * ratio,
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "demo")
    os.makedirs(out_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "*.jpg")))
    print(f"=== Hera tile-selection pipeline demo: {len(images)} images ===\n")

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        image = load_image(img_path)
        h, w = image.shape
        print(f"--- {name}  ({w}x{h} px) ---")

        # 1. Features
        features = extract_tile_features_advanced(image, tile_size=16)

        # 2a. Heuristic scoring
        scores = score_tiles(features)
        rects = merge_tiles_q(scores, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
        st = stats_for(image, scores, rects)
        print(f"  [heuristic ] {st['rects']:>3} rectangles | "
              f"selected {st['sel_pct']:5.2f}% of image | "
              f"~{st['payload_ratio_pct']:5.2f}% payload vs full frame")

        vis, quadrants = build_visualization(image, scores, rects)
        save_visualization(out_dir, name, "heuristic", vis, quadrants)

        # 2b. Isolation-Forest scoring
        try:
            scores_if = score_tiles_iforest(features)
            rects_if = merge_tiles_q(scores_if, tile_size=16, min_score_threshold=0.5, max_block_tiles=16)
            st_if = stats_for(image, scores_if, rects_if)
            print(f"  [iforest   ] {st_if['rects']:>3} rectangles | "
                  f"selected {st_if['sel_pct']:5.2f}% of image | "
                  f"~{st_if['payload_ratio_pct']:5.2f}% payload vs full frame")

            vis_if, quadrants_if = build_visualization(image, scores_if, rects_if)
            save_visualization(out_dir, name, "iforest", vis_if, quadrants_if)
        except Exception as e:
            print(f"  [iforest   ] SKIPPED: {e}")

        print()

    print("Outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
