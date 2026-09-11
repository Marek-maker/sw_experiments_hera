# OSIP Call for Ideas — Autonomous Software Experiments on Hera

**Campaign:** Ca-2026-00066
**Title:** Adaptive Tile-Selection for Bandwidth-Efficient Downlink of Asteroid Imagery
**Core 1 sandbox experiment — Phase 1 Idea submission**

| Field | Value |
|---|---|
| Proposing organisation | *(doplniť: názov / typ organizácie)* |
| Contact | Marek Racko |
| Country (ESA Member State) | Slovakia |
| Submission language | English |
| Campaign manager | Jorge Lopez Trescastro (ESA) |
| Deadline | ~15 September 2026 (submission phase, "4 days left" as of 2026-09-11) |

> **Reference to the repo:** the working prototype, source, notebook and demo outputs described here are
> published in the public fork `github.com/Marek-maker/sw_experiments_hera` (upstream
> `michalnand/sw_experiments_hera`).

---

## 1. The problem

Deep-space optical imagers generate far more data than the downlink budget can carry. Hera's Asteroid
Framing Camera (AFC) produces high-resolution frames of the Didymos/Dimorphos system, yet the comms
link from ~150 million km supports only a very small fraction of that volume per day, with one-way
light-time delays reaching up to ~40 minutes.

The current operational approach downlinks **whole frames**. The majority of each frame is either
featureless space (near-zero signal) or flat, uninformative regolith — scientifically low-value pixels
that consume the same scarce bandwidth as the crater walls, boulder fields and ejecta that matter.

Two consequences follow:

1. **Waste.** A large share of the daily downlink budget is spent on empty or uniform regions.
2. **Missed science.** Because the budget is finite, the spacecraft must choose *which frames* to send.
   That choice is today made on the ground, minutes-to-hours later, not onboard.

**The limitation this idea addresses:** the absence of a lightweight, onboard method that decides
*which parts of an image* are worth downlinking — autonomously, deterministically and within the
flight-computer constraints of a LEON3 core.

---

## 2. The solution

An **adaptive tile-selection pipeline** ("smart-crop") that runs onboard and outputs a *prioritised,
bandwidth-aware list of regions of interest (ROIs)* rather than a full frame. Only the selected tiles
are queued for downlink; everything below an interest threshold is dropped (0 bytes).

The pipeline is a five-stage, fully vectorised algorithm:

```
Input frame (2D grayscale)
        │
        ▼
[1] Tiling into a 4D tensor (grid_h, grid_w, 16, 16)   ← zero loops
        │
        ▼
[2] Per-tile feature extraction:
        mean · std · gradient (|Δx|+|Δy|) · outlier fraction
        │
        ▼
[3] Interest scoring — two selectable modes:
        (A) Heuristic  : normalised average of tile features
        (B) Isolation Forest : unsupervised "what is rare is interesting"
        │
        ▼
[4] Spatial smoothing (morphological dilation + blur)
        │
        ▼
[5] Quadtree merging with bandwidth-aware pruning
        │
        ▼
Priority-sorted ROI rectangles → downlink queue
```

**How it works, in one paragraph.** Each 16×16-pixel tile is reduced to a compact feature vector
(mean intensity, standard deviation, edge/gradient magnitude, fraction of outlier pixels). These
features are scored for "scientific interest". Two interchangeable scoring back-ends are provided:
a deterministic heuristic (fast, zero model state) and an `IsolationForest` anomaly detector
(the interesting tile is the *rare* tile — the detector isolates it in few splits). Scores are then
spatially smoothed so that a single boulder or crater forms one contiguous high-value blob instead of
scattered hot pixels. Finally a **quadtree** recursively merges homogeneous high-score regions into
few large rectangles and **prunes** any sub-region whose maximum score is below threshold — those cost
0 bytes. The output is a short, priority-ordered list of pixel rectangles plus per-rectangle metadata.

**Why this matters for Hera.** The algorithm never has to be told *what* to look for. It is an
*attention* mechanism: "send me what is unusual". On an unvisited binary asteroid that is exactly the
right prior — you do not know in advance whether the next frame holds a fresh crater, a fracture, or a
new boulder field, but you know it will not look like empty space.

### Key design choices

| Choice | Rationale |
|---|---|
| 16×16 tiles, 4D-tensor reshape | All tiles processed in single NumPy ops — no Python loop, deterministic, LEON3-friendly |
| Two scoring modes | Heuristic = lowest CPU / fully deterministic; iForest = true unsupervised novelty discovery |
| Spatial dilation + blur | Prevents quadtree over-splitting; a feature becomes one block, not 20 scattered tiles |
| Quadtree + max-score pruning | Directly optimises metadata cost: fewer, larger rectangles = fewer headers |
| Dead-tile masking (std < 1e-5) | Black space must not be scored as "the rarest thing" (a real pitfall we hit & fixed) |

---

## 3. Technical feasibility

**Target:** LEON3 (dual-core LEON-based OBC; Core 1 sandbox).

- **Runs on LEON3 architecture — yes.** The reference implementation is pure NumPy/OpenCV vectorised
  array code with no floating-point-exotic operations. Every stage maps to plain multiply-add,
  comparison and min/max reductions. The Phase 2 deliverable is C (RTEMS/buildable on LEON3), ported
  from the reference Python and validated on the provided Hera software simulation layer.
- **Asynchronous operation — yes.** The pipeline is intentionally **stateless and frame-by-frame**.
  Each invocation takes one frame (or one sub-window from the datapool) and returns a rectangle list.
  There is no state that survives a frame, so an abrupt stop (Safe Mode / anomaly) loses nothing —
  the next run simply processes the next frame.
- **Fits the 2–3 h daily slot — yes.** The workload is a bounded, pre-computable amount of arithmetic:
  for a 1024×1024 frame, 4096 tiles and 4 features/tile, the dominant cost is the feature extraction
  and the quadtree recursion over a 64×64 score grid. There is **no training on history** and no
  neural network — the iForest is fit on the current frame only (64 trees, max_samples=512). A single
  frame completes in well under a second on a desktop reference machine; the ported C version is
  sized for the LEON3 budget with margin.
- **No dynamic allocation inside loops.** All array sizes are pre-computed from the tile grid
  dimensions, in line with the flight-software memory-protection requirement.
- **No direct hardware access.** The algorithm consumes an image buffer/array and emits a rectangle
  list — it needs only the platform-provided image-acquisition service as input and a telemetry
  service as output (Section 4).

**Demonstrated.** The pipeline runs end-to-end today on 5 real Rosetta NavCam comet frames
(1024×1024 and 738×738). Reproduction: `image_compression/demo.py`, or the Colab notebook
`notebooks/hera_image_compression_demo.ipynb`.

---

## 4. Compliance

The experiment conforms to the Core 1 sandbox constraints stated in the call:

| Requirement | How the design satisfies it |
|---|---|
| **No direct spacecraft hardware access** | The algorithm is a pure function: `image → ROI list`. It needs only the platform image-acquisition service (input) and the telemetry/downlink service (output), accessed through the interfaces of ANNEX A / ANNEX C. No register, bus or Core-0 access. |
| **Automatic stop on anomaly / Safe Mode** | Stateless per-frame design. Any interruption simply terminates the current frame's computation; the next invocation starts fresh. No graceful-shutdown dependency, no on-disk or persistent state to corrupt. |
| **Strict memory protection** | All buffers are pre-sized from the grid dimensions; no dynamic allocation in inner loops; no out-of-region writes. Memory footprint is bounded and reported in Section 8. |
| **Execution-window shaped** | The algorithm is chunked to process **one frame per invocation** (or a bounded batch), returning results within the 2–3 h window with a clear completion condition (see Section 7). |
| **Downlink-only product** | Output is a compact rectangle list + metadata (a few bytes per rectangle) plus the selected tile pixels — strictly a *reduction* of the candidate telemetry, never an expansion. |

Compliance is ensured **by design**, not by runtime guard rails. The Phase 2 package will include the
V&V Test Plan mapping each constraint above to a concrete, executable test.

**Documents the final submission must attach/consult:**
`Technical and operational requirements.pdf`, `ANNEX A - Hera interface API documentation.pdf`,
`ANNEX B - Datapool.pdf`, `ANNEX C - Hera client stub, user and integration guide.pdf`,
`ANNEX D - bin2c`, `ANNEX E - example, AFC image acquisition and smart compression`,
`Hera software simulation layer.tar.gz`.

---

## 5. Benefits (quantified)

Measured on the 5-image reference set with the current prototype (`demo.py`, tile=16, threshold=0.5,
max_block=16). "Payload vs full frame" counts selected pixels + rectangle metadata (5 ints × 4 B per
rectangle):

| Frame | Size | Heuristic — payload | Isolation Forest — payload |
|---|---|---|---|
| `Comet_from_17.4_km_NavCam` | 1024×1024 | 97.7 % | **37.6 %** |
| `Comet_from_19.4_km_NavCam` | 1024×1024 | 79.8 % | **58.7 %** |
| `Comet_from_20_km_NavCam` | 1024×1024 | 95.4 % | **28.8 %** |
| `Comet_on_10_February_2016` | 1024×1024 | 41.7 % | **45.4 %** |
| `Comet_on_15_April_2015_b` | 738×738 | 11.8 % | **14.4 %** |

**Headline numbers:**

- **Best close-range case (iForest): ~28.8 % payload → ~71 % downlink reduction**, while retaining
  the high-interest geological tiles.
- **Best case overall: ~11.8 % payload → ~88 % reduction.**
- The **heuristic mode** is a conservative, fully deterministic fallback; the **Isolation Forest mode**
  is the aggressive, autonomy-driven mode. The threshold is a single tunable knob (score ≥ 0.5 by
  default) that trades reduction ratio against coverage.
- Metadata overhead is explicitly modelled (5 ints/rectangle) and kept small by the quadtree merge —
  a design that pays for itself by sending *fewer, larger* blocks.

**Beyond bandwidth.** The same ROI list is directly reusable as:
- an **onboard image-selection / prioritisation** signal (which frames to keep at all),
- a **change-detection** trigger across consecutive frames (new ROI ⇒ new event),
- a **feature-tracking** seed (ROIs are where trackable landmarks live),
- a **science-classification** front-end ("auto-geologist": space vs regolith vs boulders vs shadow).

---

## 6. Maturity

The prototype is **already implemented and running**, not a paper concept:

- Working Python reference implementation (features, scoring, quadtree merging, headless demo).
- Verified end-to-end on **real flight imagery** (Rosetta NavCam comet frames of 67P).
- Reproducible artefacts published: `demo.py`, the Colab notebook, and the produced visualisations.
- **Two independent scoring back-ends** implemented and benchmarked against each other.
- Known pitfall discovered and fixed during development (dead-space masking for the Isolation
  Forest — prevent black space being scored as the anomaly).

**Tested where:** desktop CPU reference environment + notebook/simulator-level runs. **Not yet** tested
on flight hardware — the LEON3 C port and simulator validation are Phase 2 work.

This places the idea at **TRL ~4** (component validated in a laboratory/reference environment): a
functioning, measured prototype exists and has produced quantitative results.

---

## 7. Operational concept

**Per-execution-window behaviour (one 2–3 h slot):**

1. **Startup** — the ESW is launched by the Experimental Software Framework. It performs a
   deterministic self-check (buffer allocation sizing, feature-count sanity) and reports *Housekeeping*
   telemetry (state = READY, configured tile size / threshold / mode).
2. **Data acquisition** — from the platform image-acquisition service it requests the current AFC frame
   (or a bounded batch of frames). Only the image buffer is needed; no attitude or hardware registers.
3. **Processing** — the five pipeline stages (Section 2) execute on the frame: tile features → scoring
   → smoothing → quadtree merge → ROI list. Each frame is independent; results accumulate into the
   run's output set.
4. **Output generation** — the ESW emits, per processed frame:
   - **Events**: e.g. "N ROIs selected, top score S, coverage X %" and threshold/coverage alerts,
   - **Science telemetry**: the prioritised ROI rectangle list + metadata, plus the selected tile
     pixel payload (or a reference to the pixels to be downlinked by the platform),
   - **Housekeeping**: rolling counters (frames processed, tiles pruned, cumulative reduction ratio).
5. **Completion condition** — the run ends when the requested number of frames have been processed, or
   the execution window closes, or the platform signals stop. Because the pipeline is stateless, an
   early stop is safe: the ROI lists already produced remain valid.
6. **Repeat** across consecutive execution days of the campaign.

**Acquisition strategy.** Process frames as they become available during the experiment opportunity;
optionally run a *temporal* variant (compare consecutive frames' ROI maps) to surface transient events
(dust, new ejecta, illumination-driven changes) — a natural extension the ROI map already enables.

**Expected duration.** A campaign of **~10–15 execution days** within the 4-week August 2027 window is
sufficient to collect a statistically meaningful set of reduction ratios across varying ranges
(5–30 km) and illumination geometries. Justification: each day yields a bounded batch of frames; the
scientific claim (measured downlink reduction + ROI stability) needs coverage across range and phase
angle, which is achieved by sampling one window per opportunity rather than continuously.

---

## 8. Resource estimates

Estimates for the reference 1024×1024 frame, 16-px tiles (64×64 = 4096 tiles), 4 features/tile:

| Resource | Estimate | Notes |
|---|---|---|
| **Execution time / run** | 2–3 h window used; actual compute per frame well within it | One frame per invocation; bounded batch per run |
| **CPU utilisation** | Burst, well below 100 % | Dominant cost: vectorised feature extraction + quadtree recursion over a 64×64 score grid |
| **Memory footprint (RAM)** | **~ low-MB class** | Pre-sized arrays: frame buffer (1 × 1024² × 1 B), tile-feature tensor (64×64×4 × 4 B), score grid (64×64 × 4 B), ROI list. No dynamic allocation in loops. Exact kB to be finalised in the Phase 2 DDF. |
| **Datapool categories used** | **Science images** (AFC frames) — read-only | Per ANNEX B Datapool |
| **Data generation — Housekeeping** | Small, per run | State, config echo, counters |
| **Data generation — Events** | Small, per frame / on alert | ROI count, top score, coverage, threshold alerts |
| **Data generation — Science telemetry** | Compact | ROI rectangle list (5 ints/rect) + metadata + selected-tile payload reference |

Net telemetry effect: **reduction**, not growth — the product is a prioritised subset of the candidate
imagery, with metadata overhead explicitly minimised by the quadtree merge.

---

## 9. Management and team

*(max 2–3 pages — doplniť pred podaním)*

- **Organisation:** *(názov, typ — startup / research group / individual, krajina)*
- **Team & roles:** *(Marek Racko — algorithm & software; + prípadní ďalší členovia)*
- **Relevant background & experience:** *(skúsenosti s computer vision, embedded/edge software,
  Python/C, CI, publishing)*
- **Why us:** a working prototype already exists and is measured on real flight imagery; the team
  works fast and iteratively (reference implementation → Phase 2 C port → LEON3 simulator validation).

---

## Appendix A — Mapping to the call's target capabilities

| Call capability | This experiment |
|---|---|
| Edge Computing | Onboard processing of asteroid imagery; downlink only the relevant information |
| Data Compression & Prioritisation | Explicit bandwidth-aware downlink reduction (measured 28.8–88 % of full frame) |
| Onboard Image Processing & Feature Tracking | Per-tile feature extraction; ROI list seeds visual tracking / descriptors |
| Inference for Anomaly Detection / Science Classification | Isolation-Forest anomaly detection + optional "auto-geologist" classifier front-end |
| Autonomy | Onboard decision on *what to send*, reducing reliance on ground intervention |

## Appendix B — Reproduction

```
image_compression/
  util.py · features.py · scoring.py · tiles_merging.py · main.py · demo.py
notebooks/hera_image_compression_demo.ipynb
doc/Autonomous_Edge_Inference_Tile_Downlink_Pipeline.md
doc/images_datasets.md            # datasets for Phase-2 validation (DART DRACO, OSIRIS-REx, ...)
```

Run: `cd image_compression && python demo.py` → writes visualisations + prints per-frame reduction stats.
