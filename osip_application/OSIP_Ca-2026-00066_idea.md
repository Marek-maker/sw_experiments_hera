# OSIP Call for Ideas — Autonomous Software Experiments on Hera

**Campaign:** Ca-2026-00066 · **Reference:** ESA-HERA-TECS
**Title:** Unsupervised Geological Saliency Selection for Bandwidth-Aware Downlink of Asteroid Imagery
**Target:** Hera Core 1 sandbox (GR712RC LEON3)

| Field | Value |
|---|---|
| Proposing organisation | *(doplniť: názov / typ organizácie)* |
| Contact | Marek Racko |
| Country | Slovakia |
| Submission language | English |
| Campaign manager | Jorge Lopez Trescastro (ESA) |
| Deadline | submission phase closes ~15 September 2026 |

---

## 0. Novelty statement

*The main evaluation criterion for the Idea phase is novelty. This is what is new here.*

**The core idea.** Hera will arrive at a body that has never been surveyed, and the scientific
question is: *what is on Dimorphos that nobody expected?* You cannot train a supervised
"interesting geology" classifier before you have seen the place. What you **can** do on board is
ask a statistical question of every frame: *which regions do not look like the rest of this
frame?* This experiment uses an **unsupervised novelty score per image tile** (an Isolation
Forest fitted on-board, per frame, over four cheap per-tile features) to decide, autonomously,
**which parts of an image deserve the downlink budget — and then merges those tiles under an
explicit cost model that minimises the telemetry metadata itself.**

**What is novel, precisely:**

1. **Selection is driven by unsupervised novelty, not by a pre-trained target model.** No labelled
   dataset of Dimorphos geology exists or can exist before arrival. The method needs none.
2. **The ROI merge is optimised against the telemetry budget, not just for segmentation quality.**
   A quadtree merge with bandwidth-aware pruning minimises *number of rectangles* — i.e. it treats
   the ROI header cost (5 ints per rectangle) as a first-class term, so that "interesting area" and
   "metadata overhead" are traded off explicitly. This is the part we have not seen in the
   onboard-compression literature, which optimises bytes of a *whole* frame rather than the
   *choice* of what constitutes a frame's worth of value.
3. **Stateless, per-frame, allocation-free design by construction**, so abrupt termination by the
   sandbox (anomaly / Safe Mode) cannot corrupt state.

**Measured today, on the reference AFC image set distributed with this call** (404 frames,
1020×1020, 8-bit):

| Mode | Median payload vs full frame | Frames that fit the 12 MB / 3 h science budget |
|---|---|---|
| Heuristic baseline | 50.8 % | 23.8 |
| **Unsupervised novelty (Isolation Forest)** | **15.0 %** | **80.4** (6.6× the raw-frame count) |

The full pipeline already exists as a measured reference implementation; Phase 2 is the C port to
the flight toolchain and validation on the Hera simulator.

---

## 1. The problem

Deep-space missions still operate through pre-planned procedures and heavy ground intervention.
Two constraints bite at Dimorphos specifically:

- **You do not know what the surface looks like until you get there.** The DART impact created a
  new crater and scattered ejecta whose morphology is, by definition, unobserved. Acquiring a
  frame and *deciding on the ground* whether it was worth having costs a round-trip — far too
  slow for a fast flyby or a short close-range campaign.
- **Downlink is the bottleneck.** The call's own telemetry limits are: **2048 bytes per science
  packet** and **12 MB per 3-hour execution slot**. A single raw AFC frame is 1020×1020×8 bit =
  **0.99 MB**, so the entire 3-hour budget is worth **≈12 raw frames**. Everything the spacecraft
  sends must therefore be chosen, not merely compressed.

The current state of the art addresses the *encoding* of a frame (lossless compression, e.g. the
Smart-RLE example in ANNEX E of this call). It does **not** address the *selection* problem:
deciding, autonomously and on board, **which parts of an image carry the information a geologist
would want**, given that nobody knows in advance what will be there.

## 2. The solution

A self-contained C application that, for each acquired AFC frame:

1. **Acquires** one frame and takes a read-only pointer to the image buffer.
2. **Tiles** it into 16×16 pixel blocks and computes four cheap features per tile — mean, standard
   deviation, mean gradient magnitude, and an outlier/contrast indicator — as a single vectorised
   pass with no per-tile Python-style loop and no dynamic allocation.
3. **Scores** every tile with an **Isolation Forest** (an ensemble of random isolation trees).
   The forest is **trained on the ground** on a reference set and shipped in the binary as a
   **frozen static table**; on board only the *inference* tree walk runs. Tiles that are easiest to
   isolate under the random splits are the *novel* tiles — corners, rims, boulders, unusual
   patches — with no labels required.
4. **Smooths** the score grid (dilation + blur) so a single feature becomes one coherent block
   instead of twenty scattered tiles.
5. **Merges** the scored tiles into rectangles with a **quadtree**, choosing merges that maximise
   the retained score while explicitly minimising the **number of rectangles** — because each
   rectangle costs 5 ints of metadata, which is real telemetry.
6. **Reports** the ROI rectangle list (plus a compact summary) through the platform's science
   telemetry service, so ground can reconstruct exactly which regions were prioritised.

An additional mode — a fully deterministic **heuristic** scorer (weighted feature average) — is
implemented as a **baseline and a fallback**. It needs no stored model at all and is the safest
thing to fly first; the measurements below show clearly that it is a baseline, not the product.

**Deliberate engineering choices, and why**

| Choice | Rationale |
|---|---|
| 16×16 tiles, vectorised feature pass | No inner-loop allocation; deterministic; LEON3-friendly |
| Two scoring modes | Heuristic = lowest CPU, fully deterministic, zero stored model; iForest = true unsupervised novelty |
| Forest frozen as a static table | Inference only on board; satisfies the no-`malloc` rule absolutely |
| Spatial dilation + blur before merging | Prevents the quadtree from over-splitting one feature into many rectangles |
| Quadtree with bandwidth-aware pruning | Directly optimises metadata cost: fewer, larger rectangles = fewer headers |
| Dead-tile masking (std ≈ 0) | Empty black space must not be scored as "the rarest thing" — a real pitfall we hit and fixed |

### 2.1 How this differs from ESA's own ANNEX E example

The call ships **ANNEX E: "example, AFC image acquisition and smart compression"**
(ESA-HERA-TECS-MAN-2026-002054), which implements a **lossless Smart RLE** codec: each 1020-pixel
line is run-length encoded and stored compressed only if that is smaller than raw.

That example and this experiment operate on **different axes**, and are **complementary**:

| | ANNEX E (ESA example) | This experiment |
|---|---|---|
| Question asked | *How do I encode this frame in fewer bytes?* | *Which parts of this frame are worth sending at all?* |
| Layer | Encoding / entropy coding | Semantic selection / attention |
| Loss | Lossless — reconstruction is bit-identical | Lossy — a prioritised subset is sent |
| Input | Whole frame, line by line | Per-tile novelty scores over the whole frame |
| Output | Fewer bytes for the **same** image | Fewer **scenes** for the **same** byte budget |
| Metadata | None | ROI rectangle list, explicitly costed |

The two compose naturally: our ROI rectangles define **which regions to preserve**, and a codec of
the ANNEX E type (or the spacecraft's own lossless compressor) then encodes **those regions**
efficiently. We would position this experiment as the **selection layer that sits above** the
encoding example — and we explicitly do **not** propose to replace or re-implement the ESA
provided codec.

## 3. Technical feasibility

The target environment is tightly specified by the call's *Technical and operational
requirements* document, and the design was written against those constraints:

| Requirement (from the call) | How this design satisfies it |
|---|---|
| **GR712RC LEON3**, bare metal, no OS | Single-threaded, self-contained, explicit control flow. No OS services, no threading, no timers beyond those provided |
| **C, MISRA rules, ECSS E-ST-40C Category D** | Plain C99 with fixed-width integer types; no pointer arithmetic beyond bounded indexing; a MISRA-oriented build (static analysis, documented deviations) is part of the Phase 2 deliverable |
| **No dynamic allocation** (`malloc`/`free` forbidden) | **All buffers are statically sized** from the fixed frame geometry (1020×1020, 16-px tiles → 64×64 score grid). The isolation forest is a **frozen `static const` array**, not built at run time. No allocation exists anywhere in the code path |
| **No system calls** | None used |
| **No external libraries** (only ESA's LibmCS) | Arithmetic is integer / fixed-point where possible; any transcendentals resolved through LibmCS only. The offline training tool is a separate ground-side Python script and never ships to the spacecraft |
| **No direct hardware access** | The application only calls the platform interfaces: `Hera_AFC_AcquireSingleImage()` to request a frame, `Hera_AFC_GetImageBuffer()` for a read-only pointer, `Hera_Science_Report()` and `Hera_HK_Report()` to emit results. No registers, no buses, no Core-0 access |
| **Auto-stop on anomaly / Safe Mode** | The algorithm is **stateless per frame**. Any termination simply ends the current frame's computation; the next invocation starts from a clean, fully initialised state. There is nothing to corrupt and no graceful-shutdown dependency |

**Compute budget — stated honestly.** On a desktop CPU this pipeline runs at **≈0.21 s per
1020×1020 frame** (404 frames in 86 s, single-threaded). A LEON3 at a few hundred MHz with no SIMD
is expected to be substantially slower; a conservative planning figure of **100–1000× slower**
gives roughly **0.3–3.5 minutes of CPU per frame**. Within a 2–3 hour execution window that still
leaves room for **tens of frames per run**, and the algorithm is chunked to process a **bounded
batch** with a clear completion condition. Exact timing is a **Phase 2 measurement on the Hera
software simulation layer** (supplied with this call), not a Phase 1 claim.

**Feasibility of the reference implementation.** The pipeline already exists and runs: 16-px
tiling, per-tile feature extraction, both scorers, quadtree merging and visualisation, with 404
reference frames processed end to end. The Python/NumPy form is the *algorithmic reference*; the
Phase 2 work is the C port to BCC 4.4.2 (Frontgrade Gaisler Bare C Cross Compiler for LEON3 GCC).

## 4. Compliance

| Constraint | Statement |
|---|---|
| Hera interface API (ANNEX A) | Only the documented acquisition / buffer / report entry points are used |
| Data pool (ANNEX B) | **Read-only.** No parameter writes. No data-pool access is required for the core algorithm; optionally the spacecraft range/attitude parameters could gate processing, and these would be listed in the Phase 2 ICD |
| Binary packaging (ANNEX D, bin2c) | The frozen forest table is emitted as a C header via the supplied `bin2c` tool — the intended workflow |
| Client stub / integration (ANNEX C) | Development and integration follow the stub guide; validation runs against the supplied software simulation layer |
| HK telemetry (PUS 3) | ≤ **256 bytes** per packet, ≥ **5 minutes** between reports — the design emits a small, fixed HK structure (mode, frame counter, last-result code, ROI count, top score), a few dozen bytes, at most once per processing step |
| Event reporting (PUS 5) | ≤ **50 bytes**, ≥ **20 seconds** apart, Informational/Warning only. Used for discrete conditions: run start, run complete, threshold crossing, recoverable anomaly — **not** for heartbeats |
| Science telemetry | ≤ **2048 bytes** per packet, ≤ **12 MB per 3-hour slot**. The ROI rectangle list (5 int32 per rectangle) plus the selected pixels are budgeted explicitly; the reference measurement is **0.149 MB median per frame** |
| Safe-mode tolerance | Stateless per-frame design; no persistent state; no graceful shutdown dependency |
| Memory protection | All buffers pre-sized and bounded; no out-of-region writes; no dynamic allocation |

## 5. Benefits

Measured on the **404-frame AFC reference set supplied with this call** (1020×1020, 8-bit,
guidance-image format). Payload model: selected pixels at 8 bit **plus** ROI metadata
(5 int32 per rectangle) — the same model in both rows, and deliberately conservative (it charges
1 byte per selected pixel, with **no** credit for the lossless compression that would follow).

| Scoring mode | Median payload vs full frame | Range (min–max) | Frames fitting the 12 MB / 3 h budget |
|---|---|---|---|
| Heuristic baseline | **50.8 %** | 16.3 – 97.7 % | 23.8 |
| **Unsupervised novelty (iForest)** | **15.0 %** | **5.9 – 36.5 %** | **80.4** |

**Reading of the result.** A raw 1020×1020 frame is 0.99 MB, so the call's 12 MB science budget
buys **≈12.1 raw frames** per 3-hour slot. With novelty-driven selection at the measured median,
the same budget carries **≈80 frames** — a **6.6× increase in scene coverage per downlink window**
(best observed frame: 5.9 % → 17×). At the *worst* measured frame the ratio is still 2.7× better
than raw. Equivalently: keep the current frame count and free ~85 % of the science budget for
Hera's other instruments.

**Additional benefits**

- **Ground-station time becomes optional, not mandatory**, for the decision of *what to look at*.
  The prioritisation happens during the 2–3 h window, not in the next planning cycle.
- **The ROI list is self-describing.** Ground receives coordinates plus scores, so an unexpected
  detection can be re-examined and the selection criteria re-tuned in a later run — the experiment
  produces an audit trail, not just pixels.
- **The product degrades gracefully.** Because it is a *prioritised subset*, a smaller budget is
  honoured by lowering the threshold; the most novel regions remain.

**Limitation, stated plainly (and already characterised).** The current four features are
contrast/edge-driven. On a frame containing a large, uniformly lit body, the **limb** (high
gradient) scores high and the **body interior** (low variance) scores low, so the interior can be
dropped even though it is scientifically valuable — we observed exactly this on the reference
frames. This is a known and bounded weakness with a concrete fix in Phase 2: add a
coverage/area-saliency term and re-tune the threshold, validated against the supplied simulation
layer. We report it here rather than let an evaluator find it. The comparison of interest is
therefore **not** "iForest beats heuristic" (both are our own code) but "either of our modes
versus the 100 % baseline", with the heuristic clearly labelled as a baseline.

## 6. Maturity

- **Working reference implementation exists** and has been run end to end on the full 404-frame
  reference set: tiling, feature extraction, both scoring modes, quadtree merge, ROI output,
  visualisation and per-frame statistics (CSV + summary reproduced above).
- **Reproducible**: a headless benchmark script processes the directory of AFC frames and emits
  `afc_stats.csv` and `afc_summary.txt`; a demo notebook and a headless demo script are in the
  same repository.
- **Tested against real mission data format**: the input is the call's own AFC reference set
  (1020×1020, 8-bit, guidance telemetry naming), loaded and processed without conversion.
- **What is *not* yet done, and is Phase 2**: the C port, the `bin2c`-packaged frozen model,
  MISRA/ECSS static-analysis evidence, and timing/robustness measurements on the Hera software
  simulation layer.

## 7. Operational concept

1. **Start-up.** On invocation the application initialises only static state: score grid, ROI
   buffers, and the frozen forest pointer. Cost is negligible.
2. **Acquisition strategy.** The experiment requests an AFC frame at a paced cadence (one frame
   per opportunity; the cadence is a configuration parameter chosen so that one run stays well
   inside the 2–3 h window and the science budget).
3. **Processing steps.** Feature extraction → forest inference (`score_tiles_iforest`) or, in the
   deterministic fallback configuration, the weighted-feature score → dilation/blur smoothing →
   quadtree merge with bandwidth-aware pruning → ROI rectangle list.
4. **Output generation.** A Science Data report carries the ROI list and the selected tiles; an HK
   report carries mode, counter, ROI count and top score; an Event is emitted only on run start,
   run completion, or a recoverable anomaly.
5. **Completion conditions.** The run ends after a fixed, pre-declared **number of frames**
   (bounded batch). This gives a deterministic, testable completion condition, and a bounded
   worst-case run time.
6. **Expected duration.** Seconds-to-minutes of CPU per frame, bounded to a pre-declared frame
   count, comfortably inside the 2–3 h window; exact figures to be confirmed on the simulation
   layer in Phase 2.

**Ground-side use.** Downlinked ROIs let the team reconstruct which regions the on-board
prioritiser considered novel, compare that against the images themselves, and refine the
threshold/feature set for subsequent runs — a closed loop the experiment is explicitly designed to
support.

## 8. Resource estimates

Reference: 1020×1020 AFC frame, 16-px tiles → **64×64 = 4096 tiles**, 4 features per tile.

| Resource | Estimate | Basis |
|---|---|---|
| **Execution time / run** | Bounded batch (fixed frame count) inside the 2–3 h window | Desktop 0.21 s/frame; LEON3 planning figure 100–1000× slower; exact figure from the simulator in Phase 2 |
| **CPU utilisation** | Burst, not continuous; dominated by the forest tree walk and the quadtree recursion over a 64×64 grid | Computed from the reference implementation's work profile |
| **Memory — frame buffer** | 1020×1020 × 1 B ≈ **0.99 MB** (read-only pointer to the platform buffer; not our allocation) | Fixed frame geometry |
| **Memory — our working set** | Score grid 64×64 float ≈ 16 kB; tile-feature tensor 64×64×4 float ≈ 64 kB; ROI list 5 int32 × N_rect (median 284 → ≈ 5.7 kB); code + frozen forest table | Pre-sized, static; no `malloc` anywhere |
| **Total worst-case working memory** | **Well under 1 MB**, dominated by the frame buffer we only read | Bounded by construction |
| **Data pool categories used** | None required for the core algorithm (read-only if a cadence gate is added) | ANNEX B |
| **HK telemetry** | A few dozen bytes per processing step; ≤ 256 B packet; ≥ 5 min interval | PUS 3 |
| **Event telemetry** | ≤ 50 B per event; start / completion / anomaly only; ≥ 20 s interval | PUS 5 |
| **Science telemetry** | Median **0.149 MB per frame** (ROI pixels + metadata); ≤ 2048 B per packet; ≤ 12 MB per 3 h slot | Measured, 404-frame reference set |

Net telemetry effect: a **reduction**, not growth — the product is a prioritised subset of the
candidate imagery, with metadata overhead explicitly minimised by the quadtree merge.

## 9. Management and team

*(max 2–3 pages — doplniť pred podaním)*

- **Organisation:** *(názov, typ — individual / startup / research group, krajina)*
- **Team & roles:** *(Marek Racko — algorithm & software; prípadní ďalší členovia)*
- **Relevant background & experience:** *(computer vision, embedded/C, Python, CI, prior projects)*
- **Why us:** a working, measured prototype already exists and has been exercised on the call's own
  reference image set; the method was designed against the flight constraints from the start
  (no allocation, stateless, bounded memory), so the Phase 2 C port is a translation and validation
  exercise rather than a redesign.

---

## Appendix A — Mapping to the call's target capabilities

| Call capability | This experiment |
|---|---|
| Edge Computing | On-board processing of asteroid imagery; only the relevant information is downlinked |
| Data Compression & Prioritisation | Explicit bandwidth-aware prioritisation; measured median 15.0 % of full frame on the 404-frame reference set |
| Onboard Image Processing & Feature Tracking | Per-tile feature extraction; the ROI list seeds visual tracking and descriptors |
| Inference for Anomaly Detection / Science Classification | Unsupervised Isolation-Forest novelty; no labelled data required |
| Autonomy | On-board decision on *what to send*, removing the ground round-trip from the loop |

## Appendix B — Reproduction

```
image_compression/
  utils.py · features.py · scoring.py · tiles_merging.py · main.py · demo.py
  afc_benchmark.py          # headless benchmark over a directory of AFC frames
notebooks/hera_image_compression_demo.ipynb
doc/Autonomous_Edge_Inference_Tile_Downlink_Pipeline.md
doc/images_datasets.md
```

Reproduce the numbers quoted above:

```
python afc_benchmark.py --dir <AFC_IMAGES> --vis 3
```

Outputs `results/afc/afc_stats.csv`, `results/afc/afc_summary.txt` and, for the first frames,
separate full-size visualisations (original, score heatmap, ROI overlay, selected content).
