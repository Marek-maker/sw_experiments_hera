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

**The core idea.** Hera will arrive at a body nobody has surveyed, and the scientific question is:
*what is on Dimorphos that nobody expected?* A supervised "interesting geology" classifier cannot
be trained before we have seen the place. What *can* run on board is a statistical question asked
of every frame: **which regions do not look like the rest of this frame?** This experiment scores
every image tile for **unsupervised novelty** (an Isolation Forest over four cheap per-tile
features) and uses those scores to decide, autonomously, **which parts of an image deserve the
downlink budget** — with the ROI merge itself optimised against the telemetry budget.

**What is novel, precisely:**

1. **Selection is driven by unsupervised novelty, not by a pre-trained target model.** No labelled
   dataset of Dimorphos geology exists, or can exist, before arrival. The method needs none.
2. **The ROI merge is optimised against the telemetry budget, not only for segmentation quality.**
   A quadtree merge with bandwidth-aware pruning minimises the *number of rectangles*, treating the
   ROI header cost (5 ints per rectangle) as a first-class term — so "interesting area" and
   "metadata overhead" are traded off explicitly. Onboard compression work we are aware of optimises
   the bytes of a whole frame; this optimises the *choice of what constitutes a frame's worth of
   value*.
3. **Selection by per-frame score quantile, not by an absolute threshold.** This is what makes the
   telemetry volume *predictable* — the property an operator actually needs when planning a
   downlink budget (see §5).
4. **Stateless, per-frame, allocation-free by construction**, so abrupt termination by the sandbox
   (anomaly / Safe Mode) cannot corrupt anything.

**Measured on the 404-frame AFC reference set supplied with this call** (1020×1020, 8-bit,
12-px tiles so the whole frame is covered):

| Mode | Median payload vs full frame | Worst-case payload | Frames per 12 MB / 3 h slot |
|---|---|---|---|
| Heuristic baseline | 46.9 % | 100.1 % | 25.8 |
| Per-frame-fit iForest *(analysis reference; not flight-feasible)* | 11.9 % | 28.3 % | 101.7 |
| **Frozen ground-trained iForest + quantile q=0.10** *(flight design)* | **8.8 %** | **11.0 %** | **136.8** (11.3× the raw-frame count) |

A working reference implementation exists and produced these numbers; Phase 2 is the C port to the
flight toolchain and validation on the Hera software simulation layer.

---

## 1. The problem

Deep-space missions are still operated through pre-planned procedures and heavy ground
intervention. Two constraints bite at Dimorphos specifically:

- **You do not know what the surface looks like until you get there.** The DART impact created a
  crater and scattered ejecta whose morphology is, by definition, unobserved. Acquiring a frame and
  deciding *on the ground* whether it was worth having costs a round-trip — far too slow for a
  short close-range campaign or a fast flyby.
- **Downlink is the bottleneck.** The call's own limits are **2048 bytes per science packet** and
  **12 MB per 3-hour execution slot**. One raw AFC frame is 1020×1020×8 bit = **0.99 MB**, so the
  entire 3-hour budget is worth **≈12 raw frames**. What the spacecraft sends must therefore be
  *chosen*, not merely compressed.

Today's onboard processing addresses the *encoding* of a frame (lossless compression — including
the Smart-RLE example shipped with this very call, ANNEX E). It does not address *selection*:
deciding autonomously, on board, **which parts of an image carry the information a geologist would
want**, when nobody knows in advance what will be there.

## 2. The solution

A self-contained C application; for each acquired AFC frame:

1. **Acquire** one frame, take a read-only pointer to the image buffer.
2. **Tile** it into 12×12 pixel blocks (85×85 = 7225 tiles) and compute four cheap features per
   tile — mean, standard deviation, mean gradient magnitude, high-contrast (outlier) fraction — in
   a single vectorised pass, with no per-tile loop and no dynamic allocation.
3. **Score** every tile for novelty with an **Isolation Forest whose trees are frozen at build
   time**. The forest is trained **on the ground**, on a reference image set, and shipped in the
   binary as a **static table**; on board only the *inference* tree walk runs. No random number
   generator, no tree construction and no allocation are needed in flight. Tiles that random splits
   isolate most easily are the *novel* ones — rims, boulders, corners, unusual patches — with no
   labels required.
4. **Smooth** the score grid (dilation + blur) so a single feature produces one coherent block
   rather than twenty scattered tiles.
5. **Select** the top *q* fraction of scored tiles by score quantile. Selecting a *quantile* rather
   than an absolute score threshold is deliberate: see §5.
6. **Merge** the selected tiles into rectangles with a **quadtree**, choosing merges that retain
   score while explicitly minimising the **number of rectangles** — each rectangle costs 5 ints of
   telemetry, and that is real.
7. **Report** the ROI rectangle list through the platform's science telemetry service, so ground
   can reconstruct exactly which regions were prioritised.

A second, fully deterministic **heuristic** scorer (weighted feature average, no stored model) is
kept as a **baseline and fallback**. Measurements below show plainly that it is a baseline, not the
product.

**Deliberate engineering choices**

| Choice | Rationale |
|---|---|
| **12×12 tiles** | 12 divides 1020 exactly → the **entire** frame is tiled; 16-px tiles would silently drop a 12-px edge strip (~2.3 % of pixels). 12 px also gave the lowest measured payload of the sizes tested (16 / 20 px) |
| Frozen forest as a static table | Inference only on board — satisfies the no-`malloc` rule absolutely |
| Quantile selection | OOD-robust and gives a **predictable** per-frame telemetry volume (measured spread 1.24× vs 24× for a fixed threshold) |
| Dilation + blur before merging | Prevents the quadtree over-splitting one feature into many rectangles |
| Quadtree with bandwidth-aware pruning | Directly optimises metadata cost: fewer, larger rectangles = fewer headers |
| Dead-tile masking (std ≈ 0) | Empty black space must not be scored as "the rarest thing" — a real pitfall we hit and fixed |

### 2.1 How this differs from ESA's own ANNEX E example

The call ships **ANNEX E: "example, AFC image acquisition and smart compression"**
(ESA-HERA-TECS-MAN-2026-002054) — a **lossless Smart RLE** codec: each 1020-pixel line is
run-length encoded and stored compressed only if smaller than raw.

The two operate on **different axes** and are **complementary**:

| | ANNEX E (ESA example) | This experiment |
|---|---|---|
| Question asked | *How do I encode this frame in fewer bytes?* | *Which parts of this frame are worth sending at all?* |
| Layer | Encoding / entropy coding | Semantic selection / attention |
| Loss | Lossless — bit-identical reconstruction | Lossy — a prioritised subset is sent |
| Input | Whole frame, line by line | Per-tile novelty scores over the whole frame |
| Output | Fewer bytes for the **same** image | Fewer **scenes** for the **same** byte budget |
| Metadata | None | ROI rectangle list, explicitly costed |

The two compose naturally: our ROI rectangles define **which regions to preserve**, and a codec of
the ANNEX E type (or the spacecraft's own lossless compressor) encodes **those regions** efficiently.
We position this as the **selection layer above** the encoding example, and do **not** propose to
replace or re-implement ESA's codec.

## 3. Technical feasibility

The design was written against the call's *Technical and operational requirements*
(ESA-HERA-TECS-RS-2026-002045):

| Requirement (from the call) | How this design satisfies it |
|---|---|
| **GR712RC LEON3**, bare metal, no OS | Single-threaded, self-contained, explicit control flow; no OS services, no threading |
| **C, MISRA, ECSS E-ST-40C Category D** | Plain C99 with fixed-width types; bounded indexing only; a MISRA-oriented build with static analysis and documented deviations is a Phase 2 deliverable |
| **No dynamic allocation** | All buffers statically sized from fixed frame geometry (1020×1020, 12-px tiles → 85×85 grid). The forest is a **frozen `static const` table**, not built at run time. No allocation exists anywhere in the flight code path |
| **No system calls** | None used |
| **No external libraries** (only LibmCS) | Integer / fixed-point arithmetic where possible; transcendentals via LibmCS only. The offline training tool is a ground-side Python script and never ships |
| **No direct hardware access** | Only platform interfaces: `Hera_AFC_AcquireSingleImage()`, `Hera_AFC_GetImageBuffer()`, `Hera_Science_Report()`, `Hera_HK_Report()`. No registers, no buses, no Core-0 access |
| **Auto-stop on anomaly / Safe Mode** | **Stateless per frame.** Termination simply ends the current frame's computation; the next invocation starts from clean, fully initialised state. Nothing to corrupt, no graceful-shutdown dependency |

**Why the frozen model changes nothing about the result.** A ground-trained forest is only useful if
its inferred scores are as selective as a per-frame fit. Measured, on the same 404 frames:
per-frame fit **12.0 %** median vs frozen model **10.1 %** median at the same nominal threshold —
i.e. the flight-feasible design is not a degraded version of the analysis code.

**Quantile selection is implementable without allocation.** A fixed 256-bin histogram over the
85×85 score grid yields the required quantile in O(n) with a static array and no sorting — the
standard allocation-free way to compute a quantile on board.

**Compute budget — stated honestly.** The full benchmark suite (three scoring paths, a four-point
quantile sweep and visualisations) runs at **≈0.58 s per 1020×1020 frame on a desktop CPU**,
single-threaded. The *flight* path (frozen inference + quantile + merge) is a small fraction of
that, but we do **not** claim a LEON3 figure: a conservative planning range of **100–1000× slower**
puts a single frame at roughly **tens of seconds to a few minutes**. Within a 2–3 h window that is
**tens of frames per run**, and the algorithm processes a **bounded batch** with a clear completion
condition. Exact timing is a **Phase 2 measurement on the supplied simulation layer**, not a Phase 1
claim.

**Feasibility of the reference implementation.** The pipeline already runs end to end: tiling,
feature extraction, both scorers, the frozen-model inference path, quadtree merging, quantile
selection and visualisation, with 404 reference frames processed and measured. A 20-case test suite
covers the payload model, tiling geometry, rectangle disjointness, degenerate frames (all-black,
uniform), determinism, the frozen inference path and quantile behaviour. The Python/NumPy form is
the *algorithmic reference*; Phase 2 is the C port to **BCC 4.4.2**.

## 4. Compliance

| Constraint | Statement |
|---|---|
| Hera interface API (ANNEX A) | Only the documented acquisition / buffer / report entry points are used |
| Data pool (ANNEX B) | **Read-only**, and not required by the core algorithm; any optional gating parameter would be listed in the Phase 2 ICD |
| Binary packaging (ANNEX D, bin2c) | The frozen forest table is emitted as a C header via the supplied `bin2c` tool — the intended workflow |
| Client stub / integration (ANNEX C) | Development follows the stub guide; validation against the supplied simulation layer |
| HK telemetry (PUS 3) | ≤ **256 B**/packet, ≥ **5 min** interval — a small fixed structure (mode, frame counter, last-result code, ROI count, top score), tens of bytes |
| Event reporting (PUS 5) | ≤ **50 B**, ≥ **20 s** apart, Informational/Warning only; used for run start / completion / anomaly — never heartbeats |
| Science telemetry | ≤ **2048 B**/packet, ≤ **12 MB per 3 h slot**. Payload budgeted explicitly; measured **≤ 0.11 MB per frame** at q=0.10 (see §5) |
| Safe-mode tolerance | Stateless per frame; no persistent state; no graceful-shutdown dependency |
| Memory protection | All buffers pre-sized and bounded; no out-of-region writes; no dynamic allocation |

## 5. Benefits

Measured on the **404-frame AFC reference set supplied with this call** (1020×1020, 8-bit).
Payload model, identical for every row and deliberately conservative: selected pixels at **1 byte
each** (no credit for the lossless compression that would follow in flight) **plus** ROI metadata
at 5 int32 per rectangle.

| Mode | Median payload | Worst-case payload | Frames per 12 MB / 3 h slot |
|---|---|---|---|
| Heuristic baseline | 46.9 % | 100.1 % | 25.8 |
| Per-frame-fit iForest *(analysis reference)* | 11.9 % | 28.3 % | 101.7 |
| Frozen model, fixed threshold | 10.1 % | **78.3 %** | 120.3 |
| **Frozen model + quantile q=0.10** *(recommended)* | **8.8 %** | **11.0 %** | **136.8** |
| Frozen model + quantile q=0.15 | 13.1 % | 16.3 % | 92.2 |

**Reading of the result.** A raw frame is 0.99 MB, so the call's 12 MB science budget buys
**≈12.1 raw frames** per 3-hour slot. At q=0.10 the measured median is **8.8 %** of a frame, so the
same budget carries **≈137 prioritised scenes** — an **11.3×** increase in downlink content. At
q=0.15 it is **≈92 scenes** (7.6×). Equivalently: keep the current frame count and free **~91 %** of
the science budget for Hera's other instruments.

**Why the quantile, and why it matters operationally.** With a *fixed* score threshold the payload
is wildly frame-dependent — measured worst case **78.3 %** of a frame, i.e. the algorithm would
spend 4× the budget on one frame. That happens because an unfamiliar scene makes *every* tile look
anomalous to a ground-trained model (a classic out-of-distribution effect, and precisely the
situation Hera will meet at an unvisited body). Selecting the top *q* fraction of scored tiles
removes this failure mode: the measured spread collapses to **1.24×** (7.8 – 11.0 % across all 404
frames). **A predictable per-frame volume is the difference between an experiment an operator can
schedule and one they cannot.**

**Further benefits**

- **Ground intervention becomes optional for the "what to look at" decision.** Prioritisation
  happens during the 2–3 h window, not in the next planning cycle.
- **The ROI list is self-describing.** Ground receives coordinates plus scores, so an unexpected
  detection can be re-examined and criteria re-tuned for a later run — the experiment produces an
  audit trail, not just pixels.
- **It degrades gracefully.** A smaller budget is honoured by lowering *q*; the most novel regions
  survive.

**Limitations, stated plainly**

1. **The features are contrast/edge-driven.** On a frame containing a large, uniformly lit body the
   **limb** (high gradient) scores high and the **body interior** (low variance) scores low, so the
   interior can be dropped even though it is scientifically valuable. We observed exactly this on
   the reference frames. Concrete Phase 2 fix: add a coverage/area-saliency term and re-tune,
   validated against the simulation layer.
2. **The comparison that matters is not "iForest beats heuristic".** Both are our own code, and the
   thresholds are not tuned against each other. The honest comparison is **either of our modes
   versus the 100 % baseline**, with the heuristic explicitly labelled a baseline.
3. **These are prioritised subsets, not complete frames.** "137 scenes per slot" means 137 *partial*
   frames at the same selection quality, not 137 complete images. Whether a partial frame is
   scientifically sufficient is exactly what the experiment is designed to measure in flight; the
   full frame remains available on board for later downlink.
4. **The reference frames are ESA's development/reference set, not in-flight asteroid photography**
   (single-day timetags, rendered target appearance). All numbers above are therefore *method*
   measurements on representative AFC-format data, not a prediction of flight performance.

## 6. Maturity

- **Working reference implementation**, run end to end on all 404 reference frames: tiling, feature
  extraction, heuristic + both iForest paths, quadtree merge, quantile selection, ROI output,
  visualisation, per-frame statistics.
- **Reproducible**: `afc_benchmark.py` processes a directory of AFC frames and emits
  `afc_stats.csv` and `afc_summary.txt`; a demo notebook and headless demo script are in the same
  repository.
- **Tested**: a 20-case suite (`image_compression/tests/test_pipeline.py`) covers the payload model,
  tiling geometry (including the frame-coverage check that motivated the 12-px tile choice),
  rectangle disjointness, degenerate frames, determinism, the frozen inference path (including a
  guard that it never fits anything) and quantile behaviour.
- **Input is the call's own AFC format** (1020×1020, 8-bit, guidance naming), loaded without
  conversion.
- **Not yet done (Phase 2)**: the C port, the `bin2c`-packaged frozen model, MISRA/ECSS
  static-analysis evidence, and timing/robustness measurement on the Hera simulation layer.

## 7. Operational concept

1. **Start-up.** Only static state is initialised: score grid, ROI buffers, frozen forest pointer.
   Negligible cost.
2. **Acquisition strategy.** One AFC frame is requested per opportunity, at a paced cadence chosen
   so a run stays inside the 2–3 h window and the science budget.
3. **Processing steps.** Feature extraction → frozen-forest inference → dilation/blur → per-frame
   score quantile → quadtree merge with bandwidth-aware pruning → ROI rectangle list.
4. **Output generation.** A Science Data report carries the ROI list and selected tiles; an HK
   report carries mode, counter, ROI count and top score; an Event is emitted only on run start,
   run completion, or a recoverable anomaly.
5. **Completion conditions.** The run ends after a fixed, pre-declared **number of frames** — a
   deterministic, testable completion condition with a bounded worst-case run time.
6. **Expected duration.** Bounded batch, comfortably inside the 2–3 h window; exact figures
   confirmed on the simulation layer in Phase 2.

**Ground-side use.** Downlinked ROIs let the team reconstruct which regions the on-board
prioritiser considered novel, compare that against the images, and refine *q* and the feature set
for subsequent runs — a closed loop the experiment is explicitly designed to support.

## 8. Resource estimates

Reference: 1020×1020 AFC frame, **12-px tiles → 85×85 = 7225 tiles**, 4 features per tile, full
frame coverage (0 % cropped).

| Resource | Estimate | Basis |
|---|---|---|
| **Execution time / run** | Bounded batch (fixed frame count) inside the 2–3 h window | Desktop 0.58 s/frame for the full 3-path benchmark; LEON3 planning range 100–1000× slower; exact figure from the simulator in Phase 2 |
| **CPU utilisation** | Burst, not continuous; dominated by the forest tree walk and the quadtree recursion over an 85×85 grid | Reference implementation work profile |
| **Memory — frame buffer** | 1020×1020 × 1 B ≈ **0.99 MB** (read-only pointer to the platform buffer; not our allocation) | Fixed frame geometry |
| **Memory — working set** | Score grid 85×85 float ≈ **29 kB**; tile-feature tensor 85×85×4 float ≈ **116 kB**; quantile histogram 256 bins ≈ 1 kB; ROI list 5 int32 × N_rect (median ≈400 → ≈8 kB); code + frozen forest table | Pre-sized, static; no `malloc` anywhere |
| **Total worst-case working memory** | **Well under 1 MB**, dominated by the frame buffer we only read | Bounded by construction |
| **Data pool categories used** | None required (read-only if a cadence gate is added) | ANNEX B |
| **HK telemetry** | Tens of bytes per processing step; ≤ 256 B packet; ≥ 5 min interval | PUS 3 |
| **Event telemetry** | ≤ 50 B; start / completion / anomaly only; ≥ 20 s interval | PUS 5 |
| **Science telemetry** | **≤ 0.11 MB per frame** at q=0.10 (median 0.088 MB); ≤ 2048 B/packet; ≤ 12 MB per 3 h slot | Measured, 404-frame reference set |

Net telemetry effect: a **reduction**, not growth — the product is a prioritised subset of the
candidate imagery, with metadata overhead explicitly minimised by the quadtree merge.

## 9. Management and team

*(max 2–3 pages — doplniť pred podaním)*

- **Organisation:** *(názov, typ — individual / startup / research group, krajina)*
- **Team & roles:** *(Marek Racko — algorithm & software; prípadní ďalší členovia)*
- **Relevant background & experience:** *(computer vision, embedded/C, Python, CI, prior projects)*
- **Why us:** a working, measured prototype already exists and has been exercised on the call's own
  reference image set; the method was designed against the flight constraints from the start
  (frozen model, no allocation, stateless, bounded memory), so the Phase 2 C port is a translation
  and validation exercise rather than a redesign.

---

## Appendix A — Mapping to the call's target capabilities

| Call capability | This experiment |
|---|---|
| Edge Computing | On-board processing of asteroid imagery; only the relevant information is downlinked |
| Data Compression & Prioritisation | Explicit bandwidth-aware prioritisation; measured median 8.8 % of full frame (worst case 11.0 %) on the 404-frame reference set |
| Onboard Image Processing & Feature Tracking | Per-tile feature extraction; the ROI list seeds visual tracking and descriptors |
| Inference for Anomaly Detection / Science Classification | Unsupervised Isolation-Forest novelty; no labelled data required |
| Autonomy | On-board decision on *what to send*, removing the ground round-trip from the loop |

## Appendix B — Reproduction

```
image_compression/
  utils.py · features.py · scoring.py · tiles_merging.py · main.py · demo.py
  afc_benchmark.py            # headless benchmark over a directory of AFC frames
  tests/test_pipeline.py      # 20 tests (synthetic frames only)
notebooks/hera_image_compression_demo.ipynb
doc/Autonomous_Edge_Inference_Tile_Downlink_Pipeline.md
```

Reproduce the numbers quoted above:

```
python afc_benchmark.py --dir <AFC_IMAGES> --tile-size 12 --train-frames 200 --quantile 0.05,0.10,0.15,0.20
```

Tests:

```
python -m pytest image_compression/tests -q
```

The ESA reference images are **not** distributed with this repository (they are marked *For ESA
Official Use Only*); `--dir` points at the extracted `AFC_images.tar.gz` supplied with the call.
