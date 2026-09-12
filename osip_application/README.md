# OSIP Application — Autonomous Software Experiments on Hera

**Campaign Ca-2026-00066** · Campaign manager: **Jorge Lopez Trescastro** (ESA)

## ⚠️ Deadline — verified 2026-09-11

The live OSIP campaign page showed **"4 days left"**, **Discussion Starts Sep 15**, Evaluation
Starts Oct 15 → **submission closes ≈ 15 September 2026.** The "mid-October" figure in the ESA news
article is the *selection* date, not the submission deadline.

## ⚠️ Eligibility — raise with ESA via OSIP

The call states: *"Participation in this recognition initiative is limited to teams from **ESA
Member States** and collaborating agencies."*

**Slovakia is an ESA *Associate* Member State** (in force 13 October 2022, 7-year term) — one of
five associates alongside Latvia, Lithuania, Cyprus and Canada. Associate membership normally
carries participation rights, but this call does not name it explicitly. The call also states:
*"Any questions to the Agency relating to the first step shall be addressed exclusively via OSIP."*
**Action: submit the idea and raise the eligibility question through OSIP in parallel.**

## 🎯 ANNEX E is complementary, not competing

**ANNEX E** (ESA-HERA-TECS-MAN-2026-002054) is a **lossless Smart RLE** codec: each 1020-pixel line
is run-length encoded, stored compressed only if smaller than raw.

| | ANNEX E | This experiment |
|---|---|---|
| Question | *How to encode this frame in fewer bytes?* | *Which parts are worth sending at all?* |
| Layer | Encoding (lossless) | Semantic selection (lossy subset) |
| Output | Fewer bytes, same image | Fewer scenes, same byte budget |

They **compose**: our ROI rectangles define which regions to preserve; an ANNEX-E-style codec then
encodes those regions. We explicitly do **not** propose to replace ESA's codec.

## 📐 Hard constraints from the call documents

All seven attachments were downloaded and read (stored outside the repo — they are marked *For ESA
Official Use Only* and are never committed):

| Document | Reference |
|---|---|
| ANNEX A — Hera interface API | — |
| ANNEX B — Datapool | — |
| ANNEX C — Hera client stub, user and integration guide | — |
| ANNEX D — bin2c, binary to C header converter | — |
| ANNEX E — example, AFC image acquisition and smart compression | ESA-HERA-TECS-MAN-2026-002054 |
| Technical and operational requirements | ESA-HERA-TECS-RS-2026-002045 |
| OSIP General Conditions of Participation v3 | ESA-TECSF-TOR-2022-000109 |

| Constraint | Value |
|---|---|
| Processor / compiler | GR712RC LEON3, Frontgrade Gaisler Bare C Cross Compiler **BCC 4.4.2 1.0.52** |
| Execution | Bare metal, **no OS services** |
| Language / standards | **C**, MISRA, ECSS E-ST-40C **Category D** (+ ECSS Q-ST-80C for PA) |
| Memory | **No dynamic allocation** — static or fixed pool only |
| Libraries | **No external libraries**; only ESA's **LibmCS** |
| Hardware access | None directly — only the inter-core Communication API (ANNEX A) |
| AFC camera | **1020×1020** FaintStar2 CMOS, 8-bit, 5.5°×5.5° FOV |
| HK telemetry (PUS 3) | ≤ **256 B**/packet, ≥ **5 min** between reports |
| Events (PUS 5) | ≤ **50 B**/packet, ≥ **20 s** apart, Informational/Warning only |
| Science telemetry | ≤ **2048 B**/packet, ≤ **12 MB per 3-hour slot** |
| Idea-phase main criterion | 🎯 **novelty** |

## 📊 Measured results — real AFC reference set, 404 frames

`AFC_images.tar.gz` from the call ships **404 AFC frames (1020×1020, 8-bit)**. Measured with
**12-px tiles** (85×85 grid, **0 % of the frame cropped**) and a **ground-trained model trained on
200 frames**.

| Mode | Median payload | Worst case | Frames per 12 MB / 3 h slot |
|---|---|---|---|
| Heuristic baseline | 46.9 % | 100.1 % | 25.8 |
| Per-frame-fit iForest *(analysis reference, not flight-feasible)* | 12.0 % | 31.7 % | 100.7 |
| Frozen model, fixed threshold | 10.1 % | **78.3 %** | 120.3 |
| **Frozen model + quantile q=0.10** *(recommended)* | **8.8 %** | **11.0 %** | **136.8** |
| Frozen model + quantile q=0.15 | 13.1 % | 16.3 % | 92.2 |

## 🔍 Critical review — issues found and fixed (2026-09-12)

A deliberate adversarial pass over the proposal and the code. Every item below was a **real**
defect, not a hypothetical.

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 1 | **The proposal contradicted the code.** It claimed the forest is "trained on the ground, shipped as a frozen static table", but `score_tiles_iforest()` **fits per frame at run time** — a design that needs an RNG, dynamic structures and O(n log n) tree building, all forbidden in the sandbox | **critical** | Added `train_iforest_ground()` + `score_tiles_iforest_model()` (pure inference) and **measured both**. The flight path now exists and is what the proposal quotes |
| 2 | **1020 is not divisible by 16** → the pipeline processed 1008×1008 and **silently dropped a 12-px edge strip (2.3 % of pixels)** | high | Measured tile sizes 12/15/16/20; **12 px chosen** (divides 1020 exactly *and* gave the lowest payload). A regression test documents the trap |
| 3 | **The payload metric flattered a bigger crop** — `payload_pct` divided by the full frame while fewer pixels were examined, so cropping more looked better | high | Added a `cropped_pct` column + a warning line in the summary; the proposal reports the cropped fraction explicitly |
| 4 | **A fixed score threshold is not safe.** Out-of-distribution frames made *every* tile look anomalous (worst case **78.3 %** of the frame = 4× the budget) | **critical** | Introduced **quantile selection** (keep the top *q* fraction of scored tiles). Measured spread collapses from **24×** to **1.24×** — predictable bandwidth, and an allocation-free O(n) histogram implementation |
| 5 | Ageing docstring claimed 7 features; the pipeline computes **4** | low | Corrected, with a cross-reference to the real feature list |
| 6 | The heuristic has **no** noise floor while iForest applies `scores < 0.15 → 0` — the two modes are not post-processed identically, so a naive head-to-head is confounded | medium | Disclosed in the proposal; the honest comparison stated is **each mode vs the 100 % baseline**, not mode vs mode |
| 7 | No tests at all | high | **19-case suite** (`tests/test_pipeline.py`), synthetic frames only: payload maths, tiling geometry, rectangle **disjointness** (required by the area-sum payload model), degenerate frames (all-black, uniform), determinism, frozen-inference purity (asserts it never calls `.fit`), quantile behaviour |
| 8 | Unsupported literature claim ("not seen in the onboard-compression literature") | medium | Softened to "we are aware of"; the claim is now scoped to the specific optimisation (choice of content, not bytes) |
| 9 | "6.6× more frames" conflated partial frames with complete frames | medium | Explicit caveat added: these are **prioritised subsets**, not complete images |
| 10 | An early 10-frame test showed a frozen-model score spread of 0.0015 and I nearly reported "a fixed on-board threshold is plausible" | **critical** | Re-ran on all 404 frames → spread **0.072** → the opposite conclusion. Lesson recorded: never calibrate on a 10-frame subset of a homogeneous sequence |

## ✅ Done

- 9-section proposal in the call's required structure, plus a **Novelty statement** up front (the
  Idea phase's primary criterion) and **§2.1 differentiation vs ANNEX E**.
- Feasibility written against the real toolchain (BCC 4.4.2, MISRA, no-`malloc`, LibmCS-only,
  frozen model, quantile selection).
- Benefits and resource estimates aligned to the real packet/slot limits.
- `afc_benchmark.py` (three scoring paths + threshold sweep + quantile sweep + coverage check).
- 404-frame measurements; 19 passing tests.

## ⬜ Pending

1. **Proposal metadata** — proposing organisation, contact, entity type.
2. **Section 9 — Management and team** (max 2–3 pages) — currently a placeholder.
3. **PDF export, ≤ 10 pages.**
4. **Create the draft on OSIP and submit** (needs the applicant's OSIP account).
5. **Raise the Associate-Member eligibility question via OSIP.**

## Files

| File | Content |
|---|---|
| `OSIP_Ca-2026-00066_idea.md` | the proposal |
| `../image_compression/afc_benchmark.py` | benchmark (per-frame / frozen / heuristic + sweeps) |
| `../image_compression/scoring.py` | scoring incl. the frozen-inference path |
| `../image_compression/tests/test_pipeline.py` | 19 tests |
| `../image_compression/results/afc/afc_stats.csv` | per-frame results, 404 rows |
| `../image_compression/results/afc/afc_summary.txt` | aggregate statistics |
