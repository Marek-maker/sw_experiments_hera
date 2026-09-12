# OSIP Application — Autonomous Software Experiments on Hera

**Campaign Ca-2026-00066** · Campaign manager: **Jorge Lopez Trescastro** (ESA)

## ⚠️ Deadline — verified 2026-09-11

The live OSIP campaign page showed **"4 days left"**, **Discussion Starts Sep 15**, Evaluation
Starts Oct 15. **Submission closes ≈ 15 September 2026.** The earlier "mid-October" figure found in
the ESA news article is the *selection* date, not the submission deadline.

## ⚠️ Eligibility — must be raised with ESA via OSIP

The call states: *"Participation in this recognition initiative is limited to teams from **ESA
Member States** and collaborating agencies."*

**Slovakia is an ESA *Associate* Member State** (in force 13 October 2022, 7-year term), not a full
Member State — one of five associates alongside Latvia, Lithuania, Cyprus and Canada.

Associate membership normally carries participation rights, but the call does not name it
explicitly. The call also states: *"Any questions to the Agency relating to the first step shall be
addressed exclusively via OSIP."* **Action: submit the idea and raise the eligibility question
through OSIP in parallel** — do not burn the remaining days waiting for a reply.

## 🎯 Key finding: ANNEX E is complementary, not competing

**ANNEX E — "example, AFC image acquisition and smart compression"**
(ESA-HERA-TECS-MAN-2026-002054) implements a **lossless Smart RLE** codec: each 1020-pixel line is
run-length encoded and stored compressed only if smaller than raw.

| | ANNEX E | This experiment |
|---|---|---|
| Question | *How to encode this frame in fewer bytes?* | *Which parts are worth sending at all?* |
| Layer | Encoding (lossless) | Semantic selection (lossy subset) |
| Output | Fewer bytes, same image | Fewer scenes, same byte budget |

They **compose**: our ROI rectangles define which regions to preserve; an ANNEX-E-style codec then
encodes those regions. We explicitly do **not** propose to replace ESA's codec.

## 📐 Hard constraints extracted from the call documents

All seven attachments were downloaded and read (stored outside the repo, not committed):

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
| Libraries | **No external libraries**; only ESA's **LibmCS** math library |
| Hardware access | None directly — only the inter-core Communication API (ANNEX A) |
| AFC camera | **1020×1020** FaintStar2 CMOS, 8-bit, 5.5°×5.5° FOV |
| HK telemetry (PUS 3) | ≤ **256 B** per packet, ≥ **5 min** between reports |
| Events (PUS 5) | ≤ **50 B** per packet, ≥ **20 s** apart, Informational/Warning only |
| Science telemetry | ≤ **2048 B** per packet, ≤ **12 MB per 3-hour slot** |
| Idea-phase main criterion | 🎯 **novelty** (vs. the state of the art) |

## 📊 Measured results on ESA's own AFC reference set

`AFC_images.tar.gz` from the call ships **404 AFC frames (1020×1020, 8-bit)**. The pipeline was run
on all 404 via `image_compression/afc_benchmark.py`:

| Mode | Median payload vs full frame | Range | Frames per 12 MB / 3 h slot |
|---|---|---|---|
| Heuristic baseline | 50.8 % | 16.3 – 97.7 % | 23.8 |
| **Isolation Forest (unsupervised novelty)** | **15.0 %** | **5.9 – 36.5 %** | **80.4** |

Raw frame = 0.99 MB → the budget buys **12.1 raw frames**; with selection, **80.4** → **6.6×**
(best frame 17×, worst frame 2.7×). Desktop timing: **0.21 s/frame**.

**Honest caveats, deliberately included in the proposal:**
- The four features are contrast/edge-driven. On a frame with a large uniformly lit body the
  **limb** scores high and the **interior** scores low — observed on the reference frames. Fix
  (Phase 2): add a coverage/saliency term and re-tune the threshold.
- The heuristic mode is a **baseline only** (median 50.8 %); it must not be presented as a
  bandwidth-saving mode.
- The reference frames carry a single-day timetag (2024-10-08) and a rendered appearance — they are
  the **development/reference set** ESA supplies, not in-flight asteroid photography. Worded
  accordingly in the proposal.

## ✅ Done

- Full 9-section proposal per the call's required structure, with an added **Novelty statement**
  up front (the Idea phase's primary criterion) and a **§2.1 differentiation vs ANNEX E**.
- Technical feasibility written against the real toolchain/constraints (BCC 4.4.2, MISRA,
  no-`malloc`, LibmCS-only, frozen static forest model).
- Benefits and resource estimates aligned to the real packet/slot limits.
- Benchmark script + real measurements on all 404 reference frames (CSV + summary committed).
- Compliance mapped to ANNEX A/B/C/D and the PUS service limits.

## ⬜ Pending

1. **Proposal metadata** — proposing organisation, contact, entity type (individual vs company).
2. **Section 9 — Management and team** (max 2–3 pages) — currently a placeholder.
3. **PDF export, ≤ 10 pages.**
4. **Create the draft on OSIP and submit** (requires the applicant's OSIP account).
5. **Raise the Associate-Member eligibility question via OSIP.**

## Files

| File | Content |
|---|---|
| `OSIP_Ca-2026-00066_idea.md` | the proposal |
| `../image_compression/afc_benchmark.py` | headless benchmark over a directory of AFC frames |
| `../image_compression/results/afc/afc_stats.csv` | per-frame results, 404 rows |
| `../image_compression/results/afc/afc_summary.txt` | aggregate statistics |
| `../image_compression/results/afc/vis/` | separate full-size visualisations (3 frames) |
