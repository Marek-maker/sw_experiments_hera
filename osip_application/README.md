# OSIP Application — Autonomous Software Experiments on Hera

**Deadline: ~15 September 2026** ⚠️ *(verified 2026-09-11: OSIP campaign page shows "4 days left", Discussion starts Sep 15, Evaluation starts Oct 15)*

## Verified call facts

| Field | Value |
|---|---|
| Campaign ID | **Ca-2026-00066** |
| Title | Call for Ideas: Autonomous Software Experiments on Hera |
| Status | Submission — **4 days left** (as of 2026-09-11) |
| Discussion starts | **15 Sep 2026** |
| Evaluation starts | **15 Oct 2026** |
| Campaign manager | Jorge Lopez Trescastro (ESA, Software Engineer) |
| Submission | Phase 1 idea via OSIP (online form), **max 10 pages**, in **English** |
| Eligibility | Teams from ESA Member States + collaborating agencies |
| OSIP call URL | `https://ideas.esa.int/core/servlet/hype/IMT?documentId=76590fb19b5e6424d8862a329c2884b1&documentTableId=8527279057944083272&templateName=&userAction=Browse` |
| ESA news release | https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Your_chance_to_run_software_in_deep_space_on_ESA_s_asteroid_mission (5 Aug 2026) |

## Timeline (full)

1. **~15 Sep 2026** — Phase 1 idea submission closes (500-word… no: 10-page idea, online form)
2. **15 Oct 2026** — ESA evaluation starts; selection of winning ideas
3. **31 May 2027** — deadline for full Experiment Implementation Package (Phase 2)
4. **August 2027** — one-month onboard experiment campaign (~4 weeks)

## Phase 1 idea — required sections (max 10 pages)

- [ ] **The problem** — limitation in current spaceflight
- [ ] **The solution** — proposed software / algorithm
- [ ] **Technical feasibility** — LEON3, async, fits 2–3 h daily slot
- [ ] **Compliance** — vs. attached requirements + ANNEX A API
- [ ] **Benefits** — quantified (e.g. downlink reduction %)
- [ ] **Maturity** — prior testing (simulator/drone/CubeSat)
- [ ] **Operational concept** — startup, acquisition, processing, output, completion, duration
- [ ] **Resource estimates** — exec time/run, CPU %, RAM kB, datapool categories, data generation
- [ ] **Management and team** — org + team, background (max 2–3 pages)

## OSIP attachments to consult before submitting

- `OSIP-General Conditions of Participation_v3.pdf`
- `Technical and operational requirements.pdf`
- `ANNEX A - Hera interface API documentation.pdf`
- `ANNEX B - Datapool.pdf`
- `ANNEX C - Hera client stub, user and integration guide.pdf`
- `ANNEX D - bin2c, binary to c header converter.pdf`
- `ANNEX E - example, AFC image acquisition and smart compression.pdf`
- `AFC images.tar.gz`
- `Hera software simulation layer.tar.gz`

> Note: ANNEX E is literally an "example AFC image acquisition and **smart compression**" — i.e. a
> reference example that overlaps with our pipeline. Must read it before submitting.

## Our submission

- `OSIP_Ca-2026-00066_idea.md` — the idea document (main deliverable)
- Supporting prototype: `../image_compression/` + `../notebooks/hera_image_compression_demo.ipynb`

## Remaining work before submit

1. Read ANNEX A–E + Technical requirements; align the "Compliance" section with the actual API.
2. Fill in the **Management and team** section (organisation, roles, background).
3. Produce the final **PDF** (≤10 pages) from the markdown.
4. Create the idea as a **draft** on OSIP, then submit.
