# Deep-Space Computer Vision & Surface Inference: Dataset Reference Guide
## For Hera Extended Mission Onboard Software Experiments (Core 1 Sandbox)

This document provides a comprehensive catalog of spaceflight, synthetic, and curated computer vision datasets suitable for training, testing, and benchmarking onboard algorithms for ESA's **Hera Extended Mission** software experiments.

The catalog focuses on two core mission capabilities:
1. **Onboard Image Processing and Feature Tracking** (Landmark detection, optical flow, ROI smart-cropping)
2. **Inference for Anomaly Detection or Science Classification** ("Auto-Geologist" surface texture, boulder/regolith classification, and feature identification)

---

## 1. Direct Target System Datasets (Didymos & Dimorphos)

### 1.1 DART DRACO Camera Dataset
* **Archive / Source:** [NASA Planetary Data System (PDS) Small Bodies Node (SBN)](https://pds-smallbodies.astro.umd.edu/)
* **Primary Sensor:** Didymos Reconnaissance and Asteroid Camera for Optical navigation (DRACO)
* **Data Format:** FITS (16-bit raw & calibrated), PNG/JPG derivatives
* **Key Characteristics:**
  * Contains real ground-truth optical imagery of both Didymos (primary) and Dimorphos (moonlet) captured during the final approach of NASA's DART spacecraft (Sept 2022).
  * High-resolution close-ups of Dimorphos' boulder-strewn surface down to sub-meter and centimeter scales immediately prior to impact.
  * Varying phase angles, realistic deep-space backdrops, glare, and sensor noise characteristics.
* **Primary Use Cases:**
  * **Feature Tracking:** Evaluating landmark extraction (FAST, ORB) on the exact target system prior to Hera's arrival.
  * **Smart-Crop:** Training limb/body isolation and ROI selection algorithms on authentic Didymos/Dimorphos pixel profiles.
  * **Zero-Shot Validation:** Testing surface classifiers on pre-impact Dimorphos morphology.

---

## 2. Asteroid Surface & Morphology Analog Datasets

### 2.1 OSIRIS-REx OCAMS Surface Characterization Bundle
* **Archive / Source:** NASA PDS SBN / University of Arizona
* **Primary Sensor:** OSIRIS-REx Camera Suite (PolyCam, MapCam, SamCam)
* **Data Format:** FITS, GeoTIFF, PNG, 16-bit high-dynamic-range mosaics
* **Key Characteristics:**
  * Global sub-meter resolution imagery of B-type rubble-pile asteroid **101955 Bennu**.
  * Hand-annotated and algorithmically verified datasets for **boulder distributions, crater boundaries, and regolith grains**.
  * Albedo maps and surface texture variations under diverse illumination geometries (0° to 90° phase angles).
* **Primary Use Cases:**
  * **Surface Classifier ("Auto-Geologist"):** Training micro-models (Random Forest, Quantized CNNs) to classify terrain into categories: `0: Space`, `1: Fine Regolith`, `2: Small Boulders (<2m)`, `3: Large Boulders (>2m)`, `4: Shadows`.
  * **Feature Tracking:** Testing feature point durability across significant shadow and illumination shifts.

### 2.2 Hayabusa2 ONC Image Dataset
* **Archive / Source:** JAXA Data Archives and Transmission System (DARTS) / Harvard Dataverse
* **Primary Sensor:** Optical Navigation Camera (ONC-T, ONC-W1, ONC-W2)
* **Data Format:** Fits, GeoTIFF, Calibrated Radiance Maps
* **Key Characteristics:**
  * Over 8,300 high-resolution images of C-type asteroid **162173 Ryugu**.
  * Proximity operations and touchdown sequences providing multi-scale image series (from tens of kilometers down to millimeter-scale surface detail).
* **Primary Use Cases:**
  * **Multi-Scale Processing:** Testing feature tracking algorithms as resolution shifts dynamically during orbit adjustments or flybys (5 km to 30 km range).
  * **Texture Segmentation:** Fine-grained terrain analysis in dark, low-albedo asteroid environments.

### 2.3 Rosetta OSIRIS & NAVCAM Dataset
* **Archive / Source:** ESA Planetary Science Archive (PSA)
* **Primary Sensor:** OSIRIS (Optical, Spectroscopic, and Infrared Remote Imaging System) & NAVCAM
* **Data Format:** FITS, VICAR, PDS3/PDS4
* **Key Characteristics:**
  * Long-duration proximity operations around comet **67P/Churyumov–Gerasimenko**.
  * Highly irregular geometry, deep shadow lines, steep cliffs, and dust/ejecta activity.
* **Primary Use Cases:**
  * **Ejecta & Anomaly Detection:** Testing frame-to-frame change detection algorithms for transient events (e.g., active dust outbursts, regolith shifts).
  * **Extreme Illumination Tracking:** Evaluating landmark tracking robustness under harsh light/shadow contrast boundaries.

---

## 3. Dedicated Computer Vision Benchmarking Datasets

### 3.1 Solar System Small Bodies (SSSB) Optical Navigation Dataset
* **Archive / Source:** [Zenodo Open Repository](https://zenodo.org/) (Curated by OpNav research community)
* **Data Format:** PNG, NumPy arrays (`.npy`), JSON metadata (depth maps, camera poses)
* **Key Characteristics:**
  * Approximately 50 GB of curated image sequences combining flight data from NEAR Shoemaker (Eros), Hayabusa (Itokawa), Rosetta (67P), and OSIRIS-REx (Bennu).
  * Includes **ground-truth pixel correspondences**, calibrated camera intrinsics/extrinsics, and optical flow vectors.
* **Primary Use Cases:**
  * **Quantitative Feature Tracking Benchmarking:** Evaluating tracking error (in pixels) for algorithms like ORB, FAST, Harris, and LK Optical Flow against verified ground truth.
  * **Descriptor Compression:** Testing compact descriptor generation (e.g., binary descriptors vs. float descriptors) for low-bandwidth telemetry transmission.

---

## 4. Synthetic Rendering & 3D Mesh Datasets

### 4.1 NASA PDS Small Body 3D Shape Models
* **Archive / Source:** NASA PDS Small Bodies Node
* **Formats:** Wavefront OBJ (`.obj`), Polygon File Format (`.ply`), ICQ meshes
* **Target Mesh Models Available:**
  * Didymos (Primary) & Dimorphos (Secondary) shape models (Radar & DART derived)
  * Bennu, Ryugu, Itokawa, Eros, Castalia, Toutatis
* **Pipeline Integration:**
  * Models can be imported into 3D rendering engines (**Blender**, **Mitsuba 3**, or **PBRT**) combined with **SPICE kernels** (Ancillary Information System for Solar System Missions) to render synthetic flight passes matching Hera's precise orbital geometry (5–30 km altitude).

---

## 5. Dataset Summary Matrix

| Dataset / Resource | Source Archive | Primary Purpose | Image Format / Size | Key Benchmark Value for Hera |
| :--- | :--- | :--- | :--- | :--- |
| **DART DRACO** | NASA PDS SBN | Target Ground Truth | FITS, PNG (~10 GB) | Authentic Didymos/Dimorphos pixel morphology & lighting |
| **OSIRIS-REx OCAMS** | NASA PDS SBN | Surface Classification | FITS, Mosaics (~200 GB) | Dense boulder/regolith labels for terrain classification models |
| **Hayabusa2 ONC** | JAXA DARTS | Feature Tracking & Scaling | FITS, GeoTIFF (~50 GB) | Multi-scale imagery across variable orbital altitudes |
| **Rosetta OSIRIS** | ESA PSA | Anomaly & Change Detection | FITS (~500 GB) | High-contrast shadowing and dust/transient event detection |
| **SSSB OpNav Dataset** | Zenodo | CV Algorithm Benchmarking | PNG + JSON (~50 GB) | Ground-truth optical flow & pixel correspondence maps |
| **NASA 3D Shape Models** | NASA PDS SBN | Synthetic Data Generation | `.obj`, `.ply` (<1 GB) | Unlimited trajectory-specific image generation via Blender/SPICE |

---

## 6. Model Training & Validation Workflow for Core 1 (LEON3)

To ensure high compliance and zero risk to the Hera spacecraft, data should be utilized across a three-phase pipeline:

```
[Phase A: Off-Board Model Training]
  ├── Train Surface Classifier (Random Forest / Micro-CNN) on OSIRIS-REx & DART datasets
  └── Optimize Feature Trackers (ORB/FAST) on SSSB OpNav dataset

[Phase B: Synthetic Pipeline & Data Augmentation]
  ├── Render 1,000+ synthetic Didymos passes using 3D Shape Models + SPICE Kernels
  ├── Apply radiation noise, blur, and 12-bit to 8-bit dynamic compression
  └── Benchmark execution bounds (memory/cycles) on CPU target simulator

[Phase C: Hardware-in-the-Loop Validation]
  ├── Deploy C code compiled for LEON3 (RTEMS/BCC compiler)
  ├── Run image processing on static test batches within 2-3 hour execution window
  └── Verify 0% dynamic memory allocation and 100% memory boundary compliance
```

---

## 7. Draft Text Snippet for ESA Phase 1 Proposal (Maturity & Credibility)

Below is ready-to-use text for the **Maturity** section of your ESA submission:

> *"The proposed software algorithms leverage a rigorous multi-dataset validation methodology prior to flight software packaging. Initial training and hyperparameter optimization for surface terrain classification and landmark tracking will utilize open-source planetary datasets, including the **Solar System Small Bodies (SSSB) OpNav Dataset** and high-resolution **OSIRIS-REx OCAMS surface characterization bundles**. Zero-shot performance and smart-crop parameters will be benchmarked directly against calibrated **DART/DRACO flight imagery of the Didymos system**.

To simulate Hera's specific extended mission trajectory (5 km to 30 km range), a synthetic image rendering pipeline utilizing official **NASA PDS 3D shape meshes of Dimorphos** combined with SPICE orbital kernels will generate thousands of realistic flight frames under varying phase angles and illumination conditions. This ensures that algorithm convergence, processing latency, and telemetry reduction ratios are fully quantified on a LEON3 hardware emulator prior to flight delivery."*