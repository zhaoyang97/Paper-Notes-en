---
title: >-
  [Paper Note] Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry
description: >-
  [ICCV 2025][Audio & Speech][Speckle vibrometry] This paper proposes a non-contact system based on laser speckle vibrometry that simultaneously senses micro-vibrations on the surfaces of multiple opaque containers via a 2…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "Speckle vibrometry"
  - "liquid level inference"
  - "Transformer"
  - "2D grid sensing"
  - "non-contact detection"
date: 2026-05-08
content_hash: 06f4d366ce993776
---

# Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry

**Conference**: ICCV 2025
**arXiv**: [2507.20757](https://arxiv.org/abs/2507.20757)  
**Code**: [Project Page](https://matankic.github.io/see_inside_containers)  
**Area**: Computational Imaging / Vibration Sensing
**Keywords**: Speckle vibrometry, liquid level inference, Transformer, 2D grid sensing, non-contact detection

## TL;DR

This paper proposes a non-contact system based on laser speckle vibrometry that simultaneously senses micro-vibrations on the surfaces of multiple opaque containers via a 2D grid, then employs a Vibration Transformer to infer container type and hidden liquid fill level from vibration spectra — establishing "seeing inside opaque containers" as a novel computer vision task.

## Background & Motivation

**Background**: Computer vision aims to infer scene information from images, but conventional cameras are limited to the visible surfaces of objects. A vision system can detect and recognize a soda can in a scene, yet cannot determine whether it is full or empty. Techniques such as hyperspectral, polarization, and thermal imaging can probe certain surface material properties, but these methods are confined to optically accessible surface characteristics and cannot penetrate into object interiors.

**Limitations of Prior Work**: Probing internal object properties requires a signal that can permeate the interior yet be sensed optically at the surface. Object vibration is precisely such a signal — liquid content alters the resonance characteristics of a container. Prior work (e.g., Davis et al.) has demonstrated the feasibility of inferring material properties by sensing object vibration via high-speed cameras and laser speckle. However, these efforts focused only on low-level physical attributes (motion spectra, stiffness, density), and existing speckle vibrometry systems can measure only a single row of points, precluding simultaneous scanning of multiple containers. Additionally, approaches that estimate fill level from pouring sounds or tap acoustics require physical contact or close-range microphones.

**Key Challenge**: Everyday containers (e.g., soda cans, shampoo bottles) exhibit complex geometries and diverse materials, yielding a highly non-trivial relationship between vibration response and fill level — unlike wine glasses, which have a clear monotonic correspondence between resonant frequency and liquid level. Even containers from the same production batch may exhibit different resonant frequencies due to manufacturing variation, rendering traditional physics-based analytical methods ineffective.

**Goal**: (1) Develop a novel speckle imaging system capable of simultaneously sensing vibrations at multiple points arranged in a 2D grid; (2) Design a learning method to infer the hidden fill level of containers from multi-point vibration signals; (3) Validate the model's generalization to unseen container instances, fill levels, and acoustic sources.

**Key Insight**: Exploiting the extreme sensitivity of laser speckle to minute surface tilts, the system encodes vibrations as image-domain displacements in speckle patterns via defocused imaging. An ROI readout strategy is designed to increase camera frame rate by 25×, enabling high-speed vibration sampling.

**Core Idea**: A 2D-grid speckle vibrometry system remotely and non-contactly measures surface vibrations of multiple containers; a dual-stage, modal-analysis-inspired Vibration Transformer then learns the complex relationship between vibration spectra and fill level / container type.

## Method

### Overall Architecture

The system comprises a hardware acquisition stage and a learning inference stage. On the hardware side, a laser passes through a diffractive beam splitter to generate a 6×6 dot array projected onto a row of containers; a defocused camera captures speckle patterns at approximately 57 kHz via an ROI readout strategy; GPU-accelerated phase correlation combined with Lucas–Kanade optical flow then recovers biaxial vibration signals $\mathbf{v}_i \in \mathbb{R}^{2 \times N}$ for each laser point. On the learning side, the DFT of each measurement point's vibration signal yields a frequency magnitude spectrum $V_i[f] = |\mathcal{F}\{\mathbf{v}_i\}|$, which serves as input to the Vibration Transformer, whose outputs are container-type classification and fill-level prediction.

### Key Designs

1. **2D Grid Speckle Vibrometry System**:

    - Function: Simultaneously measure high-frequency vibrations at multiple surface points across multiple containers in the scene, from a distance.
    - Mechanism: A laser passes through a diffractive beam splitter to produce a 6×6 dot array, which is then stretched by an anamorphic prism pair from a square grid into a rectangular grid matched to the container arrangement. Camera defocus causes each laser dot to form a speckle patch. The key innovation is the ROI readout strategy: the camera is configured to output only $M$ regions of interest of $W \times P$ pixels (one ROI per row of speckle dots), boosting frame rate by approximately $H/(MP)$. For example, with 6 ROIs each 6 pixels tall, frame rate rises from 2,247 Hz to 57,699 Hz. To efficiently handle the large volume of displacement computations (6×6 grid × 20 kHz × 2 s = 1.44 million calls), a GPU-parallelized PCLK+ (phase correlation + Lucas–Kanade) is implemented, achieving a 20× speedup.
    - Design Motivation: Prior speckle vibrometry systems could only measure a single row of points or required dual-camera calibration. This system achieves 2D grid sensing with a single laser, single camera, and diffractive beam splitter, while avoiding the light loss and calibration complexity of dual-camera setups.

2. **Vibration Transformer Architecture**:

    - Function: Classify container type and infer fill level from multi-point vibration spectra.
    - Mechanism: A dual-stage Transformer design inspired by modal analysis. The first stage, PointTransformer, independently processes the spectrum $V_i$ of each measurement point: the 2×4800 frequency magnitude matrix is divided into non-overlapping 2×100 patches, linearly projected into 512-dimensional tokens (48 tokens total), augmented with learnable positional encodings and a [pnt] classification token, and processed by 8 layers of self-attention (4 heads) to extract frequency features per point — analogous to analyzing the resonant modal frequencies at each point. The second stage, ShapeTransformer, processes three [pnt] tokens: a positional encoding representing grid location and a global [cls] token are added, and 8 layers of self-attention fuse multi-point information — analogous to analyzing modal shapes. Two MLPs then output container-type and fill-level predictions respectively.
    - Design Motivation: Physically, object vibration is governed jointly by modal frequencies (measurable independently at each point) and modal shapes (requiring multi-point analysis). PointTransformer corresponds to frequency analysis; ShapeTransformer corresponds to shape analysis — elegantly embedding physical intuition into the network architecture.

3. **SORD Ordinal Regression Loss**:

    - Function: Exploit the ordinal nature of fill levels to improve prediction accuracy and generalization.
    - Mechanism: Fill level is inherently ordinal (0% < 20% < 40% < …) and should not be treated as an unordered categorical variable. The SORD (Sorted Ordinals) loss constructs a soft target distribution for ground-truth fill level $l$ as $q_l[h] \propto e^{-50(l - L[h])^2}$, and computes cross-entropy between the predicted probability vector $\hat{p}$ and this soft target. Consequently, misclassifying 60% as 40% incurs far less penalty than misclassifying it as 0%, naturally encoding ordinal relationships. At inference, MAP estimation ($\hat{l}_{MAP}$) yields discrete predictions, while expectation estimation ($\hat{l}_{\mathbb{E}} = \sum_h L[h] \cdot \hat{p}[h]$) yields continuous predictions.
    - Design Motivation: Standard cross-entropy penalizes all misclassifications equally, failing to distinguish between "off by one level" and "off by five levels." SORD realizes distance-aware penalization of prediction errors and enables the model to extrapolate to intermediate fill levels unseen during training.

### Loss & Training

The total loss is a weighted sum of the SORD loss (weight 0.9) and the container classification cross-entropy loss (weight 0.1). The Adam optimizer is used with a learning rate of $10^{-5}$ for 7,500 epochs. Data augmentation includes random smoothing filters to simulate different acoustic sources and environmental variations, as well as random dropout of 50% of PointTransformer input tokens.

## Key Experimental Results

### Main Results

Results on a self-collected container vibration dataset (23 container types, 5,910 samples) across multiple test scenarios:

| Test Scenario | Fill-Level Accuracy | Fill-Level MAE | Container Classification Accuracy | Notes |
|---|---|---|---|---|
| (a) In-distribution | 0.98 | 0.01 (1%) | 1.00 | Trained containers + new source positions |
| (b) Unseen instances | 0.79 | 0.09 (9%) | 0.95 | New containers of same type (e.g., 6th can in a 6-pack) |
| (c) Unseen fill levels | N/A | 0.12 (12%) | 0.81 | Fill level absent from training |
| (d) Ambient noise | 0.92 | 0.04 (4%) | 0.97 | Supermarket ambient sound as excitation |
| (e) Unseen fill + ambient | N/A | 0.15 (15%) | 0.67 | Hardest combination |
| (f) Unseen instance + ambient | 0.59 | 0.16 (16%) | 0.77 | Hardest combination |
| CNN baseline | 0.17 | 0.33 (33%) | 0.86 | Near-random performance |

### Ablation Study

| Configuration | MAE (In-distribution) | MAE (Unseen instances) | MAE (Unseen instance + ambient) |
|---|---|---|---|
| Full model (3 points) | 0.02 | 0.09 | 0.16 |
| Single-point measurement | 0.03 | 0.11 | 0.18 |
| Continuous regression (no SORD) | 0.20 | — | — |
| With phase information | ≈ 0.02 | — | — |

### Key Findings

- The Vibration Transformer achieves only 1% MAE in-distribution, vastly outperforming the CNN baseline (33% MAE, near-random), validating the superiority of Transformers for spectral signal modeling.
- Multi-point measurement primarily provides gains in challenging scenarios (unseen instances): 9% vs. 11% MAE, as multi-point data encodes modal shape information.
- The SORD ordinal loss is critical: replacing it with continuous regression increases MAE from 0.01 to 0.20.
- The frequency magnitude spectrum already contains sufficient information; additionally incorporating phase information provides no benefit.
- The model learns meaningful latent representations — PCA visualization shows six discrete fill levels forming clear clusters, with unseen fill-level samples naturally interpolating between clusters.
- Highly resonant containers (e.g., certain bottles) are more difficult to generalize to new instances, as manufacturing variation causes resonant frequency shifts.

## Highlights & Insights

- **Pioneer Problem Formulation**: Defining "seeing through opaque containers to determine fill level" as a new computer vision task extends visual perception from visible surfaces to object interiors. This problem definition alone carries significant value — it motivates a series of new questions: detecting contents inside sealed packages, judging fruit ripeness, sensing food seal integrity, and more.
- **Physics-Driven Network Design**: The dual-stage design of the Vibration Transformer directly mirrors the physical decomposition of "modal frequencies + modal shapes" from modal analysis, serving as an exemplary case of encoding domain knowledge into network architecture. This design philosophy is transferable to any physics-driven signal analysis task.
- **25× Frame Rate Boost via ROI Readout**: By reading only the rows of interest, camera frame rate is increased from 2,247 Hz to 57,699 Hz — resolving the core bottleneck of high-speed data acquisition through a straightforward engineering technique.

## Limitations & Future Work

- The dataset is limited in scale (5,910 samples, 23 container types); generalization to entirely unseen container categories (e.g., from soda cans to kettles) remains unvalidated.
- Laser safety is a concern — at 14 mW per point, direct eye exposure is unsafe; practical deployment requires eye-safe solutions.
- Glass, polished metal, and very-low-reflectance materials produce weak speckle signals, necessitating reflective stickers as aids.
- Double-walled insulated containers are vibrationally insensitive to fill level due to their thermal isolation structure, though low-frequency differences remain; inference difficulty is substantially increased.
- Future directions include: liquid type classification (water vs. soda vs. oil), granular material detection (sand), container fingerprinting, and richer semantic inference tasks.

## Related Work & Insights

- **vs. Davis et al. (Visual Vibrometry)**: Prior work estimated low-level material properties such as density and Young's modulus from high-speed video; this paper advances vibration analysis to infer high-level semantic attributes (fill level) while employing a more efficient speckle sensing system.
- **vs. Acoustic Fill-Level Detection**: Wilson et al. infer fill level from pouring sounds; Garcia et al. use tap acoustics. These methods require physical contact or close-range microphones. The proposed method is fully non-contact, remote, and capable of simultaneously inspecting multiple containers.
- **vs. Sheinin et al. (Dual-Shutter)**: The prior dual-shutter speckle system could only measure a single row and required scanning to cover multiple containers. The proposed 2D grid system covers all containers in a single acquisition, while avoiding the dual-camera calibration complexity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Novel task formulation, novel sensing system, and physics-driven network design all represent genuine innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Six test scenarios provide comprehensive coverage with thorough ablations, though the dataset scale is limited and large-scale validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — The narrative flows seamlessly from problem motivation through physical principles to system design; figures and tables are well crafted.
- Value: ⭐⭐⭐⭐ — Highly inspiring as a proof of concept, though engineering challenges such as laser safety and scalability must be addressed before practical industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inductive Transfer Learning for Graph-Based Recommenders](../../NeurIPS2025/audio_speech/inductive_transfer_learning_for_graph-based_recommenders.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[ICLR 2026\] PACE: Pretrained Audio Continual Learning](../../ICLR2026/audio_speech/pace_pretrained_audio_continual_learning.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/audio_speech/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

</div>

<!-- RELATED:END -->
