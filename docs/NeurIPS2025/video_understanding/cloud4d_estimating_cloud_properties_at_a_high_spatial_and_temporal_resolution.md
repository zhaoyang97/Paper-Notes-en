---
title: >-
  [Paper Note] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution
description: >-
  [NeurIPS 2025][Video Understanding][cloud physical property estimation] The first learning framework based on ground-level multi-view cameras that reconstructs four-dimensional (3D spatial + temporal) cloud liquid water content distributions via a homography-guided 2D-to-3D Transformer. The method achieves less than 10% error relative to radar at 25 m spatial and 5 s temporal resolution, improving spatiotemporal resolution by an order of magnitude over satellite observations.
tags:
  - NeurIPS 2025
  - Video Understanding
  - cloud physical property estimation
  - multi-view 3D reconstruction
  - homography-guided Transformer
  - liquid water content
  - meteorological observation
date: 2026-05-08
content_hash: a859d5eeeeec186e
---

# Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution

**Conference**: NeurIPS 2025
**arXiv**: [2511.19431](https://arxiv.org/abs/2511.19431)
**Code**: [Project Page](https://cloud4d.jacob-lin.com/)
**Area**: Video Understanding
**Keywords**: cloud physical property estimation, multi-view 3D reconstruction, homography-guided Transformer, liquid water content, meteorological observation

## TL;DR

The first learning framework based on ground-level multi-view cameras that reconstructs four-dimensional (3D spatial + temporal) cloud liquid water content distributions via a homography-guided 2D-to-3D Transformer. The method achieves less than 10% error relative to radar at 25 m spatial and 5 s temporal resolution, improving spatiotemporal resolution by an order of magnitude over satellite observations.

## Background & Motivation

**Scale bottleneck in meteorological modeling**: Current weather and climate models (GraphCast, Pangu-Weather, etc.) operate at kilometer scales and cannot directly simulate shallow cumulus clouds spanning less than one kilometer, relying instead on coarse "parameterization" approximations — a primary source of error in weather forecasting and climate prediction.

**Severe observational data gaps**: High-resolution satellites have revisit periods of several days; scanning radars and aircraft observations cover only a small fraction of any cloud; complete high-resolution observations of a single cloud's full lifecycle are lacking.

**Bias inherited by ML models**: Modern ML weather systems are trained on reanalysis data (ERA5), which is itself derived from physical models and inherits the same parameterization biases.

**Importance of shallow cumulus clouds**: Shallow cumulus clouds cover approximately 40% of Earth's surface and play a critical role in regulating planetary temperature, yet their small size and short lifespan make them extremely difficult to capture comprehensively with existing instruments.

**Potential of ground-level cameras**: Ground-level cameras are low-cost and deployable at scale, capturing images every 5 seconds at a temporal resolution far exceeding that of satellites (hours to days). However, they have never previously been used to estimate cloud physical properties.

**Challenges of mapping 2D images to 3D physical quantities**: Ground-level camera viewpoints vary far more than orthogonal satellite views, making implicit learning of camera geometry extremely difficult and necessitating explicit modeling approaches.

## Method

### Overall Architecture

Cloud4D adopts a two-stage architecture: (1) **Cloud Layer Model** — leveraging the layered spatial structure of clouds, multi-view image features are mapped onto cloud-layer planes via homography transformations to predict 2.5D cloud properties (liquid water path LWP, cloud base height CBH, and cloud layer thickness $\Delta h$); (2) **3D Refinement** — the 2.5D properties are lifted into an initial 3D liquid water content distribution, which is then refined by a sparse Transformer that learns a complete 3D prior. At inference time, height-varying horizontal wind profiles are additionally extracted by tracking temporal 3D reconstruction results.

### Key Design 1: Homography-Guided Cloud Layer Model

- **Function**: Projects multi-view 2D image features onto horizontal planes at different altitudes via homography transformations, constructing a feature volume, then applies a 2D CNN to predict 2.5D cloud properties.
- **Mechanism**: Exploits the physical prior that cloud layers form horizontal strata at specific altitudes, simplifying the complex 3D estimation into a more tractable 2D-to-2D task. Homography mappings from image to plane are established at 18 altitude levels (400 m–3800 m, sampled every 200 m), lifting DINOv2 features (upsampled via LoftUp) into world coordinates.
- **Design Motivation**: Unlike the cost volumes in multi-view stereo matching, the homographies here are explicitly aligned with the physical altitudes of cloud layers. Since cloud layers are thin vertically but extensive horizontally, the 2.5D representation is both efficient and physically well-motivated. Features at each layer are conditioned on the sampled altitude via Adaptive Layer Normalization, enabling the network to be altitude-aware.

### Key Design 2: Sparse Transformer 3D Refinement

- **Function**: Lifts 2.5D predictions into an initial 3D liquid water content field, then applies a sparse Transformer to learn the full 3D distribution and output final voxel-level LWC estimates.
- **Mechanism**: 3D voxels are initialized from LWP, CBH, and $\Delta h$ — LWP is uniformly distributed within the cloud layer, augmented with a physically motivated prior of linearly increasing LWC toward the cloud top (consistent with observed LWC profiles); empty voxels are discarded to yield a sparse structure ($M \ll N_x \cdot N_y \cdot N_z$), substantially reducing computational complexity. Each sparse voxel concatenates back-projected DINOv2 features and sinusoidal positional encodings as input.
- **Design Motivation**: A purely 2.5D representation cannot capture fine-grained vertical variations in LWC within the cloud. The sparse Transformer learns a 3D prior without processing a full dense voxel grid. Outputs are softmax-normalized along the height dimension and then rescaled to preserve the original LWP, combining the strong features of the 2.5D model with the flexibility of 3D refinement.

### Key Design 3: Wind Field Retrieval from 3D Reconstructions

- **Function**: Retrieves height-varying horizontal wind profiles by tracking motion across temporal sequences of 3D liquid water content reconstructions.
- **Mechanism**: The 3D LWC volume is sliced into 2D image sequences at different altitude levels; the off-the-shelf point tracker CoTracker3 is applied to track horizontal cloud motion, and pixel-space motion rates are directly converted to horizontal wind speeds.
- **Design Motivation**: Compared with conventional satellite cloud motion vectors (tracked in 2D images), tracking on the full 3D LWC field directly yields height-varying wind profiles with richer physical content. Temporal cues and efficient long-sequence processing are exploited without requiring additional training.

### Loss & Training

- **Two-stage training**: Stage 1 trains the Cloud Layer Model to predict 2.5D properties (60k steps); Stage 2 freezes Stage 1 weights and trains the sparse Transformer for 3D refinement (30k steps).
- **Stage 1 loss**: $\mathcal{L}_{2D} = \mathcal{L}_{LWP} + 0.1 \cdot \mathcal{L}_{CBH} + 0.1 \cdot \mathcal{L}_{\Delta h}$, where all three terms are L1 losses and the weighting coefficients scale the different physical quantities to comparable magnitudes.
- **Stage 2 loss**: $\mathcal{L}_{3D} = \|\rho - \hat{\rho}\|_1$, an L1 loss computed directly on 3D liquid water content voxels.
- **Training data**: Synthetic training data are generated using the large-eddy simulation (LES) software MicroHH (three cumulus scenarios), supplemented with Terragen-rendered data for pretraining to increase diversity, totaling 15,000 images. Training runs for 3 days on 4× H100 GPUs.

## Key Experimental Results

### Table 1: Quantitative Comparison Against Existing Methods (Radar Baseline)

| Method | Occupancy F1 ↑ | LWC MAE (g/m³) ↓ | LWP MAE (kg/m²) ↓ | CBH MAE (m) ↓ | CTH MAE (m) ↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| VIP-CT (satellite method) | 0.40 | 0.13 | 0.39 | 791.23 | 1021.49 |
| **Cloud4D (Ours)** | **0.70** | **0.03** | **0.06** | **189.58** | **295.77** |

Cloud4D substantially outperforms the satellite-based method VIP-CT on all metrics: LWC error is reduced by 77%, LWP error by 85%, and cloud base and cloud top height errors by 76% and 71%, respectively. The relative error against radar is only 8.9% (MAE 0.029 / radar mean 0.321 g/m³).

### Table 2: Key System Parameter Comparison

| Metric | Cloud4D | Satellite (Sentinel-2/MODIS) | ERA5 |
|------|:---:|:---:|:---:|
| Spatial resolution | 25 m × 25 m × 25 m | ~10 m–1 km (2D) | ~31 km |
| Temporal resolution | 5 seconds | Hours to 5 days | 1 hour |
| Coverage | 5 km × 5 km | Global | Global |
| 3D physical quantities | ✓ (LWC volume) | ✗ (2D images) | ✓ (coarse) |
| Relative error vs. radar | <10% | — | Captures only coarse properties |

## Highlights & Insights

1. **Elegant integration of physical priors**: The layered structure of clouds is encoded as homography transformations, decomposing the difficult 3D reconstruction problem into 2.5D prediction followed by sparse 3D refinement — computationally efficient and physically principled.
2. **Order-of-magnitude improvement in spatiotemporal resolution**: Compared with satellite products, temporal resolution improves from hours/days to 5 seconds, spatial resolution reaches 25 m, and 3D physical quantities are simultaneously retrieved.
3. **Low-cost, scalable solution**: Only six ground-level cameras are required to replace expensive radar equipment, providing a viable pathway toward a globally dense cloud observation network.
4. **Bonus: wind field retrieval**: Height-varying horizontal wind profiles are additionally obtained from 3D reconstruction sequences using an off-the-shelf point tracker, without requiring dedicated instrumentation.
5. **Cross-domain application of visual foundation models**: DINOv2 + LoftUp demonstrates strong 3D perception capability in meteorological scenes, validating the transferability of visual foundation models to scientific domains.

## Limitations & Future Work

1. **Restricted cloud types**: The method is currently trained and evaluated only on shallow cumulus clouds, without covering other important cloud types such as stratus and cirrus.
2. **Single-layer assumption**: Only the lowest cloud layer is processed; clouds at higher altitudes are obscured and cannot be estimated, limiting applicability in multi-layer cloud scenarios.
3. **Sensitivity to environmental conditions**: Ground-level cameras are susceptible to occlusion under adverse weather conditions such as rain, fog, and snow, reducing robustness.
4. **Synthetic-to-real domain gap**: Training relies entirely on LES synthetic data, which may lead to generalization issues for extreme or rare cloud morphologies.
5. **Limited spatial coverage**: A 5 km × 5 km footprint is far smaller than satellite coverage; analysis of large-scale weather systems still requires satellite data.

## Related Work & Insights

- **VIP-CT / 3DeepCT**: Pioneer works on cloud physical property estimation from satellite perspectives, but they learn camera geometry implicitly and fail in the ground-camera setting — highlighting the critical importance of explicit geometric modeling when viewpoint variation is large.
- **Multi-view stereo (MVSNet, etc.)**: The homography design in Cloud4D shares conceptual roots with cost volumes in MVS, but the key innovation is aligning the plane-sweep assumption with the physical altitudes of cloud layers.
- **GraphCast / NeuralGCM and other ML weather models**: These global models operate at kilometer resolution; Cloud4D fills the fine-scale observational gap they cannot address, and could serve as high-fidelity training data or a validation tool in the future.
- **CoTracker3**: Employed as a plug-and-play temporal tracking module for wind field retrieval, demonstrating a composable paradigm for applying visual foundation models in scientific applications.
- **Interdisciplinary inspiration**: Transferring well-established CV techniques for multi-view reconstruction (homographies, sparse Transformers, feature volumes) to atmospheric science represents an exemplary instance of AI for Science.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneers the task of ground-camera-to-4D cloud physical property estimation; the homography-guided 2.5D-to-3D decomposition is a distinctive design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two months of real-world deployment and 12 days of cumulus data provide solid evaluation, though cloud type and scenario diversity remain limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear, physical motivation is well-developed, figures are polished, and quantitative and qualitative comparisons are comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new paradigm for low-cost, high-resolution cloud observation with significant implications for both meteorology and AI for Science.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/video_understanding/alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[NeurIPS 2025\] ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](convis-bench_estimating_video_similarity_through_semantic_concepts.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[AAAI 2026\] VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation](../../AAAI2026/video_understanding/vtinker_guided_flow_upsampling_and_texture_mapping_for_high-resolution_video_fra.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)

<!-- RELATED:END -->
