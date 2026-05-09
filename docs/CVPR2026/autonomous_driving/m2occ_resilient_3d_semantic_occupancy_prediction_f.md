---
title: >-
  [Paper Note] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs
description: >-
  [CVPR 2026][Autonomous Driving][semantic occupancy prediction] To address incomplete inputs caused by camera failures in autonomous driving, M²-Occ introduces a Multi-view Masked Reconstruction (MMR) module that exploits the overlapping fields of view between adjacent cameras to recover missing view features, and a Feature Memory Module (FMM) that refines voxel representations using class-level semantic prototypes. The framework achieves a 4.93% IoU gain when the rear camera is missing, without degrading full-view performance.
tags:
  - CVPR 2026
  - Autonomous Driving
  - semantic occupancy prediction
  - missing camera view
  - multi-view reconstruction
  - feature memory
  - robustness
date: 2026-05-08
content_hash: 5503d509130887af
---

# M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

**Conference**: CVPR 2026
**arXiv**: [2603.09737](https://arxiv.org/abs/2603.09737)
**Code**: [https://github.com/qixi7up/M2-Occ](https://github.com/qixi7up/M2-Occ)
**Area**: 3D Vision / Autonomous Driving / Semantic Occupancy Prediction
**Keywords**: semantic occupancy prediction, missing camera view, multi-view reconstruction, feature memory, robustness

## TL;DR
To address incomplete inputs caused by camera failures in autonomous driving, M²-Occ introduces a Multi-view Masked Reconstruction (MMR) module that exploits the overlapping fields of view between adjacent cameras to recover missing view features, and a Feature Memory Module (FMM) that refines voxel representations using class-level semantic prototypes. The framework achieves a 4.93% IoU gain when the rear camera is missing, without degrading full-view performance.

## Background & Motivation

**Background**: 3D semantic occupancy prediction provides dense voxel-level scene understanding for autonomous driving, offering more comprehensive coverage than BEV perception by handling arbitrarily shaped obstacles and fine-grained semantics.

**Limitations of Prior Work**: Existing multi-camera methods (SurroundOcc, TPVFormer, etc.) implicitly assume all six surround-view cameras operate normally. In real deployments, however, camera failures due to lens occlusion, hardware faults, or communication interruptions are common. Preliminary experiments show that even a well-established model like SurroundOcc suffers severe performance degradation upon losing a single critical viewpoint — rear camera loss drops IoU from 31.45% to 23.94%, creating a critical geometric blind spot. While some robustness works exist for the BEV domain (M-BEV, MetaBEV, SafeMap), the robustness of 3D semantic occupancy prediction to sensor failure remains essentially unexplored.

## Core Problem
How to maintain geometric integrity and semantic consistency in 3D semantic occupancy prediction when one or more cameras fail completely? This is a safety-critical issue — rear camera failure renders the vehicle effectively blind to its surroundings behind, potentially causing serious accidents.

## Method

### Overall Architecture
The method follows the standard 2D→3D pipeline: multi-view images are processed by a shared ResNet-101+FPN backbone to extract multi-scale 2D features, which are then lifted into 3D voxel representations via spatial cross-attention, and finally decoded by a 3D occupancy head to predict per-voxel semantic labels. Two new modules are inserted at key stages: (1) the MMR module is inserted after feature extraction to recover missing-view features; (2) the FMM module is inserted during voxel refinement to apply semantic prototype-based correction. During training, Random View Masking (RVM) randomly drops views to simulate failures; during testing, specific views are masked to evaluate robustness.

### Key Designs

1. **Multi-view Masked Reconstruction (MMR)**: Recovers missing-view features by exploiting the physical overlapping fields of view between adjacent cameras. The process consists of three steps:

   - **View Relationship Modeling**: The six cameras are modeled as a cyclic graph, where the neighborhood of view $v_i$ consists of its left and right neighbors: $\mathcal{N}(v_i) = \{v_{(i-1)\text{mod}N}, v_{(i+1)\text{mod}N}\}$
   - **Overlap Region Feature Aggregation**: Overlapping boundary regions (of width $w_{ov}$ corresponding to the physical overlap) are cropped from the feature maps of both neighbors and concatenated with a learnable mask token: $\mathbf{f}_{ref} = \text{Concat}(\mathbf{f}_{left}[:,-w_{ov}:], \mathbf{e}_{mask}, \mathbf{f}_{right}[:,:w_{ov}])$
   - **Transformer Decoding for Reconstruction**: A 6-layer Transformer block (8-head attention, MLP ratio=4) with learnable positional encodings reconstructs the missing features from the coarse structural prior: $\hat{\mathbf{f}}_i = \mathcal{D}(\mathbf{f}_{ref} + \mathbf{p}_{pos})$
   - The MMR loss computes MSE only over masked views: $\mathcal{L}_{MMR} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|\hat{\mathbf{f}}_i - \mathbf{f}_i^{gt}\|^2$

2. **Feature Memory Module (FMM)**: Since MMR-recovered features may be blurry or semantically ambiguous, FMM employs global semantic prototypes as "long-term memory" to refine voxel representations. Two strategies are considered:

   - **Single-Proto**: Each semantic class maintains a global centroid $\mathbf{m}_k$, updated via momentum moving average: $\mathbf{m}_k^{(t)} = (1-\lambda)\mathbf{m}_k^{(t-1)} + \lambda \cdot \bar{\mathbf{f}}_k$, with $\lambda=0.1$. This is stable but cannot capture intra-class diversity.
   - **Multi-Proto**: Each class maintains $N_p$ sub-prototypes $\mathbf{m}_{k,j}$, with retrieval weights $\alpha_{k,j}$ computed via cosine similarity and softmax temperature $\tau$. This can model intra-class variation (e.g., different vehicle types), but may suffer from noisy routing and over-fragmentation under severe missing-view conditions.
   - **Memory-Augmented Features**: Predicted class probabilities $P(k)$ serve as gates to aggregate weighted prototypes as a residual correction: $\mathbf{x}' = \mathbf{x} + \sum_{k=1}^{K}(P(k)\sum_{j=1}^{N_p}\alpha_{k,j}\mathbf{m}_{k,j})$

### Loss & Training
- Primary loss: standard semantic occupancy cross-entropy loss + $\mathcal{L}_{MMR}$
- Built on the official SurroundOcc implementation with ResNet-101 + FCOS3D pretrained weights
- AdamW optimizer, learning rate $2 \times 10^{-4}$, weight decay 0.01
- Trained for 24 epochs; voxel size 200×200×16, range [−50m, 50m]
- Random View Masking (RVM) during training; fixed masking patterns during evaluation

## Key Experimental Results

| Missing View | Metric | M²-Occ | SurroundOcc Baseline | Gain |
|---|---|---|---|---|
| Front | IoU↑ | 30.40 | 25.03 | **+5.37** |
| Back (safety-critical) | IoU↑ | 28.87 | 23.94 | **+4.93** |
| Front Left | IoU↑ | 31.25 | 30.74 | +0.51 |
| Front Right | IoU↑ | 31.17 | 30.56 | +0.61 |
| Back Left | IoU↑ | 31.08 | 30.35 | +0.73 |
| Back Right | IoU↑ | 31.19 | 30.62 | +0.57 |
| 1 View (avg) | IoU↑ | 30.66 | 28.42 | +2.24 |
| 3 Views | IoU↑ | 26.06 | 20.52 | **+5.54** |
| 5 Views | IoU↑ | 18.36 | 13.35 | **+5.01** |

### Ablation Study
- **MMR alone**: IoU recovers from 26.76 to 28.19 (+1.43), primarily restoring large-scale spatial structure (roads, vehicle volumes).
- **MMR + FMM (Single-Proto)**: IoU further improves to 28.38 (+0.19), confirming the effectiveness of semantic refinement.
- **Multi-Proto vs. Single-Proto**: Multi-Proto (IoU 27.76) underperforms Single-Proto (28.38); fine-grained prototype routing introduces noise under missing-view conditions.
- **Computational overhead**: Only ~0.15 GB additional GPU memory (~2.5%), with inference time scaling linearly with the number of missing views.
- **Large objects recover well**: e.g., drivable surface (27.51→35.02, +7.51), whereas small objects degrade (pedestrian 12.50→10.51, traffic cone 8.70→5.71).

## Highlights & Insights
- **Precise problem formulation**: The first work to systematically study sensor failure robustness in 3D semantic occupancy prediction, addressing an important gap.
- **Physically grounded MMR design**: Rather than hallucinating raw pixels, MMR leverages the true overlapping regions from adjacent cameras in feature space as structural priors — a targeted and principled design choice.
- **Minimal overhead**: A mere 0.15 GB memory increase yields significant robustness gains, making the approach deployment-friendly.
- **Comprehensive evaluation protocol**: Covers both deterministic single-view failures and random multi-view dropout, constituting a sound benchmark design.
- **Honest analysis**: The paper explicitly acknowledges limitations on small objects and provides reasonable explanations.

## Limitations & Future Work
- Reconstruction quality for small objects degrades noticeably: MMR relies on boundary overlap regions, which lack sufficient resolution to capture fine details of distant small objects.
- Multi-Proto underperforms Single-Proto under missing-view conditions, suggesting that the prototype routing mechanism requires improvement for incomplete observations.
- Generalizability beyond SurroundOcc to other occupancy prediction methods (e.g., BEVFormer, TPVFormer) remains to be validated.
- Temporal information is not exploited — features from adjacent frames could compensate for missing views in the current frame.
- MMR inference time scales linearly with the number of missing views, increasing by approximately 62% when five views are absent.
- Scenarios involving consecutive missing frames (rather than independent single-frame dropout) are not discussed.

## Related Work & Insights
- **vs. M-BEV**: M-BEV performs whole-view masked reconstruction in BEV space; this work operates in feature space and targets 3D occupancy prediction rather than BEV detection.
- **vs. MetaBEV**: MetaBEV addresses sensor corruption and modality missing for joint BEV tasks (detection + segmentation); M²-Occ focuses on voxel-level reconstruction for 3D semantic occupancy.
- **vs. SafeMap/FlexMap**: These methods target BEV map construction and do not involve dense 3D semantic occupancy prediction.

**Inspirations**: The semantic prototype idea in FMM could be combined with the reflectance channel in LR-SGS — using reflectance as an additional class-level feature to enrich prototypes. Temporal information represents a clear direction for future improvement, leveraging complete observations from prior frames to compensate for current-frame missing views.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic study of sensor failure robustness in 3D occupancy prediction; the physically grounded MMR design is targeted and principled.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 single-view and multiple multi-view dropout scenarios with complete ablations, though only a single baseline is evaluated.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, experimental analysis is honest (acknowledging small-object limitations), and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Directly relevant to safe autonomous driving deployment; the low-overhead design has strong practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth-_and_region-guided_3d_occupancy_from_surround-view_cameras_for_auton.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)

<!-- RELATED:END -->
