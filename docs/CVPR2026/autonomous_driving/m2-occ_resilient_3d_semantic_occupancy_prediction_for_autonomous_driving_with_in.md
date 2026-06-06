---
title: >-
  [Paper Note] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs
description: >-
  [CVPR 2026][Autonomous Driving][Semantic occupancy prediction] M²-Occ addresses real-world scenarios where camera failures cause missing views by proposing MMR (reconstructing missing view representations in feature spac…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "Semantic occupancy prediction"
  - "sensor failure"
  - "missing view reconstruction"
  - "semantic prototypes"
  - "robust perception"
date: 2026-05-08
content_hash: 84358337080c955d
---

# M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

**Conference**: CVPR 2026
**arXiv**: [2603.09737](https://arxiv.org/abs/2603.09737)  
**Code**: [github.com/qixi7up/M2-Occ](https://github.com/qixi7up/M2-Occ)  
**Area**: Autonomous Driving / 3D Perception
**Keywords**: Semantic occupancy prediction, sensor failure, missing view reconstruction, semantic prototypes, robust perception

## TL;DR

M²-Occ addresses real-world scenarios where camera failures cause missing views by proposing MMR (reconstructing missing view representations in feature space using adjacent camera FoV overlaps) and FMM (refining ambiguous voxel features via a learnable semantic prototype memory bank). On the SurroundOcc baseline, it achieves +4.93% IoU when the rear camera is missing, maintains 18.36% IoU under five missing cameras (versus a baseline collapse to 13.35%), and does not compromise performance under complete-view inputs.

## Background & Motivation

**Background**: 3D semantic occupancy prediction (SOP) is a critical task in autonomous driving, describing the geometric structure and semantic information around a vehicle at the voxel level. Camera-based approaches (SurroundOcc, TPVFormer, VoxFormer) have achieved notable progress, typically assuming all six surround-view cameras operate normally.

**A Neglected Yet Critical Problem**: In practice, cameras frequently fail due to lens occlusion, hardware damage, or communication interruption. Even a single camera failure causes catastrophic performance degradation—e.g., losing the rear camera drops SurroundOcc's IoU from 32.38% to 23.94% (−26%), which is unacceptable for safety-critical systems.

**Existing Robustness Work Focuses on BEV, Not 3D Occupancy**: Methods handling missing views such as M-BEV, MetaBEV, and SafeMap target BEV detection or map construction, and do not address the more challenging task of dense 3D semantic occupancy prediction.

**Mechanism**: The paper draws on the human ability to infer unseen regions from context and supplement information from memory: (a) MMR reconstructs missing view representations in feature space by leveraging FoV overlap among adjacent cameras; (b) FMM employs global semantic prototypes as prior knowledge to refine voxel features that remain ambiguous after reconstruction.

## Method

### Overall Architecture

Input: $N$ surround-view camera images (some potentially missing) → shared ResNet-101 + FPN for 2D feature extraction → MMR module reconstructs feature representations for missing views → 2D-to-3D view transformation (spatial cross-attention) to build a unified 3D volume → FMM module refines voxel features with semantic prototypes → 3D occupancy head predicts per-voxel semantic labels.

### Key Designs

1. **MMR (Multi-view Masked Reconstruction)**:

    - **Function**: Restores 2D feature representations of missing views in feature space.
    - **Design Motivation**: The six surround-view cameras in nuScenes exhibit significant FoV overlaps (e.g., the right boundary regions of the front-left and front cameras overlap). When a camera fails, partial information about its coverage area can be recovered from the boundary features of adjacent cameras.
    - **Core Pipeline**:
        - **View Relationship Modeling**: The six cameras are modeled as a circular graph $\mathcal{N}(v_i) = \{v_{(i-1) \bmod N},\ v_{(i+1) \bmod N}\}$.
        - **Overlap Region Feature Extraction**: Overlapping boundary regions (width $w_{ov}$) are cropped from the feature maps of the left and right adjacent cameras and concatenated with a learnable mask token $\mathbf{e}_{mask}$: $\mathbf{f}_{ref} = \text{Concat}(\mathbf{f}_{left}[:,-w_{ov}:],\ \mathbf{e}_{mask},\ \mathbf{f}_{right}[:,:w_{ov}])$.
        - **Transformer Decoder Reconstruction**: A 6-layer Transformer decoder (8-head attention) with learnable positional encodings refines the coarse structural prior into a reconstructed result $\hat{\mathbf{f}}_i = \mathcal{D}(\mathbf{f}_{ref} + \mathbf{p}_{pos})$ that approximates the original complete features.
    - **Distinction from Generative Methods**: MMR reconstructs in **feature space** rather than generating raw pixels, avoiding the high computational cost and noise introduction associated with image generation.
    - **Reconstruction Loss**: $\mathcal{L}_{MMR} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|\hat{\mathbf{f}}_i - \mathbf{f}_i^{gt}\|_2^2$, computed only on masked views to prevent the model from learning an identity mapping.

2. **FMM (Feature Memory Module)**:

    - **Function**: Refines semantically ambiguous regions in 3D voxel features using global semantic prototypes as prior knowledge.
    - **Design Motivation**: Although MMR recovers geometric structure, reconstructed features may remain ambiguous or semantically equivocal, particularly in central blind zones far from overlap regions. FMM acts as a "long-term memory" that stores the "ideal" feature representation for each object class to correct such deviations.
    - **Single-Proto Strategy**: A single global centroid prototype $\mathbf{m}_k$ is maintained per semantic class and updated via momentum moving average: $\mathbf{m}_k^{(t)} = (1-\lambda)\mathbf{m}_k^{(t-1)} + \lambda \cdot \bar{\mathbf{f}}_k$ ($\lambda=0.1$), filtering mini-batch noise. **Experiments demonstrate that Single-Proto is more stable than Multi-Proto.**
    - **Multi-Proto Strategy**: Each class maintains $N_p$ sub-prototypes to capture intra-class variation (e.g., "truck" encompasses pickups and semi-trailers). Retrieval is performed by computing cosine similarity + temperature-$\tau$ softmax weighting. However, under missing-view conditions, similarity-based routing can amplify noise and lead to excessive fragmentation.
    - **Memory-Enhanced Feature**: $\mathbf{x}' = \mathbf{x} + \sum_{k=1}^{K} P(k) \sum_{j=1}^{N_p} \alpha_{k,j} \mathbf{m}_{k,j}$, where predicted class probability $P(k)$ serves as a gate, injecting semantic priors in a residual manner.

3. **Training Strategy — Random View Masking (RVM)**:

    - During training, images from a random subset of views are dropped to simulate real-world failure scenarios.
    - During testing, specific masking patterns (single-view / multi-view) are applied to evaluate robustness.
    - This masking strategy is conceptually similar to MAE but operates at the level of entire camera views rather than image patches.

### Loss & Training

Total loss = original SurroundOcc occupancy prediction loss + $\mathcal{L}_{MMR}$ (feature reconstruction loss).

## Key Experimental Results

### Main Results — Single View Missing

| Missing View | Metric (IoU) | M²-Occ | SurroundOcc Baseline | Gain |
|---|---|---|---|---|
| Back | IoU | **28.87** | 23.94 | **+4.93** |
| Front | IoU | **30.40** | 25.03 | **+5.37** |
| Front Left | IoU | **31.25** | 30.74 | +0.51 |
| Front Right | IoU | **31.17** | 30.56 | +0.61 |
| Back Left | IoU | **31.08** | 30.35 | +0.73 |
| Back Right | IoU | **31.19** | 30.62 | +0.57 |
| Standard (no missing) | IoU | 32.38 | 32.38 | 0 (no compromise) |

- The largest gains occur when the rear or front camera is missing (+4.93/+5.37), as these positions have the least overlap with adjacent cameras, causing the greatest degradation in the baseline model.
- Performance does not degrade under complete-view inputs, indicating that MMR and FMM introduce no negative interference.

### Multi-View Missing Extension

| Missing Cameras | Metric (IoU) | M²-Occ | Baseline | Gain |
|---|---|---|---|---|
| 1 | IoU | **30.66** | 28.42 | +2.24 |
| 3 | IoU | **26.06** | 20.52 | **+5.54** |
| 5 | IoU | **18.36** | 13.35 | **+5.01** |

- The robustness advantage grows as the number of missing cameras increases.
- In the extreme case of five missing cameras (only one remaining), the baseline collapses to 13.35% IoU, whereas M²-Occ retains 18.36%, preserving critical structural information.

### Ablation Study

| Configuration | IoU | mIoU | Note |
|---|---|---|---|
| No-missing baseline | 30.13 | 15.31 | Complete input reference |
| Missing + no recovery | 26.76 | 13.21 | Missing causes −3.37 IoU |
| + MMR | 28.19 | 13.79 | Recovers geometric structure +1.43 |
| + MMR + Single-Proto | **28.38** | **13.55** | Optimal combination |
| + MMR + Multi-Proto | 27.76 | 12.15 | Multi-proto is less stable |

### Key Findings

- **MMR Primarily Recovers Large-Scale Geometric Structure**: IoU improves substantially for large objects such as drivable surface and vehicles (e.g., drive.surf. rises from 27.51% to 35.02% when the rear camera is missing), but small objects (pedestrians, traffic cones) may decline, as reconstructed features lose high-frequency details.
- **Single-Proto Outperforms Multi-Proto**: Under missing-view conditions, visual evidence is inherently sparse; the similarity-based routing of Multi-Proto amplifies noise instead. A simple, stable single centroid proves more robust than fine-grained but fragile sub-prototypes.
- **Manageable Computational Overhead**: GPU memory increases by only ~0.15 GB (2.5%), and inference latency increases linearly with the number of missing views (one MMR reconstruction pass per missing view).

## Highlights & Insights

- **Precise Problem Formulation**: This work is the first to systematically study robustness of 3D semantic occupancy prediction under camera-missing conditions, establishing a comprehensive evaluation protocol (deterministic single-view failure + random multi-view dropout) that fills an important research gap.
- **Feature-Space Rather Than Pixel-Space Reconstruction**: MMR operates in feature space, elegantly exploiting the natural redundancy provided by adjacent camera FoV overlaps, while avoiding the high cost and instability of image generation.
- **Global Semantic Prototypes as a Fallback**: The design philosophy of FMM is practically motivated—when locally reconstructed feature quality is poor, the model falls back on global statistical priors ("this region should resemble a vehicle / drivable surface"), ensuring semantic consistency.

## Limitations & Future Work

- **Small Object Performance Degradation**: Features reconstructed by MMR lose high-frequency information, potentially causing IoU to decrease for small objects such as pedestrians and traffic cones (e.g., pedestrian IoU drops from 12.50% to 10.51% when the rear camera is missing), which is a safety concern in critical scenarios.
- **Multi-Proto Strategy Underperforms**: Ablation results show Multi-Proto is inferior to Single-Proto, but the underlying cause is not analyzed in depth—better prototype update strategies or noise suppression mechanisms may be needed.
- **Temporal Information Not Exploited**: Adjacent frames can provide additional context to compensate for missing views in the current frame, but M²-Occ is a purely single-frame method.
- **Validation on a Single Baseline Only**: Generalization to other mainstream occupancy methods such as TPVFormer and OccFormer is not demonstrated.
- **Linear Increase in Inference Latency**: Each missing view requires a separate Transformer decoder reconstruction pass; with five missing views, latency increases from 0.50 s to 1.25 s (2.5×), potentially precluding real-time deployment.

## Related Work & Insights

- **vs. M-BEV**: M-BEV also employs masked view reconstruction but targets BEV detection; M²-Occ extends this to dense 3D occupancy prediction and adds FMM for semantic regularization.
- **vs. MetaBEV**: MetaBEV handles sensor failures via LiDAR–camera cross-modal fusion and thus requires LiDAR hardware; M²-Occ is a camera-only solution at lower cost.
- **vs. MAE**: MAE applies patch-level masking for self-supervised pretraining; MMR applies masking at the entire camera-view level in a supervised manner (using GT features from complete views as supervision).
- **Inspiration**: The semantic prototype memory bank in FMM can be transferred to other perception tasks involving input degradation (e.g., 3D perception under rain, fog, or nighttime conditions) as a general semantic stabilization module.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of sensor-missing robustness for occupancy prediction; the MMR+FMM combination is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic single/multi-view missing evaluation protocol with complete ablations, though validated on only one baseline.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear, method figures are intuitive, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐ Addresses a real and overlooked safety problem with practical significance for autonomous driving deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth_region_guided_3d_occupancy.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](test-time_3d_occupancy_prediction.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
