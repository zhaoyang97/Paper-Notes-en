---
title: >-
  [Paper Note] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception
description: >-
  [ICCV 2025][Autonomous Driving][collaborative perception] This paper proposes INSTINCT, a LiDAR-based instance-level collaborative perception framework that achieves state-of-the-art performance across multiple datasets…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "collaborative perception"
  - "instance-level fusion"
  - "V2X"
  - "LiDAR"
  - "bandwidth efficiency"
date: 2026-05-08
content_hash: c32f5dd09543a6b2
---

# INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception

**Conference**: ICCV 2025
**arXiv**: [2509.23700](https://arxiv.org/abs/2509.23700)
**Code**: [https://github.com/CrazyShout/INSTINCT](https://github.com/CrazyShout/INSTINCT)
**Area**: Autonomous Driving
**Keywords**: collaborative perception, instance-level fusion, V2X, LiDAR, bandwidth efficiency

## TL;DR

This paper proposes INSTINCT, a LiDAR-based instance-level collaborative perception framework that achieves state-of-the-art performance across multiple datasets through three core modules — quality-aware filtering, dual-branch detection routing, and cross-agent local instance fusion — while reducing communication bandwidth to approximately 1/264–1/281 of that required by existing methods.

## Background & Motivation

Collaborative perception systems overcome the limitations of single-vehicle perception in long-range detection and occlusion scenarios by fusing sensor data from multiple agents. However, frequent collaborative interactions and real-time requirements (≥10 Hz) impose stringent communication bandwidth constraints. Existing intermediate fusion methods transmit complete feature maps, resulting in substantial bandwidth overhead. While query-based instance-level interaction has been explored in camera-based modalities, it remains underdeveloped for LiDAR-based collaborative perception, with performance lagging behind state-of-the-art methods. INSTINCT aims to bridge this gap in LiDAR-based instance-level collaborative perception while simultaneously achieving high accuracy and low bandwidth consumption.

## Method

### Overall Architecture

INSTINCT operates in five stages:
1. A single-agent detector extracts instance features $\mathbf{Q}_i$ from LiDAR data.
2. The quality-aware filtering module selects high-quality instance features $\tilde{\mathbf{Q}}_i$ and generates a unified spatial position map $\mathcal{S}_{j \to i}$.
3. The dual-branch detection routing (DDR) module partitions instances into collaboration-relevant $\mathbf{Q}_i^{coop}$ and collaboration-irrelevant $\mathbf{Q}_i^{single}$ subsets.
4. The cross-agent local instance fusion (CALIF) module performs cross-domain adaptation and Gaussian distance-based local fusion on the collaborative features.
5. The final detection head produces prediction outputs.

### Key Designs

1. **Quality-Aware Filtering (QAF)**: Instance feature quality is ensured via an IoU-penalized classification loss (MAL Loss). MAL is applied exclusively to the last decoder layer, and BEV IoU is adopted to avoid numerical instability associated with 3D IoU. A sparse 2D relative position map is constructed for coordinate unification and range filtering, reducing bandwidth by approximately 94.1%. The MAL loss is formulated as: $\text{MAL}(p,q,y) = -q^\gamma \log(p) + (1-q^\gamma)\log(1-p)$ (when $q>0$).

2. **Dual-Branch Detection Routing (DDR)**: Motivated by the observation that an instance with no corresponding counterpart across the collaborative scene cannot benefit from collaboration and may instead introduce interference, all received instance features are concatenated into a feature table and passed through a shared-parameter detection head to obtain detection boxes. An IoU matrix $\mathcal{M}_{iou}$ is computed between all instance pairs; instances whose IoU values fall below threshold $\lambda$ with all others are assigned to $\mathbf{Q}_i^{single}$, while the remainder are assigned to $\mathbf{Q}_i^{coop}$.

3. **Cross-Agent Local Instance Fusion (CALIF)**: This module comprises two sub-components: (a) Cross-Domain Adaptation (CDA), which employs a self-attention mechanism to bridge domain gaps caused by heterogeneous hardware and environmental conditions, augmented with spatial positional encodings and agent-aware encodings; (b) Gaussian Distance-based Attention (GDA), which generates attention weights based on the Euclidean distance between the circumscribed circle centers of instance bounding boxes: $\mathcal{W}_{k,v} = \exp\!\left(-\frac{\sqrt{(x_k-x_v)^2+(y_k-y_v)^2}}{\beta r_k^2}\right)$. This enables asymmetric local interaction, naturally suppressing contributions from distant instances or those with large detection deviations.

### Loss & Training

- Classification loss: combination of Focal Loss and MAL Loss.
- Regression loss: L1 Loss.
- A **Co-GT Sampling** strategy is introduced: a cross-agent object-level point cloud database is constructed, from which samples are drawn during training with spatial consistency verification, increasing the diversity of mixed instance features.
- A **Fade** strategy is employed near training convergence to obtain samples closer to the true data distribution.
- Optimizer: Adam with initial learning rate 0.001 and OneCycle scheduling.

## Key Experimental Results

### Main Results

| Model | Fusion Type | DAIR-V2X AP@0.5/0.7 | V2XSet AP@0.5/0.7 | V2V4Real AP@0.5/0.7 | Bandwidth (log₂) |
|-------|-------------|---------------------|-------------------|---------------------|------------------|
| No Fusion | — | 0.635/0.496 | 0.652/0.520 | 0.398/0.220 | 0 |
| V2VNet | Intermediate | 0.634/0.423 | 0.827/0.658 | 0.647/0.336 | 24.62/25.10/25.10 |
| Where2comm | Intermediate | 0.790/0.665 | 0.926/0.849 | 0.702/0.380 | 21.72/21.19/22.86 |
| CoAlign | Intermediate | 0.780/0.655 | 0.929/0.847 | 0.721/0.466 | 24.62/25.10/25.10 |
| **INSTINCT** | **Instance** | **0.819/0.753** | **0.923/0.873** | **0.809/0.620** | **13.58/14.16/14.81** |

On the real-world datasets DAIR-V2X and V2V4Real, INSTINCT surpasses the previous state-of-the-art by 13.23% and 33.08% in AP@0.7, respectively, with bandwidth as low as $2^{13}$–$2^{14}$ bytes/frame (approximately 16 KB/frame).

### Ablation Study

| QAF | DDR | CDA | GDA | GT | AP@0.5/0.7 | Bandwidth (log₂) |
|-----|-----|-----|-----|-----|------------|------------------|
| | | | | | 0.730/0.598 | 17.67 |
| ✓ | | | | | 0.696/0.604 | 13.58 |
| ✓ | ✓ | | | | 0.720/0.632 | 13.58 |
| ✓ | ✓ | ✓ | | | 0.790/0.710 | 13.58 |
| ✓ | ✓ | ✓ | ✓ | | 0.811/0.739 | 13.58 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **0.819/0.753** | 13.58 |

Individual module contributions: QAF reduces bandwidth by 94.1%; DDR yields +3.43% AP@0.7; adding CDA to DDR yields +11.23%; the full CALIF module yields +14.16%; Co-GT Sampling achieves a final gain of +15.51%.

### Key Findings

- In pose noise robustness evaluations, INSTINCT maintains top accuracy across all noise levels, demonstrating superior environmental robustness.
- Performance gains on the synthetic dataset V2XSet are relatively modest (+2.81%), attributed to the uniform scene distribution raising the performance ceiling.
- Instance-level interaction demonstrates substantially greater adaptability than conventional feature-level interaction in complex real-world scenarios.

## Highlights & Insights

- INSTINCT is the first collaborative perception architecture to achieve LiDAR-based instance-level interaction in V2X scenarios.
- Communication bandwidth is reduced to approximately 16 KB/frame, roughly 1/264–1/281 of that required by feature-level methods.
- The dual-branch routing strategy is both concise and effective, isolating collaboration-irrelevant instances to prevent interference.
- The Gaussian distance-based asymmetric local attention mechanism is an elegant design that leverages geometric priors to guide feature interaction.

## Limitations & Future Work

- Detection performance remains limited in extremely sparse point cloud scenarios.
- The IoU threshold $\lambda$ in DDR is a manually specified hyperparameter; adaptive selection warrants further investigation.
- Co-GT Sampling increases training complexity; simplification strategies merit exploration.
- Integration with camera modalities has not been investigated.

## Related Work & Insights

- The instance-level interaction paradigm is extensible to other sensor fusion scenarios (e.g., radar + camera).
- The techniques for adapting MAL Loss to 3D detection in QAF (applying only to the last decoder layer with BEV IoU) have broad applicability.
- Co-GT Sampling introduces a new paradigm for data augmentation in collaborative perception settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First LiDAR-V2X instance-level collaborative framework; systematic and novel three-module design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets (real-world + synthetic), comprehensive ablation, and pose noise robustness evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, complete derivations, and intuitive figures.
- **Value**: ⭐⭐⭐⭐ Substantial improvements over state-of-the-art on real-world datasets with significant bandwidth reduction; high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)
- [\[NeurIPS 2025\] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](../../NeurIPS2025/autonomous_driving/sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)
- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)
- [\[ICLR 2026\] SiMO: Single-Modality-Operable Multimodal Collaborative Perception](../../ICLR2026/autonomous_driving/simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)

</div>

<!-- RELATED:END -->
