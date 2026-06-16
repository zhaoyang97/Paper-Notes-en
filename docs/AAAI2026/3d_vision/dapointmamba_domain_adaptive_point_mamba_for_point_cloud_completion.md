---
title: >-
  [Paper Note] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion
description: >-
  [AAAI 2026][3D Vision][Point Cloud Completion] This work presents the first integration of Mamba (SSM) into unsupervised domain adaptive point cloud completion (UDA PCC). The proposed DAPointMamba framework achieves high…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Domain Adaptation"
  - "State Space Model"
  - "Mamba"
  - "Cross-Domain Alignment"
date: 2026-05-08
content_hash: 22b6d7f9c687aaa9
---

# DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion

**Conference**: AAAI 2026
**arXiv**: [2511.20278](https://arxiv.org/abs/2511.20278)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: Point Cloud Completion, Domain Adaptation, State Space Model, Mamba, Cross-Domain Alignment

## TL;DR

This work presents the first integration of Mamba (SSM) into unsupervised domain adaptive point cloud completion (UDA PCC). The proposed DAPointMamba framework achieves high-quality cross-domain point cloud completion through three modules—Cross-Domain Patch-Level Scanning, Spatial SSM Alignment, and Channel SSM Alignment—while maintaining linear complexity and a global receptive field.

## Background & Motivation

Point cloud completion (PCC) is a fundamental task in 3D vision with broad applications in autonomous driving, robotics, and virtual reality. However, existing supervised methods suffer significant performance degradation under cross-domain deployment due to distribution shifts caused by heterogeneous sensors and scenes.

Existing UDA PCC methods face two primary bottlenecks:

**CNN architectures are constrained by local receptive fields**: Convolution-based backbones cannot model global geometric structures, limiting the learning of domain-invariant features.

**Transformer architectures suffer from quadratic complexity**: Although DAPoinTr introduces global modeling, the quadratic complexity of attention mechanisms leads to poor computational efficiency, particularly for long patch sequences.

Mamba/SSM models naturally offer global receptive fields and linear complexity, yet directly applying SSM to UDA PCC encounters the following challenges:

- **Spatial topology disruption**: Directly serializing sparse, unstructured 3D point clouds into 1D sequences destroys spatial topology and local geometric features.
- **Absence of domain-invariant feature design**: Existing Point Mamba architectures lack specialized designs for domain transfer.

## Method

### Overall Architecture

DAPointMamba consists of three core components that form a hierarchical cross-domain alignment system, progressing from local to global:

1. **Cross-Domain Patch-Level Scanning (CDPS)**: Ensures spatial correspondence during serialization.
2. **Cross-Domain Spatial SSM Alignment (CDSA)**: Addresses fine-grained spatial inconsistency.
3. **Cross-Domain Channel SSM Alignment (CDCA)**: Addresses global semantic inconsistency.

### Key Designs

#### CDPS: Cross-Domain Patch-Level Scanning

The core idea of CDPS is to ensure spatial alignment of patches between the source and target domains via shared coordinate normalization and Z-order serialization.

Specific steps:
1. Compute the shared minimum coordinate across both domains: $C_{min} = min(min(X_s, dim=1), min(X_t, dim=1))$
2. Normalize and discretize into a shared grid space: $G_s = [X_s - C_{min} * scale]$, $G_t = [X_t - C_{min} * scale]$
3. Map 3D coordinates to 1D sequences using a consistent Z-order curve encoding.
4. After sorting by Z-order values, divide the sequence into $G$ patches, each containing $K$ points.

Through unified normalization and Z-order serialization, the $g$-th patch corresponds to the same spatial region in both domains, enabling precise patch-level alignment.

#### CDSA: Cross-Domain Spatial SSM Alignment

CDSA enforces local spatial alignment via similarity-based feature modulation:

1. Apply depthwise separable 1D convolution to patch-level features: $\mathcal{D}_s = DWConv(P_s^G)$, $\mathcal{D}_t = DWConv(P_t^G)$
2. Compute cosine similarity as a spatial similarity weight: $\mathcal{W}_{spatial} = cos(D_s, D_t)$
3. Modulate patch features using the similarity weight: $\tilde{X}_s = P_s^G \odot W_{spatial}$
4. Encourage cross-domain local feature consistency via MSE loss: $\mathcal{L}_{sp} = \frac{1}{BDG}\sum(\tilde{X}_s - \tilde{X}_t)^2$

Design intuition: High-similarity regions are preserved while features in low-similarity regions are suppressed, guiding the model to focus on spatially consistent structures across domains.

#### CDCA: Cross-Domain Channel SSM Alignment

CDCA targets global semantic inconsistency through channel mixing and adaptive modulation:

1. **Global feature computation**: Average over patches to obtain $g_s, g_t \in \mathbb{R}^{B \times D}$
2. **Alignment strength estimation**: $\alpha = Sigmoid(MLP([g_s, g_t])) \in \mathbb{R}^{B \times 1}$
3. **Channel cross-mixing**: Divide feature channels into $S$ segments and interleave source and target domain segments:

    - $X_{s,mix} = [X_s^{(1)}, X_t^{(2)}, X_s^{(3)}, X_t^{(4)}, \cdots]$
4. **Adaptive similarity modulation**: Compute cosine similarity of mixed representations and combine with $\alpha$ to generate adaptive weights.
5. **Channel alignment loss**: $\mathcal{L}_{ch} = \frac{1}{BDG}\sum(\tilde{F}_s - \tilde{F}_t)^2$

Design highlight: Cross-mixing of information dissolves domain boundaries, enabling mutual awareness between the global semantic features of the source and target domains.

### Loss & Training

Total loss function:

$$\mathcal{L}_{total} = Loss_{(CD)} + \lambda L_{sp} + \beta L_{ch}$$

- $Loss_{(CD)}$: Chamfer Distance reconstruction loss
- $\lambda = 0.1$, $\beta = 0.1$
- Initial learning rate $1 \times 10^{-3}$, weight decay $5 \times 10^{-2}$, batch size 32
- Backbone: refinement module of PointMamba

## Key Experimental Results

### Main Results

**3D-FUTURE Dataset (CD↓, ×10⁴):**

| Method | Avg | Cabinet | Chair | Lamp | Sofa | Table |
|------|-----|---------|-------|------|------|-------|
| DAPoinTr | 22.35 | 18.46 | 17.60 | 27.91 | 23.08 | 24.71 |
| **DAPointMamba** | **20.40** | 19.35 | **16.21** | **22.81** | **22.38** | **21.25** |

**ModelNet Dataset (CD↓, ×10⁴):**

| Method | Avg | Plane | Car | Chair | Lamp | Sofa | Table |
|------|-----|-------|-----|-------|------|------|-------|
| DAPoinTr | 13.79 | 2.38 | 8.04 | 13.83 | 33.26 | 12.72 | 12.51 |
| **DAPointMamba** | **13.11** | **2.30** | **7.58** | **13.15** | **32.04** | **12.48** | **11.08** |

**Real-World Scans (UCD↓/UHD↓, ×10⁴/×10²):**

| Method | ScanNet-Chair | KITTI-Car |
|------|--------------|-----------|
| DAPoinTr | 1.1/2.7 | 0.45/1.8 |
| **DAPointMamba** | **0.95**/2.8 | **0.40**/2.1 |

### Ablation Study

**Incremental module contribution (3D-FUTURE Avg CD↓):**

| Baseline | +CDPS | +CDSA | +CDCA |
|----------|-------|-------|-------|
| 23.38 | 21.73 | 21.17 | **20.40** |

**Computational efficiency comparison:**

| Model | Params(M) | FLOPs(G) | Time(ms) |
|------|-----------|----------|----------|
| DAPoinTr | 36.904 | 24.912 | 23.774 |
| **DAPointMamba** | **9.571** | **5.192** | **3.820** |

### Key Findings

- Compared to DAPoinTr, DAPointMamba reduces parameter count by 74%, FLOPs by 79%, and inference latency by 84%.
- CDPS contributes the largest individual gain (−1.65 CD); CDCA yields the most notable improvements on high-variance categories (lamp, table).
- On real-world scan data, UCD consistently outperforms prior methods; however, UHD (maximum point error) is slightly inferior, as the method optimizes overall shape rather than extreme points.

## Highlights & Insights

1. **First exploration of SSM adaptation for UDA PCC**, filling a research gap for Mamba in domain-adaptive point cloud tasks.
2. **The three-tier alignment architecture is elegantly structured**: CDPS (patch spatial correspondence) → CDSA (fine-grained spatial alignment) → CDCA (global semantic alignment), progressing systematically from local to global.
3. **Shared Z-order curve normalization** is a concise yet effective means of cross-domain spatial alignment.
4. **Balance between linear complexity and high accuracy**: DAPointMamba surpasses the Transformer-based baseline in performance (CD reduced by 1.95) while substantially lowering computational cost.

## Limitations & Future Work

1. Performance on the UHD metric (maximum point error) remains moderate, potentially requiring specialized handling of boundary points.
2. Evaluation is limited to synthetic→real and synthetic→synthetic transfer; real→real scenarios are not assessed.
3. The channel mixing strategy (alternating even/odd segments) is relatively fixed; adaptive mixing ratios warrant further exploration.
4. The Cabinet category underperforms relative to DAPoinTr, indicating that adaptation for certain geometric shapes still has room for improvement.

## Related Work & Insights

- **DAPoinTr** (SOTA baseline): A pioneering Transformer-based approach for UDA PCC, but quadratic complexity remains a bottleneck.
- **PointMamba**: Validates the effectiveness of SSM for point cloud analysis; DAPointMamba extends this by introducing domain adaptation capability.
- **Z-order curve**: A classical spatial indexing method, here cleverly repurposed for unified serialization to achieve cross-domain spatial alignment.
- Inspiration: The adaptability of SSM/Mamba to other 3D cross-domain tasks (e.g., detection, segmentation) merits further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐ (First integration of Mamba into UDA PCC; three modules exhibit notable originality)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple benchmarks + real-world data + efficiency comparison + visualization)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic, rich figures and tables)
- Value: ⭐⭐⭐⭐ (Opens a new direction for Mamba in domain-adaptive 3D tasks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](../../ICCV2025/3d_vision/dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)

</div>

<!-- RELATED:END -->
