---
title: >-
  [Paper Note] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion
description: >-
  [AAAI 2026][3D Vision][Point Cloud Completion] This work introduces Mamba (SSM) into Unsupervised Domain Adaptive Point Cloud Completion (UDA PCC) for the first time, proposing the DAPointMamba framework. By utilizing three modules—cross-domain patch-level scanning, cross-domain spatial SSM alignment, and cross-domain channel SSM alignment—it achieves high-quality cross-domain point cloud completion while maintaining linear complexity and a global receptive field.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Domain Adaptation"
  - "State Space Model"
  - "Mamba"
  - "Cross-Domain Alignment"
date: 2026-05-08
content_hash: 0a8ee0ac7a0dea9f
---

# DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion

**Conference**: AAAI 2026  
**arXiv**: [2511.20278](https://arxiv.org/abs/2511.20278)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Point Cloud Completion, Domain Adaptation, State Space Model, Mamba, Cross-Domain Alignment

## TL;DR

This work introduces Mamba (SSM) into Unsupervised Domain Adaptive Point Cloud Completion (UDA PCC) for the first time, proposing the DAPointMamba framework. By utilizing three modules—cross-domain patch-level scanning, cross-domain spatial SSM alignment, and cross-domain channel SSM alignment—it achieves high-quality cross-domain point cloud completion while maintaining linear complexity and a global receptive field.

## Background & Motivation

Point cloud completion (PCC) is a fundamental task in 3D vision, widely applied in autonomous driving, robotics, and virtual reality. However, existing supervised methods suffer from severe performance degradation when deployed across domains due to the distribution shifts introduced by different sensors and scenarios.

Existing UDA PCC methods mainly face two bottlenecks:

**CNN architectures are limited by local receptive fields**: Convolution-based backbones cannot model global geometric structures, which restricts domain-invariant feature learning.

**Transformer architectures suffer from quadratic complexity**: Although DAPoinTr introduces global modeling, the quadratic complexity of the self-attention mechanism leads to low computational efficiency, especially for long sequences of patches.

Mamba/SSM models naturally possess the advantages of a global receptive field and linear complexity, but directly applying SSM to UDA PCC faces the following challenges:

- **Spatial topology destruction**: Directly serializing sparse, unstructured 3D point clouds into 1D sequences destroys the spatial topology and local geometric features.
- **Lack of domain-invariant feature design**: Existing Point Mamba architectures lack specialized designs for domain adaptation.

## Method

### Overall Architecture

DAPointMamba contains three core components, forming a cross-domain alignment system from local to global levels:

1. **Cross-Domain Patch-Level Scanning (CDPS)**: Ensures spatial correspondence during serialization.
2. **Cross-Domain Spatial SSM Alignment (CDSA)**: Resolves fine-grained spatial inconsistencies.
3. **Cross-Domain Channel SSM Alignment (CDCA)**: Resolves global semantic inconsistencies.

### Key Designs

#### CDPS: Cross-Domain Patch-Level Scanning

The core idea of CDPS is to ensure that the patches of the source and target domains are spatially aligned through shared coordinate normalization and Z-order serialization.

Specific steps:
1. Compute the shared minimum coordinates of the source and target domains: $C_{min} = min(min(X_s, dim=1), min(X_t, dim=1))$
2. Normalize and discretize into the shared grid space: $G_s = [X_s - C_{min} * scale]$, $G_t = [X_t - C_{min} * scale]$
3. Use a consistent Z-order curve encoding to map 3D coordinates into 1D sequences.
4. After sorting by Z-order values, divide the sequence into $G$ patches, each containing $K$ points.

Due to unified normalization and Z-order serialization, the $g$-th patch in both domains corresponds to the same spatial region, achieving precise patch-level alignment.

#### CDSA: Cross-Domain Spatial SSM Alignment

CDSA enforces local spatial alignment through similarity-based feature modulation:

1. Apply depthwise separable 1D convolution to patch-level features: $\mathcal{D}_s = DWConv(P_s^G)$, $\mathcal{D}_t = DWConv(P_t^G)$
2. Compute cosine similarity as spatial similarity weights: $\mathcal{W}_{spatial} = cos(D_s, D_t)$
3. Modulate patch features using the similarity weights: $\tilde{X}_s = P_s^G \odot W_{spatial}$
4. Promote cross-domain local feature consistency using MSE loss: $\mathcal{L}_{sp} = \frac{1}{BDG}\sum(\tilde{X}_s - \tilde{X}_t)^2$

Design Intuition: Regions with high similarity remain unchanged, while features in low-similarity regions are suppressed, thereby guiding the model to focus on consistent spatial structures across domains.

#### CDCA: Cross-Domain Channel SSM Alignment

CDCA targets global semantic inconsistencies, achieving global alignment through channel mixing and adaptive modulation:

1. **Global Feature Computation**: Average the patches to obtain $g_s, g_t \in \mathbb{R}^{B \times D}$.
2. **Alignment Intensity Estimation**: $\alpha = Sigmoid(MLP([g_s, g_t])) \in \mathbb{R}^{B \times 1}$
3. **Cross-Channel Mixing**: Divide feature channels into $S$ segments, and alternately concatenate source and target domain segments.

    - $X_{s,mix} = [X_s^{(1)}, X_t^{(2)}, X_s^{(3)}, X_t^{(4)}, \cdots]$
4. **Adaptive Similarity Modulation**: Compute the cosine similarity of the mixed representation and combine it with $\alpha$ to generate adaptive weights.
5. **Channel Alignment Loss**: $\mathcal{L}_{ch} = \frac{1}{BDG}\sum(\tilde{F}_s - \tilde{F}_t)^2$

Design Highlight: Breaking domain boundaries through feature cross-mixing allows the global semantic features of the source and target domains to perceive each other.

### Loss & Training

Total loss function:

$$\mathcal{L}_{total} = Loss_{(CD)} + \lambda L_{sp} + \beta L_{ch}$$

- $Loss_{(CD)}$: Chamfer Distance reconstruction loss.
- $\lambda = 0.1$, $\beta = 0.1$.
- Initial learning rate $1 \times 10^{-3}$, weight decay $5 \times 10^{-2}$, batch size 32.
- Backbone: Refinement module of PointMamba.

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

**Incremental Effects of Component Addition (3D-FUTURE Avg CD↓):**

| Baseline | +CDPS | +CDSA | +CDCA |
|----------|-------|-------|-------|
| 23.38 | 21.73 | 21.17 | **20.40** |

**Computational Efficiency Comparison:**

| Model | Params(M) | FLOPs(G) | Time(ms) |
|------|-----------|----------|----------|
| DAPoinTr | 36.904 | 24.912 | 23.774 |
| **DAPointMamba** | **9.571** | **5.192** | **3.820** |

### Key Findings

- Compared to DAPoinTr, the parameters are reduced by 74%, FLOPs by 79%, and inference latency by 84%.
- CDPS contributes the most (reducing CD by 1.65), and CDCA shows the most significant improvement on high-variance categories (lamp, table).
- On real-world scan datasets, UCD comprehensively outperforms prior methods, though UHD (maximum error) is slightly inferior—this is because the method optimizes overall shape rather than extreme outlier points.

## Highlights & Insights

1. **First exploration of the adaptation of SSM in UDA PCC**, filling the research gap of Mamba in domain adaptive point cloud tasks.
2. **The three-layer alignment framework is exquisitely designed**: CDPS (patch spatial correspondence) $\rightarrow$ CDSA (fine-grained spatial alignment) $\rightarrow$ CDCA (global semantic alignment), progressing step-by-step from local to global.
3. **Shared normalization of the Z-order curve** is a simple but effective means of cross-domain spatial alignment.
4. **Balance of linear complexity and high precision**: While surpassing the performance of Transformer-based solutions (CD decreased by 1.95), the complexity is significantly reduced.

## Limitations & Future Work

1. The performance on UHD metrics (maximum point error) is average, indicating a possible need for specialized handling of boundary points.
2. Only synthetic $\rightarrow$ real and synthetic $\rightarrow$ synthetic transfer were validated; real $\rightarrow$ real scenarios are still missing.
3. The channel mixing strategy (alternating even/odd) is relatively fixed, and adaptive mixing ratios could be explored.
4. The performance on the Cabinet category under evaluation metrics is inferior to DAPoinTr, showing that adaptation to certain geometric shapes still has room for improvement.

## Related Work & Insights

- **DAPoinTr** (SOTA baseline): A pioneer of Transformer architectures in UDA PCC, but its quadratic complexity is a bottleneck.
- **PointMamba**: Validated the effectiveness of SSM in point cloud analysis; DAPointMamba introduces domain adaptive capability on top of it.
- **Z-order Curve**: A classic spatial indexing method, ingeniously applied here for unified serialization in cross-domain spatial alignment.
- Insight: The adaptability of SSM/Mamba in other 3D cross-domain tasks (such as detection, segmentation) is worth exploring further.

## Rating

- Novelty: ⭐⭐⭐⭐ (First introduction of Mamba to UDA PCC, with novel designs for the three modules)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple benchmarks + real-world data + efficiency comparison + visualization)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic, rich figures and tables)
- Value: ⭐⭐⭐⭐ (Opens a new direction for the application of Mamba in domain-adaptive 3D tasks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](../../ICCV2025/3d_vision/dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)

</div>

<!-- RELATED:END -->
