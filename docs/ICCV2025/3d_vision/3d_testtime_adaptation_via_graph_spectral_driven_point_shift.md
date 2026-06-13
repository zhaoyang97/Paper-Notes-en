---
title: >-
  [Paper Note] 3D Test-time Adaptation via Graph Spectral Driven Point Shift
description: >-
  [ICCV 2025][3D Vision][Test-time adaptation] This paper proposes GSDTTA, which is the first work to shift 3D point cloud test-time adaptation (TTA) from the spatial domain to the graph spectral domain. By optimizing only…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Test-time adaptation"
  - "graph spectral analysis"
  - "point cloud classification"
  - "graph Fourier transform"
  - "feature map guided self-training"
date: 2026-05-08
content_hash: 1fb5bafd0a7795fd
---

# 3D Test-time Adaptation via Graph Spectral Driven Point Shift

**Conference**: ICCV 2025
**arXiv**: [2507.18225](https://arxiv.org/abs/2507.18225)  
**Code**: Unavailable (not provided in the paper)  
**Area**: 3D Vision / Domain Adaptation
**Keywords**: Test-time adaptation, graph spectral analysis, point cloud classification, graph Fourier transform, feature map guided self-training
**Authors**: Xin Wei, Qin Yang, Yijie Fang, Mingrui Zhu, Nannan Wang (Xidian University)

## TL;DR

This paper proposes GSDTTA, which is the first work to shift 3D point cloud test-time adaptation (TTA) from the spatial domain to the graph spectral domain. By optimizing only the lowest 10% frequency components (reducing parameters by ~90%), GSDTTA achieves global structural adjustment. Combined with a feature map guided self-training strategy for pseudo-label generation, it significantly outperforms existing 3D TTA methods on ModelNet40-C and ScanObjectNN-C.

## Background & Motivation

**Core Challenge**: Deep point cloud classification models (e.g., DGCNN) trained on clean data can suffer performance drops exceeding 35% when confronted with real-world corruption (sensor errors, scene complexity, etc.).

**Limitations of Prior Work**:
1. **2D TTA methods cannot be directly transferred**: The irregular, unordered structure of point clouds prevents direct application of 2D TTA methods (TENT, SHOT, etc.) to 3D settings.
2. **High computational cost**: CloudFixer and 3DD-TTA rely on diffusion models to repair point clouds in the spatial domain, requiring optimization of high-dimensional transformations ($\Delta P \in \mathbb{R}^{N \times 3}$, with $N$ typically $> 1024$).
3. **Dependency on source domain data**: MATE, CloudFixer, and 3DD-TTA all require access to source domain training data or additional auxiliary task training, which violates the strict TTA setting.
4. **Error accumulation**: BFTT3D avoids backpropagation but offers limited accuracy.

**Two Key Observations in This Work**:
1. **Spectral energy compactness**: Approximately 95% of the energy in the graph spectrum of a point cloud is concentrated in low-frequency components; the global shape can be reconstructed using only the lowest 10% frequency coefficients, reducing the number of optimization parameters by ~90%.
2. **Domain invariance**: Graph Laplacian eigenmaps are isometry-invariant shape descriptors that are inherently domain-agnostic, naturally compensating for the source-domain bias in deep features.

## Core Problem

How to efficiently adapt 3D point cloud models to distribution shift at test time with fewer parameters and without relying on source domain data?

## Method

### Overall Architecture

GSDTTA consists of two core modules that are alternately optimized:
1. **GSDPS (Graph Spectral Driven Point Shift)**: Adjusts the input point cloud in the graph spectral domain.
2. **GSGMA (Graph Spectral Guided Model Adaptation)**: Updates model parameters guided by graph spectral information.

For each test batch, 4 steps of input adaptation (GSDPS) are performed, followed by 1 step of model adaptation (GSGMA), repeated for 10 cycles in total.

### Key Designs

**1. Outlier-Aware Graph Construction**

Given a point cloud $X \in \mathbb{R}^{N \times 3}$, an RBF-weighted kNN graph is constructed:
- Edge weight: $w_{ij} = \exp(-d^2(x_i, x_j) / 2\delta^2)$
- Adjacency matrix: $A_{ij} = w_{ij} \cdot \mathbf{I}(x_j \in \mathcal{N}(x_i))$
- Outlier removal: low-degree nodes are removed via a degree threshold $\tau = \gamma / (Nk) \cdot \sum A_{ij}$ (outliers, being far from inliers, have very low degree)
- Final adjacency matrix incorporates constraint $\mathbf{I}(\sum_j A_{ij} > \tau)$, effectively handling background noise

**2. Graph Spectral Driven Point Shift (Core Contribution)**

- Compute graph Laplacian on the outlier-aware graph: $L_o = D_o - A_o$
- Eigen-decomposition: $L_o = U_o \Lambda_o U_o^T$
- Transform to spectral domain via GFT: $\hat{X} = U_o^T X$
- Learn low-frequency spectral adjustment: $\hat{X}_a = \hat{X} + [\Delta\hat{X},\ \mathbf{0}]$, where $\Delta\hat{X} \in \mathbb{R}^{M \times 3}$, $M=100 \ll N$ (only the lowest 10% frequencies are adjusted)
- Transform back to spatial domain via IGFT: $X_s = U_o \hat{X}_a$

**Advantage**: The optimization variable has only $M \times 3 = 300$ dimensions, far smaller than the $N \times 3 > 3072$ dimensions in spatial-domain methods.

**3. Feature Map Guided Self-Training Strategy**

- **Global deep descriptor** $f_d$: extracted by the pre-trained model $f_\theta$
- **Global spectral descriptor** $f_s$: obtained by element-wise max-pooling over the eigenmap (similar to Global Point Signature)
- Soft class centroids $q_d^c$ and $q_s^c$ are computed in both spaces (weighted by model prediction probabilities)
- Pseudo-label generation: a convex combination of deep and spectral similarities:
  $$\hat{y}_i = \arg\min_c \left( \alpha \cdot \cos(f_d^i, q_d^c) + (1-\alpha) \cdot \cos(f_s^i, q_s^c) \right)$$
- Key insight: spectral descriptors are domain-agnostic and provide robust complementary supervision in the early stages of adaptation when the model has not yet been sufficiently adjusted.

### Loss & Training

**Input Adaptation Loss** (optimizing $\Delta\hat{X}$):
$$\mathcal{L}_{IA} = \mathcal{L}_{pl} + \beta_1(\mathcal{L}_{ent} + \mathcal{L}_{div}) + \beta_2 \mathcal{L}_{cd}$$
- $\mathcal{L}_{pl}$: pseudo-label cross-entropy loss
- $\mathcal{L}_{ent}$: entropy loss, encouraging more confident predictions
- $\mathcal{L}_{div}$: diversity loss, preventing prediction collapse to a single class
- $\mathcal{L}_{cd}$: one-sided Chamfer distance (from original to adapted point cloud), constraining the magnitude of point shifts

**Model Adaptation Loss** (optimizing $\theta$):
$$\mathcal{L}_{MA} = \mathcal{L}_{pl} + \beta_3(\mathcal{L}_{ent} + \mathcal{L}_{div})$$

**Hyperparameter Settings**: $k=10$, $\delta=0.1$, $\gamma=0.6$, $M=100$, $\alpha=0.5$, $\beta_1=0.3$, $\beta_2=1000$, $\beta_3=3$, learning rate $0.0001$ (AdamW), batch size $32$.

## Key Experimental Results

### ModelNet40-C (Mean Accuracy %)

| Backbone | Source-only | TENT | SHOT | CloudFixer | 3DD-TTA | **GSDTTA** | vs CloudFixer |
|----------|------------|------|------|------------|---------|-----------|--------------|
| DGCNN | 66.51 | 75.91 | 77.36 | 76.54 | 71.69 | **79.07** | +2.53 |
| CurveNet | 71.38 | 77.88 | 81.24 | 77.91 | - | **82.63** | +4.72 |
| PointNeXt | 66.99 | 81.08 | 80.46 | 76.04 | - | **82.51** | +6.47 |

### ScanObjectNN-C (Mean Accuracy %)

| Backbone | Source-only | TENT | SHOT | CloudFixer | 3DD-TTA | **GSDTTA** | vs CloudFixer |
|----------|------------|------|------|------------|---------|-----------|--------------|
| DGCNN | 55.72 | 57.31 | 56.32 | 60.73 | 57.08 | **61.83** | +1.10 |
| CurveNet | 53.06 | 58.71 | 58.17 | 58.96 | - | **62.59** | +3.63 |
| PointNeXt | 50.66 | 58.05 | 57.06 | 59.01 | - | **60.84** | +1.83 |

### Key Findings

- Strong performance on the Background corruption: 88.57% with DGCNN vs. 74.55% for CloudFixer (+14%), owing to the outlier-aware graph effectively removing background noise.
- On semantic corruptions such as Shear and Cutout, GSDTTA outperforms CloudFixer by an average of 4.68% and 3.60%, respectively.

### Ablation Study

On ScanObjectNN-C (DGCNN):

| Configuration | GSDPS | GSGMA | Eigenmap | Mean Acc |
|------|-------|-------|----------|----------|
| Source-only | ✗ | ✗ | - | 55.72 |
| w/o GSDPS | ✗ | ✓ | ✓ | 57.28 |
| w/o GSGMA | ✓ | ✗ | ✓ | 58.35 |
| w/o Eigenmap | ✓ | ✓ | ✗ | 61.20 |
| **Full GSDTTA** | ✓ | ✓ | ✓ | **61.83** |

- GSDPS contributes +4.55%; GSGMA contributes +3.48%.
- Eigenmap guidance vs. deep-feature-only guidance: +0.63%.
- Removing the outlier-aware graph ($\gamma=0$): Background corruption accuracy drops sharply from 69.54% to 18.24%.
- Hyperparameter sensitivity: performance is stable (std $< 1\%$) across $\beta_1 \in [0,1]$, $\beta_2 \in [0,3000]$, and $\beta_3 \in [0,5]$.

## Highlights & Insights

1. **Novel perspective**: This is the first work to shift 3D TTA from the spatial domain to the graph spectral domain, providing a new paradigm for point cloud adaptation grounded in graph signal processing theory.
2. **Parameter efficiency**: Only 300 parameters ($100 \times 3$) are optimized, reducing the number of optimization variables by ~90% compared to spatial-domain methods.
3. **Theory-driven design**: The method is motivated by two clear theoretical properties: spectral energy compactness and the domain invariance of eigenmaps.
4. **Outlier-aware design**: Outlier nodes are naturally excluded via a degree threshold, proving highly effective against Background corruption (+14%).
5. **Strong generalizability**: Consistent improvements are achieved across three different backbones (DGCNN, CurveNet, PointNeXt).

## Limitations & Future Work

1. **Scalability**: The global spectral operation (eigen-decomposition) has $O(N^3)$ complexity, limiting its applicability to large-scale point clouds. The authors suggest that unsupervised segmentation combined with multi-scale local spectral analysis could mitigate this.
2. **Weaker handling of basic noise**: On Uniform, Gaussian, and Impulse corruptions, CloudFixer (leveraging the denoising capability of diffusion models) outperforms GSDTTA by an average of 8.35%, as spectral-domain methods are less effective than dedicated generative models for point-level denoising.
3. **Extreme corruptions (Dropout/Cutout)**: Performance on Cutout_15 and Cutout_30 remains poor (31–46%), posing a challenge for all methods.
4. **No support for dynamic/streaming scenarios**: The current approach requires batch-wise processing; single-sample adaptation has not been explored.
5. **Limited to classification**: The method has not been extended to downstream tasks such as 3D segmentation or detection.

## Related Work & Insights

| Method | Domain | Requires Source Data | Adaptation Target | Core Mechanism |
|------|------|------------|---------|---------|
| TENT | Model params | ✗ | BN layers | Entropy minimization |
| SHOT | Model params | ✗ | Feature extractor | Information maximization + pseudo-labels |
| MATE | Model params | ✓ (auxiliary task training) | Encoder | Masked autoencoder reconstruction |
| BFTT3D | Model params | ✓ (feature prototypes) | Non-parametric network | No backpropagation |
| CloudFixer | Input data | ✓ (pre-trained diffusion model) | Point cloud | Diffusion model denoising |
| 3DD-TTA | Input data | ✓ (pre-trained diffusion model) | Point cloud | Diffusion model |
| **GSDTTA** | **Input + Model** | **✗** | **Spectral coefficients + model params** | **Spectral domain adjustment + eigenmap self-training** |

The unique advantages of GSDTTA are: (1) simultaneous adaptation of both input and model; (2) no access to source domain data required; (3) highest parameter efficiency among compared methods.

**Further Insights**:
1. **Generalizability of frequency-domain thinking**: The principle that low frequencies in the graph spectrum correspond to global structure may transfer to other 3D tasks (segmentation, detection, registration), where low-frequency adjustment could serve as a general domain-agnostic strategy.
2. **Complementary multi-modal features**: The convex combination of deep features (source-domain biased but highly discriminative) and spectral features (domain-agnostic but weakly discriminative) parallels multi-view/multi-modal fusion and may generalize to other TTA scenarios.
3. **Connection to frequency-domain prompting**: Analogous to frequency-domain prompt tuning in 2D, GSDTTA's spectral adjustment is essentially a 3D "spectral prompt."
4. **Potential combination with diffusion models**: CloudFixer excels at point-level denoising while GSDTTA excels on structural corruptions; combining the two (global spectral adjustment followed by diffusion-based denoising) could yield even stronger results.

## Rating

| Dimension | Score | Comments |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First to introduce graph spectral domain into 3D TTA; highly original perspective |
| Technical Depth | ⭐⭐⭐⭐ | Clear theoretical motivation; elegant method design |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Three backbones, two datasets, comprehensive ablations |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured; motivation clearly articulated |
| Practical Value | ⭐⭐⭐ | Limited by computational cost of eigen-decomposition |
| **Overall** | **⭐⭐⭐⭐** | High-quality work that opens a new direction for spectral-domain TTA |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](../../NeurIPS2025/3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene](event-driven_storytelling_with_multiple_lifelike_humans_in_a_3d_scene.md)
- [\[AAAI 2026\] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](../../AAAI2026/3d_vision/adapt-as-you-walk_through_the_clouds_training-free_online_te.md)

</div>

<!-- RELATED:END -->
