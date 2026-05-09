---
title: >-
  [Paper Note] 3D Test-time Adaptation via Graph Spectral Driven Point Shift
description: >-
  [ICCV 2025][3D Vision][Test-time adaptation] This paper proposes GSDTTA, which shifts 3D point cloud test-time adaptation from the spatial domain to the graph spectral domain. By optimizing only the lowest 10% frequency components to adapt global structure, combined with an eigenvector-map-guided self-training strategy, GSDTTA achieves state-of-the-art performance on ModelNet40-C and ScanObjectNN-C.
tags:
  - ICCV 2025
  - 3D Vision
  - Test-time adaptation
  - point cloud classification
  - graph spectral domain
  - graph Fourier transform
  - domain shift
date: 2026-05-08
content_hash: eca884461eb9e8e3
---

# 3D Test-time Adaptation via Graph Spectral Driven Point Shift

**Conference**: ICCV 2025
**arXiv**: [2507.18225](https://arxiv.org/abs/2507.18225)
**Code**: None
**Area**: 3D Vision
**Keywords**: Test-time adaptation, point cloud classification, graph spectral domain, graph Fourier transform, domain shift

## TL;DR
This paper proposes GSDTTA, which shifts 3D point cloud test-time adaptation from the spatial domain to the graph spectral domain. By optimizing only the lowest 10% frequency components to adapt global structure, combined with an eigenvector-map-guided self-training strategy, GSDTTA achieves state-of-the-art performance on ModelNet40-C and ScanObjectNN-C.

## Background & Motivation
Point cloud classification models (e.g., DGCNN) trained on clean data can suffer accuracy drops exceeding 35% when encountering real-world noise such as background interference, occlusion, and LiDAR noise. Test-time adaptation (TTA) methods can dynamically adapt models during inference, yet existing 3D TTA approaches exhibit the following limitations:

**High-dimensional spatial optimization**: CloudFixer and 3DD-TTA require learning per-point offsets $\Delta P \in \mathbb{R}^{N \times 3}$ (where $N$ typically exceeds 1024), resulting in a large optimization space and slow convergence.

**Dependence on auxiliary training data**: MATE requires source-domain auxiliary tasks, and BFTT3D requires extracting source-domain prototypes, both of which violate strict TTA settings.

**Reliance on diffusion models**: CloudFixer and 3DD-TTA leverage pre-trained diffusion models for point cloud restoration, incurring additional computational overhead.

**Core Insight**: The graph spectral domain exhibits two key properties—(1) energy is highly concentrated in low-frequency components (approximately 95% of energy resides in low frequencies), such that optimizing only the lowest 10% of frequencies suffices to control global shape, reducing parameter count by approximately 90%; (2) Laplacian eigenmaps are domain-agnostic descriptors unaffected by source-domain bias, making them suitable for guiding pseudo-label generation in early adaptation stages.

**Core Idea**: Perform learnable low-frequency adjustments in the graph spectral domain, generating point shifts via inverse graph Fourier transform (Graph Spectral Driven Point Shift), while employing an eigenmap-guided self-training strategy that alternately optimizes input and model parameters.

## Method

### Overall Architecture
GSDTTA consists of two core modules: **Graph Spectral Driven Point Shift (GSDPS)** and **Graph Spectral Guided Model Adaptation (GSGMA)**. These two modules operate in alternating iterations: for each batch of test data, four steps of input adaptation are performed followed by one step of model adaptation, cycling for a total of ten rounds.

### Key Designs

1. **Anomaly-Aware Graph Construction**: A kNN graph is constructed from the input point cloud, with edge weights defined by the radial basis function $w_{ij} = \exp(-d^2(x_i,x_j) / 2\delta^2)$. Outlier points are filtered using a degree threshold $\tau = \frac{\gamma}{Nk}\sum A_{ij}$ (points whose degree falls below the threshold are removed), improving the robustness of subsequent spectral analysis.

2. **Low-Frequency Adjustment in the Graph Spectral Domain**: The Laplacian matrix $L_o = D_o - A_o$ of the anomaly-aware graph is computed and eigen-decomposed to obtain eigenvector matrix $U_o$. The point cloud is transformed to the spectral domain via GFT as $\hat{X} = U_o^T X$; learnable adjustments $\Delta\hat{X} \in \mathbb{R}^{M \times 3}$ are then added exclusively to the first $M=100$ low-frequency components, leaving high-frequency components unchanged. The adjusted representation is mapped back to the spatial domain via IGFT as $X_s = U_o \hat{X}_a$.

    - **Design Motivation**: Approximately 95% of energy is concentrated in low-frequency components (verified in Figure 2). Optimizing only $M=100$ parameters (rather than $N=1024$ points $\times$ 3 dimensions $= 3072$ parameters) substantially reduces optimization difficulty.

3. **Eigenmap-Guided Self-Training Strategy**: Pseudo-labels are generated via a convex combination of a deep feature descriptor $f_d$ (global features extracted by the model) and a spectral descriptor $f_s$ (eigenmap max-pooling):
$$\hat{y}_i = \arg\min_c \left(\alpha \frac{f_d^T q_d^c}{\|f_d\|\|q_d^c\|} + (1-\alpha) \frac{f_s^T q_s^c}{\|f_s\|\|q_s^c\|}\right)$$
where $q_d^c$ and $q_s^c$ denote class centroids.

    - **Design Motivation**: Spectral descriptors are domain-agnostic and are particularly valuable in early adaptation stages when the model has not yet been adjusted, compensating for source-domain bias in deep features.

### Loss & Training
- **Input Adaptation Objective**: $\mathcal{L}_{IA} = \mathcal{L}_{pl} + \beta_1(\mathcal{L}_{ent} + \mathcal{L}_{div}) + \beta_2\mathcal{L}_{cd}$
    - $\mathcal{L}_{pl}$: Pseudo-label cross-entropy loss
    - $\mathcal{L}_{ent} + \mathcal{L}_{div}$: Information maximization loss (individual confidence + collective diversity)
    - $\mathcal{L}_{cd}$: Chamfer Distance, ensuring the adapted point cloud does not deviate excessively from the original
- **Model Adaptation Objective**: $\mathcal{L}_{MA} = \mathcal{L}_{pl} + \beta_3(\mathcal{L}_{ent} + \mathcal{L}_{div})$
- Optimizer: AdamW, learning rate 0.0001, batch size 32

## Key Experimental Results

### Main Results

| Backbone | Method | Background | Occlusion | LiDAR | Mean |
|----------|--------|-----------|-----------|-------|------|
| DGCNN | Source-only | 49.71 | 33.26 | 14.91 | 66.51 |
| DGCNN | TENT | 60.65 | 42.94 | 33.38 | 75.91 |
| DGCNN | CloudFixer | 74.55 | 35.94 | 37.48 | 76.54 |
| DGCNN | **GSDTTA** | **88.57** | **45.38** | 31.52 | **79.07** |
| CurveNet | SHOT | 66.49 | 58.63 | 56.04 | 81.24 |
| CurveNet | CloudFixer | 66.07 | 37.13 | 38.76 | 77.91 |
| CurveNet | **GSDTTA** | **87.84** | **50.73** | **44.45** | **82.63** |
| PointNeXt | TENT | 80.43 | 51.90 | 46.92 | 81.08 |
| PointNeXt | CloudFixer | 79.28 | 38.32 | 35.73 | 76.04 |
| PointNeXt | **GSDTTA** | **91.29** | **55.06** | **46.84** | **82.51** |

On ModelNet40-C with the DGCNN backbone, GSDTTA achieves a mean accuracy of 79.07%, surpassing CloudFixer by 2.53% and 3DD-TTA by 7.38%. Gains are particularly pronounced on the Background corruption (88.57% vs. CloudFixer's 74.55%).

### Ablation Study

| Component Configuration | Mean Acc (DGCNN) |
|------------------------|-----------------|
| Spatial-domain adaptation only (Baseline) | ~75 |
| Spectral-domain adaptation (w/o eigenmap guidance) | ~77 |
| Spectral-domain adaptation + deep feature pseudo-labels | ~78 |
| Spectral-domain adaptation + eigenmap-guided pseudo-labels (Full) | **79.07** |

On ScanObjectNN-C, GSDTTA consistently outperforms all competing methods, achieving a mean accuracy of 61.83% under the DGCNN backbone (vs. CloudFixer's 60.73%).

### Key Findings
- Low-frequency spectral adjustment requires only approximately 100 parameters—about 3% of the 3,072 parameters required by spatial-domain methods—making optimization significantly easier given limited test data.
- Eigenmap descriptors contribute most on the Background corruption (+14%), as this corruption alters global structure while preserving local geometry.
- Alternating iterative optimization (input + model) outperforms optimizing either component independently.

## Highlights & Insights
- **Addressing TTA from a graph signal processing perspective** represents the first attempt to introduce graph spectral analysis into 3D TTA, offering a genuinely novel viewpoint.
- The parameter efficiency of low-frequency adjustment (90% parameter reduction) makes it well-suited for online/streaming TTA scenarios.
- The use of eigenmaps as domain-agnostic descriptors introduces a new paradigm for pseudo-label generation in the TTA literature.
- No additional training data or pre-trained diffusion models are required, strictly adhering to the TTA setting.

## Limitations & Future Work
- Graph Laplacian eigen-decomposition has a computational complexity of $O(N^3)$, which becomes costly for large-scale point clouds.
- The outlier filtering threshold $\gamma$ and the kNN parameter $k$ require manual tuning.
- Validation is currently limited to classification tasks; extension to dense prediction tasks such as segmentation and detection has not yet been explored.
- Gains on LiDAR and Occlusion corruptions are relatively smaller than on other corruption types, suggesting that spectral-domain adjustment has limited effectiveness in extremely sparse or heavily incomplete scenarios.

## Related Work & Insights
- **CloudFixer / 3DD-TTA**: Spatial-domain point cloud restoration methods relying on diffusion models; the proposed method achieves superior results without requiring diffusion models.
- **Graph Spectral Analysis**: Draws upon graph signal processing ideas such as Global Point Signature, elevating spectral analysis from a descriptor tool to an adaptation mechanism.
- **Information Maximization**: The $\mathcal{L}_{ent} + \mathcal{L}_{div}$ combination is adopted from TTA methods such as LAME.
- **Insights**: The spectral-domain operation paradigm introduced here can potentially generalize to other 3D tasks (e.g., point cloud denoising and completion) and may also be explored for 2D image TTA.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to introduce graph spectral domain analysis into 3D TTA; highly original perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three backbones × two datasets with comprehensive coverage; efficiency comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐ A parameter-efficient TTA solution with practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](../../NeurIPS2025/3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](constraint-aware_feature_learning_for_parametric_point_cloud.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene](event-driven_storytelling_with_multiple_lifelike_humans_in_a_3d_scene.md)

</div>

<!-- RELATED:END -->
