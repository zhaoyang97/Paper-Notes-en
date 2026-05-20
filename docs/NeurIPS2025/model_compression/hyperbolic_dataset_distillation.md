---
title: >-
  [Paper Note] Hyperbolic Dataset Distillation
description: >-
  [NeurIPS 2025][Model Compression][dataset distillation] This paper proposes HDD, the first method to incorporate hyperbolic space into dataset distillation. By matching the Riemannian centroids of real and synthetic data…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "dataset distillation"
  - "hyperbolic space"
  - "distribution matching"
  - "Lorentz model"
  - "hierarchical pruning"
date: 2026-05-08
content_hash: 73072b85c206b54d
---

# Hyperbolic Dataset Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2505.24623](https://arxiv.org/abs/2505.24623)  
**Code**: [https://github.com/Guang000/HDD](https://github.com/Guang000/HDD)  
**Area**: Model Compression
**Keywords**: dataset distillation, hyperbolic space, distribution matching, Lorentz model, hierarchical pruning

## TL;DR
This paper proposes HDD, the first method to incorporate hyperbolic space into dataset distillation. By matching the Riemannian centroids of real and synthetic data in the Lorentz hyperbolic space—rather than performing distribution matching in Euclidean space—HDD leverages the hierarchical weighting property of hyperbolic geometry to assign higher influence to more representative, low-level samples. The method consistently improves over DM/IDM baselines across multiple datasets.

## Background & Motivation

**Background**: Dataset Distillation (DD) aims to compress a large training set into a minimal synthetic set such that models trained on the synthetic set retain the performance of those trained on the original. Existing methods fall into three categories: gradient matching, trajectory matching, and distribution matching. Distribution matching (DM) is computationally more efficient as it avoids bi-level optimization, but tends to yield lower accuracy.

**Limitations of Prior Work**: DM methods (including MSE- and MMD-based variants) measure distributional discrepancy in Euclidean space and treat all samples as i.i.d. points, ignoring the inherent hierarchical or tree-like structure within datasets—samples closer to a "class prototype" should carry more weight than noisy or marginal samples.

**Key Challenge**: Euclidean space assigns uniform weight to all samples and cannot differentiate the varying contributions of samples at different hierarchical levels to the class representation.

**Key Insight**: Hyperbolic space, by virtue of its negative curvature and exponentially growing volume, naturally encodes tree-like hierarchical structures. When computing centroids in hyperbolic space, samples near the origin (low-level, more representative) exert greater influence on the centroid, while peripheral samples (high-level, noisier) contribute less.

**Core Idea**: Map features into the Lorentz hyperbolic space and optimize the synthetic dataset by minimizing the geodesic distance between the hyperbolic centroids of the real and synthetic data.

## Method

### Overall Architecture
Given a real dataset $\mathcal{R}$ and a synthetic dataset $\mathcal{S}$ to be optimized ($|\mathcal{S}| \ll |\mathcal{R}|$), a shallow network $\phi$ extracts features, which are then embedded into the Lorentz hyperbolic space via the exponential map. The Riemannian (Karcher) means of both datasets are computed in hyperbolic space, and the synthetic data is updated by minimizing the geodesic distance between these centroids. HDD functions as a plug-in compatible with most existing DM frameworks.

### Key Designs

1. **Lorentz Hyperbolic Space Embedding**:

    - Function: Maps Euclidean features $v_i$ to hyperbolic space points $z_i$.
    - Mechanism: Applies the exponential map $z_i = \exp_{p_0}(v_i) = \cosh(\sqrt{-K}\|v_i\|)p_0 + \sinh(\sqrt{-K}\|v_i\|)\frac{v_i}{\sqrt{-K}\|v_i\|}$, where $p_0 = (\sqrt{-1/K}, 0, \ldots, 0)$ is the base point and $K < 0$ is the curvature.
    - Design Motivation: The Lorentz model offers better numerical stability and analytical tractability compared to the Poincaré ball model.

2. **Hyperbolic Centroid Matching**:

    - Function: Computes the distributional centers of real and synthetic data in hyperbolic space and measures their discrepancy via geodesic distance.
    - Mechanism: The Riemannian mean is defined as $\bar{z} = \arg\min_{z} \sum_i d_L^2(z, z_i)$; in practice, the closed-form approximation $\mathbf{c} = \sqrt{-K} \cdot \frac{\bar{\mathbf{z}}}{\sqrt{|\langle \bar{\mathbf{z}}, \bar{\mathbf{z}} \rangle_\mathcal{L}| + \epsilon}}$ from Law et al. is used to avoid iterative overhead. The geodesic distance $d_L(m,n) = \frac{1}{\sqrt{-K}} \text{acosh}(-K\langle m,n\rangle_\mathcal{L})$ faithfully reflects distances on the manifold.
    - Design Motivation: The hyperbolic centroid naturally gravitates toward low-level samples near the origin (i.e., class prototypes), automatically realizing hierarchical weighting.

3. **Hierarchical Weight Analysis**:

    - Function: Theoretically explains the differential contribution of samples at different hierarchical levels to the loss.
    - Mechanism: Via a tangent space approximation, each sample's influence is modulated by a scalar weight $w(d) = \frac{\sqrt{|K|}d}{\sinh(\sqrt{|K|}d)}$, which is strictly decreasing in distance $d$—samples closer to the origin receive higher weights.
    - Design Motivation: Provides a theoretical explanation for why hyperbolic matching naturally outperforms Euclidean matching, which treats all samples as i.i.d.

### Loss & Training
$$\mathcal{L}_{\text{Lhd}} = \lambda \cdot d_L(\bar{z}^{\text{real}}, \bar{z}^{\text{syn}}) = \frac{\lambda}{\sqrt{-K}} \text{acosh}(-K\langle \bar{z}^{\text{real}}, \bar{z}^{\text{syn}} \rangle_\mathcal{L})$$
where $\lambda$ is a gradient scaling factor to compensate for the extremely small distances arising from centroids concentrated near the origin in hyperbolic space. The final optimization objective is $\mathcal{S}^* = \arg\min \mathbb{E}_{\phi_Q}[\lambda \cdot d_L(\bar{z}^{\text{real}}, \bar{z}^{\text{syn}})]$.

## Key Experimental Results

### Main Results

| Dataset | IPC | IDM | IDM+HDD | Gain |
|--------|-----|-----|---------|------|
| Fashion-MNIST | 1 | 77.4% | 78.5% | +1.1% |
| SVHN | 1 | 65.3% | 67.8% | +2.5% |
| CIFAR-10 | 1 | 45.2% | 47.0% | +1.8% |
| CIFAR-100 | 1 | 22.1% | 25.3% | +3.2% |
| CIFAR-10 | 10 | 57.3% | 61.3% | +4.0% |
| SVHN | 50 | 85.2% | 87.6% | +2.4% |
| CIFAR-10 | 50 | 67.2% | 69.7% | +2.5% |

DM+HDD is also effective: SVHN IPC=1 improves from 21.9% → 25.0% (+3.1%), and CIFAR-10 IPC=1 from 26.4% → 28.7% (+2.3%).

### Ablation Study — Hyperbolic Pruning

| Pruning Ratio | DM | DM+HDD | IDM | IDM+HDD |
|----------|-----|---------|------|---------|
| 0% (Full) | 48.5% | 50.3% | 57.3% | 61.3% |
| 20% | 47.2% | 49.6% | 55.8% | 60.5% |
| 40% | 46.6% | 48.6% | 54.1% | 58.4% |
| 80% | 44.3% | 45.8% | 47.2% | 50.3% |

Retaining only 20% of the distilled coreset largely preserves model performance while significantly improving training stability.

### Key Findings
- HDD yields particularly pronounced improvements at low IPC (1 or 10), gaining +2–4%, indicating that hierarchical information is especially critical when data is scarce.
- Gains are consistent in cross-architecture evaluations (ConvNet→AlexNet/VGG/ResNet).
- Hyperbolic pruning reveals that the hierarchical structure of datasets is exploitable—low-level samples contain the majority of useful information.

## Highlights & Insights
- **First introduction of hyperbolic geometry into dataset distillation**, opening a new perspective—leveraging the natural hierarchical bias of non-Euclidean geometry to optimize information compression.
- **Plug-and-play design**: HDD can be added as a loss term on top of any existing DM framework without architectural modifications.
- **Rigorous theoretical analysis**: The weight function $w(d)$ is derived via tangent space approximation, and the hierarchical influence mechanism is further validated from a gradient perspective.
- **Hierarchical pruning finding**: The observation that 20% of the coreset suffices to retain performance suggests the possibility of exploiting hierarchical structure for even more aggressive compression in a single step.

## Limitations & Future Work
- Validation is limited to classification tasks; effectiveness on dense prediction tasks such as detection and segmentation remains unknown.
- Cross-architecture generalization experiments are restricted to ConvNet/AlexNet/VGG/ResNet; performance on modern architectures such as ViT has not been evaluated.
- The curvature $K$ and scaling factor $\lambda$ require manual tuning for different datasets, with no adaptive strategy proposed.
- The approximate computation of the hyperbolic centroid may suffer from accuracy degradation in high-dimensional or large-scale settings.
- Experiments are conducted only on small-scale datasets (CIFAR-10/100, TinyImageNet); effectiveness on large-scale benchmarks such as ImageNet-1K remains unclear.

## Related Work & Insights
- **vs. DM/IDM**: DM/IDM match distributional means via MMD in Euclidean space; HDD instead matches centroids in hyperbolic space, introducing hierarchical weighting.
- **vs. Generative Distillation (D4M, SRe2L)**: Generative methods synthesize data via diffusion models at substantial computational cost; HDD improves the matching-based paradigm, and the two approaches are orthogonal.
- **vs. Hyperbolic Machine Learning**: Prior hyperbolic methods are predominantly applied to graph networks and metric learning; this work is the first to apply them to dataset distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce hyperbolic space into dataset distillation—novel direction, though the core operations (exponential map + centroid matching) are well-established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, cross-architecture transfer, ablation, and pruning analysis; lacks large-scale validation.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ Practical plug-and-play solution with good utility and meaningful contributions to the dataset distillation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)
- [\[NeurIPS 2025\] Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)
- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](../../AAAI2026/model_compression/tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)

</div>

<!-- RELATED:END -->
