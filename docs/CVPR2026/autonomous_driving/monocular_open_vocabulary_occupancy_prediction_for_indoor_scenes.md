---
title: >-
  [Paper Note] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)
description: >-
  [CVPR 2026][Autonomous Driving][Open-vocabulary occupancy prediction] This paper proposes LegoOcc, which leverages Language-Embedded Gaussians (LE-Gaussians) as a unified geometric-semantic intermediate representation. Combined with a Poisson-process-based Gaussian-to-Occupancy (G2O) operator and a progressive temperature decay strategy, LegoOcc achieves monocular open-vocabulary occupancy prediction for indoor scenes using only binary occupancy labels (without semantic annotations), attaining 59.50 IoU / 21.05 mIoU on Occ-ScanNet.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Open-vocabulary occupancy prediction
  - 3D Gaussian representation
  - Poisson aggregation
  - temperature decay
  - indoor scenes
date: 2026-05-08
content_hash: 81feea3c16137f35
---

# Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)

**Conference**: CVPR 2026
**arXiv**: [2602.22667](https://arxiv.org/abs/2602.22667)
**Code**: [https://github.com/JuIvyy/LegoOcc](https://github.com/JuIvyy/LegoOcc)
**Area**: Autonomous Driving / Indoor Scene Understanding
**Keywords**: Open-vocabulary occupancy prediction, 3D Gaussian representation, Poisson aggregation, temperature decay, indoor scenes

## TL;DR

This paper proposes LegoOcc, which leverages Language-Embedded Gaussians (LE-Gaussians) as a unified geometric-semantic intermediate representation. Combined with a Poisson-process-based Gaussian-to-Occupancy (G2O) operator and a progressive temperature decay strategy, LegoOcc achieves monocular open-vocabulary occupancy prediction for indoor scenes using only binary occupancy labels (without semantic annotations), attaining 59.50 IoU / 21.05 mIoU on Occ-ScanNet.

## Background & Motivation

3D semantic occupancy prediction for indoor scenes is critical for embodied agents, yet faces three major challenges:

**Indoor vs. Outdoor Gap**: Indoor scenes exhibit denser geometry, more complex layouts, finer-grained semantic categories, and severe long-tail distributions. Existing outdoor open-vocabulary occupancy methods (e.g., POP-3D, LOcc) transfer poorly to indoor settings, achieving only 5.96/9.25 mIoU.

**Closed-Set Vocabulary Limitation**: Existing indoor occupancy methods (ISO, EmbodiedOcc, etc.) rely on fixed-category annotation during training and cannot recognize objects outside the training vocabulary, making them unsuitable for real-world deployment.

**High Cost of Semantic Annotation**: The large number of categories and long-tail distributions in indoor scenes make dense semantic annotation prohibitively expensive. In contrast, binary occupancy labels can be obtained automatically via depth reconstruction at much lower cost.

This paper therefore adopts a **geometry-only supervision** paradigm—using only binary occupancy labels without semantic annotations—and investigates how open-vocabulary occupancy prediction can be achieved under this weak supervision setting.

## Method

### Overall Architecture

LegoOcc takes monocular RGB images as input. A feed-forward Gaussian predictor generates a set of Language-Embedded Gaussians (LE-Gaussians), where each Gaussian is parameterized as:

$$\mathcal{G}_i = (\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i, \mathbf{f}_i)$$

Here $\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, \alpha_i$ encode geometric information, and $\mathbf{f}_i \in \mathbb{R}^d$ is a language-aligned semantic embedding. The same set of Gaussians is used simultaneously for:
- **Geometry learning**: predicting 3D occupancy via the Poisson-based G2O operator, supervised by binary labels.
- **Semantic learning**: rendering Gaussian features onto the image plane and aligning them with features from an open-vocabulary segmentation model (Trident).

At inference, cosine similarity between the embeddings of occupied voxels and text prompts enables semantic querying for arbitrary categories.

### Key Designs

1. **Poisson-based Gaussian-to-Occupancy (G2O) Operator**: Addresses instability in voxel aggregation under weak supervision.

   Existing G2O methods have notable shortcomings:
   - **GaussianFormer2**: Aggregation ignores opacity $\alpha_i$ and relies solely on the spatial kernel $p_i(\mathbf{x})$, introducing inconsistency between geometric aggregation and rendering.
   - **Bernoulli method**: Introduces $\tilde{\alpha}_i = \alpha_i p_i(\mathbf{x})$ with a complementary probability union rule, but the union saturates rapidly to 1 when multiple Gaussians overlap, forcing opacity values to become very small and degrading feature rendering quality.

   This paper models the local contribution of each Gaussian as an event intensity in a non-homogeneous Poisson process:

   $h_i(\mathbf{x}) \triangleq \alpha_i p_i(\mathbf{x}), \quad z(\mathbf{x}) = \sum_{i=1}^N h_i(\mathbf{x})$

   The occupancy probability is defined as the probability of at least one event occurring:

   $p(\mathbf{x}) = 1 - \exp\left(-\sum_{i=1}^N \alpha_i p_i(\mathbf{x})\right)$

   Compared to the Bernoulli product form $1 - \prod(1-\alpha_i p_i)$, the Poisson exponential-sum formulation does not saturate under multi-Gaussian overlap, allowing opacity to retain discriminative values and thereby stabilizing both geometric aggregation and semantic rendering.

2. **Progressive Temperature Decay**: Addresses the feature blending problem in rendering.

   Standard $\alpha$-blending renders pixel features as a weighted mixture of multiple Gaussian embeddings along a ray, causing the resulting features to be blends rather than language-aligned representations of individual Gaussians. This paper introduces a tempered sigmoid:

   $\alpha_i = \sigma\left(\frac{\alpha_i^{\text{logit}}}{\tau}\right)$

   with an exponential decay schedule:

   $\tau(r) = \max\{T_{\min}, T_{\max} \cdot (T_{\min}/T_{\max})^r\}$

   where $r \in [0,1]$ denotes training progress, with defaults $T_{\max}=1$ and $T_{\min}=10^{-3}$. At early training stages, a high temperature ensures smooth optimization; at later stages, a low temperature drives opacity toward binary $\{0,1\}$ values, reducing feature blending. Compared to hard Top-$k$ selection (e.g., Dr. Splat), this approach remains end-to-end differentiable; compared to linear decay, the exponential schedule allocates more iterations in the low-temperature regime, yielding better performance.

3. **Multi-View Feature Consistency**: Neighboring frames (5 by default) are re-rendered and subjected to the same feature alignment loss, enhancing cross-view semantic consistency without requiring additional 2D annotations.

### Loss & Training

$$L_{\text{total}} = \lambda_{\text{focal}} L_{\text{focal}} + \lambda_{\text{lov}} L_{\text{lov}} + \lambda_{\text{scal}} L_{\text{scal}} + \lambda_{\text{feat}} L_{\text{feat}} + \lambda_{\text{depth}} L_{\text{depth}}$$

- $L_{\text{focal}}$: Focal Loss for binary occupancy supervision.
- $L_{\text{lov}}$: Lovász-Softmax loss for IoU optimization.
- $L_{\text{scal}}$: Scene-class affinity regularization for spatial consistency.
- $L_{\text{feat}}$: Cosine alignment loss between rendered features and open-vocabulary segmentation features (Trident).
- $L_{\text{depth}}$: Huber depth loss for stable geometry learning.

Training configuration: Depth-Anything V2 as the depth backbone, AdamW optimizer, lr $2 \times 10^{-4}$ with cosine decay, 4× RTX 4090, 10 epochs.

## Key Experimental Results

### Main Results

| Method | Setting | IoU | mIoU | FPS |
|--------|---------|-----|------|-----|
| ISO | Closed-set (full annotation) | 42.16 | 28.71 | 3.81 |
| EmbodiedOcc | Closed-set (full annotation) | 53.55 | 45.15 | 11.48 |
| RoboOcc | Closed-set (full annotation) | 56.48 | 47.76 | - |
| POP-3D† | Open-vocabulary | 35.32 | 5.96 | 10.21 |
| LOcc† | Open-vocabulary | 36.70 | 9.25 | 8.93 |
| **LegoOcc (Ours)** | **Open-vocabulary** | **59.50** | **21.05** | **22.47** |

Under the open-vocabulary setting, LegoOcc surpasses all methods in IoU—including closed-set approaches—achieves more than double the mIoU of the previous best open-vocabulary method (+11.80), and runs at the fastest inference speed.

### Ablation Study

| G2O Operator | Setting | IoU | mIoU | Note |
|--------------|---------|-----|------|------|
| GaussianFormer2 | Open-vocabulary | 0.00 | 0.00 | Complete collapse due to opacity inconsistency |
| Bernoulli | Open-vocabulary | 46.65 | 17.25 | Functional but opacity suppressed |
| **Poisson** | **Open-vocabulary** | **59.50** | **21.05** | Best; stable aggregation |

| Temperature Schedule | $T_{\min}$ | $T_{\max}$ | IoU | mIoU | Note |
|----------------------|------------|------------|-----|------|------|
| No schedule ($\tau=1$) | 1.0 | 1.0 | 59.19 | 18.15 | Good geometry, poor semantics |
| Constant low temp ($\tau=10^{-3}$) | 1e-3 | 1e-3 | 0.00 | 0.00 | Optimization collapse |
| Linear decay | 1e-3 | 1.0 | 7.60 | 2.30 | Insufficient low-temperature iterations |
| **Exponential decay** | **1e-3** | **1.0** | **59.50** | **21.05** | Optimal configuration |

### Key Findings

- The choice of G2O operator is critical for open-vocabulary performance: GaussianFormer2 (without opacity) collapses entirely to 0 under the open-vocabulary setting.
- Temperature scheduling is central to semantic learning: without scheduling, mIoU is only 18.15; with exponential decay, it improves to 21.05.
- LegoOcc's open-vocabulary IoU (59.50) surpasses all closed-set fully-supervised methods.
- A gap of approximately 26 mIoU remains between open-vocabulary and closed-set methods, primarily attributable to textual ambiguity in fine-grained indoor categories.

## Highlights & Insights

1. **Poisson process modeling of occupancy is elegant**: Treating Gaussian contributions as event intensities and voxel occupancy as "at least one event" yields clear physical intuition, a concise mathematical formulation, and natural compatibility with opacity.
2. **Temperature scheduling bridges the rendering–aggregation gap**: Progressively sharpening opacity transitions features from "blended mixtures" to "single-voxel representations," serving as a differentiable analog of hard assignment.
3. **Weak supervision surpasses strong supervision in IoU**: The open-vocabulary model outperforms closed-set fully-annotated methods in geometric accuracy, demonstrating the strong representational capacity of language-embedded Gaussians as an intermediate representation.
4. **Fastest inference speed** (22.47 FPS), approximately 6× faster than ISO (3.81), achieving both high performance and efficiency.

## Limitations & Future Work

1. **mIoU remains below closed-set levels**: The open-vocabulary mIoU (21.05) lags substantially behind closed-set methods (47.76), particularly for fine-grained categories such as tvs (5.36), furniture (5.88), and objects (6.94).
2. **Dependence on external models**: The pipeline requires Depth-Anything V2 for depth priors, Trident for open-vocabulary segmentation features, and Qwen2.5-VL for object noun extraction, resulting in a lengthy pipeline.
3. **Evaluation on a single dataset**: All experiments are conducted on Occ-ScanNet; generalization to other indoor environments (e.g., Matterport3D, Replica) remains unverified.
4. **Difficulty in fine-grained semantic alignment**: When semantically similar categories overlap in image space (e.g., furniture vs. objects), feature confusion is difficult to fully eliminate even with temperature decay.
5. **Monocular limitation**: The potential benefits of multi-view input for open-vocabulary occupancy prediction are not explored.

## Related Work & Insights

- **EmbodiedOcc / EmbodiedOcc++**: State-of-the-art closed-set indoor occupancy methods using Gaussian volumetric prediction; this work extends the paradigm to open-vocabulary settings.
- **GaussianFormer2**: Proposes the G2O operator without opacity; this paper analyzes its failure under open-vocabulary conditions and proposes the Poisson-based improvement.
- **Dr. Splat**: Registers CLIP features to Gaussians via hard Top-$k$ selection; this paper replaces it with continuous temperature decay to preserve differentiability.
- **POP-3D / LOcc**: Outdoor open-vocabulary occupancy methods; experiments confirm their poor transferability to indoor scenes.
- **Insights**: Poisson process modeling can be generalized to occupancy estimation in other neural implicit representations; the temperature scheduling technique is applicable to any scenario involving $\alpha$-blending feature rendering.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of Poisson G2O and temperature decay is theoretically grounded and thoroughly analyzed.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive ablations on Occ-ScanNet, but limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear, with a well-structured progression from GaussianFormer2 → Bernoulli → Poisson.
- Value: ⭐⭐⭐⭐ Represents the first practical open-vocabulary occupancy prediction system for large-scale indoor scenes, advancing the deployment of embodied intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes](rescene4d_temporally_consistent_semantic_instance_segmentation_of_evolving_indoo.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
