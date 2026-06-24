---
title: >-
  [Paper Note] SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion
description: >-
  [AAAI 2026 Oral][3D Vision][Semantic Scene Completion] This paper proposes SplatSSC, which addresses the issue of inefficient random initialization and floating artifacts from outlier primitives in the object-centric paradigm through a depth-guided Gaussian primitive initialization strategy and a Decoupled Gaussian Aggregator (DGA). It achieves an IoU gain of 6.3% and a mIoU gain of 4.1% on Occ-ScanNet, while reducing latency and memory costs by over 9.3%.
tags:
  - "AAAI 2026 Oral"
  - "3D Vision"
  - "Semantic Scene Completion"
  - "3D Gaussian Splatting"
  - "Depth Guidance"
  - "Decoupled Aggregation"
  - "Indoor Scene Understanding"
date: 2026-05-08
content_hash: 13eeed058163e30f
---

# SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion

**Conference**: AAAI 2026 Oral  
**arXiv**: [2508.02261](https://arxiv.org/abs/2508.02261)  
**Code**: [GitHub](https://github.com/Made-Gpt/SplatSSC)  
**Area**: 3D Vision  
**Keywords**: Semantic Scene Completion, 3D Gaussian Splatting, Depth Guidance, Decoupled Aggregation, Indoor Scene Understanding

## TL;DR

This paper proposes SplatSSC, which addresses the issue of inefficient random initialization and floating artifacts from outlier primitives in the object-centric paradigm through a depth-guided Gaussian primitive initialization strategy and a Decoupled Gaussian Aggregator (DGA). It achieves an IoU gain of 6.3% and a mIoU gain of 4.1% on Occ-ScanNet, while reducing latency and memory costs by over 9.3%.

## Background & Motivation

Monocular 3D semantic scene complete (SSC) aims to infer full 3D geometry and semantic descriptions from a single image. Recently, the object-centric paradigm (represented by GaussianFormer) has achieved breakthroughs in efficiency and performance by representing scenes with 3D Gaussian primitives. However, this paradigm suffers from two fundamental issues:

**Problem 1: Inefficient primitive initialization**
- To cover the entire 3D space without geometric clues, prior methods randomly distribute a large number of primitives within the 3D volume.
- Most primitives are wasted on representing empty or unknown space, leading to severe redundancy.
- For instance, GaussianFormer utilizes 19,200 primitives, many of which are useless.

**Problem 2: Fragile aggregation of outlier primitives**
- The Gaussian-to-voxel splatting strategy (GaussianFormer, GaussianFormer-2) lacks an effective mechanism to exclude outliers.
- Isolated outlier primitives splat incorrect semantics onto distant voxels, generating "floaters".
- The Probability Gaussian Splatting (PGS) of GaussianFormer-2 has a design flaw: the opacity $\mathbf{a}_i$ is canceled out during posterior probability normalization, causing low-confidence outlier primitives to still produce high occupancy values.

The authors present a rigorous mathematical analysis of the PGS flaw: for an isolated outlier primitive $G_n$, the likelihood of other primitives at its neighboring point $\mathbf{x}^f$ approaches zero, leading the posterior probability to collapse to 1:
$$p(G_n|\mathbf{x}^f) \approx \frac{p(\mathbf{x}^f|G_n)\mathbf{a}_n}{p(\mathbf{x}^f|G_n)\mathbf{a}_n + 0} = 1$$

Even if $\mathbf{a}_n$ is very low, it is canceled out after normalization, and the expected semantics degenerate into the semantic label of that outlier primitive.

## Method

### Overall Architecture

SplatSSC comprises the following main components:
1. **Image Encoder** (EfficientNet + FPN) extracts multi-scale image features.
2. **Frozen Depth-Anything-V2** extracts depth features.
3. **Depth Branch**: The GMF module fuses image and depth features to output refined depth maps.
4. **Lifter**: Initializes sparse Gaussian primitives based on depth priors.
5. **Multi-stage Encoder**: Iteratively refines Gaussian attributes.
6. **DGA**: Decouples geometry and semantics, aggregating Gaussian primitives into semantic voxel grids.

### Key Designs

#### 1. **Depth Branch and GMF Module (Group-wise Multi-scale Fusion)**: Efficient Multimodal Fusion

**GCA Layer (Group Cross-Attention)**:
- Samples depth features and multi-scale image features at pre-defined reference points.
- Splits features along the channel dimension into $G$ groups, where each group has a feature dimension of $D_g = D/G$.
- Queries come from depth features, while Keys and Values come from image features of various scales.
- Uses lightweight linear projection instead of standard dot-product attention:

$$A_g^l = \mathbb{S}_l(W_a(Q_g + K_g^l))$$

$$\mathcal{F}_d' = \mathbb{C}_g(\sum_{l=1}^{L} A_g^l \circ V_g^l) W_o$$

**Efficiency Analysis**: The complexity of standard cross-attention is $\mathcal{O}(LN^2D)$, which is reduced by GCA to $\mathcal{O}(ND^2(L+2)/G)$. The weight matrix $W_a$ is shared across groups and scales, significantly reducing the parameter size.

**Data Significance**: GMF improves the $\delta_1$ metric of frozen Depth-Anything-V2 from 0.075 to 0.981 (a gain of 0.906), and to 0.993 with a fine-tuned version.

#### 2. **Decoupled Gaussian Aggregator (DGA)**: The Key to Eliminating Floaters

DGA factors semantic occupancy prediction into two independent paths:

**Geometric Occupancy Prediction**:
$$\alpha'(x) = 1 - \prod_{i \in \mathcal{N}(\mathbf{x})} (1 - \alpha(\mathbf{x}; G_i) \cdot \mathbf{a}_i)$$

Key difference: the influence of each primitive is modulated by its learned opacity $\mathbf{a}_i$. Low-confidence outlier primitives are naturally suppressed.

**Conditional Semantic Distribution**:
$$e^k(\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_i) \cdot \tilde{\mathbf{c}}_i^k}{\sum_{j \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_j)}$$

Semantic prediction **does not use opacity**, relying solely on geometric proximity and normalized semantic weights.

**Probability Fusion**:
$$\hat{\mathbf{y}}_x^k = \alpha'(\mathbf{x}) \cdot e^k(\mathbf{x}), \quad \hat{\mathbf{y}}_x^{empty} = 1 - \alpha'(\mathbf{x})$$

This is an elegant gating mechanism: low occupancy probabilities directly suppress any incorrect semantic predictions, eliminating floaters without requiring extra heuristic rules.

#### 3. **Probability Scale Loss**: Progressive Geometric Supervision

It extends MonoScene's geometry-aware scale loss to the occupancy probability predictions of all $n$ encoder layers:

$$\mathcal{L}_{scal}^{prob} = \frac{1}{2}\sum_{i=1}^{n-1} \frac{i}{n} \cdot \mathcal{L}_{scal}^{geo,i} + \mathcal{L}_{scal}^{geo,n}$$

Linear weight scheduling puts weaker constraints on earlier layers and progressively enforces consistency in deeper layers.

### Loss & Training

**Two-stage Training**:

Stage 1: Depth branch pretraining
$$\mathcal{L}_d = 10 \mathcal{L}_{\text{huber}}^{\text{depth}} + 20 \mathcal{L}_{\text{huber}}^{\text{pts}} + 0.5 \mathcal{L}_{\text{grad}}$$

Stage 2: End-to-end SSC training
$$\mathcal{L}_{ssc} = 100 \mathcal{L}_{\text{focal}} + 2 \mathcal{L}_{\text{lovasz}} + 0.5 \mathcal{L}_{scal}^{prob}$$

Note: The depth loss $\mathcal{L}_d$ is removed in Stage 2 to prevent the model from being over-constrained by the initial depth prediction. Depth-Anything-V2 remains frozen throughout.

## Key Experimental Results

### Main Results (Occ-ScanNet)

| Method | Input | IoU↑ | mIoU↑ | Note |
|------|------|------|-------|------|
| TPVFormer | RGB | 33.39 | 24.94 | Transformer baseline |
| GaussianFormer | RGB | 40.91 | 29.93 | Pioneered object-centric paradigm |
| MonoScene | RGB | 41.60 | 24.62 | Dense 2D-to-3D lifting |
| EmbodiedOcc | RGB | 53.95 | 45.48 | Previous representative method |
| EmbodiedOcc++ | RGB | 54.90 | 46.20 | Enhanced version |
| RoboOcc | RGB | 56.48 | 47.67 | Prev. SOTA |
| **SplatSSC** | RGB | **62.83** | **51.83** | Ours (Leads significantly) |

IoU increases by 6.35% (absolute value) and mIoU increases by 4.16%. Consistent improvements are observed across all semantic categories.

### Ablation Study

**Component Ablation**:

| GMF | Aggregator | IoU↑ | mIoU↑ | Note |
|-----|------------|------|-------|------|
| ✗ | GF.agg | 11.64 | 12.62 | No GMF + original aggregator nearly fails |
| ✗ | GF2.agg | 27.54 | 17.27 | No GMF + PGS aggregator |
| ✗ | DGA | 48.85 | 36.91 | Effective with DGA even without GMF |
| ✓ | GF.agg | 16.63 | 10.45 | GMF + original aggregator remains poor |
| ✓ | GF2.agg | 57.70 | 45.13 | GMF + PGS |
| ✓ | **DGA** | **60.61** | **48.01** | Full method (Ours) is optimal |

**Ablation on Gaussian Parameters**:

| No. of Primitives | Scale Range | Memory (MiB) | Latency (ms) | IoU | mIoU |
|-------------------|-------------|--------------|--------------|-----|------|
| 19200 | [0.01,0.08] | 3.122 | 135.18 | 62.77 | 47.69 |
| 4800 | [0.01,0.08] | 3.158 | 123.27 | 62.23 | 47.20 |
| **1200** | **[0.01,0.16]** | **3.112** | **115.56** | **61.47** | **48.87** |
| 19200 | [0.01,0.32] | 14.380 | 134.51 | OOM | — |

The highest mIoU is achieved using only 1,200 primitives, which outperforms 19,200 primitives while significantly reducing computations.

### Key Findings

1. DGA outperforms GF2.agg in both IoU and mIoU by around 2.8%, proving that floaters are a key bottleneck in sparse splatting.
2. The GMF module has a huge impact on performance—removing it leads to an 11%+ drop even with DGA.
3. Using only 1,200 primitives paired with a moderate scale range of [0.01, 0.16] yields the optimal configuration.
4. Explicit depth loss is unexpectedly counterproductive in Stage 2, whereas Probability Scale Loss is more suitable.
5. Significant efficiency gains: compared to EmbodiedOcc, latency is reduced by 9.32%, and memory is reduced by 9.64%.

## Highlights & Insights

1. **Precise Mathematical Analysis of PGS Flaw**: Provides a highly rigorous and convincing proof for the issues where opacity is canceled out via normalization in the GaussianFormer-2 aggregator.
2. **Elegance of Decoupled Design**: Fully factors geometric occupancy from semantic prediction, where opacity acts only as a gating mechanism in the geometry path. This presents a natural and principled solution.
3. **Less is More**: 1,200 depth-guided primitives outperform 19,200 random primitives, intuitively showcasing the importance of initialization quality.
4. **Efficiency Design in GCA**: Group-shared attention weights yield substantial savings in parameters and computations.
5. **Progressive Design of Probability Scale Loss**: Imposes different supervision weights across different encoder layers, aligning naturally with the process of layer-by-layer refinement.

## Limitations & Future Work

1. Currently evaluated only on indoor scenes (Occ-ScanNet) without validation on outdoor scenes (e.g., nuScenes).
2. Relies on a frozen Depth-Anything-V2, limiting the upper-bound quality of depth priors.
3. 1,200 primitives might be insufficient to handle large-scale or highly complex scenes.
4. Temporal consistency (consistency of occupancy predictions across consecutive frames) is not discussed.
5. Sensitivity analyses for hyperparameters, such as the number of groups $G$ in GCA, are not detailed.

## Related Work & Insights

- **GaussianFormer/GaussianFormer-2**: Pioneers of the object-centric SSC paradigm, whose aggregator design flaws are exposed by SplatSSC.
- **EmbodiedOcc/EmbodiedOcc++**: Representative works that introduce the object-centric paradigm into indoor scenes.
- **VoxFormer**: A sparse-to-dense Transformer method that first introduces geometric priors for proposal generation.
- **Depth-Anything-V2**: A highly capable monocular depth estimator providing depth features and depth priors.
- **Insights**: The design philosophy of depth-guided sparse initialization and decoupled aggregation is transferable to other object-centric 3D perception tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The PGS flaw analysis and DGA decoupled design are novel and profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Extensive ablation studies are provided, but the datasets are limited to indoor scenarios.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous problem analysis, clear mathematical derivation, and well-motivated methods.
- **Value**: ⭐⭐⭐⭐ — A significant advance for indoor 3D scene understanding, with direct value for embodied intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TGSFormer: Scalable Temporal Gaussian Splatting for Embodied Semantic Scene Completion](../../CVPR2026/3d_vision/tgsformer_scalable_temporal_gaussian_splatting_for_embodied_semantic_scene_compl.md)
- [\[AAAI 2026\] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)
- [\[CVPR 2026\] Multi-modal Frequency Decomposition Network for Semantic Scene Completion](../../CVPR2026/3d_vision/multi-modal_frequency_decomposition_network_for_semantic_scene_completion.md)
- [\[CVPR 2026\] Learning Spatial-Temporal Consistency for 3D Semantic Scene Completion](../../CVPR2026/3d_vision/learning_spatial-temporal_consistency_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](../../ICCV2025/3d_vision/monocular_semantic_scene_completion_via_masked_recurrent_networks.md)

</div>

<!-- RELATED:END -->
