---
title: >-
  [Paper Note] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models
description: >-
  [AAAI 2026][3D Vision][Test-time adaptation] This paper proposes Uni-Adapter, a training-free online test-time adaptation (TTA) framework for 3D vision-language foundation models (VLFMs). It addresses distribution shifts via clustering-based dynamic prototype caching and graph-regularized label smoothing, achieving state-of-the-art performance on multiple 3D corruption benchmarks.
tags:
  - AAAI 2026
  - 3D Vision
  - Test-time adaptation
  - 3D point cloud
  - vision-language foundation models
  - dynamic prototype learning
  - training-free adaptation
  - online clustering
  - graph regularization
date: 2026-05-08
content_hash: b771b88b68d85dfa
---

# Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models

**Conference**: AAAI 2026
**arXiv**: [2511.15311v2](https://arxiv.org/abs/2511.15311v2)
**Code**: [Available](https://github.com/Mehran-TAM/Uni-Adapter)
**Area**: 3D Vision / Test-Time Adaptation / Vision-Language Foundation Models
**Keywords**: Test-time adaptation, 3D point cloud, vision-language foundation models, dynamic prototype learning, training-free adaptation, online clustering, graph regularization

## TL;DR

This paper proposes Uni-Adapter, a training-free online test-time adaptation (TTA) framework for 3D vision-language foundation models (VLFMs). It addresses distribution shifts via clustering-based dynamic prototype caching and graph-regularized label smoothing, achieving state-of-the-art performance on multiple 3D corruption benchmarks.

## Background & Motivation

**Gap between zero-shot capability and real-world deployment of 3D VLFMs**: Models such as Uni3D, ULIP-2, and OpenShape achieve strong zero-shot recognition through point cloud–image–text tri-modal alignment, yet real-world point clouds are subject to sensor noise, sparsity, and low resolution, causing significant distribution shift from training data.

**High computational cost of training-based TTA**: Methods such as TPT require backpropagation at inference time to optimize prompts or parameters, demanding gradient iterations for each test sample, which is unsuitable for real-time or streaming inference.

**Insufficient mode coverage in high-confidence caching**: Existing training-free methods (TDA, Point-Cache) cache only high-confidence samples as prototypes; however, 3D features within the same semantic class often exhibit multi-modal distributions (e.g., the "airplane" class forms multiple sub-clusters in feature space), so high-confidence prototypes represent only a subset of modes and introduce decision boundary bias.

**Pseudo-label noise contaminating the cache**: Cache-based methods rely on pseudo-labels to assign samples to corresponding classes, but pseudo-labels are unreliable under domain shift, causing misclassified samples to persistently corrupt the cache in a negative feedback loop.

**Near-absence of cache-based TTA in 3D**: Cache-based TTA has been explored primarily for 2D CLIP; Point-Cache is the only prior work on 3D VLFMs, leaving substantial room for exploration.

**Inefficiency of Point-Cache's dual-cache design**: Point-Cache employs a global cache plus a local cache, with per-sample k-means applied to patch features in the local cache, resulting in a throughput of only 25% relative to zero-shot inference.

## Method

### Overall Architecture

Uni-Adapter consists of three core modules applied sequentially during the forward pass at inference, requiring no backpropagation whatsoever:

1. **Online Prototyping**: Maintains up to $N$ cluster centers per class as prototypes, updated continuously via confidence-weighted moving averages.
2. **Prototype Reassignment**: Constructs a cosine similarity graph over all prototypes and applies graph Laplacian regularization to correct noisy pseudo-labels.
3. **Entropy-Based Fusion**: Combines cache logits and original VLFM logits with weights inversely proportional to their respective entropies—lower confidence yields a smaller contribution.

### Key Design 1: Clustering-Based Online Prototype Cache

- **Function**: Maintains up to $N$ cluster centers per class; upon arrival of a new sample, the nearest prototype is updated, or a new prototype is initialized if a slot is available.
- **Mechanism**: After encoding input $\mathbf{X}_t$ as $\mathbf{f}_t$, a pseudo-class $k$ is predicted via cosine similarity with text embeddings; the most similar prototype within class $k$ is then updated via confidence-weighted averaging:

$$\mathbf{c}_{k,n}^{\text{new}} = \frac{\alpha_t \mathbf{f}_t + b_{k,n} \alpha_{k,n} \mathbf{c}_{k,n}^{\text{old}}}{\alpha_t + b_{k,n} \alpha_{k,n}}$$

where $\alpha_t = \exp(-\beta \cdot H_t)$ is a confidence weight based on prediction entropy and $b_{k,n}$ is the accumulated sample count.
- **Design Motivation**: High-confidence caching covers only the "peak" regions of the distribution, whereas cluster centers cover all distributional modes. t-SNE visualizations clearly show that the same class forms multiple sub-clusters in feature space; a multi-prototype design captures this intra-class diversity.

### Key Design 2: Graph-Regularized Prototype Label Smoothing

- **Function**: Constructs a similarity graph over all prototypes and applies graph regularization to correct noisy pseudo-labels.
- **Mechanism**: All prototype features $\mathbf{U} \in \mathbb{R}^{M \times d}$ are collected; a cosine similarity matrix $\mathbf{A} = \mathbf{U}\mathbf{U}^\top$ is computed, sparsified with threshold $\gamma$, and used to form the normalized graph Laplacian $\mathbf{L}_{\text{norm}}$. The following system is then solved:

$$\mathbf{Z}^* = (\mathbf{I} + \lambda_{\text{reg}} \mathbf{L}_{\text{norm}})^{-1} \mathbf{Z}^{(0)}$$

Solved efficiently via conjugate gradient, reducing complexity from $O(M^3)$ to $O(\rho \cdot \text{nnz}(\mathbf{L}_{\text{norm}}))$.
- **Design Motivation**: Online clustering is susceptible to pseudo-label noise; prototypes with incorrect labels mislead subsequent sample assignments. By allowing similar prototypes to "pull" each other toward consistent label assignments through the graph structure, this approach is more robust than simple confidence filtering.

### Key Design 3: Cache Logit Computation and Entropy-Weighted Fusion

- **Function**: Computes cache logits from prototype–input similarities and fuses them with the original VLFM logits.
- **Mechanism**: Cache logits are normalized by the per-class prototype count: $\mathbf{s}^{\text{cache}} = \mathbf{\Lambda} \mathbf{Z}^{*\top} (\mathbf{U} \mathbf{f}_t)$. Final predictions are obtained via entropy-cross-weighted fusion:

$$\mathbf{s}^{\text{final}} = \frac{H_{\text{cache}} \cdot \mathbf{s}^{\text{main}} + H_t \cdot \mathbf{s}^{\text{cache}}}{H_{\text{cache}} + H_t}$$

- **Design Motivation**: The more uncertain branch (higher entropy) grants greater influence to the other. When the cache is newly initialized (unreliable), the method automatically defers to the original model; as the cache accumulates sufficient prototypes, it progressively contributes more, enabling adaptive gradual adaptation.

### Loss & Training

This method **requires no training whatsoever** and involves no loss functions or gradient computations. All adaptation operations are completed during the forward pass at inference:

- Online cluster update: confidence-weighted moving average
- Graph regularization: conjugate gradient, up to 100 iterations
- Per-sample adaptation: batch size = 1, supporting streaming inference

Key hyperparameters: number of cluster centers $N=30$, sparsification threshold $\gamma=0.5$, confidence decay $\beta=10$, label smoothing coefficient $\lambda_{\text{reg}}=0.3$.

## Key Experimental Results

### Main Results: Distribution Shift Adaptation on Corrupted Datasets (Uni3D-Large, batch=1)

| Dataset | Source-Only | TDA* (CVPR24) | Point-Cache* (CVPR25) | **Uni-Adapter** | Gain |
|---|---|---|---|---|---|
| ModelNet-40C | 59.15% | 63.63% | 66.73% | **69.70%** | +10.55% |
| ScanObjectNN-C | 38.07% | 40.62% | 42.13% | **46.33%** | +8.26% |
| ShapeNet-C | 57.92% | 59.43% | 57.70% | **62.41%** | +4.49% |

Uni-Adapter consistently leads across all 15 corruption types, surpassing the strongest baseline Point-Cache by approximately 3 percentage points on ModelNet-40C.

### Clean and Large-Scale Datasets

| Dataset | Scale | Source-Only | Point-Cache | **Uni-Adapter** |
|---|---|---|---|---|
| ModelNet40 | 40 classes | 83.47% | 83.43% | **83.96%** |
| ScanObjectNN | 15 classes | 61.46% | 61.46% | **64.03%** |
| ShapeNet | 55 classes | 81.23% | 80.96% | **81.23%** |
| Objaverse-LVIS | 1156 classes | 51.59% | 51.65% | **52.44%** |

Performance is not degraded on clean data without distribution shift; ScanObjectNN even improves by 2.57%.

### Computational Efficiency

| Method | Throughput (test/s) | Ratio vs. Zero-Shot |
|---|---|---|
| Zero-shot | 39.19 | 100% |
| Point-Cache | 9.73 | 25% |
| **Uni-Adapter** | **36.93** | **94%** |

Uni-Adapter incurs only ~6% throughput reduction, compared to 75% for Point-Cache.

### Ablation Study

1. **Component contributions**: Online Prototyping accounts for the primary gain (59.15→68.48); Prototype Reassignment adds a further 1.22% (→69.70).
2. **Clustering vs. confidence-based caching**: Clustering-based caching consistently outperforms confidence-based caching across all corruption types on ShapeNet-C.
3. **Number of cluster centers $N$**: $N=30$ is optimal; too few fail to cover intra-class distribution, too many introduce noise.
4. **Label smoothing $\lambda_{\text{reg}}$**: 0.3 is optimal; values near 0 eliminate smoothing, values near 1 cause over-smoothing.
5. **Conjugate gradient vs. direct inversion**: Conjugate gradient is faster (27.07 ms vs. 29.20 ms) with MAE < 0.0005%.
6. **Statistical significance**: All comparisons yield $p$-values well below 0.05; the strongest comparison against Point-Cache on ModelNet-40C yields $p = 8.04 \times 10^{-7}$.
7. **Cross-model validation**: Effective on ULIP-2 and OpenShape as well, with gains of +7.97% and +4.64% on ModelNet-40C, respectively.

## Highlights & Insights

1. **Truly training-free adaptation**: Requires no backpropagation, no modification of model parameters, and no labeled data; operates with batch size = 1.
2. **Elegant clustering cache design**: Resolves the mode-coverage limitation of high-confidence caching by using online clustering to capture multi-modal intra-class distributions.
3. **Graph-regularized label smoothing**: Leverages topological relationships among prototypes to correct pseudo-labels, which is more principled than simple confidence filtering.
4. **Exceptional computational efficiency**: Throughput remains close to zero-shot inference (only 6% reduction), far superior to Point-Cache (75% reduction), with negligible memory overhead.
5. **Model-agnostic**: Demonstrated effective across three 3D VLFMs—Uni3D, ULIP-2, and OpenShape.
6. **Comprehensive evaluation**: Covers corrupted datasets (15 corruption types × 5 severity levels), clean datasets, and large-scale datasets (1156 classes), with statistical significance testing and thorough ablations.

## Limitations & Future Work

1. **Cold-start instability**: During the cache initialization phase (before prototypes have sufficiently accumulated), performance is unstable under severely noisy inputs—a limitation acknowledged by the authors.
2. **Accumulated pseudo-label bias**: While graph smoothing corrects some errors, argmax-based pseudo-label generation may persistently fail when domain shift is extreme.
3. **Fixed cluster count $N$**: All classes share the same maximum cluster count, despite potentially varying distributional complexity across classes.
4. **Classification tasks only**: Downstream 3D tasks such as segmentation and detection are not addressed.
5. **No continuous domain drift**: Experiments assume fixed corruption types; scenarios with continuously evolving domains are not evaluated.

## Related Work & Insights

| Method | Type | Training-Free | 3D-Specific | VLFM-Specific | ModelNet-40C |
|---|---|---|---|---|---|
| TENT (ICLR21) | Training-based TTA | ❌ | ❌ | ❌ | 59.48 |
| T3A (NeurIPS21) | Training-free TTA | ✅ | ❌ | ❌ | 64.12 |
| TPT (NeurIPS22) | Training-based TTA | ❌ | ❌ | ✅ | 61.02 |
| TDA (CVPR24) | Training-free TTA | ✅ | ❌ | ✅ | 63.63 |
| CloudFixer (ECCV24) | Input adaptation | ✅ | ✅ | ❌ | 56.09 |
| Point-Cache (CVPR25) | Training-free TTA | ✅ | ✅ | ✅ | 66.73 |
| **Uni-Adapter** | Training-free TTA | ✅ | ✅ | ✅ | **69.70** |

Key distinction: Point-Cache uses high-confidence caching with per-sample k-means on local patch features (dual-cache structure), whereas Uni-Adapter uses online clustering for global prototypes with graph smoothing (a single unified cache), achieving substantially higher throughput.

## Rating

- Novelty: ⭐⭐⭐⭐ Replacing confidence-based caching with clustering-based caching and adding graph smoothing for correction represents meaningful innovation, though the overall framework remains an improvement upon the cache-based TTA paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three corrupted datasets, three clean datasets, two large-scale datasets, three 3D VLFMs, statistical significance testing, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated design choices, and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Training-free TTA for 3D VLFMs is a timely and practical research direction; the proposed method is both efficient and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[ICCV 2025\] 3D Test-time Adaptation via Graph Spectral Driven Point Shift](../../ICCV2025/3d_vision/3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)
- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](../../CVPR2026/3d_vision/tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)
- [\[AAAI 2026\] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization](enhancing_generalization_of_depth_estimation_foundation_model_via_weakly-supervi.md)

</div>

<!-- RELATED:END -->
