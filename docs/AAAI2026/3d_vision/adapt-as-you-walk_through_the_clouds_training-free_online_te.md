---
title: >-
  [Paper Note] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models
description: >-
  [AAAI 2026][3D Vision][Test-Time Adaptation] Uni-Adapter is proposed—a training-free online test-time adaptation framework for 3D Vision-Language Foundation Models (VLFMs). It registers SOTA performance across multiple 3D corruption benchmarks by tackling distribution shifts with cluster-based dynamic prototype caching and graph-regularized label smoothing.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Test-Time Adaptation"
  - "3D Point Clouds"
  - "Vision-Language Foundation Models"
  - "Dynamic Prototype Learning"
  - "Training-Free Adaptation"
  - "Online Clustering"
  - "Graph Regularization"
date: 2026-05-08
content_hash: 9843f9f55574232b
---

# Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models

**Conference**: AAAI 2026  
**arXiv**: [2511.15311v2](https://arxiv.org/abs/2511.15311v2)  
**Code**: [Yes](https://github.com/Mehran-TAM/Uni-Adapter)  
**Area**: 3D Vision / Test-Time Adaptation / Vision-Language Foundation Models  
**Keywords**: Test-Time Adaptation, 3D Point Clouds, Vision-Language Foundation Models, Dynamic Prototype Learning, Training-Free Adaptation, Online Clustering, Graph Regularization  

## TL;DR

Uni-Adapter is proposed—a training-free online test-time adaptation framework for 3D Vision-Language Foundation Models (VLFMs). It registers SOTA performance across multiple 3D corruption benchmarks by tackling distribution shifts with cluster-based dynamic prototype caching and graph-regularized label smoothing.

## Background & Motivation

**Gap between the zero-shot capability of 3D VLFMs and practical deployment**: Models like Uni3D, ULIP-2, and OpenShape achieve strong zero-shot recognition through point cloud-image-text tri-modal alignment. However, realistically collected point clouds are often affected by sensor noise, sparsity, and low resolution, leading to significant distribution shifts away from the training distribution.

**High computational overhead of training-based TTA**: Methods such as TPT require backpropagation to optimize prompts or parameters at inference time, performing gradient iterations for every test sample. This makes them unsuitable for real-time deployment and streaming inference scenarios.

**Key Challenge (Insufficient mode coverage of high-confidence caching)**: Existing training-free methods (e.g., TDA, Point-Cache) only cache high-confidence samples as prototypes. However, 3D features of the same semantic class often present multi-modal distributions (e.g., the "airplane" class forms multiple sub-clusters in the feature space). High-confidence prototypes can only represent partial modes, leading to biased decision boundaries.

**Cache contamination by pseudo-label noise**: Cache-based methods rely on pseudo-labels to assign samples to their corresponding categories. Under domain shifts, pseudo-labels become unreliable, and misclassified samples continually contaminate the cache, forming negative feedback loops.

**Cache-based TTA in the 3D domain is nearly non-existent**: Cache-based TTA has been mainly explored for 2D CLIP. In 3D VLFMs, Point-Cache is the only prior work, leaving a massive search space for exploration.

**Inefficient dual-cache design of Point-Cache**: Point-Cache employs a dual structure of global and local caches. Crucially, the local cache requires k-means clustering on patch features for every sample, which reduces throughput to only 25% of the zero-shot baseline.

## Method

### Overall Architecture

Uni-Adapter consists of three core modules connected in series, executing all adaptation operations during the forward pass of inference without requiring backpropagation:

1. **Online Prototyping**: Maintains at most $N$ cluster centers as prototypes for each class, which are continuously updated using confidence-weighted moving averages.
2. **Prototype Reassignment**: Constructs a cosine similarity graph among prototypes and applies graph Laplacian regularization smoothing to correct noisy pseudo-labels.
3. **Entropy-Based Fusion**: Computes a weighted fusion of the cache logits and the original VLFM logits based on their respective entropy values, where lower confidence results in a smaller weight.

### Key Designs 1: Cluster-Based Online Prototype Caching

- **Function**: Maintains a maximum of $N$ cluster centers for each class. As new samples arrive, they are matched with the nearest prototype and updated, or initialized as a new prototype if there is an empty slot.
- **Mechanism**: For a new input $\mathbf{X}_t$ encoded as $\mathbf{f}_t$, the pseudo-class $k$ is first predicted by cosine similarity with the text embeddings. Then, the most similar prototype within that class is retrieved and updated via confidence-weighted moving average:

$$\mathbf{c}_{k,n}^{\text{new}} = \frac{\alpha_t \mathbf{f}_t + b_{k,n} \alpha_{k,n} \mathbf{c}_{k,n}^{\text{old}}}{\alpha_t + b_{k,n} \alpha_{k,n}}$$

where $\alpha_t = \exp(-\beta \cdot H_t)$ is the confidence weight based on predicted entropy, and $b_{k,n}$ is the cumulative sample count.
- **Design Motivation**: High-confidence caching only covers "peak" areas of the distribution, whereas cluster centers can cover various modes within the distribution. t-SNE visualizations clearly show that the same class forms multiple sub-clusters in the feature space. The multi-prototype design captures this intra-class diversity.

### Key Designs 2: Graph-Regularized Prototype Label Smoothing

- **Function**: Builds a similarity graph among all prototypes and optimizes through graph regularization to correct noisy pseudo-labels.
- **Mechanism**: Collects all prototype features $\mathbf{U} \in \mathbb{R}^{M \times d}$, computes the cosine similarity matrix $\mathbf{A} = \mathbf{U}\mathbf{U}^\top$, sparsifies it using a threshold $\gamma$, and constructs a normalized graph Laplacian $\mathbf{L}_{\text{norm}}$ to solve:

$$\mathbf{Z}^* = (\mathbf{I} + \lambda_{\text{reg}} \mathbf{L}_{\text{norm}})^{-1} \mathbf{Z}^{(0)}$$

This is solved efficiently using the conjugate gradient method, reducing the complexity from $O(M^3)$ to $O(\rho \cdot \text{nnz}(\mathbf{L}_{\text{norm}}))$.
- **Design Motivation**: Online clustering is susceptible to pseudo-label noise; prototypes with erroneous labels can mislead subsequent sample assignments. Propagating labels through graph structures makes similar prototypes "pull each other" toward consistent label assignments, which is more robust than simple confidence filtering.

### Key Designs 3: Cache Logit Computation and Entropy-Weighted Fusion

- **Function**: Computes cache logits based on the similarity between prototypes and inputs, then fuses them with the original VLFM logits.
- **Mechanism**: Cache logits are normalized by the number of prototypes per class: $\mathbf{s}^{\text{cache}} = \mathbf{\Lambda} \mathbf{Z}^{*\top} (\mathbf{U} \mathbf{f}_t)$. Finally, they are fused through entropy cross-weighting:

$$\mathbf{s}^{\text{final}} = \frac{H_{\text{cache}} \cdot \mathbf{s}^{\text{main}} + H_t \cdot \mathbf{s}^{\text{cache}}}{H_{\text{cache}} + H_t}$$

- **Design Motivation**: The side with higher entropy (greater uncertainty) gives the other side more say. When the cache is just initialized (unreliable), the system automatically leans toward the original model. As the cache accumulates sufficient information, it gradually plays a larger role, achieving adaptive, progressive adaptation.

### Loss & Training

This method represents a **completely training-free** approach, requiring no loss functions or gradient computations. All adaptation actions occur during the forward pass of inference:

- Online clustering update: Confidence-weighted moving average.
- Graph regularization solving: Conjugate gradient method, up to 100 iterations.
- Instance-by-instance adaptation: Batch size = 1, supporting streaming inference.

Key hyperparameters: Cluster center limit $N=30$, sparsity threshold $\gamma=0.5$, confidence decay $\beta=10$, label smoothing coefficient $\lambda_{\text{reg}}=0.3$.

## Key Experimental Results

### Main Results: Adaptation to Distribution Shifts on Corrupted Datasets (Uni3D-Large, batch=1)

| Dataset | Source-Only | TDA* (CVPR24) | Point-Cache* (CVPR25) | **Uni-Adapter** | Gain |
|---|---|---|---|---|---|
| ModelNet-40C | 59.15% | 63.63% | 66.73% | **69.70%** | +10.55% |
| ScanObjectNN-C | 38.07% | 40.62% | 42.13% | **46.33%** | +8.26% |
| ShapeNet-C | 57.92% | 59.43% | 57.70% | **62.41%** | +4.49% |

Consistently leads across 15 corruption types. Outperforms the strongest baseline, Point-Cache, by approximately 3 percentage points on ModelNet-40C.

### Clean Datasets and Large-Scale Datasets

| Dataset | Size | Source-Only | Point-Cache | **Uni-Adapter** |
|---|---|---|---|---|
| ModelNet40 | 40 classes | 83.47% | 83.43% | **83.96%** |
| ScanObjectNN | 15 classes | 61.46% | 61.46% | **64.03%** |
| ShapeNet | 55 classes | 81.23% | 80.96% | **81.23%** |
| Objaverse-LVIS | 1156 classes | 51.59% | 51.65% | **52.44%** |

Does not degrade performance even on clean data without distribution shifts, and actually improves it by 2.57% on ScanObjectNN.

### Computational Efficiency

| Method | Throughput (test/s) | Relative Zero-Shot Ratio |
|---|---|---|
| Zero-shot | 39.19 | 100% |
| Point-Cache | 9.73 | 25% |
| **Uni-Adapter** | **36.93** | **94%** |

Uni-Adapter loses only about 6% throughput, compared to a 75% reduction for Point-Cache.

### Ablation Study

1. **Component Contributions**: Online Prototyping yields the core performance gains (59.15% $\rightarrow$ 68.48%), and Prototype Reassignment adds another 1.22% ($\rightarrow$ 69.70%).
2. **Clustering vs. Confidence Caching**: On all corruption types of ShapeNet-C, the cluster-based cache consistently outperforms the confidence-based cache.
3. **Number of Cluster Centers $N$**: $N=30$ is optimal; too few cannot cover the intra-class distributions, whereas too many introduce noise.
4. **Label Smoothing $\lambda_{\text{reg}}$**: 0.3 is optimal. As it approaches 0, the smoothing effect vanishes; as it approaches 1, over-smoothing occurs.
5. **Conjugate Gradient vs. Direct Inversion**: Conjugate gradient is faster (27.07ms vs. 29.20ms) with MAE < 0.0005%.
6. **Statistical Significance**: All comparison p-values are far below 0.05. Against the strongest competitor, Point-Cache, the result on ModelNet-40C reports a p-value of $8.04 \times 10^{-7}$.
7. **Cross-Model Validation**: The method scales effectively on ULIP-2 and OpenShape, yielding gains of +7.97% and +4.64% on ModelNet-40C, respectively.

## Highlights & Insights

1. **True Training-Free Adaptation**: Requires no backpropagation, incurs no parameter modifications, works without annotations, and accommodates streaming setups with a batch size of 1.
2. **Exquisite Cluster Caching Design**: Effectively solves the insufficient mode coverage of high-confidence caching by leveraging online clustering to capture intra-class multi-modal distributions.
3. **Graph-Regularized Label Smoothing**: Leverages the topological relationships among prototypes to correct pseudo-labels, presenting a more elegant solution than simple confidence thresholding.
4. **Outstanding Computational Efficiency**: Achieves real-time execution speeds (retaining 94% of zero-shot throughput) compared to Point-Cache which yields a 75% reduction, with negligible memory overhead.
5. **Model-Agnostic**: Demonstrates generalizability across three major 3D VLFMs (Uni3D, ULIP-2, OpenShape).
6. **Comprehensive Experiments**: Covers corrupted datasets (15 corruptions $\times$ 5 severity levels), clean datasets, and large-scale datasets (1156 categories), bolstered by statistical significance tests.

## Limitations & Future Work

1. **Unstable Cold Start**: At the initialization phase when the cache prototypes are not yet fully populated, performance can fluctuate under highly noisy inputs—a limitation acknowledged by the authors.
2. **Accumulated Pseudo-Label Bias**: Although graph smoothing corrects some errors, argmax-based pseudo-label generation can still propagate accumulated errors in cases of extreme domain shifts.
3. **Fixed Number of Clusters $N$**: All classes share the same maximum cluster limit, which ignores the varying complexity of class-wise distributions.
4. **Evaluation Limited to Classification**: Downstream tasks such as 3D point cloud segmentation or object detection have not yet been evaluated.
5. **Continuous Domain Shift Not Addressed**: The experiments assume fixed corruption types and do not test scenarios where domain distributions shift continuously over time.

## Related Work & Insights

| Method | Type | Training-Free? | 3D-Specific | VLFM-Specific | ModelNet-40C |
|---|---|---|---|---|---|
| TENT (ICLR21) | Training-based TTA | ❌ | ❌ | ❌ | 59.48 |
| T3A (NeurIPS21) | Training-free TTA | ✅ | ❌ | ❌ | 64.12 |
| TPT (NeurIPS22) | Training-based TTA | ❌ | ❌ | ✅ | 61.02 |
| TDA (CVPR24) | Training-free TTA | ✅ | ❌ | ✅ | 63.63 |
| CloudFixer (ECCV24) | Input Adaptation | ✅ | ✅ | ❌ | 56.09 |
| Point-Cache (CVPR25) | Training-free TTA | ✅ | ✅ | ✅ | 66.73 |
| **Uni-Adapter** | Training-free TTA | ✅ | ✅ | ✅ | **69.70** |

Key distinctions: Point-Cache uses high-confidence caching + local patch feature k-means (a dual-cache structure), whereas Uni-Adapter utilizes online clustering for global prototypes + graph smoothing (a single unified cache), achieving vastly superior throughput.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Replacing confidence caching with clustering combined with graph smoothing contains reasonable innovation, though the overall framework is still an evolution of cache-based TTA.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage with 3 corrupted datasets + 3 clean datasets + 2 large-scale datasets, validated on 3 VLFMs, accompanied by comprehensive ablation studies and statistical significance.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, clearly motivated, and mathematically mathematically complete.
- **Value**: ⭐⭐⭐⭐ Training-free TTA for 3D VLFMs is a practical and timely research direction; this work offers an exceptionally efficient solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[ECCV 2024\] CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation](../../ECCV2024/3d_vision/cloudfixer_test-time_adaptation_for_3d_point_clouds_via_diffusion-guided_geometr.md)
- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[ICLR 2026\] TTT3R: 3D Reconstruction as Test-Time Training](../../ICLR2026/3d_vision/ttt3r_3d_reconstruction_as_test-time_training.md)
- [\[CVPR 2026\] Low-Rank Test-Time Training for Pre-Trained Point Cloud Models](../../CVPR2026/3d_vision/low-rank_test-time_training_for_pre-trained_point_cloud_models.md)

</div>

<!-- RELATED:END -->
