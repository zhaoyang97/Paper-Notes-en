---
title: >-
  [Paper Note] Divide and Contrast: Learning Robust Temporal Features Without Augmentation
description: >-
  [ICML 2026][Time Series][Time Series Representation Learning] Di-COT efficiently learns robust time series representations without data augmentation by **randomly partitioning sequences into overlapping sub-blocks** and…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Representation Learning"
  - "Contrastive Learning"
  - "Self-supervised"
  - "Augmentation-free"
  - "Sub-block Partitioning"
date: 2026-05-08
content_hash: 4aedfa4f09fb9bea
---

# Divide and Contrast: Learning Robust Temporal Features Without Augmentation

**Conference**: ICML 2026  
**arXiv**: [2605.21241](https://arxiv.org/abs/2605.21241)  
**Code**: To be confirmed  
**Area**: Time Series / Self-supervised Learning  
**Keywords**: Time Series Representation Learning, Contrastive Learning, Self-supervised, Augmentation-free, Sub-block Partitioning

## TL;DR
Di-COT efficiently learns robust time series representations without data augmentation by **randomly partitioning sequences into overlapping sub-blocks** and performing contrastive learning on them. It is 2.5x faster and achieves higher accuracy compared to existing methods, with comprehensive validation on 6 large-scale datasets + 124 UCR + 28 UEA.

## Background & Motivation

**Background**: Self-supervised representation learning for time series has become a significant research direction, with contrastive learning being widely applied. Existing methods such as TNC, TS-TCC, and TS2Vec utilize temporal proximity or data augmentation to construct positive and negative pairs.

**Limitations of Prior Work**:
- Requirement for complex data augmentation (time warping, magnitude transformation) leads to representation distortion.
- Use of Dynamic Time Warping or multiple encoder forward passes results in high computational overhead.
- The recent method CaTT avoids augmentation but assumes that temporal adjacency is equivalent to semantic similarity, which fails on UCR/UEA datasets.

**Key Challenge**: On datasets with high temporal volatility (frequent event transitions), step-wise contrastive learning generates false positives at temporal transitions. Methods solely relying on temporal proximity cannot handle such scenarios. Meanwhile, the computational complexity of existing loss functions is quadratic $O(T^2)$ with respect to sequence length $T$, which is unfriendly to long sequences.

**Goal**: To design a self-supervised time series learning framework that requires no data augmentation, no multiple encoder passes, and is independent of sequence length.

**Key Insight**: Instead of performing contrastive learning on individual time steps, it is better to partition the sequence into sub-block units with **semantic integrity**. This avoids false positives at temporal transitions while retaining sufficient learning signals.

**Core Idea**: Replace **step-wise or augmentation-based contrastive learning** with **contrastive learning on dynamic overlapping sub-blocks**, and reformulate it as a **multi-class classification task** to achieve efficient, length-independent computation.

## Method

### Overall Architecture
Three stages: (1) **Sub-block Partitioning**: The input sequence $\mathbf{x}^{(i)} \in \mathbb{R}^{T \times D}$ is randomly partitioned into $k$ overlapping sub-blocks, where $k$ is uniformly sampled from $\{k_{\min}, \ldots, k_{\max}\}$; (2) **Encoding and Similarity Computation**: Each sub-block is encoded and pooled to obtain $k$ embeddings, followed by calculating a temperature-scaled similarity matrix $\mathbf{S}^{(i)} \in \mathbb{R}^{k \times k}$; (3) **Cross-entropy Contrastive Objective**: Predicting adjacent sub-blocks is reformulated as multi-class classification, with each sub-block acting as an anchor to generate dense supervision.

### Key Designs

1. **Dynamic Overlapping Sub-block Partitioning**:

    - **Function**: Decomposes long sequences into short sub-sequences with semantic integrity for efficient contrastive learning.
    - **Mechanism**: In each iteration, the number of sub-blocks $k \ll T$ is sampled from $\mathcal{U}\{k_{\min}, \ldots, k_{\max}\}$. The sub-block length $L = \frac{T}{1 + (k-1)(1-\rho)}$ and stride $s = \lfloor L(1-\rho) \rfloor$, where $\rho \in (0, 1)$ is the overlap ratio. After encoding all sub-blocks, embeddings $\mathbf{z}^{(i)} = \{z_1^{(i)}, \ldots, z_k^{(i)}\}$ are obtained, where $z_j^{(i)} = f_\theta(\tilde{x}_j^{(i)}) \in \mathbb{R}^F$.
    - **Design Motivation**: The overlapping design ensures sufficient context sharing between adjacent sub-blocks to avoid artificial boundaries. Randomly sampling $k$ per iteration exposes the model to various temporal granularities, implicitly learning multi-scale robustness. By reducing the granularity of the contrastive objective (from $T$ to $k \ll T$), false positives at temporal transitions are naturally avoided.

2. **Cross-entropy Contrastive Objective**:

    - **Function**: Reformulates traditional InfoNCE contrastive learning as a multi-class classification problem to achieve efficient, length-independent computation.
    - **Mechanism**: For each pair of sub-blocks $(j, p)$, the temperature-scaled similarity is computed as $S_{j, p}^{(i)} = \frac{\mathbf{z}_j^{(i)\top} \mathbf{z}_p^{(i)}}{\tau}$. The previous sub-block is defined as the positive label $p^*(j) = j - 1$ (for $j > 0$), and the others as negatives. The loss is $\mathcal{L}_{\text{CE}} = -\frac{1}{B k} \sum_i \sum_j \log \frac{\exp(S_{j, p^*(j)}^{(i)})}{\sum_p \exp(S_{j, p}^{(i)})}$.
    - **Design Motivation**: Compared to the $O(B T^2 d)$ complexity of CaTT, Di-COT achieves $O(B k^2 d)$ where $k \ll T$. It provides dense supervision—every sub-block acts as an anchor generating $B \times k$ positive pairs, whereas augmentation-based methods only yield $2B$ pairs.

3. **Augmentation-free Strategy**:

    - **Function**: Eliminates data augmentation, masking, and multiple encoder passes to prevent representation distortion and reduce computational overhead.
    - **Mechanism**: Contrastive learning is performed entirely within the original sequence space without generating any augmented views. Different parts (sub-blocks) of the sequence are compared instead of augmented versions, and all sub-blocks share the same semantic context.
    - **Design Motivation**: Augmentations for time series (jittering, permutation) may destroy valid temporal structures. Through sub-block overlap and multi-granularity sampling, the model naturally learns invariance to small displacements. This also avoids multiple encoder passes.

## Key Experimental Results

### Main Results (Linear Evaluation on 6 Large-scale Datasets)

| Dataset | **Ours** | CaTT | TS2Vec | TF-C | Gain vs CaTT |
|--------|------|------|--------|------|----------|
| ECG | **85.28** | 80.89 | 71.83 | 74.67 | +4.39% |
| HARTH | **93.23** | 93.13 | 90.27 | 92.24 | +0.10% |
| PAMAP2 | **71.38** | 69.86 | 70.37 | 71.30 | +1.52% |
| SKODA | **99.41** | 94.87 | 98.96 | 98.23 | +4.54% |
| SLEEP | **85.21** | 85.17 | 84.81 | 85.18 | +0.04% |
| WISDM2 | **63.92** | 63.25 | 62.39 | 62.54 | +0.67% |
| **Avg. Acc.** | **83.07** | 81.20 | 79.77 | 80.69 | **+1.87%** |
| **Training Time (h)** | **2.88** | 3.47 | 3.28 | 6.52 | **-17%** |

### Low-label Regime (1% Labeled Data)

| Dataset | Ours | TF-C | TNC | Supervised Baseline |
|--------|------|------|------|----------|
| ECG | **73.33** | 74.50 | 61.06 | 54.28 |
| HARTH | **87.23** | 78.00 | 83.04 | 75.37 |
| SKODA | **98.01** | 93.50 | 96.11 | 92.77 |
| **Avg. Acc.** | **76.36** | 73.55 | 72.73 | 70.39 |

In the low-label setting, improvement is +5.97% compared to the supervised baseline, and it is 2.5x faster than TF-C.

### Ablation Study

| Configuration | Large Datasets | UCR | UEA | Description |
|------|------------|------|------|------|
| Full Model | 83.07 | 81.33 | 71.24 | Standard Di-COT |
| Remove Overlap ($\rho=0$) | 81.22 | 81.12 | 70.13 | -2.23% / -1.56% |
| Remove Temperature | 82.47 | 81.33 | 70.79 | -0.72% (Small impact) |
| Fixed Global Partition | 82.80 | 81.32 | 69.69 | Random sampling is better |
| Contrast Shuffled Blocks | 81.80 | 81.19 | 70.09 | Temporal proximity is important |
| Non-linear Projection | 81.85 | 79.88 | 69.73 | Different from CV, not applicable |

### Key Findings
- **Sub-block overlap is most critical**: It contributes the most, especially on large-scale datasets (-2.23%).
- **Temporal adjacency is key**: Using temporally adjacent sub-blocks as positive pairs is significantly better than random pairing (-1.53%).
- **Backbone selection**: InceptionTime outperforms ResNet (-3.56%) and FCN (-2.58%).
- **No need for non-linear projection**: Unlike SimCLR, projection actually decreases performance.

## Highlights & Insights
- **Clever Granularity Scaling**: By reducing the contrastive granularity from time steps ($T$) to sub-blocks ($k \ll T$), false positives at temporal transitions are naturally avoided while retaining sufficient signals—making it more robust than CaTT's assumptions.
- **Length-independent Computation**: Reducing loss complexity from $O(B T^2 d)$ to $O(B k^2 d)$ enables the processing of long sequences; the cross-entropy reformulation is more efficient than traditional InfoNCE.
- **Benefits of Augmentation-free**: Completely discarding data augmentation prevents representation distortion and reduces computational overhead—suggesting that not all CV tricks are suitable for sequences.
- **Multi-granularity Robustness**: Randomly sampling $k$ per iteration allows the model to naturally learn multi-scale temporal features, implying multi-resolution learning without explicit design.

## Limitations & Future Work
- Di-COT is based on contrastive learning and learns discriminative representations, making it less suitable for time series forecasting tasks.
- The method relies heavily on the "temporal proximity = semantic similarity" assumption; it may still fail on data with high-frequency state jumps.
- The number of sub-blocks $k$ and overlap ratio $\rho$ require dataset-dependent tuning.
- Improvements: Adaptive sub-block partitioning strategies; hybrid contrastive strategies; extending to non-sequential tasks to verify generalizability.

## Related Work & Insights
- **vs CaTT** (Shamba et al. 2025): Also avoids augmentation and multiple encodings but contrasts all time steps; Di-COT avoids false positives via sub-blocks, has sequence-length independent complexity, and achieves better performance.
- **vs TS2Vec** (Yue et al. 2022): Contrast cross-views at the same timestamp, requiring two augmented views; Di-COT is more efficient and avoids augmentation bias.
- **vs Augmentation-based methods** (TS-TCC, TF-C): Traditional methods rely on complex augmentations; Di-COT proves that a simple augmentation-free strategy with reasonable granularity selection can outperform them on time series.
- **Insight**: Caution should be exercised when applying successful CV methods to other domains—sometimes "less" (no augmentation) is better than "more" (complex augmentation).

## Rating
- Novelty: ⭐⭐⭐⭐ Resolves the core trade-off (efficiency vs. accuracy) in time series contrastive learning by improving granularity selection and loss formulation; the idea is direct yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 large-scale + 124 UCR + 28 UEA datasets across 5 downstream tasks with sufficient ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with sufficient differentiation from prior work; some paragraphs could be more concise.
- Value: ⭐⭐⭐⭐⭐ High practical deployment value—both fast and accurate; code is open-source and directly applicable to various time series tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)
- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](../../AAAI2026/time_series/task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[ICML 2026\] Doubly Outlier-Robust Online Infinite Hidden Markov Model](doubly_outlier-robust_online_infinite_hidden_markov_model.md)
- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)

</div>

<!-- RELATED:END -->
