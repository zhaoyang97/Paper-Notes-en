---
title: >-
  [Paper Note] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation
description: >-
  [ICCV 2025][Segmentation][Unsupervised temporal action segmentation] This paper proposes Skeleton Motion Quantization (SMQ), which achieves unsupervised temporal action segmentation on skeleton sequences via a joint-deco…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Unsupervised temporal action segmentation"
  - "skeleton sequences"
  - "motion quantization"
  - "temporal autoencoder"
  - "motion words"
date: 2026-05-08
content_hash: 47db1027a16e5b67
---

# Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation

**Conference**: ICCV 2025
**arXiv**: [2508.04513](https://arxiv.org/abs/2508.04513)
**Code**: [github.com/bachlab/SMQ](https://github.com/bachlab/SMQ)
**Area**: Image Segmentation
**Keywords**: Unsupervised temporal action segmentation, skeleton sequences, motion quantization, temporal autoencoder, motion words

## TL;DR

This paper proposes Skeleton Motion Quantization (SMQ), which achieves unsupervised temporal action segmentation on skeleton sequences via a joint-decoupled temporal autoencoder and a skeleton motion word quantization module, substantially outperforming existing unsupervised methods on HuGaDB, LARa, and BABEL.

## Background & Motivation

Temporal Action Segmentation (TAS) aims to partition long videos or sequences into distinct action segments. Existing methods for skeleton-based segmentation rely predominantly on **fully supervised** learning, incurring high annotation costs. Although unsupervised methods exist for RGB video (e.g., CTE, TOT, ASOT), they are **not optimized for the structural properties of skeleton data** and perform poorly when applied directly.

The authors identify two critical failure modes of existing unsupervised video segmentation methods on skeleton data:

Methods that use **Viterbi decoding as post-processing** (CTE, TOT) assume each action occurs only once per sequence, making them incapable of recognizing repeated actions.

Methods **without Viterbi decoding** tend to produce excessively fragmented segments, yielding poor segmentation quality.

Skeleton data possesses an inherent **joint–temporal structure** that prior methods do not exploit. This motivates the development of an unsupervised segmentation approach specifically designed for skeleton data.

## Method

### Overall Architecture

The SMQ framework consists of three core components: a **joint-decoupled temporal encoder** → a **skeleton motion quantization module** (comprising temporal patching and codebook quantization) → a **temporal decoder**. The input skeleton sequence $\mathbf{X} \in \mathbb{R}^{N \times C \times T \times V}$ is mapped to per-frame action labels $\mathbf{Y} \in \mathbb{N}^{N \times T}$.

### Key Designs

1. **Joint-Decoupled Encoder**: The input is reshaped to $\mathbf{X}' \in \mathbb{R}^{(N \cdot V) \times C \times T}$, treating each joint's temporal sequence as an independent sample fed into a TCN encoder. Dilated residual layers with exponentially growing dilation factors capture multi-scale temporal dependencies, producing embeddings $\mathbf{Z}_{nv} \in \mathbb{R}^{D \times T}$. **Core Idea**: Information from different joints is kept decoupled in the embedding space to prevent the representation from being dominated by a subset of joints.

2. **Temporal Patching**: After concatenating joint embeddings into $\mathbf{Z}_{concat} \in \mathbb{R}^{N \times T \times (V \cdot D)}$, the sequence is divided into $M = T/P$ non-overlapping temporal patches $\mathbf{Z}_p \in \mathbb{R}^{N \times M \times P \times (V \cdot D)}$. Compared to frame-wise quantization, patch-based representations better capture the temporal continuity of actions and reduce over-segmentation.

3. **Skeleton Motion Word Quantization**: A codebook $\mathcal{C} = \{\mathbf{c}_k\}_{k=1}^{K}$ of size $K$ is learned, where each entry $\mathbf{c}_k \in \mathbb{R}^{P \times (V \cdot D)}$ represents a "skeleton motion word." Quantization is performed via nearest-neighbor assignment:
$$\mathbf{q}_i = \mathbf{c}_{k_i} \quad \text{with} \quad k_i = \arg\min_k \|\mathbf{p}_i - \mathbf{c}_k\|_2$$
The codebook is updated via **EMA (Exponential Moving Average)** with decay factor $\alpha = 0.5$, improving training stability.

4. **Mirrored Decoder**: Symmetric in structure to the encoder, the decoder reconstructs the original skeleton sequence from quantized representations while maintaining joint decoupling.

### Loss & Training

The total loss consists of two terms:

$$L_{total} = \lambda L_{rec} + L_{commit}$$

- **Inter-Joint Distance MSE Reconstruction Loss**: Measures pairwise inter-joint distance discrepancies per frame rather than comparing coordinates directly. This loss is inherently **translation- and rotation-invariant**, capturing only postural differences:
$$L_{rec} = \frac{1}{N \cdot T \cdot V^2} \sum_{n,t,v,w} (dX_{ntvw} - d\hat{X}_{ntvw})^2$$

- **Commitment Loss**: Encourages encoder outputs to stay close to their assigned motion words, with $\lambda = 0.001$:
$$L_{commit} = \sum_{\mathbf{p}_i} \|\text{sg}[\mathbf{c}_{k_i}] - \mathbf{p}_i\|_2^2$$

## Key Experimental Results

### Main Results

| Dataset | Metric | SMQ (Ours) | ASOT | TOT+Viterbi | CTE+Viterbi |
|--------|------|-----------|------|-------------|-------------|
| HuGaDB | MoF | **42.0** | 33.9 | 33.8 | 39.2 |
| HuGaDB | Edit | **36.1** | 17.4 | 20.8 | 21.7 |
| HuGaDB | F1@50 | **24.3** | 3.0 | 7.5 | 7.5 |
| LARa | MoF | **37.4** | 22.9 | 32.6 | 23.0 |
| LARa | Edit | **39.4** | 23.4 | 17.7 | 17.7 |
| LARa | F1@50 | **16.4** | 5.7 | 3.2 | 1.6 |
| BABEL-S2 | MoF | **49.1** | 43.1 | 35.3 | 42.4 |
| BABEL-S2 | F1@50 | **27.4** | 23.4 | 19.8 | 12.8 |

### Ablation Study

**Joint-Decoupled vs. Entangled Embeddings**:

| Embedding | MoF (HuGaDB) | Edit | F1@50 |
|----------|-------------|------|-------|
| Entangled | 38.9 | 34.2 | 18.9 |
| **Disentangled** | **42.0** | **36.1** | **24.3** |

**Effect of Patch Size on LARa**:

| Patch Size | MoF | Edit | F1@50 | Note |
|-----------|-----|------|-------|------|
| 1 (frame-wise) | 33.9 | 25.6 | 8.6 | Over-segmentation |
| 50 (1 sec) | **37.4** | **39.4** | **16.4** | Optimal |
| 100 (2 sec) | 30.8 | 36.1 | 15.4 | Insufficient resolution |

**Loss Function Variants**:

| Commitment | Reconstruction Loss | MoF (LARa) | Edit | F1@50 |
|-----------|---------|-----------|------|-------|
| ✓ | ✗ | 29.9 | 29.9 | 8.8 |
| ✗ | Inter-Joint Dist. MSE | 31.0 | 34.6 | 14.4 |
| ✓ | MSE | 34.6 | 39.4 | 14.4 |
| ✓ | **Inter-Joint Dist. MSE** | **37.4** | **39.4** | **16.4** |

### Key Findings

- SMQ **comprehensively outperforms** all existing unsupervised temporal segmentation and self-supervised skeleton representation learning methods across all datasets and metrics.
- Joint-decoupled embeddings yield gains of up to **+5.4** F1@50 on HuGaDB, demonstrating the importance of exploiting skeletal structural priors.
- Frame-wise quantization (patch size = 1) causes severe over-segmentation; a 1-second patch size is optimal.
- Inter-joint distance MSE outperforms standard MSE due to its translation and rotation invariance.

## Highlights & Insights

- **Novelty of the Motion Words concept**: Framing temporal segmentation as temporal clustering and using discrete motion words to represent prototypical motion patterns is both intuitive and effective.
- **Simplicity of joint-decoupled design**: Independent per-joint embeddings are achieved via a reshape operation, without requiring complex graph convolutional networks.
- **Elegant inter-joint distance loss**: Translation and rotation invariance are obtained without explicit data augmentation.

## Limitations & Future Work

- Action boundaries can only occur at patch boundaries, making it **impossible to precisely localize action transition points**.
- The codebook size $K$ must be specified in advance and set to match the number of actions in each dataset, which is unknown in practice.
- Validation is limited to motion capture and IMU data; applicability to skeletons estimated from RGB video has not been explored.
- A substantial gap relative to supervised methods remains (e.g., MS-GCN achieves MoF = 90.4 vs. SMQ = 42.0 on HuGaDB).

## Related Work & Insights

- The quantization mechanism is analogous to VQ-VAE, but here temporal patches rather than individual frames are quantized.
- The joint-decoupling strategy may generalize to other skeleton-based temporal tasks such as action recognition and motion prediction.
- The EMA codebook update strategy is adopted from van den Oord et al.'s VQ-VAE work.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of skeleton motion words and joint-decoupled design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets with comprehensive ablation studies.
- Value: ⭐⭐⭐ — Requires prior knowledge of action count; significant gap to supervised methods.
- Overall: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation](clot_closed_loop_optimal_transport_for_unsupervised_action_segmentation.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)

</div>

<!-- RELATED:END -->
