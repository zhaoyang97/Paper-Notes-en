---
title: >-
  [Paper Note] Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration
description: >-
  [ICCV 2025][Model Compression][Data Pruning] This paper proposes Partial Forward Blocking (PFB), which computes sample importance at shallow layers during forward propagation and prunes low-importance samples by blocking their subsequent deep-layer forward passes. On ImageNet with 40% pruning, PFB achieves a 0.5% accuracy improvement and a 33% reduction in training time.
tags:
  - ICCV 2025
  - Model Compression
  - Data Pruning
  - Training Acceleration
  - Probability Density
  - Kernel Density Estimation
  - Forward Blocking
date: 2026-05-08
content_hash: 82cc57e42d25cfb2
---

# Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration

**Conference**: ICCV 2025
**arXiv**: [2506.23674](https://arxiv.org/abs/2506.23674)
**Code**: None
**Area**: Training Acceleration / Data Pruning
**Keywords**: Data Pruning, Training Acceleration, Probability Density, Kernel Density Estimation, Forward Blocking

## TL;DR

This paper proposes Partial Forward Blocking (PFB), which computes sample importance at shallow layers during forward propagation and prunes low-importance samples by blocking their subsequent deep-layer forward passes. On ImageNet with 40% pruning, PFB achieves a 0.5% accuracy improvement and a 33% reduction in training time.

## Background & Motivation

Large-scale datasets improve model generalization but incur substantial computational costs. Key limitations of existing data pruning methods:
- **Gradient-based methods** (EL2N, GraNd, etc.): Require full forward and backward passes to compute gradients, introducing significant overhead.
- **Proxy model methods** (SVP, YOCO, etc.): Require training an additional proxy model, with overhead comparable to training the target model itself.
- **Existing dynamic methods** (InfoBatch, DivBS, etc.): Require complete forward passes over all samples, even without gradients or proxy models.

Core question: Can training samples be efficiently selected at lower cost while preserving model generalization performance?

## Method

### Overall Architecture

PFB partitions the training network into a **shallow sub-network** $Net^{sh}$ and a **deep sub-network** $Net^{dp}$. At each training iteration:
1. All samples pass through the shallow sub-network to extract features.
2. Importance scores are computed from these features.
3. Low-importance samples are pruned, **blocking** their deep-layer forward passes.
4. Retained samples **reuse** their shallow features for the remaining deep forward and backward passes.

### Key Designs

1. **Partial Forward Blocking Strategy**: The core innovation lies in moving the pruning decision to an early stage of the forward pass. Pruned samples traverse only the shallow network (e.g., stage-1), with all deep-layer computation entirely skipped. By contrast:

    - Gradient-based methods require a full forward pass plus an additional backward pass.
    - Proxy-based methods require a full forward pass through the proxy model.
    - Methods such as InfoBatch require a complete forward pass for all samples.

   The computational cost of PFB equals the full forward pass for retained samples plus the shallow forward pass for pruned samples, which is substantially lower than all existing methods.

2. **Probability Density Importance**: The probability density of a sample in feature space is used as a measure of redundancy:

    - High probability density = dense region in feature space = high redundancy = low importance
    - Low probability density = sparse region = rare sample = high importance

   Importance is defined as:
    $\mathcal{I}(z_i^t) = \frac{1}{f_\mathbf{X}(\mathbf{x}_i^t) + r}$

   where $r = \alpha \cdot \max_{z_i \in B} f_\mathbf{X}(\mathbf{x}_i^t)$, and $\alpha \sim U(0, 0.01)$ introduces randomness to further promote diversity.

3. **Adaptive Distribution Estimation (ADE)**: Kernel Density Estimation (KDE) is employed to efficiently estimate probability densities:

    - Shallow features are spatially average-pooled and channel-reduced to obtain a compact representation $\mathbf{x}_i^t \in \mathbb{R}^{1 \times D}$.
    - A set of cluster centers $C^t = \{\mathbf{c}_j^t\}$ is maintained, and KDE is computed using a standard multivariate Gaussian kernel:
    $\hat{f}_\mathbf{X}(\mathbf{x}_i^t) = \sum_{j=1}^{N_C} \frac{w_j^t}{N_C} K_\mathbf{H}(\mathbf{x}_i^t - \mathbf{c}_j^t)$
    - The bandwidth matrix $\mathbf{H}$ is set via Silverman's rule.
    - Centers are updated using only **retained samples** via an exponential moving average (EMA, $\beta=0.01$).
    - Weights $w_j^{t+1} = n_j^t / (t \cdot (1-p) N_B)$ balance contributions across kernels.

### Loss & Training

- Standard cross-entropy loss computed only over retained samples $S^t$.
- The pruning ratio $p$ is a hyperparameter (recommended range: 30%–50%).
- The number of cluster centers $N_C$ is set to a small value (e.g., 100) to ensure efficient KDE.
- The stage-1 output is used as the feature extraction point for the shallow network.

## Key Experimental Results

### Main Results

**ResNet-50 on ImageNet-1k:**

| Method | 30% Pruned | 40% Pruned | 50% Pruned |
|--------|-----------|-----------|-----------|
| Full Data | 76.4 | 76.4 | 76.4 |
| Random | 72.2 (↓4.2) | - | 69.1 (↓7.3) |
| Forgetting | 74.8 (↓1.6) | - | 72.0 (↓4.4) |
| InfoBatch | 76.5 (↑0.1) | - | 75.8 (↓0.6) |
| MoSo | 76.5 (↑0.1) | - | 73.5 (↓2.9) |
| **PFB (Ours)** | **77.0 (↑0.6)** | - | **76.1 (↓0.3)** |

**Swin-T on ImageNet-1k:**

| Method | 30% Pruned | 40% Pruned | 50% Pruned |
|--------|-----------|-----------|-----------|
| Full Data | 79.6 | 79.6 | 79.6 |
| Dyn-Unc | 79.1 (↓0.5) | 78.5 (↓1.1) | 77.6 (↓2.0) |
| InfoBatch | 78.6 (↓1.0) | 78.2 (↓1.4) | 77.5 (↓2.1) |
| **PFB (Ours)** | **79.6 (±0.0)** | **79.2 (↓0.4)** | **78.2 (↓1.4)** |

### Ablation Study

**ResNet-18 on CIFAR-10/100 compared with other methods:**

| Method | CIFAR-10 30%/50%/70% | CIFAR-100 30%/50%/70% |
|--------|---------------------|----------------------|
| Random | 94.6/93.3/90.2 | 73.8/72.1/69.7 |
| InfoBatch | 95.6/95.1/94.7 | 78.2/78.1/76.5 |
| DivBS | 95.4/95.2/95.1 | 78.5/78.2/77.2 |
| **PFB** | **95.9/95.5/95.2** | **79.1/78.8/77.9** |
| Full Data | 95.6 | 78.2 |

**Training time comparison (ImageNet, ResNet-50, 40% pruning):**

| Method | Top-1 Acc | Training(h) | Overhead(h) | Total(n·h) | Reduction |
|--------|----------|------------|------------|-----------|-----------|
| Full Data | 76.4 | 13.9 | - | 55.6 | - |
| InfoBatch | 76.5 | 10.1 | 0.07 | 40.7 | 26.8% ↓ |
| DivBS | 76.4 | 11.2 | 0.72 | 47.6 | 14.4% ↓ |
| **PFB** | **76.9** | **9.2** | **0.06** | **37.1** | **33.2% ↓** |

### Key Findings

- **Lossless or positive gain**: PFB consistently outperforms full-data training at 30% pruning (ImageNet R50: +0.6%, CIFAR-100: +0.9%).
- **Maximum time savings**: 40% pruning yields a 33.2% reduction in training time with only 0.06h of additional overhead, far below DivBS (0.72h).
- **Generality across architectures**: Strong performance on both ResNet-50 and Swin-T.
- **Density outperforms loss/gradient**: Rare samples with low probability density better preserve generalization than high-loss samples.

## Highlights & Insights

- **Paradigm innovation**: The pruning decision is moved from *after* the forward pass to *within* it, fundamentally reshaping the computational cost structure.
- **Unique density perspective**: Measuring sample redundancy from a distributional standpoint more intrinsically captures sample information value than gradient- or loss-based metrics.
- **Elegant ADE design**: Cluster centers updated via EMA with weighted KDE enable adaptive tracking of the training distribution at negligible cost.
- **Positive-gain phenomenon**: Moderate pruning can improve performance, suggesting that redundant samples in the full dataset may interfere with learning.

## Limitations & Future Work

- The optimal choice of shallow sub-network depth (stage-1 vs. stage-2) may vary across architectures.
- Probability density estimation in very high-dimensional feature spaces may suffer from the curse of dimensionality, necessitating dimensionality reduction.
- Validation is limited to classification and segmentation; applicability to dense prediction tasks such as object detection remains unexplored.
- The number of cluster centers $N_C$ requires tuning as a hyperparameter.
- Effectiveness in self-supervised learning and large language model pre-training has yet to be verified.

## Related Work & Insights

- The comparison with InfoBatch is most illuminating: InfoBatch prunes by loss value but still requires a full forward pass, whereas PFB fundamentally reduces forward computation.
- Probability density importance can serve as a general data sampling strategy applicable to active learning, curriculum learning, and related settings.
- The "deep forward blocking" mechanism can be extended to early-exit strategies at inference time.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Dual innovation: forward blocking paradigm + probability density importance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers CIFAR-10/100/ImageNet, CNN/Transformer, and segmentation tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear with thorough analysis of computational overhead.
- **Value**: ⭐⭐⭐⭐⭐ Achieves genuinely lossless training acceleration: 33% time reduction with accuracy gains.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](../../NeurIPS2025/model_compression/casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[CVPR 2026\] A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos](../../CVPR2026/model_compression/a_paradigm_shift_fully_end-to-end_training_for_temporal_sentence_grounding_in_vi.md)
- [\[CVPR 2026\] Batch Loss Score for Dynamic Data Pruning](../../CVPR2026/model_compression/batch_loss_score_for_dynamic_data_pruning.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[ICCV 2025\] Scheduling Weight Transitions for Quantization-Aware Training](scheduling_weight_transitions_for_quantization-aware_training.md)

<!-- RELATED:END -->
