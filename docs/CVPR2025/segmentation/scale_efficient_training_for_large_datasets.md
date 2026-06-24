---
title: >-
  [Paper Note] Scale Efficient Training for Large Datasets
description: >-
  [CVPR 2025][Segmentation][Efficient Training] Proposes SeTa (Scale Efficient Training), a loss-based dynamic sample pruning framework. Through a three-step strategy consisting of random sampling for de-redundancy, loss clustering for difficulty division, and sliding window progressive curriculum learning, it achieves up to 50% reduction in training costs without performance loss across 11 datasets, 10 task categories, and 14 models.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Efficient Training"
  - "Dynamic Sample Pruning"
  - "Curriculum Learning"
  - "Data-Efficient"
  - "Sliding Window Strategy"
date: 2026-05-08
content_hash: 0b2bff59f8680ef6
---

# Scale Efficient Training for Large Datasets

**Conference**: CVPR 2025  
**arXiv**: [2503.13385](https://arxiv.org/abs/2503.13385)  
**Code**: [GitHub](https://github.com/mrazhou/SeTa)  
**Area**: Segmentation/General Training Acceleration  
**Keywords**: Efficient Training, Dynamic Sample Pruning, Curriculum Learning, Data-Efficient, Sliding Window Strategy

## TL;DR

Proposes SeTa (Scale Efficient Training), a loss-based dynamic sample pruning framework. Through a three-step strategy consisting of random sampling for de-redundancy, loss clustering for difficulty division, and sliding window progressive curriculum learning, it achieves up to 50% reduction in training costs without performance loss across 11 datasets, 10 task categories, and 14 models.

## Background & Motivation

Large-scale datasets are the foundation of deep learning. However, as the volume of data grows, the conflict between training efficiency and data scale becomes increasingly prominent. The existence of a large number of low-value samples causes computational waste during training. These low-value samples consist of three categories: (1) redundant/duplicate samples, which offer diminishing marginal information; (2) over-difficult samples, which consume substantial computation but contribute negligibly to model optimization; and (3) over-easy samples, which have been fully learned and no longer produce effective gradients.

Existing dynamic pruning methods (such as InfoBatch) only eliminate well-learned samples based on average loss, neglecting redundant and over-difficult samples. Static coreset selection methods require expensive preprocessing and suffer from poor generalizability across architectures.

The design philosophy of SeTa is to use loss as a computation-free difficulty proxy, simultaneously eliminating the three types of low-value samples through random sampling, loss clustering, and sliding window curriculum learning. Notably, as shown in Figure 1, continuing to increase data after the ToCa dataset size exceeds 3M yields saturated performance improvements, indicating that redundancy is an inherent problem of large-scale datasets.

## Method

### Overall Architecture

SeTa is a plug-and-play training acceleration framework that can be integrated by modifying only 3 lines of code. The workflow consists of three steps: (1) random sampling to remove redundant samples with a ratio of $r$; (2) performing K-means clustering on the remaining samples based on loss to obtain $k$ difficulty groups; (3) using a sliding window to progressively select sample groups from easy to difficult for training, followed by an annealing phase at the end.

### Key Design 1: Loss-Guided Sample Clustering

**Function**: Organize training samples hierarchically according to learning difficulty.

**Mechanism**: First, perform uniform downsampling with a ratio of $r$ to remove redundancy. Then, cluster the remaining samples using K-means based on their current loss value $l_i^t$: $\mathcal{C}^* = \arg\min \sum_{j=1}^{k} \sum_{x_i \in \mathcal{G}_j} \|l_i^t - c_j\|^2$. This clustering yields $k$ sample groups $\{\mathcal{G}_1, ..., \mathcal{G}_k\}$ sorted in ascending order of difficulty.

**Design Motivation**: Loss is a universal metric in deep learning, costing zero additional computation (already computed during training). Compared to the division method of InfoBatch (higher/lower than average), K-means provides finer-grained difficulty stratification, laying the foundation for subsequent curriculum learning. Random sampling is performed before clustering to avoid extra cost for information density evaluation.

### Key Design 2: Sliding Window Curriculum Learning

**Function**: Progressively expose training samples from easy to hard, while simultaneously excluding over-easy and over-difficult samples.

**Mechanism**: Define the window size $w = \lceil \alpha k \rceil$ (where $\alpha \in (0,1]$ controls the ratio of selection at each step). The window position $s_t = n \mod (k - w + 1)$ increases cyclically, selecting difficulty groups from index $s_t$ to $s_t + w - 1$. The window slides periodically—resetting to the simple groups after reaching the most difficult group, achieving a cyclic curriculum from easy to difficult over multiple rounds.

**Design Motivation**: Training exclusively on simple samples leads to underfitting, while training only on difficult samples leads to optimization instability. The key advantage of the sliding window is to simultaneously exclude the "over-easy group" and the "over-difficult group"—at any moment, only the intermediate-difficulty groups that lie on the model's current learning frontier are selected. The cyclic mechanism allows the model to repeatedly consolidate simple knowledge while progressively adapting to more difficult samples.

### Key Design 3: Partial Annealing Strategy

**Function**: Reduce optimization bias introduced by localized sample selection and ensure stable convergence.

**Mechanism**: In the final stage of training, instead of using the sliding window, random sampling is conducted from all samples with probability $r$: $\mathcal{S}_t^{anneal} = \{x_i | x_i \in \mathcal{G}, u_i < r\}$, where $u_i \sim \text{Uniform}(0,1)$.

**Design Motivation**: Although highly efficient, the sliding window can introduce distribution bias (constantly lacking extremely easy and extremely difficult samples). Partial annealing restores exposure to the full distribution in the final phase, yet retains efficiency through the sampling rate $r$. Compared to InfoBatch's full-dataset annealing, partial annealing is more efficient.

### Loss

SeTa does not modify any task-specific loss functions; it only changes the subset of samples $S_t$ participating in training per epoch. Its training savings are equivalent to the pruning ratio $\rho_O \approx \bar{\rho}$ (since the data selection overhead $O_d \ll O_m$).

## Key Experimental Results

### Main Results: Large-Scale Synthetic Datasets

| Dataset | Task | Baseline | SeTa Pruning Rate | SeTa Performance |
|--------|------|----------|-----------|---------|
| ToCa (3M) | Image Captioning COCO CIDEr | 112.7 | 50% | **114.3** (+1.6) |
| SS1M (3M) | Zero-shot Captioning CIDEr | 91.2 | 50% | **92.1** (+0.9) |
| ST+MJ (15M) | Scene Text Recognition Avg Acc | 96.3 | 50% | **96.5** (+0.2) |

### ImageNet Classification Experiments

| Method | Backbone | Pruning Rate | Top-1 Acc |
|------|------|--------|-----------|
| Baseline | ResNet50 | 0% | 76.4 |
| InfoBatch | ResNet50 | 40% | 76.3 (-0.1) |
| **SeTa** | ResNet50 | 40% | **76.5** (+0.1) |
| Baseline | ViT-S | 0% | 79.9 |
| **SeTa** | ViT-S | 50% | **80.0** (+0.1) |
| Baseline | Vim-S | 0% | 80.5 |
| **SeTa** | Vim-S | 40% | **80.7** (+0.2) |

### CIFAR-100 Comparison

| Method | 30% Pruning | 50% Pruning | 70% Pruning |
|------|---------|---------|---------|
| Static Random | 73.8 (-4.4) | 72.1 (-6.1) | 69.7 (-8.5) |
| InfoBatch | 77.5 (-0.7) | 76.2 (-2.0) | - |
| **SeTa** | **78.7** (+0.5) | **78.0** (-0.2) | **76.3** (-1.9) |

### Key Findings

- On synthetic datasets of scale 3M+, SeTa under a 50% pruning rate achieves not only lossless performance but even performance improvements (+0.2 to +1.6), indicating that eliminating low-value samples has positive effects.
- Extremely strong generalization across architectures: highly effective on CNN (ResNet), Transformer (ViT, Swin), and Mamba (Vim).
- Outstanding cross-task generalizability: covers 10 categories of tasks such as classification, captioning, segmentation, retrieval, stereo matching, and geolocalization.
- Even under an extreme pruning rate of 70%, performance degradation is minimal (only -1.9 on CIFAR-100).
- In LLM instruction-tuning (LLaMA-7B on Alpaca 52K), 50% pruning still maintains the MT-bench score, indicating efficacy on smaller datasets as well.

## Highlights & Insights

1. **Minimalist Design**: A plug-and-play framework integrated by modifying only 3 lines of code, offering extremely high engineering practicality.
2. **Unified Treatment of Three Types of Low-Value Samples**: Random sampling to remove redundancy and a sliding window to exclude over-easy/over-difficult samples, which is more comprehensive than InfoBatch which only handles over-easy samples.
3. **Beyond Lossless**: Performance actually improves with 50% pruning in multiple scenarios, indicating that too many low-quality samples actually hinder model training.

## Limitations & Future Work

- While universal, using loss as a difficulty proxy might not be fine-grained enough—high loss may stem from annotation noise rather than genuine learning difficulty.
- K-means clustering assumes that the loss distribution is unimodal or separable, which might be suboptimal for multimodal distributions.
- Hyperparameters of the sliding window ($k$, $\alpha$, $r$, and the number of annealing epochs) require tuning, although the paper claims to use default settings.
- For extremely small datasets (such as CIFAR-10, 50K), the necessity of random sampling for de-redundancy is weak.

## Related Work & Insights

- **InfoBatch (ICLR'24)**: A pioneer in dynamic pruning based on average loss. SeTa further identifies and excludes redundant and over-difficult samples through K-means clustering and a sliding window.
- **EfficientTrain++**: A sample-level method (frequency cropping + curriculum learning), whose cross-domain generalizability is inferior to SeTa.
- **Curriculum Learning**: SeTa's sliding window is an efficient implementation of curriculum learning. The cyclic easy-to-hard approach avoids the monotonic progressive limitations of traditional curriculum learning.

## Rating

⭐⭐⭐⭐ — Extremely minimalist method yet with exceptionally broad coverage (11 datasets × 10 tasks × 14 models), offering high practical value. The core insight—simultaneously removing three types of low-value samples—is more comprehensive than existing approaches. The developer friendliness of its 3-line-code integration is a genuine plus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping](../../ICCV2025/segmentation/ragnet_large-scale_reasoning-based_affordance_segmentation_benchmark_towards_gen.md)
- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[CVPR 2025\] F-LMM: Grounding Frozen Large Multimodal Models](f-lmm_grounding_frozen_large_multimodal_models.md)
- [\[CVPR 2026\] Making Training-Free Diffusion Segmentors Scale with the Generative Power](../../CVPR2026/segmentation/making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)
- [\[CVPR 2025\] ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect](shiftwiseconv_small_convolutional_kernel_with_large_kernel_effect.md)

</div>

<!-- RELATED:END -->
