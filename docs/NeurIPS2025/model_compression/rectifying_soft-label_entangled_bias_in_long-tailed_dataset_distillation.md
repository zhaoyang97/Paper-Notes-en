---
title: >-
  [Paper Note] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation
description: >-
  [NeurIPS 2025][Model Compression][Dataset Distillation] This paper identifies a dual entangled bias in soft labels within long-tailed dataset distillation — originating from both the distillation model and the distilled images — and proposes ADSA, an Adaptive Soft-label Alignment module that eliminates this bias via post-hoc calibration in logit space. As a plug-and-play module, ADSA integrates seamlessly into existing distillation pipelines, achieving up to 11.8% accuracy improvement on tail classes on ImageNet-1k-LT.
tags:
  - NeurIPS 2025
  - Model Compression
  - Dataset Distillation
  - Long-Tailed Distribution
  - Soft Label Calibration
  - Generalization Bound
  - Logit Adjustment
date: 2026-05-08
content_hash: 92741eaadb448542
---

# Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2511.17914](https://arxiv.org/abs/2511.17914)
**Code**: [Available](https://github.com/j-cyoung/ADSA_DD.git)
**Area**: Dataset Distillation / Long-Tailed Learning
**Keywords**: Dataset Distillation, Long-Tailed Distribution, Soft Label Calibration, Generalization Bound, Logit Adjustment

## TL;DR

This paper identifies a dual entangled bias in soft labels within long-tailed dataset distillation — originating from both the distillation model and the distilled images — and proposes ADSA, an Adaptive Soft-label Alignment module that eliminates this bias via post-hoc calibration in logit space. As a plug-and-play module, ADSA integrates seamlessly into existing distillation pipelines, achieving up to 11.8% accuracy improvement on tail classes on ImageNet-1k-LT.

## Background & Motivation

Dataset distillation aims to compress large-scale datasets into compact synthetic datasets that retain critical information, thereby reducing storage and training costs. However, existing methods focus primarily on balanced datasets and perform poorly under long-tailed distributions commonly found in real-world scenarios.

**Core Problem: Bias in Soft Labels under Long-Tailed Distillation**

Soft labels are a key component of recent dataset distillation methods (SRe2L, EDC, GVBSM, etc.) and contribute significantly to performance. Under long-tailed distributions, however, soft labels introduce systematic bias.

The authors first derive an **imbalance-aware generalization bound** (Theorem 3.1). Under the long-tailed assumption ($p_{tr}(x|y) = p_{te}(x|y)$ but $p_{tr}(y) \neq p_{te}(y)$), the discrepancy term $R_{dd}$ decomposes as:

$$R_{dd} = D_{KL}(p_{te}(y|x) \| p_{dd}(y|x)) + D_{KL}(p_{te}(x) \| p_{dd}(x)) + \text{const}$$

The first term indicates that the posterior $p_{dd}(y|x)$ learned from the distilled dataset should align with the test-set label distribution — a condition that long-tailed soft labels systematically violate.

**Perturbation Analysis Reveals Dual Bias Sources**

Through a carefully designed experiment, the distillation pipeline is decomposed into two independent components (image generation and label generation), with four configurations that vary the degree of imbalance:

- Config (1): Imbalanced images + balanced model annotation → bias from images
- Config (2): Balanced images + imbalanced model annotation → bias from model (larger effect)
- Config (3): Both imbalanced → dual bias
- Config (4): Both balanced → baseline

Key finding: both imbalanced images and imbalanced models cause soft labels to be overconfident on head classes and underconfident on tail classes. The observed bias approximately decomposes as:

$$p_{DD}^{obs}(y|x) = p_{DD}^{target}(y|x) + \epsilon_T(y|x) + \epsilon_I(y|x)$$

## Method

### Overall Architecture

ADSA is a post-hoc module that does not participate in model training or image distillation. Its core mechanism is to use the distilled images themselves as a "validation set" to diagnose and calibrate class-level output imbalance.

### Key Designs

#### 1. **Logit Calibration Preserves Semantic Relationships**

Following the logit calibration approach of Menon et al., adjustments are made in logit space to preserve inter-class semantic relationships:

$$p(y|x;\tau) = \frac{\exp(f_y(x) - \tau \log \pi_y)}{\sum_{y' \in [K]} \exp(f_{y'}(x) - \tau \log \pi_{y'})}$$

where $\pi_y$ is the empirical frequency of class $y$ and $\tau$ is the calibration hyperparameter.

**Design Motivation**: Directly modifying probability values risks destroying the inter-class relational information encoded in soft labels — which constitutes their core value. A translation in logit space smoothly adjusts all dimensions while preserving relative relationships.

#### 2. **Distilled Images as a Validation Set for Bias Diagnosis**

Since distilled images exhibit distributional shift relative to the original training data, they can serve as a hold-out validation set to detect class-level bias in model outputs. The class-averaged soft label is computed as:

$$p(\bar{y}=i|x;\tau) = \mathbb{E}_{x \sim \mathcal{D}_i}[p(y=i|x;\tau)]$$

#### 3. **Adaptive Calibration Strength Optimization**

The optimal $\tau^*$ is found by minimizing the variance of per-class confidence:

$$\tau^* = \arg\min_\tau \sqrt{\frac{1}{K} \sum_{i=0}^{K-1} \left(p(\bar{y}=i|x;\tau) - \frac{1}{K}\sum_{j=0}^{K-1} p(\bar{y}=j|x;\tau)\right)^2}$$

**Design Motivation**: Under a balanced test set, an unbiased model should exhibit approximately equal average confidence across all classes. Minimizing variance automatically identifies the optimal calibration strength for bias elimination.

### Loss & Training

ADSA involves no training — it is a purely post-hoc module. A single one-dimensional optimization is performed at the annotation stage to find $\tau^*$, after which the original soft labels are replaced with $p(y|x;\tau^*)$.

Three key properties:
1. Eliminates entangled bias, approximating the true test-set posterior distribution
2. Preserves inter-class semantic relationships, maintaining the informativeness of soft labels
3. Adaptively adjusts to different datasets, IPC values, and imbalance factors

## Key Experimental Results

### Main Results

Performance of various methods with and without ADSA on CIFAR-10-LT (Top-1 Accuracy %):

| Method | IPC=10, IF=100 | IPC=50, IF=50 | IPC=50, IF=100 | Gain |
|---|---|---|---|---|
| EDC | 50.9 | 65.6 | 56.0 | - |
| EDC+ADSA | **68.7** | **76.4** | **74.8** | +17.8/+10.8/+18.8 |
| GVBSM | 29.4 | 37.2 | 30.9 | - |
| GVBSM+ADSA | **40.4** | **51.4** | **46.9** | +11.0/+14.2/+16.0 |
| SRe2L | 22.6 | 36.6 | 34.6 | - |
| SRe2L+ADSA | **25.9** | **38.8** | **45.3** | +3.3/+2.2/+10.7 |

Results on ImageNet-1k-LT (EDC, IPC=50):

| Class Group | EDC | EDC+ADSA | Gain |
|---|---|---|---|
| Head | 55.5 | 51.3 | -4.2 |
| Mid | 32.3 | 38.1 | +5.8 |
| Tail | 12.4 | **24.2** | **+11.8** |
| Overall | 38.6 | **41.4** | +2.8 |

### Ablation Study

Compatibility with alternative distillation and long-tailed methods (CIFAR-10-LT, IF=50):

| Method | IPC=10 | IPC=50 | Note |
|---|---|---|---|
| MTT | 33.4 | 53.0 | Baseline |
| MTT+soft label | 37.9 | 51.4 | Naïve soft labels hurt performance |
| MTT+ADSA | **40.4** | **56.6** | Calibration yields consistent gains |
| DREAM | 56.0 | 58.6 | Baseline |
| DREAM+ADSA | **59.9** | **65.7** | Substantial improvement |

### Key Findings

1. **Larger imbalance factors yield larger ADSA gains**: At IF=100, EDC on CIFAR-10-LT improves from 56.0% to 74.8% (+18.8%).
2. **Tail class improvements are most pronounced**: On ImageNet-LT, tail class accuracy nearly doubles from 12.4% to 24.2%.
3. **Effective even under limited soft-label budgets**: ADSA provides positive gains even when soft labels are generated for only 1 epoch.
4. **Contrast with naïve soft labels**: Uncalibrated soft labels can actually degrade performance in long-tailed settings (MTT+soft label).

## Highlights & Insights

1. **Elegant experimental design for bias decomposition**: By independently controlling the source models for image distillation and label generation, the contributions of the two bias sources are quantitatively separated — a generalizable analytical tool.
2. **Seamless theory-experiment integration**: Method design is guided by the generalization bound, with perturbation experiments validating theoretical predictions.
3. **Extreme simplicity**: One-dimensional optimization, post-hoc processing, plug-and-play — a simple method with substantial effect.
4. **Data-centric perspective**: Rather than modifying model architectures or loss functions, the method directly calibrates the data (soft labels), offering a novel angle on long-tailed learning.

## Limitations & Future Work

1. Head class accuracy occasionally decreases in exchange for large tail class gains; whether a Pareto improvement is achievable remains open.
2. The additive bias decomposition assumption does not hold perfectly in practice (dual bias is not always the worst case in experiments).
3. The current method assumes a balanced test set; extension to imbalanced test distributions warrants investigation.
4. The global scalar $\tau$ could be extended to per-class calibration parameters for finer-grained adjustment.

## Related Work & Insights

- Builds upon the logit adjustment paradigm in long-tailed learning, but innovatively applies it to the calibration of distilled soft labels.
- Unlike LTDD, which focuses on parameter distribution alignment, this paper directly targets soft-label bias, yielding a lighter and more general approach.
- The insight of using distilled images as a validation set is particularly inspiring — the distributional shift of synthetic data is repurposed as a diagnostic tool.

## Rating

- Novelty: ⭐⭐⭐⭐ (Problem analysis is novel; the method is simple yet built on deep insights)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, methods, and IPC/IF combinations; complete ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Theoretical derivations are clear; experimental design and analysis are logically rigorous)
- Value: ⭐⭐⭐⭐ (Plug-and-play module with practical applicability; provides key insights for long-tailed distillation)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](../../AAAI2026/model_compression/rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)
- [\[ICCV 2025\] Heavy Labels Out! Dataset Distillation with Label Space Lightening](../../ICCV2025/model_compression/heavy_labels_out_dataset_distillation_with_label_space_lightening.md)
- [\[NeurIPS 2025\] Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)

<!-- RELATED:END -->
