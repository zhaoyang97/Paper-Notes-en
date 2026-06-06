---
title: >-
  [Paper Note] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers
description: >-
  [AAAI2026][Model Compression][class-incremental learning] This paper proposes Sequential Learning with Drift Compensation (SLDC), which learns latent space transformation operators (linear / weakly nonlinear) to compensa…
tags:
  - "AAAI2026"
  - "Model Compression"
  - "class-incremental learning"
  - "distribution drift"
  - "vision transformer"
  - "knowledge distillation"
  - "sequential fine-tuning"
date: 2026-05-08
content_hash: 099bf7afca6aa341
---

# Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers

**Conference**: AAAI2026
**arXiv**: [2511.09926](https://arxiv.org/abs/2511.09926)  
**Code**: [raoxuan98-hash/sldc](https://github.com/raoxuan98-hash/sldc)  
**Area**: Model Compression
**Keywords**: class-incremental learning, distribution drift, vision transformer, knowledge distillation, sequential fine-tuning

## TL;DR

This paper proposes Sequential Learning with Drift Compensation (SLDC), which learns latent space transformation operators (linear / weakly nonlinear) to compensate for distribution drifts induced by sequential fine-tuning of pre-trained ViTs in class-incremental learning. Combined with knowledge distillation, the approach achieves performance close to the joint-training upper bound.

## Background & Motivation

Recent work on pre-trained ViT-based class-incremental learning (CIL) has demonstrated that sequential fine-tuning (SeqFT) followed by Gaussian-approximation-based classifier refinement is an effective paradigm (e.g., SLCA/SLCA++). However, this paradigm has a critical flaw: **sequentially optimizing the shared backbone parameters causes distribution drift**—the class feature distributions learned from previous tasks no longer align with the feature space of the updated model, leading to progressively degraded classifier performance as more tasks arrive.

Most existing methods attempt to *prevent* representation drift via distillation, model ensembling, or gradient projection. This paper takes a different perspective, focusing on **how to compensate for the negative effects of drift after it has already occurred**—a viewpoint that remains largely unexplored in PTM-based CIL.

## Core Problem

1. **Distribution drift**: After fine-tuning on a new task, the Gaussian distributions ($\mu_c$, $\Sigma_c$) of old classes no longer align with the current feature space, introducing bias when synthesizing features for classifier refinement.
2. **Overfitting of nonlinear transformations**: Directly training an MLP to map between consecutive feature spaces leads to overfitting, yielding worse distribution accuracy than a linear mapping.
3. **Limited samples**: Under the exemplar-free setting, no old-task data is retained, which may limit the diversity of samples available for operator estimation.

## Method

### Overall Architecture

SLDC consists of three stages:
1. **Sequential fine-tuning**: Fine-tune the ViT backbone on the current task using LoRA (optionally with distillation → SeqKD).
2. **Distribution compensation**: Learn a transformation operator that aligns old-class Gaussian distributions to the new feature space.
3. **Classifier refinement**: Sample synthetic features from the compensated Gaussian distributions to train a unified classifier.

### Latent Space Transformation Operator

A transformation operator $\mathcal{P}_{t-1 \to t}: \mathcal{F}_{t-1} \to \mathcal{F}_t$ is defined to map the feature space of task $t-1$ to that of task $t$, ideally propagating the first-order (mean) and second-order (covariance) moments of old-class Gaussian distributions to the new space.

Since the full input space is inaccessible in practice, the operator is approximated using only current-task data $\mathcal{D}_t$ and two frozen models $\mathcal{F}_{t-1}$ and $\mathcal{F}_t$.

### α₁-SLDC (Linear Variant)

Features extracted from both models on current-task data are L2-normalized, and a linear operator is solved via regularized least squares:

$$\mathbf{A}_t = \tilde{F}^t (\tilde{F}^{t-1})^\top \left( \tilde{F}^{t-1} (\tilde{F}^{t-1})^\top + \gamma_{\alpha_1} I_d \right)^{-1}$$

To handle small-sample scenarios, a heuristic re-weighting smoothing is applied: $\mathbf{A}_t = (1-w)\mathbf{A}_t + wI_d$, where $w = \exp(-n_t / (\alpha_{\text{temp}} d))$.

The compensation of old-class Gaussian distributions admits a closed-form update: $\mu_c \leftarrow \mathbf{A}_t \mu_c$, $\Sigma_c \leftarrow \mathbf{A}_t \Sigma_c \mathbf{A}_t^\top$.

### α₂-SLDC (Weakly Nonlinear Variant)

Assuming the ideal transformation operator lies between purely linear and fully nonlinear, a weakly nonlinear transformation is constructed:

$$\mathcal{T}(f) = c_1 \mathbf{A} f + c_2 \psi(f)$$

where $c_1 + c_2 = 1$, $\mathbf{A}$ is a learnable matrix, and $\psi(f)$ is a two-layer ReLU MLP. A regularization term $\gamma_{\alpha_2}(c_1 - 1)^2$ is added to the objective to constrain the contribution of the nonlinear component and prevent overfitting.

Since the weakly nonlinear transformation no longer admits closed-form Gaussian propagation, **Monte Carlo sampling** is used to estimate the compensated distribution: $N \gg d$ samples are drawn from the old Gaussian, passed through $\mathcal{T}$, and the mean and covariance are re-estimated.

### β₁/β₂-SLDC (Distillation-Enhanced Variants)

Feature-level knowledge distillation is incorporated into the fine-tuning stage to constrain representation updates:

$$\mathcal{L}_{\text{All}} = \mathcal{L}_{\text{CE}} + \gamma_{\text{kd}} \mathcal{L}_{\text{KD}} + \gamma_{\text{norm}} \mathcal{L}_{\text{Norm}}$$

Distillation mitigates representation drift while SLDC compensates residual distribution drift; the two are complementary.

### Auxiliary Data Enhancement (ADE)

When task data is limited, unlabeled auxiliary data from any source can be used to enrich the estimation of the transformation operator. ADE requires no labels and retains no old-task data, making it compatible with the exemplar-free CIL framework.

## Key Experimental Results

10-task CIL evaluations are conducted on four datasets (CUB-200, Cars-196, CIFAR-100, ImageNet-R) using two pre-trained ViT-B/16 models (MoCo-V3 self-supervised / ImageNet-21K supervised).

**Main results with MoCo-V3 pre-training (Last-Acc):**

| Method | CUB-200 | Cars-196 | CIFAR-100 | ImageNet-R |
|--------|---------|----------|-----------|------------|
| Joint-Training (upper bound) | 81.82 | 81.16 | 88.86 | 75.95 |
| SeqFT (baseline) | 64.40 | 60.42 | 73.36 | 61.37 |
| α₂-SLDC | 78.98 (+14.58) | 77.53 (+17.11) | 81.75 (+8.39) | 71.38 (+10.01) |
| β₂-SLDC | **81.82** (+4.85) | **80.10** (+6.23) | 85.16 (+4.81) | **73.01** (+6.08) |
| β₂-SLDC + ADE | **82.32** | **80.61** | **86.12** | **73.14** |
| SLCA++ | 75.48 | 69.71 | 84.77 | 69.01 |
| CoFiMA | 77.65 | 69.51 | 87.44 | 70.87 |

**Key Findings:**

- β₂-SLDC achieves 81.82% on CUB-200, exactly matching joint training.
- Across all datasets, the gap between β-SLDC and joint training ranges from +0.50% to −3.29%.
- α₂-SLDC yields large gains over the SeqFT baseline (+8–17%), demonstrating the effectiveness of distribution compensation.
- Direct MLP-based transformation (MLPDC) degrades performance when combined with distillation (e.g., −4.41% on CUB-200), confirming the overfitting concern.
- ADE further improves performance in most settings.

## Highlights & Insights

1. **Novel perspective**: Compensating drift rather than preventing it represents a distinctive viewpoint in the CIL literature.
2. **Strong theory–practice alignment**: The weakly nonlinear assumption is supported by NTK theory, and the linear variant admits closed-form Gaussian propagation guarantees.
3. **Performance approaching the joint-training upper bound**: This is a landmark result in CIL research, validating the viability of the SeqFT + distribution compensation paradigm.
4. **Strong generality**: SLDC can be integrated as a plug-and-play module into existing SeqFT methods.
5. **Insightful observation on MLP overfitting**: The finding reveals that the degree of nonlinearity in distribution compensation must be carefully controlled, and that weak nonlinearity is a better trade-off.

## Limitations & Future Work

1. **Gaussian assumption**: Deep feature distributions are not necessarily Gaussian; for multimodal or long-tailed class distributions, the Gaussian approximation may be insufficiently accurate.
2. **Task-data dependency of the transformation operator**: If the current task data distribution differs greatly from old-task distributions, the estimated operator may not generalize well to old classes.
3. **Limited effectiveness of the linear variant on fine-grained datasets**: α₁-SLDC even degrades performance by 7.75% on Cars-196 under ImageNet-21K pre-training, indicating that the linear assumption does not always hold.
4. **Fixed MLP architecture in the weakly nonlinear variant**: The choice of a two-layer ReLU MLP is relatively coarse; more sophisticated architecture search may be beneficial.
5. **ADE requires additional data**: Although no annotation is needed, the acquisition and selection of auxiliary data still requires careful design.

## Related Work & Insights

| Dimension | SLCA/SLCA++ | CoFiMA | RanPAC | SLDC |
|-----------|-------------|--------|--------|------|
| Backbone update | Low LR / LoRA | Model averaging | Frozen | LoRA + distillation |
| Drift handling | Implicit (slow update) | Model fusion | None (frozen) | Explicit compensation operator |
| Classifier refinement | Gaussian sampling | Gaussian sampling | Random projection | Compensated Gaussian sampling |
| Computational cost | Low | Medium | Low | Medium (operator estimation) |
| Task ID dependency | No | No | No | No |

The core advantage of SLDC lies in explicitly modeling the evolution of the feature space, serving as a complement to distillation rather than a simple replacement.

1. **Generalizability of distribution compensation**: Beyond Gaussian assumptions, it is worth exploring compensation for more complex distributions (e.g., Gaussian mixtures, normalizing flows).
2. **Design philosophy of weak nonlinearity**: The principle of finding a balance between overfitting and underfitting is worth borrowing in other domains such as domain adaptation and feature alignment in transfer learning.
3. **Integration with prompt-based CIL**: SLDC currently operates within the SeqFT paradigm; whether it can be combined with methods such as L2P/CODA-Prompt warrants further exploration.
4. **Implications of matching joint training**: This result suggests that the core bottleneck in CIL may not be catastrophic forgetting per se, but rather the failure to properly account for distribution drift.

## Rating
- Novelty: ⭐⭐⭐⭐ (Compensating drift is a fresh angle; the weakly nonlinear assumption offers original insight)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets × 2 pre-trained models × multiple variant comparisons; rigorous and comprehensive experimental design)
- Writing Quality: ⭐⭐⭐⭐ (Clear definitions, complete derivations; notation-heavy but well-organized)
- Value: ⭐⭐⭐⭐ (Matching the joint-training upper bound is a milestone result; the method is plug-and-play)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[ICML 2026\] AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning](../../ICML2026/model_compression/area_attribute_extraction_and_aggregation_for_clip-based_class-incremental_learn.md)
- [\[AAAI 2026\] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)

</div>

<!-- RELATED:END -->
