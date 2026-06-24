---
title: >-
  [Paper Note] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning
description: >-
  [CVPR 2025][Model Compression][Few-Shot Class-Incremental Learning] This paper proposes the Tri-WE method, which updates the entire model (instead of freezing the feature extractor) by interpolating three classification heads—base, previous session, and current session—in the weight space, and mitigates forgetting in few-shot scenarios using Amplified Data Knowledge Distillation (ADKD), achieving SOTA performance in FSCIL on miniImageNet/CUB200/CIFAR100.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Few-Shot Class-Incremental Learning"
  - "Weight-Space Ensemble"
  - "Knowledge Distillation"
  - "Catastrophic Forgetting"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 7289e161748f2408
---

# Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning

**Conference**: CVPR 2025  
**arXiv**: [2506.15720](https://arxiv.org/abs/2506.15720)  
**Code**: None  
**Area**: Continual Learning / Few-Shot Class-Incremental Learning  
**Keywords**: Few-Shot Class-Incremental Learning, Weight-Space Ensemble, Knowledge Distillation, Catastrophic Forgetting, Data Augmentation

## TL;DR

This paper proposes the Tri-WE method, which updates the entire model (instead of freezing the feature extractor) by interpolating three classification heads—base, previous session, and current session—in the weight space, and mitigates forgetting in few-shot scenarios using Amplified Data Knowledge Distillation (ADKD), achieving SOTA performance in FSCIL on miniImageNet/CUB200/CIFAR100.

## Background & Motivation

**Background**: Few-Shot Class-Incremental Learning (FSCIL) requires the model to continually extend its classification capabilities given only a few samples of new classes in each incremental session. Existing mainstream methods generally fall into two categories: (1) freezing the feature extractor and using only class means as new class classifiers (CEC, ALICE); (2) updating a subset of parameters or ensembling multiple classifiers (S3C, SAVC, BiDistill).

**Limitations of Prior Work**: (1) Freezing the feature extractor restricts the model's adaptation to new classes; (2) updating the entire model easily leads to catastrophic forgetting and overfitting (as shown in Figure 2, where the base class accuracy drops significantly after updating the model); (3) although Knowledge Distillation (KD) is a common anti-forgetting technique in CIL, in few-shot scenarios, KD tends to overfit to limited samples, resulting in the distillation of biased knowledge rather than generalized knowledge.

**Key Challenge**: Updating the model improves new class performance but degrades old class performance, whereas keeping the model frozen limits adaptation to new classes. This is fundamentally the stability-plasticity dilemma.

**Goal**: To allow the entire model to be updated while effectively mitigating catastrophic forgetting and overfitting, thereby balancing stability and plasticity.

**Key Insight**: Inspired by WiseFT (weight interpolation between fine-tuned and original CLIP models), the generalization capability of the base model, the accumulated knowledge of the previous session model, and the new class adaptability of the current session model are fused through weight-space interpolation. Concurrently, data mixup augmentation is utilized to enrich the data sources for KD.

**Core Idea**: Tripartite weight-space ensemble (weight interpolation of base + previous + current classification heads) + augmented data knowledge distillation (KD performed after mixing few-shot data), enabling a smooth transition rather than abrupt changes in the model during continual learning.

## Method

### Overall Architecture

In the $t$-th incremental session, the model is initialized from the $(t-1)$-th session, given the $N$-way $K$-shot new class data $\mathcal{D}^{(t)}$ and the old class prototype buffer $\mathcal{M}$. The weights of three classification heads are interpolated via Tri-WE to form a unified classification head $h_\phi^{(t)}$, which is trained using $\mathcal{L}_{Cls}$ and $\mathcal{L}_{Cls-Old}$. Concurrently, few-shot data is augmented and subjected to ADKD (feature + logit distillation) to prevent forgetting. Only a single classification head is required during deployment.

### Key Designs

1. **Tripartite Weight-Space Ensemble (Tri-WE)**:

    - **Function**: Fuses the classification heads of three session models in the weight space to achieve continuous and smooth knowledge transfer.
    - **Mechanism**: Maintains three classification heads: $h_{\phi_0}$ (fixed from the base session), $h_{\phi_{old}}^{(t)}$ (initialized from session $t-1$, focusing on old classes), and $h_{\phi_{all}}^{(t)}$ (covering all classes). Interpolation is performed under three scenarios based on class types: base classes (tripartite weighting $\bar\alpha_1 \phi_0^n + \bar\alpha_2 \phi_{old}^n + \bar\alpha_3 \phi_{all}^n$), intermediate session old classes (bipartite weighting $\bar\alpha_4 \phi_{old}^n + \bar\alpha_5 \phi_{all}^n$), and new classes (directly using $\phi_{all}^n$). The interpolation weights are automatically adjusted via two learnable scalars $\alpha_1, \alpha_2$ (initialized to 1.0).
    - **Design Motivation**: The weight space of the base model possesses excellent generalization due to extensive training on base data, the previous model carries knowledge of all previously learned classes, and the current model adapts to new classes. Tripartite interpolation avoids abrupt shifts in decision boundaries while avoiding additional inference computational overhead (only a single $h_\phi^{(t)}$ is deployed).

2. **Amplified Data Knowledge Distillation (ADKD)**:

    - **Function**: Distills generalized knowledge from the previous model to the current model while avoiding overfitting to the few-shot data.
    - **Mechanism**: Augments $NK$ few-shot training samples to $16NK$ samples using random mixing (e.g., CutMix, MixUp). Two levels of distillation losses are computed on the augmented data: feature-level $\mathcal{L}_{feat} = \mathbb{E}\|g_\theta^{(t-1)}(x) - g_\theta^{(t)}(x)\|_2$ and logit-level $\mathcal{L}_{logit} = \mathbb{E}[-f_{\theta,\phi}^{(t-1)}(x) \log f_{\theta,\phi}^{(t)}(x)]$. The parameters of the previous model are frozen.
    - **Design Motivation**: Directly applying KD on a small set of original samples leads to overfitting to specific patterns of those samples, failing to distill generalized knowledge. Data mixup augmentation generates a richer data distribution, enabling KD to extract the generalization capabilities of the previous model.

3. **Base Session Training Enhancement**:

    - **Function**: Ensures the base model possesses strong generalization capability.
    - **Mechanism**: Employs the base training technique of ALICE, additionally adding a geometric classification auxiliary head—applying $B$ types of geometric transformations (rotations, etc.) to the inputs, and training an auxiliary classifier to predict the transformation type. This head is used only during base training and is disabled in incremental sessions.
    - **Design Motivation**: Geometric transformation classification helps the model learn the intrinsic structure of the data, improving its generalization capability to new tasks.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{Cls} + \gamma_1 \mathcal{L}_{Cls-Old} + \gamma_2 \mathcal{L}_{ADKD}$, where $\gamma_1 = 1.2, \gamma_2 = 10.0$. In the incremental sessions, the learning rate for the feature extractor is 0.001, and for the classification head \phi_{all} is 0.1. The backbone is ResNet18, and the SGD optimizer is used.

## Key Experimental Results

### Main Results

**miniImageNet (9 sessions)**:

| Method | Session 0 | Session 8 (Final) | Avg |
|------|----------|----------------|-----|
| ALICE (ECCV'22) | 80.60 | 55.70 | 63.99 |
| NC-FSCIL (ICLR'23) | 84.02 | 58.31 | 67.82 |
| SAVC (CVPR'23) | 81.12 | 57.11 | 67.05 |
| OrCo (CVPR'24) | 83.30 | 58.08 | 67.12 |
| **Ours** | **84.13** | **60.13** | **70.62** |

The final session shows an improvement of +1.82 over NC-FSCIL, and +2.05 over OrCo.

### Ablation Study

SOTA performance is likewise achieved on CUB200 and CIFAR100. On CUB200, base/last/avg accuracies all outperform NC-FSCIL, SAVC, and YourSelf.

### Key Findings

- Updating the entire model indeed improves new class performance (upper part of Figure 2); the key lies in how to manage forgetting.
- The tripartite interpolation of Tri-WE is more suitable for multi-session incremental scenarios than bipartite interpolation (WiseFT).
- The learnable interpolation weights $\alpha_1, \alpha_2$ can automatically adapt to the requirements of different sessions.
- KD with augmented data performs significantly better than KD using original data.
- There is a consistent and significant improvement on the final session compared to all previous methods.

## Highlights & Insights

- The **weight-space ensemble** concept is simple yet effective—matching the three classification heads in the weight space rather than the output space, incurring no extra inference overhead.
- Extending WiseFT from bipartite to tripartite (by incorporating the base model anchor point) fits the multi-session characteristics of incremental learning.
- ADKD effectively enriches KD data sources using simple CutMix/MixUp, with an extremely low implementation cost.
- The design of class-stratified interpolation (using different interpolation combinations for base classes, intermediate old classes, and new classes) reflects precise judgment regarding the sources of knowledge for different categories.

## Limitations & Future Work

- A prototype still needs to be maintained for each old class, causing memory overhead to scale linearly with the number of sessions.
- Whether the base training strategy using geometric auxiliary classification is effective for all datasets is not fully discussed.
- Performance on longer incremental sequences (e.g., 20+ sessions) remains unverified.
- Although the feature extractor is updated, its learning rate is extremely small (0.001), which is closer to 'fine-tuning' rather than a 'true update'.
- The sensitivity of hyperparameters such as $\gamma_1, \gamma_2$ is not analyzed.

## Related Work & Insights

- The weight interpolation concept of **WiseFT** is generalized from CLIP adaptation to incremental learning.
- The multi-model weight averaging concept of **Model Soup** is adopted as tripartite weighting.
- The integration of ADKD with CutMix/MixUp provides a simple solution for performing effective KD under low-data regimes.

## Rating

- **Novelty**: 7/10 — Tri-WE is a natural extension of WiseFT, and ADKD is relatively straightforward.
- **Experimental Thoroughness**: 8/10 — Evaluated on three standard benchmarks and compared against a large number of methods.
- **Writing Quality**: 8/10 — Clear motivation and explicit description of methods.
- **Value**: 7/10 — Achieves SOTA in FSCIL, though the generalizability of the method remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[CVPR 2025\] Logits DeConfusion with CLIP for Few-Shot Learning](logits_deconfusion_with_clip_for_few-shot_learning.md)
- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)
- [\[ICLR 2026\] SAFA-SNN: Sparse-aware Fast Adaptive Spiking Neural Network for On-device Few-Shot Class-Incremental Learning](../../ICLR2026/model_compression/safa-snn_sparsity-aware_on-device_few-shot_class-incremental_learning_with_fast-.md)
- [\[CVPR 2025\] Incremental Object Keypoint Learning (KAMP)](incremental_object_keypoint_learning.md)

</div>

<!-- RELATED:END -->
