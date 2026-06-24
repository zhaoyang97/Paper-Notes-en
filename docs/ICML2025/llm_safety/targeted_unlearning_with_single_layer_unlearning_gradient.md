---
title: >-
  [Paper Note] Targeted Unlearning with Single Layer Unlearning Gradient
description: >-
  [ICML 2025][LLM Safety][machine unlearning] This paper proposes the SLUG (Single Layer Unlearning Gradient) method, which identifies the optimal single layer using layer importance and gradient alignment metrics. It achieves highly efficient and precise targeted unlearning using only a single gradient computation and single-layer parameter update, applicable to CLIP, Stable Diffusion, and VLMs.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "machine unlearning"
  - "CLIP"
  - "Stable Diffusion"
  - "VLM"
  - "Single Layer Update"
  - "privacy protection"
date: 2026-05-08
content_hash: ead8c43084fe7182
---

# Targeted Unlearning with Single Layer Unlearning Gradient

**Conference**: ICML 2025  
**arXiv**: [2407.11867](https://arxiv.org/abs/2407.11867)  
**Code**: [github.com/CSIPlab/SLUG](https://github.com/CSIPlab/SLUG)  
**Area**: Machine Unlearning, Multimodal Foundation Models, Trustworthy AI  
**Keywords**: machine unlearning, CLIP, Stable Diffusion, VLM, Single Layer Update, privacy protection

## TL;DR

This paper proposes the SLUG (Single Layer Unlearning Gradient) method, which identifies the optimal single layer using layer importance and gradient alignment metrics. It achieves highly efficient and precise targeted unlearning using only a single gradient computation and single-layer parameter update, applicable to CLIP, Stable Diffusion, and VLMs.

## Background & Motivation

Large foundation models (LLMs, text-to-image models, and VLMs) are trained on massive datasets and inevitably memorize private data or copyrighted content. Since retraining from scratch is prohibitively expensive, **machine unlearning** has emerged as a necessary alternative.

Existing unlearning methods face three major challenges:
- **Low computational efficiency**: Methods like fine-tuning (FT) and Gradient Ascent (GA) require multiple iterations of updates across the entire model.
- **Large side effects**: Full-model parameter updates degrade the performance on unrelated concepts.
- **Requires heavy hyperparameter tuning**: Such as learning rates, number of iterations, and mask thresholds.

Key Insight: **Different layers in deep networks learn different features, meaning that modifying only the single most critical layer can achieve targeted unlearning while minimizing the impact on the overall model capability.**

## Method

### Overall Architecture

SLUG consists of three steps:
1. One-time computation of both forget and retain gradients.
2. Identification of the optimal single layer via the Pareto frontier.
3. A single-step update with the step size determined by binary search.

### Loss & Training

**Retain set loss** — Standard contrastive loss (maintaining vision-text alignment):

$$\mathcal{L}_{\text{retain}} = \frac{1}{2N_r}\sum_{i=1}^{N_r}(\ell_{i2t}(i) + \ell_{t2i}(i))$$

**Forget set loss** — Cosine embedding loss (breaking vision-text alignment):

$$\mathcal{L}_{\text{forget}} = \frac{1}{N_f}\sum_{i=1}^{N_f} 1 - \cos(\mathbf{v}_i, \mathbf{t}_j)$$

### Single Layer Identification

**Layer Importance** (Sensitivity of layer $l$ to the forget set):

$$\text{Importance}(l) = \frac{\|\nabla_{\theta_l} \mathcal{L}_{\text{forget}}\|_2}{\|\theta_l\|_2}$$

**Gradient Alignment** (Angle between the forget and retain gradients):

$$\text{Alignment}(l) = \cos(\nabla_{\theta_l}\mathcal{L}_{\text{forget}}, \nabla_{\theta_l}\mathcal{L}_{\text{retain}})$$

Goal: **Maximize Importance and minimize Alignment** $\rightarrow$ Search for the Pareto frontier across all layers.

### Single-Step Gradient Update

$$\theta_l^* \leftarrow \theta_l^{(0)} - \lambda^* \nabla_{\theta_l}\mathcal{L}_{\text{forget}}\big|_{\theta=\theta^{(0)}}$$

The step size $\lambda^*$ is determined via binary search (fixing $S=10$ steps) to find a balance where forget accuracy is near 0 while test accuracy is maintained.

### Generalization to SD and VLM

- **Stable Diffusion**: Applies SLUG to the text encoder (CLIP) to achieve layer-level plug-and-play unlearning.
- **VLM (LLaVA)**: Applies SLUG to the vision encoder, affecting downstream text generation.

## Key Experimental Results

### CLIP Zero-Shot Classification

| Method | FA@1↓ | TA_IN@1↑ | TA_CA@1↑ | Computational Complexity |
|------|-------|----------|----------|-----------|
| GA | 0.00 | 35.88 | 24.92 | $O(k \cdot N_f)$ |
| SalUn | 0.00 | 55.45 | 26.11 | $O(N_f) + O(k \cdot (N_f+N_r))$ |
| SSD | 0.00 | 51.84 | 35.96 | $O(N_f+N_r)$ |
| **SLUG** | **0.00** | **59.96** | **58.32** | $O(N_f+N_r)$ |

### UnlearnCanvas Benchmark

| Method | Style UA↑ | Efficiency (Time/s)↓ | Storage/GB↓ |
|------|---------|-------------|----------|
| ESD | 98.58 | 6163 | 4.3 |
| SalUn | 86.26 | 667 | 4.0 |
| **SLUG** | **86.29** | **39** | **0.04** |

### VLM Unlearning (LLaVA-1.5-7B)

The unlearning accuracy of 10 celebrity identities was reduced from 99.50% to 2.8%, while competitiveness on VLM benchmarks was maintained.

### Key Findings

- SLUG maintains nearly 60% ImageNet accuracy and 58.32% CelebA accuracy on CLIP, **significantly outperforming other methods**.
- Extremely high efficiency: Requires only 39 seconds and 0.04 GB of storage on UnlearnCanvas.
- Cross-model generalization: Shows consistent performance from ViT-B/32 to EVA01-g-14.

## Highlights & Insights

1. **Extreme Simplicity**: Only one gradient computation, one single-layer update, and one binary search, with no iterative training required.
2. **Modular Design**: Only modifies single-layer weights, serving as a plug-and-play "unlearning patch".
3. **Cross-Model Universality**: Complete validation pipeline established across CLIP $\rightarrow$ SD $\rightarrow$ VLM.
4. **Clear Theoretical Intuition**: Utilizes Fisher Information Matrix and Pareto optimality for layer selection.

## Limitations & Future Work

- Unlearning performance on complexly entangled concepts may be limited.
- The upper bound of representation capability for single-layer updates is limited.
- Binary search requires a small amount of validation data.

## Related Work & Insights

- Fine-tuning / Gradient Ascent unlearning
- SalUn (Saliency Unlearning, ICLR 2024)
- SSD (Selective Synaptic Dampening)
- Task Arithmetic

## Rating

⭐⭐⭐⭐⭐ — Extremely simple yet highly effective method. The comprehensive validation across three classes of foundation models (CLIP, SD, VLM) is highly convincing. The efficiency figures of 39 seconds/0.04 GB set a new benchmark for unlearning methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP](../../ACL2025/llm_safety/cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)
- [\[CVPR 2026\] Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](../../CVPR2026/llm_safety/machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)
- [\[ICML 2025\] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning](negmerge_sign-consensual_weight_merging_for_machine_unlearning.md)
- [\[ICML 2025\] Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](../../ACL2025/llm_safety/zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)

</div>

<!-- RELATED:END -->
