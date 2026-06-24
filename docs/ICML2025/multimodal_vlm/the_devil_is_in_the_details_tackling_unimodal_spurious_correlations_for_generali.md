---
title: >-
  [Paper Note] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models
description: >-
  [ICML 2025][Multimodal VLM][Multimodal Reward Model] This work discovers that Multimodal Reward Models (MM-RMs) over-rely on unimodal text shortcuts during training, leading to poor out-of-distribution (OOD) generalization. To address this, the authors propose a Shortcut-aware MM-RM learning algorithm that reduces reliance on unimodal spurious correlations through dynamic sample reweighting, improving OOD accuracy from 68.1% to 78.5%.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Multimodal Reward Model"
  - "Spurious Correlations"
  - "Shortcut Learning"
  - "Generalization"
  - "Preference Alignment"
date: 2026-05-08
content_hash: a0606cf1bf72e016
---

# The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models

**Conference**: ICML 2025  
**arXiv**: [2503.03122](https://arxiv.org/abs/2503.03122)  
**Code**: [github.com/alignrm/Generalizable-MM-RM](https://github.com/alignrm/Generalizable-MM-RM)  
**Area**: Multimodal Reward Models, LLM Alignment, Robust Learning  
**Keywords**: Multimodal Reward Model, Spurious Correlations, Shortcut Learning, Generalization, Preference Alignment

## TL;DR

This work discovers that Multimodal Reward Models (MM-RMs) over-rely on unimodal text shortcuts during training, leading to poor out-of-distribution (OOD) generalization. To address this, the authors propose a Shortcut-aware MM-RM learning algorithm that reduces reliance on unimodal spurious correlations through dynamic sample reweighting, improving OOD accuracy from 68.1% to 78.5%.

## Background & Motivation

Reward Models (RMs) serve as critical proxies for LLM alignment, and multimodal RMs are essential for addressing visual hallucinations and safety concerns. However, generalization failure of RMs on OOD data leads to **reward hacking**, where the policy model compromises human intentions to maximize the reward.

This paper identifies an overlooked critical problem: **even when trained on multimodal data, MM-RMs still exploit text-only unimodal shortcuts**.

Core Evidence:
- IID test accuracy is up to 91.4%, but OOD accuracy is only 68.1%, exhibiting a 23.2% gap.
- RMs trained solely on text achieve comparable IID accuracy to multimodal RMs (with only a ~1.2% difference).
- On the POVID dataset, the model achieves 100.0% IID accuracy but drops to 47.8% (worse than random) in OOD settings.

## Method

### Problem Analysis Framework

A **cross-distribution generalization framework** is constructed: three preference datasets (VLFeedback, POVID, and RLHF-V) are used to represent different environments $\{D^e\}_{e \in \mathcal{E}}$, training on one environment and testing on others.

The **Shortcut-Failure Degradation (SFD)** metric is proposed to measure the performance degradation of MM-RMs when text shortcuts fail, with values ranging from 14.2 to 57.5 and a mean of 39.5.

### Shortcut-aware MM-RM Algorithm

The architecture utilizes **dual-branch training**:
- **Main branch $\mathcal{M}$**: Standard multimodal RM (Shortcut-aware MM-RM)
- **Auxiliary branch $\mathcal{M}_t$**: A text-only RM with images removed (serving as a shortcut proxy)

**Shortcut-Failure Coefficient (SFC)** Definition:

$$\text{SFC}(\boldsymbol{x}_i, y_i) = \frac{\mathcal{L}_t(\boldsymbol{x}_i, y_i)}{\mathcal{L}(\boldsymbol{x}_i, y_i) + \mathcal{L}_t(\boldsymbol{x}_i, y_i)}$$

High SFC $\rightarrow$ Text shortcut fails $\rightarrow$ Multimodal understanding required $\rightarrow$ Increase weight  
Low SFC $\rightarrow$ Text shortcut holds $\rightarrow$ Likely spurious correlation $\rightarrow$ Decrease weight

**Shortcut-aware Loss Function**:

$$\mathcal{L}_{sa} = \mathbb{E}_{(\boldsymbol{x}_i, y_i) \in \mathcal{S}_{train}}[\text{SFC}(\boldsymbol{x}_i, y_i) \cdot \mathcal{L}(\boldsymbol{x}_i, y_i)]$$

The gradients of SFC are detached; it serves solely as a weighting coefficient and does not participate in backpropagation.

### Inference Phase

The auxiliary branch is discarded after training. The inference process is identical to a standard MM-RM, incurring **no extra computational overhead**.

## Key Experimental Results

### Cross-Distribution Generalization

| Setting | Standard MM-RM | Shortcut-aware MM-RM |
|------|-----------|---------------------|
| Mean IID Accuracy | 91.4% | 90.2% |
| Mean OOD Accuracy | 68.1% | **78.5%** |
| Gain | — | **+10.4%** |

### Best-of-64 Downstream Evaluation (VLFeedback Training)

| Method | MM-Vet | LLaVA-bench | MMHal-V |
|------|--------|-------------|---------|
| Standard | 49.0 | 80.5 | 3.70 |
| **Shortcut-aware** | **50.2** | **84.7** | **3.74** |

### Key Findings

- The average SFD drops from 39.5 in the standard MM-RM to a significantly lower value in the Shortcut-aware variant.
- Consistently effective across three model scales: 2B, 4B, and 8B.
- Shortcut-aware MM-RM exhibits greater robustness against reward overoptimization.

## Highlights & Insights

1. **Precise Problem Definition**: This work is the first to systematically reveal the unimodal spurious correlation problem in MM-RMs and its detrimental impact on generalization.
2. **Ingenious SFD Metric**: Leveraging a text-only RM as a shortcut proxy, SFD quantifies the impact of text shortcuts on MM-RMs.
3. **Turning a Curse into a Blessing**: The existence of text shortcuts is transformed into a training signal—where the shortcut fails is precisely where multimodal understanding is most needed.
4. **Plug-and-Play + Zero Inference Overhead**: The auxiliary branch is only required during training and is discarded during inference.

## Limitations & Future Work

- Only text-only unimodal shortcuts are considered; visual-only shortcuts are not explored in depth.
- Dual-branch training increases the computational overhead during training (approx. $2\times$ forward passes).
- The three datasets used for cross-distribution evaluation may lack sufficient diversity.

## Related Work & Insights

- Reward Modeling (RLHF, Bradley-Terry Model)
- Spurious Correlations / Shortcut Learning (Geirhos et al., Arjovsky et al.)
- Multimodal Bias (Unimodal Bias in VQA, Length Bias)
- Invariant Risk Minimization (IRM)

## Rating

⭐⭐⭐⭐ — Profound problem identification with elegant designs for SFD and SFC. The experimental design (cross-distribution generalization matrix) is systematic and comprehensive. However, the technical solution is essentially weighted resampling, representing a moderate algorithmic novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs](../../ICML2026/multimodal_vlm/density-aware_translation_of_spurious_correlations_in_zero-shot_vlms.md)
- [\[ICLR 2026\] Label-Free Mitigation of Spurious Correlations in VLMs using Sparse Autoencoders](../../ICLR2026/multimodal_vlm/label-free_mitigation_of_spurious_correlations_in_vlms_using_sparse_autoencoders.md)
- [\[ICML 2025\] ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics](elemental_interactive_learning_from_demonstrations_and_vision-language_models_fo.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](../../ICCV2025/multimodal_vlm/controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ACL 2025\] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model](../../ACL2025/multimodal_vlm/internlm-xcomposer25-reward_a_simple_yet_effective_multi-modal_reward_model.md)

</div>

<!-- RELATED:END -->
