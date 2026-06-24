---
title: >-
  [Paper Note] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models
description: >-
  [CVPR 2025][LLM Safety][Adversarial Robustness] The first test-time adversarial defense method for VLMs. By minimizing the entropy consistency of multi-view augmentations and aligning adversarial-clean embedding statistics, it learns a defensive prompt for each test sample. With only a single optimization step, it improves the robustness of CLIP against AutoAttack from 0.1% to 48.9%.
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "Adversarial Robustness"
  - "Test-time Defense"
  - "Prompt Tuning"
  - "VLM Safety"
  - "Zero-shot Inference"
date: 2026-05-08
content_hash: 099821a99827928e
---

# TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2411.13136](https://arxiv.org/abs/2411.13136)  
**Code**: None  
**Area**: Multimodal VLMs  
**Keywords**: Adversarial Robustness, Test-time Defense, Prompt Tuning, VLM Safety, Zero-shot Inference

## TL;DR
The first test-time adversarial defense method for VLMs. By minimizing the entropy consistency of multi-view augmentations and aligning adversarial-clean embedding statistics, it learns a defensive prompt for each test sample. With only a single optimization step, it improves the robustness of CLIP against AutoAttack from 0.1% to 48.9%.

## Background & Motivation

**Background**: VLMs face threats from adversarial attacks—subtle pixel perturbations can degrade the classification accuracy of CLIP from 68% to 0%. Existing defense methods require training adversarial prompts (APT) on task-specific data, but the target task is unknown during inference.

**Limitations of Prior Work**: APT requires task-specific labeled data for adversarial training and generalizes poorly to unseen attack types. How can defense be achieved at test-time without labels or training data?

**Key Challenge**: Effective adversarial defense typically requires knowing attack patterns, yet nothing is known about the attacks during inference; simultaneously, clean sample performance must not be sacrificed.

**Goal**: Adaptively learn a defensive prompt for each test sample during inference, without relying on any task-specific data.

**Key Insight**: Different augmented views of adversarial samples produce inconsistent predictions (high entropy), whereas augmented views of clean samples yield consistent predictions. This discrepancy is exploited to restore correct predictions of adversarial samples by minimizing multi-view entropy.

**Core Idea**: Learn a defensive bimodal prompt for each test sample using a single-step gradient optimization, simultaneously enhancing robustness and maintaining clean performance through entropy minimization + adversarial-clean statistical alignment.

## Method

### Overall Architecture
Pre-computation: Initialize with an adversarially trained pre-tuned prompt on ImageNet $\rightarrow$ calculate layer-wise adversarial/clean embedding statistics (means/variances). Inference: Generate $M$ augmented views for the test sample $\rightarrow$ select top-$K$ low-entropy views $\rightarrow$ perform single-step optimization of $\mathcal{L}_{TAPT} = \mathcal{L}_{entropy} + \alpha \cdot \mathcal{L}_{adv} + (1-\alpha) \cdot \mathcal{L}_{clean}$ $\rightarrow$ perform inference utilizing the optimized prompt.

### Key Designs

1. **Multi-View Entropy Minimization**:

    - **Function**: Detect and repair adversarial perturbations by leveraging prediction consistency.
    - **Mechanism**: Generate $M$ random augmented views (cropping/flipping) of the test sample, select the top-$K$ views with the lowest prediction entropy, and minimize their joint prediction entropy. After adversarial perturbations are "diluted" by augmentations, signals of the correct category become more consistent across multiple views.
    - **Design Motivation**: The effect of adversarial perturbations in augmented views is uncertain (they may be amplified or weakened). Selecting low-entropy views filters out the views where perturbation effects are amplified.

2. **Adversarial-Clean Embedding Alignment**:

    - **Function**: Shift the embedding statistics of the test sample from the adversarial domain toward the clean domain.
    - **Mechanism**: Pre-compute the layer-wise mean and variance of adversarial/clean embeddings of ImageNet samples. At test time, align the layer-wise embedding statistics of the current sample with the pre-computed statistics: $\mathcal{L}_{adv}$ pulls toward adversarial statistics (as the input might be attacked), and $\mathcal{L}_{clean}$ pulls toward clean statistics (to preserve original semantics). $\alpha$ controls the balance between the two.
    - **Design Motivation**: Relying solely on entropy minimization might be insufficient (as certain attacks can maintain low entropy). Statistical alignment provides additional constraints from the representation space perspective.

3. **Single-Step Optimization**:

    - **Function**: Ensure inference efficiency.
    - **Mechanism**: The entire prompt optimization requires only a single gradient descent step to complete, as the APT pre-initialization already provides a good starting point.
    - **Design Motivation**: Test-time defense must run in real-time, making multi-step optimization unacceptable.

### Loss & Training
$\mathcal{L}_{TAPT} = \mathcal{L}_{entropy} + \alpha \cdot \mathcal{L}_{adv} + (1-\alpha) \cdot \mathcal{L}_{clean}$. No task-specific training data is required.

## Key Experimental Results

### Main Results

| Method | ImageNet Robust Acc | 11 Dataset Average Robust Acc |
|------|-----------------|-------------------|
| Vanilla CLIP | 0.0% | 0.1% |
| APT-V | 14.8% | 16.3% |
| **TAPT-V** | **49.2%** | **48.9%** |
| Gain vs. APT | **+34.4%** | **+32.6%** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Entropy minimization only | ~35% | Effective but insufficient |
| + Adversarial alignment | ~43% | Significant improvement |
| + Clean alignment | ~48.9% | Preserves clean performance |
| Number of views M=64, K=32 | Optimal | More views offer higher stability |

### Key Findings
- **From 0% to 49%**: CLIP's robustness to AutoAttack is improved from nearly zero to approximately 50%, without requiring any task-specific data.
- **No Clean Performance Trade-off**: Accuracy on clean samples remains largely unchanged.
- **Task-Agnostic**: Statistics pre-computed using ImageNet perform effectively across 11 datasets from diverse domains.

## Highlights & Insights
- **First Test-Time Adversarial Defense for VLMs**: Fills an important security gap, as previous defense methods required training-time data.
- **Ingenious Statistical Alignment Idea**: Uses in-domain (ImageNet) statistics as anchors to pull the embeddings of unknown attacks back into normal ranges.

## Limitations & Future Work
- Pre-initialization still relies on ImageNet adversarial training, which may be insufficient for domains that differ significantly from ImageNet.
- A single optimization step might be insufficient against strong adaptive attacks.
- The value of $\alpha$ needs to be set manually; adaptive determination could be more robust.

## Related Work & Insights
- **vs. APT**: APT requires task-specific data for adversarial training, whereas TAPT operates zero-shot on any task, yielding a robustness improvement of over 30 percentage points.
- **vs. TPT**: TPT is a test-time prompt tuning method but does not target adversarial attacks. TAPT introduces adversarial-clean statistical alignment to make it robust against attacks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First test-time adversarial defense for VLMs; the combination of entropy minimization and statistical alignment is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 11 datasets and multiple attack methods, though lack of evaluation under adaptive attacks.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and solid explanation of the methodology.
- Value: ⭐⭐⭐⭐⭐ High practical significance for the secure deployment of VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/llm_safety/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[NeurIPS 2025\] Buffer Layers for Test-Time Adaptation](../../NeurIPS2025/llm_safety/buffer_layers_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
