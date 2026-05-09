---
title: >-
  [Paper Note] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting
description: >-
  [NeurIPS 2025][Multimodal VLM][CLIP] This paper proposes CAW (Confidence-Aware Weighting), an adversarial fine-tuning loss function for CLIP that focuses on hard adversarial examples via confidence-aware weighting, combined with feature alignment regularization to preserve pre-trained semantic knowledge. CAW achieves state-of-the-art zero-shot robustness under AutoAttack with lower memory overhead.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - CLIP
  - adversarial robustness
  - zero-shot
  - adversarial fine-tuning
  - confidence-aware weighting
date: 2026-05-08
content_hash: c8250cd88c5cbb4c
---

# Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting

**Conference**: NeurIPS 2025
**arXiv**: [2510.02913](https://arxiv.org/abs/2510.02913)
**Code**: Not available
**Area**: Multimodal VLM
**Keywords**: CLIP, adversarial robustness, zero-shot, adversarial fine-tuning, confidence-aware weighting

## TL;DR

This paper proposes CAW (Confidence-Aware Weighting), an adversarial fine-tuning loss function for CLIP that focuses on hard adversarial examples via confidence-aware weighting, combined with feature alignment regularization to preserve pre-trained semantic knowledge. CAW achieves state-of-the-art zero-shot robustness under AutoAttack with lower memory overhead.

## Background & Motivation

Vision-language models such as CLIP exhibit strong zero-shot generalization, yet remain highly vulnerable to adversarial attacks—small, imperceptible perturbations can cause severe prediction errors. Adversarial training is the most effective approach for improving robustness, but applying it directly to large-scale pre-trained models like CLIP poses two key challenges: (1) overfitting and forgetting of pre-trained knowledge; and (2) difficulty in simultaneously maintaining clean accuracy and adversarial robustness.

Limitations of prior work:
- **TeCoA** is the first to study zero-shot robustness of large-scale VLMs, introducing contrastive adversarial loss with text supervision, but fails to improve both clean and robust accuracy simultaneously.
- **PMG-AFT** extends TeCoA with additional loss terms to enhance robustness, but incurs substantial memory overhead.
- **TGA-ZSR** leverages semantic text supervision to improve robustness and interpretability, but robust accuracy under strong attacks remains suboptimal.

Core starting point: **not all adversarial examples are equally important**—the model is already highly confident on some samples while highly uncertain on others. Focusing training on the "hard" adversarial examples (i.e., those with lowest model confidence) enables more efficient robustness improvement.

## Method

### Overall Architecture

CAW employs a frozen original CLIP model alongside a fine-tunable target CLIP model (image encoder only), jointly optimized via two novel loss terms: (1) a confidence-aware loss that emphasizes hard samples; and (2) a feature alignment regularization that preserves pre-trained knowledge. Training follows the standard adversarial training framework—the inner loop generates adversarial examples via PGD (maximizing cross-entropy loss), and the outer loop updates model parameters using the new loss function.

### Key Designs

1. **Confidence-Aware Loss**: The core idea is to align the prediction distribution $P^{\text{clean}}$ of the frozen CLIP on clean images with the prediction distribution $P^{\text{adv}}$ of the fine-tuned CLIP on adversarial images via KL divergence, using $(1 - P^{\text{adv}}_{i,y_i})$ as a weighting factor to amplify the contribution of hard samples:

    $L_{\text{CA}} = \frac{1}{N}\sum_{i=1}^{N}\left[\text{KL}(P^{\text{adv}}_i \| P^{\text{clean}}_i)(1 - P^{\text{adv}}_{i,y_i})\right]$

   where $P^{\text{adv}}_{i,y_i}$ is the predicted probability of the ground-truth class under adversarial input—lower probability indicates higher uncertainty, yielding a larger weight and greater training attention on that sample. Unlike ARoW, CAW uses the direction $\text{KL}(P^{\text{adv}} \| P^{\text{clean}})$ (adversarial distribution first), which empirically yields better performance.

2. **Feature Alignment Regularization**: By minimizing the $\ell_2$ distance between the features of the fine-tuned and frozen encoders on adversarial inputs, semantic consistency is maintained at the image feature level prior to text alignment:

    $L_{\text{Reg}} = \frac{1}{N}\sum_{i=0}^{N}\|f(x_{\text{adv}})_{\text{tar}} - f(x_{\text{adv}})_{\text{ori}}\|_2$

   Performing distillation in feature space rather than at the logit level better preserves the visual-semantic knowledge of pre-trained CLIP and reduces overfitting risk during fine-tuning.

3. **Total Loss**:

    $L_{\text{total}} = L_{\text{CE}} + \alpha \cdot L_{\text{CA}} + \beta \cdot L_{\text{Reg}}$

   where $L_{\text{CE}}$ is the standard cross-entropy loss, and $\alpha$ and $\beta$ control the contribution of each term.

### Loss & Training

- Inner loop: PGD-2 is used to generate adversarial examples with perturbation budget $\epsilon = 1/255$.
- Outer loop: Image encoder parameters are updated using $L_{\text{total}}$.
- Class descriptions are constructed using CLIP's standard text templates.
- The model is trained exclusively on TinyImageNet and then zero-shot transferred to 14 other datasets.

## Key Experimental Results

### Main Results

**Zero-shot robust accuracy under AutoAttack ($\epsilon=1/255$, 15 datasets)**

| Method | TinyImageNet | CIFAR-10 | CIFAR-100 | STL-10 | SUN397 | Caltech-101 | Avg. |
|--------|-------------|----------|-----------|--------|--------|-------------|------|
| CLIP | 0.02 | 0.01 | 0.08 | 0.03 | 0.04 | 0.43 | 0.09 |
| FT-Adv | 50.48 | 37.55 | 20.39 | 69.14 | 16.25 | 49.90 | 29.08 |
| TeCoA | 35.03 | 28.18 | 16.09 | 66.08 | 17.41 | 54.54 | 27.23 |
| PMG-AFT | 44.26 | 44.12 | 23.66 | 73.90 | 19.63 | 60.57 | 31.55 |
| TGA-ZSR | 49.45 | 40.53 | 22.38 | 72.06 | 20.36 | 57.16 | 31.63 |
| **CAW** | **50.52** | **47.35** | **26.35** | **74.27** | **19.64** | **62.79** | **33.51** |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| $L_{\text{CE}}$ only | Baseline robust accuracy | Standard adversarial training |
| $L_{\text{CE}} + L_{\text{CA}}$ | Robust accuracy ↑ | Hard samples better learned via confidence weighting |
| $L_{\text{CE}} + L_{\text{Reg}}$ | Clean accuracy ↑ | Feature regularization effectively preserves pre-trained knowledge |
| $L_{\text{CE}} + L_{\text{CA}} + L_{\text{Reg}}$ | Both ↑ | Two loss terms are complementary; best overall performance |
| KL(clean‖adv) direction | Robust accuracy ↓ | Placing adversarial distribution first in KL is superior |

**Memory comparison with baselines**

| Method | Memory Usage (relative) | Avg. Robust Accuracy |
|--------|------------------------|----------------------|
| PMG-AFT | High | 31.55 |
| TGA-ZSR | High | 31.63 |
| **CAW** | **Low** | **33.51** |

### Key Findings

- Vanilla CLIP achieves near-zero accuracy under AutoAttack (average 0.09%), demonstrating the severity of the zero-shot robustness problem.
- CAW achieves an average robust accuracy of 33.51% across 15 datasets, approximately 2% higher than the previous best, TGA-ZSR.
- Fine-tuning solely on TinyImageNet generalizes to 14 datasets from different distributions, indicating that the learned robust features transfer well.
- On CIFAR-10, CAW outperforms PMG-AFT by 3.23 percentage points (47.35 vs. 44.12).

## Highlights & Insights

- **Simple yet effective core idea**: Weighting by $(1 - P^{\text{adv}}_{i,y_i})$ naturally focuses training on hard samples without requiring additional sample mining or curriculum learning strategies.
- Regularization in feature space rather than logit space better preserves CLIP's rich pre-trained semantics.
- The overall method is lightweight—no additional text generation, attention mechanisms, or complex multi-stage training is needed.
- Memory efficiency surpasses PMG-AFT and TGA-ZSR, making CAW more suitable for resource-constrained deployment.

## Limitations & Future Work

- Validation is limited to classification tasks; extension to more complex downstream tasks such as detection and segmentation remains unexplored.
- The paper primarily addresses $\ell_\infty$-norm perturbations; robustness to other attack types such as $\ell_2$ is not thoroughly discussed.
- As a workshop paper, the experimental scale and depth of analysis are limited.
- No comparison is made with more recent adversarial training methods, such as those that leverage generative models to augment adversarial training data.
- The selection strategy for hyperparameters $\alpha$ and $\beta$ is not elaborated.

## Related Work & Insights

- TeCoA establishes the foundational research paradigm for zero-shot robustness of large-scale VLMs.
- PMG-AFT introduces a pre-trained model-guided fine-tuning framework.
- ARoW proposes the idea of prioritizing vulnerable samples; CAW adapts and improves this concept for vision-language models.
- Insight: The confidence-weighting idea is simple and general, and can be extended to other adversarial training and robust learning settings.

## Rating

- Novelty: ⭐⭐⭐ The core mechanism (confidence weighting + feature regularization) is relatively intuitive, but the combination proves effective in the VLM setting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across 15 datasets, multiple attack types, and memory comparison.
- Writing Quality: ⭐⭐⭐⭐ Concise and clear; method formulations are well-presented.
- Value: ⭐⭐⭐⭐ Provides a low-resource approach to CLIP robustness enhancement with significant practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](../../CVPR2026/multimodal_vlm/agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[NeurIPS 2025\] iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning](ifinder_structured_zero-shot_vision-based_llm_grounding_for_dash-cam_video_reaso.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)

<!-- RELATED:END -->
