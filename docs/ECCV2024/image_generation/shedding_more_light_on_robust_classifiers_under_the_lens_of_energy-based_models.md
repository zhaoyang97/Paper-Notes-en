---
title: >-
  [Paper Note] Shedding More Light on Robust Classifiers under the lens of Energy-based Models
description: >-
  [ECCV 2024][Image Generation] By reinterpreting robust discriminative classifiers as energy-based models (EBMs), this paper reveals the energy dynamics of adversarial training, proposes an energy-weighted adversarial training method (WEAT), and demonstrates the implicit generative capabilities of robust classifiers.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 4537de585cae4de9
---

# Shedding More Light on Robust Classifiers under the lens of Energy-based Models

**Conference**: ECCV 2024  
**arXiv**: [2407.06315](https://arxiv.org/abs/2407.06315)  
**Area**: Image Generation

## TL;DR

By reinterpreting robust discriminative classifiers as energy-based models (EBMs), this paper reveals the energy dynamics of adversarial training, proposes an energy-weighted adversarial training method (WEAT), and demonstrates the implicit generative capabilities of robust classifiers.

## Background & Motivation

- Adversarial Training (AT) is the core method for enhancing the robustness of neural networks. However, algorithmic progress has stalled in recent years, with top-tier methods relying primarily on more data or better architectures to achieve improvements.
- Prior research suffers from an insufficient understanding of AT; in particular, there lacks a unified explanatory framework for why robust classifiers exhibit generative capabilities, as well as the fundamental causes of robust overfitting.
- Previous work (Zimmermann et al.) established a preliminary connection between AT and EBMs, but lacks a systematic analysis.
- **Core Motivation**: To utilize the EBM framework to provide a unified explanation for various phenomena in AT (robust overfitting, the advantages of TRADES, sample weighting, and generative capabilities), and to design better training strategies accordingly.

## Method

### Overall Architecture

Reinterpreting the logits of a standard discriminative classifier as an energy function:
- **Joint Energy**: $E_\theta(\mathbf{x}, y) = -\theta(\mathbf{x})[y]$ (negative logit of the ground-truth class)
- **Marginal Energy**: $E_\theta(\mathbf{x}) = -\log\sum_k \exp(\theta(\mathbf{x})[k])$ (negative LogSumExp)
- The cross-entropy loss can be decomposed as: $\mathcal{L}_{CE} = E_\theta(\mathbf{x}, y) - E_\theta(\mathbf{x})$

### Key Designs

**1. Discovery of Three-Stage AT Energy Dynamics**

By tracking the energy difference between natural and adversarial data during training, $\Delta E_\theta(\mathbf{x}) = E_\theta(\mathbf{x}) - E_\theta(\mathbf{x}^\star)$, it is discovered that SAT training consists of three phases:
- Phase 1: Small fluctuations in the energy difference.
- Phase 2: The energy difference stabilizes.
- **Phase 3**: The energy difference drops sharply $\rightarrow$ corresponding to the occurrence of robust overfitting.

**2. EBM Reinterpretation of TRADES**

Rewriting the KL divergence term of TRADES in the form of energy (Proposition 1):

$$KL(p(y|\mathbf{x}) \| p(y|\mathbf{x}^\star)) = \underbrace{\mathbb{E}_{k \sim p(y|\mathbf{x})}[E_\theta(\mathbf{x}^\star, k) - E_\theta(\mathbf{x}, k)]}_{\text{conditional term}} + \underbrace{E_\theta(\mathbf{x}) - E_\theta(\mathbf{x}^\star)}_{\text{marginal term}}$$

This reveals that TRADES implicitly aligns the energy of natural and adversarial data, thereby mitigating overfitting.

**3. Smoothness of the Energy Landscape**

- It is discovered that all SOTA robust models on RobustBench share a common characteristic: a **smooth marginal energy landscape**.
- Increasing robustness $\leftrightarrow$ $\Delta E_\theta(\mathbf{x})$ approaching zero $\leftrightarrow$ a smoother energy landscape.

**4. WEAT (Weighted Energy Adversarial Training)**

Based on the discovery that low-energy samples are prone to overfitting while high-energy samples contribute more heavily to robustness, an energy-weighting scheme is proposed:
- Weight function: $w = \log(1 + \exp(|E_\theta(\mathbf{x})|))^{-1}$
- Higher weights are assigned to samples with energy close to zero.
- $E_\theta(\mathbf{x})$ is detached from the computational graph to prevent trivial solutions.

### Loss & Training

WEAT has two variants:
- **WEAT_NAT**: Based on TRADES, computing the CE loss on natural data.
- **WEAT_ADV**: Computing the CE loss on adversarial data + KL divergence.
- KL divergence is used as the inner loss to generate adversarial examples, and the weighting function is applied to the overall outer loss.

## Key Experimental Results

### Main Results

Comparison results using ResNet-18 on CIFAR-10, CIFAR-100, and SVHN datasets:

| Method | CIFAR-10 PGD | CIFAR-10 AA | CIFAR-100 PGD | CIFAR-100 AA | SVHN PGD | SVHN AA |
|------|-------------|------------|--------------|-------------|---------|--------|
| SAT | 49.03 | 45.37 | 23.89 | 20.99 | 50.54 | 44.87 |
| TRADES | 52.65 | 49.46 | 28.53 | 24.29 | 55.52 | 48.13 |
| MAIL-TR | 53.09 | 49.42 | 28.79 | 24.24 | 54.94 | 47.48 |
| **WEAT_NAT** | 52.43 | 49.02 | **29.71** | **24.88** | 55.31 | 48.61 |
| **WEAT_ADV** | **53.35** | **49.75** | **30.90** | **25.63** | **56.40** | **49.60** |

On Tiny-ImageNet (using ResNet-18), WEAT_ADV achieves an AA accuracy of 18.45%, outperforming TRADES (17.24%) and MART (17.79%).

### Ablation Study

Analysis of the impact of different attack methods on energy distribution:

| Attack Method | Marginal Energy $E_\theta(\mathbf{x})$ Shift | Joint Energy $E_\theta(\mathbf{x},y)$ Shift | Robust Accuracy |
|---------|--------------------------------|-----------------------------------|---------|
| PGD (Untargeted) | Significant left shift (energy decrease) | Shift right | 0% |
| TRADES (KL) | Significant left shift | Bimodal distribution | 30% |
| APGD | Minimal left shift | Shift right | Close to 0% |
| APGD-T (Targeted) | **Right shift** (energy increase) | Shift right to target class | — |
| CW | Almost unchanged | Minimal shift | — |

Key Findings: Untargeted attacks make adversarial examples look more "in-distribution" (lower energy) from the model's perspective, whereas targeted attacks show the opposite behavior.

### Key Findings

1. **Energy smoothness is a hallmark of robustness**: The distribution of $\Delta E_\theta(\mathbf{x})$ approaches zero for all SOTA robust models.
2. **TRADES essentially aligns energy**: The EBM perspective explains that the advantage of TRADES over SAT lies in its implicit mitigation of energy divergence.
3. **High-energy samples $\approx$ misclassified samples**: Removing high-energy correctly classified samples has an equivalent effect on robustness as removing misclassified samples.
4. **Generative capability**: Through an improved SGLD sampling (PCA initialization + momentum), robust classifiers can achieve remarkable IS and FID scores without dedicated generative training.

## Highlights & Insights

- The EBM perspective provides a unified analytical framework for AT research, connecting previously fragmented observations (overfitting, sample weighting, TRADES advantages, and generative capabilities).
- The discovery of the three-stage training dynamics and its correlation between energy divergence and overfitting holds significant theoretical value.
- The WEAT method is simple and elegant: it is solely based on marginal energy weighting, requires no label information, and does not need a warm-up period.
- The analysis of generative capability reveals an interesting phenomenon: models trained with KL divergence (such as TRADES) actually exhibit weaker generative capabilities than those trained with cross-entropy (CE).

## Limitations & Future Work

- WEAT matches but does not exceed the SOTA on CIFAR-10; its advantages are primarily demonstrated on CIFAR-100 and Tiny-ImageNet.
- The generative capability is still far inferior to dedicated generative models (e.g., diffusion models) and is more of an interesting byproduct.
- The theoretical analysis is mainly based on empirical observations, and the causal relationship between energy smoothness and robustness has not been rigorously proven.
- Experiments are only conducted using ResNet-18, lacking validation on larger models such as WideResNet-70-16.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Both the EBM-perspective AT analysis and the WEAT method exhibit clear novelty.
- **Practicality**: ⭐⭐⭐⭐ — WEAT is plug-and-play, and the energy weighting does not rely on labels.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive analysis across four datasets, multiple attack types, and comparisons with various SOTA methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear theoretical derivations, rich visualizations, and excellent coherence throughout the paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

</div>

<!-- RELATED:END -->
