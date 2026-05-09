---
title: >-
  [Paper Note] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation
description: >-
  [ICLR 2026][Image Generation][mean flow] This paper proposes RMFlow, which appends a noise-injection refinement step after 1-NFE MeanFlow transport to compensate for single-step transport errors, while incorporating a maximum likelihood objective during training to minimize the KL divergence between the learned and target distributions. RMFlow achieves near-SOTA 1-NFE results on text-to-image generation, molecular generation, and time-series generation.
tags:
  - ICLR 2026
  - Image Generation
  - mean flow
  - noise injection refinement
  - 1-NFE
  - likelihood maximization
  - multimodal generation
date: 2026-05-08
content_hash: b9416e188117ee8e
---

# RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation

**Conference**: ICLR 2026
**arXiv**: [2602.00849](https://arxiv.org/abs/2602.00849)
**Code**: None
**Area**: Diffusion Models / One-Step Generation / Mean Flow Improvement
**Keywords**: mean flow, noise injection refinement, 1-NFE, likelihood maximization, multimodal generation

## TL;DR
This paper proposes RMFlow, which appends a noise-injection refinement step after 1-NFE MeanFlow transport to compensate for single-step transport errors, while incorporating a maximum likelihood objective during training to minimize the KL divergence between the learned and target distributions. RMFlow achieves near-SOTA 1-NFE results on text-to-image generation, molecular generation, and time-series generation.

## Background & Motivation

**Background**: MeanFlow learns a mean velocity field to enable few-step generation without pretraining or distillation. However, performance degrades significantly at 1-NFE — single-step transport is insufficiently precise, causing generated samples to deviate from the target distribution.

**Limitations of Prior Work**: 1-NFE MeanFlow exhibits large bias on Gaussian mixture distributions and produces invalid molecular structures (broken molecules) in molecular generation. Multi-step inference (8/32 NFE) performs well but sacrifices efficiency.

**Key Challenge**: The deterministic output of 1-NFE transport deviates from the true distribution due to approximation errors in the mean velocity field, yet increasing NFE is not an option.

**Goal**: Improve MeanFlow generation quality while maintaining 1-NFE inference.

**Key Insight**: The 1-NFE transport is treated as a "coarse transport," followed by a noise-injection step for "refinement" — essentially converting the deterministic output of MeanFlow into a stochastic output, using noise to compensate for transport errors. An additional maximum likelihood objective is incorporated during training to minimize KL divergence.

**Core Idea**: Deterministic 1-NFE MeanFlow output + Gaussian noise injection ≈ improved approximation of the target distribution.

## Method

### Overall Architecture
$x_{\text{gen}} = x_0 + \hat{u}_{0,1}(x_0; \theta) + \sigma \epsilon$ (one-step transport + noise injection), with training loss = MeanFlow loss + $\lambda_1$ negative log-likelihood loss + $\lambda_2$ guidance regularization.

### Key Designs

1. **Noise-Injection Refinement**: After 1-NFE transport, $\sigma \epsilon$ ($\epsilon \sim \mathcal{N}(0, I)$) is added to convert the deterministic point estimate into a distribution with variance, compensating for transport errors.
2. **Maximum Likelihood Training Objective**: $\mathcal{L}_{\text{NLL}} = \mathbb{E}[\|x_{\text{tgt}} - (x_0 + \hat{u}_{0,1})\|^2]$; it is theoretically shown that this loss lower-bounds the expected log-likelihood of the target distribution.
3. **Joint Loss**: The MeanFlow loss controls the Wasserstein distance along the probability path, while the NLL loss controls the KL divergence of the terminal distribution.

### Loss & Training
- For large-scale tasks, MeanFlow is first trained, then fine-tuned with LoRA incorporating the NLL loss.
- T2I: COCO dataset, 480M U-Net, SD-VAE latent space.
- Molecular generation: QM9 + GEOM-Drugs.

## Key Experimental Results

### Main Results

| Method | NFE | Mixture Gaussian TV ↓ | QM9 Molecule Stability ↑ |
|--------|-----|----------------------|--------------------------|
| MeanFlow | 1 | 1.44 | Low (broken structures) |
| MeanFlow | 8 | 0.80 | Moderate |
| MeanFlow | 32 | 0.67 | High |
| **RMFlow** | **1** | **0.76** | **Near 32-NFE level** |

T2I (COCO FID-30K): RMFlow achieves FID comparable to Distilled SD and StyleGAN-T without requiring auxiliary models.

### Key Findings
- 1-NFE RMFlow surpasses 8-NFE MeanFlow (TV 0.76 vs. 0.80), approaching 32-NFE performance.
- Noise injection effectively prevents structural fragmentation in molecular generation.
- Training cost is comparable to MeanFlow, as noise injection introduces negligible overhead.

## Highlights & Insights
- **Minimal Modification**: Adding a single $\sigma\epsilon$ step substantially improves 1-NFE quality — essentially reformulating a point estimation problem as distribution estimation, using noise to compensate for model error.
- **Multimodal Generality**: The same framework handles images, molecules, and time series, demonstrating that noise-injection refinement is a modality-agnostic technique.
- A theoretical connection between the NLL loss and KL divergence is established, providing a principled justification for noise injection.

## Limitations & Future Work
- The noise injection scale $\sigma$ is a hyperparameter requiring tuning.
- T2I experiments use COCO with a small U-Net; validation on ImageNet or larger models is absent.
- Direct comparisons with recent one-step methods such as SoFlow and TwinFlow are lacking.
- Whether noise injection is universally beneficial remains unclear; it may introduce blurring in high-dimensional image generation.

## Related Work & Insights
- **vs. MeanFlow**: 1-NFE RMFlow outperforms 8-NFE MeanFlow; the key improvements are noise injection and the NLL loss.
- **vs. SoFlow**: Different improvement strategies — SoFlow learns a solution function, while RMFlow learns a mean velocity field with refinement.

## Rating
- Novelty: ⭐⭐⭐ The noise-injection idea is straightforward, but represents a first application to MeanFlow.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multimodal validation (images/molecules/time series), though image experiments are limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-organized, with rigorous theoretical derivations.
- Value: ⭐⭐⭐⭐ Offers a simple and effective improvement strategy for MeanFlow.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [\[ICLR 2026\] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)

<!-- RELATED:END -->
