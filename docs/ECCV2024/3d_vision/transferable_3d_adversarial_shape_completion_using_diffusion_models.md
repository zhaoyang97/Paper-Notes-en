---
title: >-
  [Paper Note] Transferable 3D Adversarial Shape Completion using Diffusion Models
description: >-
  [ECCV 2024][3D Vision] This work proposes 3DAdvDiff, which leverages 3D diffusion models to generate high-quality, transferable 3D adversarial point clouds via adversarial shape completion. By combining model uncertainty, ensemble adversarial guidance, and saliency scoring strategies, it achieves state-of-the-art (SOTA) attack success rates against modern 3D models under black-box settings.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: bd71ba348761ce4f
---

# Transferable 3D Adversarial Shape Completion using Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2407.10077](https://arxiv.org/abs/2407.10077)  
**Code**: None  
**Area**: 3D Vision

## TL;DR

This work proposes 3DAdvDiff, which leverages 3D diffusion models to generate high-quality, transferable 3D adversarial point clouds via adversarial shape completion. By combining model uncertainty, ensemble adversarial guidance, and saliency scoring strategies, it achieves state-of-the-art (SOTA) attack success rates against modern 3D models under black-box settings.

## Background & Motivation

### Background

**Background**: Existing 3D adversarial attack methods primarily add perturbations directly to the xyz coordinates, which severely degrades visual quality and makes them easily detectable by defenses.

### Proposed Solution

**Proposed Solution**: Most existing attacks focus on white-box scenarios, and their transferability to recently proposed models (such as PCT, PRC, and GDANet) is extremely poor.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing black-box attacks (e.g., AdvPC, PF-Attack) are only effective against early models, with their attack success rates (ASRs) dropping below 5% on newer models.

### Key Challenge

**Key Challenge**: How to generate high-quality and highly transferable 3D adversarial point clouds without disrupting the natural appearance of the original shapes.

## Method

### Overall Architecture

Taking a partial shape as a prior, the method utilizes a pre-trained 3D shape completion diffusion model (PVD) and injects adversarial guidance during the reverse generation process to generate a complete point cloud that "looks natural but can deceive classifiers."

### Key Designs

**Adversarial Shape Completion**: During the reverse denoising process of the diffusion model, adversarial gradient guidance is applied to each intermediate step:
- Keep the partial shape $z_0$ unchanged, applying guidance only to the completed part $\tilde{x}$.
- Apply guidance only in the early time steps $T_{\text{adv}} = (0, 0.2T]$ (since early noise is too large to be meaningful).

**Model Uncertainty for Enhancing Transferability**:
- Utilizing the permutation invariance (unordered nature) of point clouds, apply $M$ times simple random sampling (Bernoulli 0.5) to each intermediate point cloud.
- Calculate the Monte Carlo estimate of the average adversarial gradient under the $M$ random samplings, resembling Bayesian inference in MC-Dropout.

**Ensemble Adversarial Guidance**:
- Integrate logits from multiple surrogate models (PointNet, DGCNN, PRC) to compute the adversarial loss.
- Use adaptive weights (weighted by the correct classification ratio of each model).

**Generation Quality Preservation**:
- Compute the saliency score (sum of gradients) of each point, and apply perturbations only to the top-$N$ key points.
- Impose an $\ell_\infty$-norm constraint (0.16) on the perturbation at each step.

### Loss & Training

The loss at each step of the adversarial generation process is the gradient of the cross-entropy loss, which modifies the denoising mean via I-FGSM-style gradient guidance. The overall process maximizes the classification loss to generate misclassified adversarial exemplars.

## Key Experimental Results

### Main Results

ShapeNet all-class transfer attack ASR (%), surrogate model PointNet:

| Method | PointNet | PointNet++ | DGCNN | PointConv | CurveNet | PCT | PRC | GDANet | Avg |
|------|----------|------------|-------|-----------|----------|-----|-----|--------|-----|
| PGD | 99.9 | 2.1 | 0.7 | 0.8 | 0.5 | 0.4 | 0.7 | 1.6 | 0.9 |
| PF-Attack | 99.6 | 24.2 | 6.7 | 5.1 | 3.8 | 1.2 | 2.4 | 1.9 | 6.2 |
| 3DAdvDiff | 99.9 | 73.2 | 12.6 | 55.3 | 40.5 | 32.6 | 25.9 | 16.0 | 36.6 |
| **3DAdvDiff_ens** | **99.9** | **97.0** | **99.9** | **94.5** | **93.5** | **80.5** | **99.9** | **85.2** | **90.1** |

### Ablation Study

White-box attack ASR (%) against defenses:

| Method | No Defense | SRS | SOR | DUP-Net | IF-Defense | HybridTraining |
|------|--------|-----|-----|---------|------------|----------------|
| PGD | 99.9 | 5.9 | 1.0 | 0.7 | 13.8 | 1.9 |
| GeoA3 | 99.8 | 4.9 | 1.6 | 0.8 | 13.6 | 2.2 |
| SI-Adv | 92.5 | 6.3 | 1.8 | 1.4 | 17.3 | 3.3 |
| **3DAdvDiff** | **99.9** | **78.1** | **55.5** | **52.7** | **79.1** | **15.9** |

### Key Findings

- The ensemble version, 3DAdvDiff_ens, achieves an average ASR of 90.1% across 7 black-box models, compared to only 6.2% for the previous state-of-the-art, PF-Attack.
- The attack success rates against 5 defenses far exceed traditional methods (e.g., SRS: 78.1% vs 5.9%).
- The ShapeNet dataset exhibits a long-tail distribution: the top 5 classes account for 50% of the data but contribute only 14% of the successful adversarial examples.
- The gradient cosine similarity between models with different architectures is extremely low, explaining why conventional transfer attacks fail.
- Generating adversarial examples via shape completion rather than coordinate perturbation fundamentally improves visual quality.

## Highlights & Insights

- The work shifts the paradigm by redefining adversarial attacks as "generating unseen data" rather than "perturbing existing data."
- Capitalizing on the step-by-step guidance of diffusion models, only tiny adversarial perturbations are required at each step, yet they accumulate to a significant effect.
- Model uncertainty is realized via random downsampling of point clouds, which naturally and effectively exploits the unordered property of point clouds.
- A new benchmark is established for evaluating the robustness of 3D point cloud classification models.

## Unique Challenges of 3D Black-Box Attacks

The authors provide an in-depth analysis of why 3D black-box attacks are significantly more challenging than 2D ones:
1. The ShapeNet dataset exhibits a long-tail distribution, where the top 5 classes constitute 50% of the data but yield only 14% of the successful adversarial samples.
2. The gradient cosine similarity between 3D models with different architectures (e.g., PointNet vs. DGCNN vs. PCT) is extremely low.
3. Coordinate perturbations in 3D space are far more perceptible than pixel perturbations in 2D images.

This explains why conventional transfer attack methods (e.g., PGD, GeoA3) exhibit ASRs below 2% on newer models. 3DAdvDiff bypasses these fundamental barriers through the progressive generation of diffusion models (instead of direct perturbation) and model uncertainty.

## Limitations & Future Work

- The quality of the generated samples relies on the pre-trained diffusion model, and the experiments are currently validated on only a few categories of ShapeNet.
- The generation speed is slow (requiring a full 1000-step diffusion process).
- The adversarial success rate is still influenced by the choice of the partial shape.
- Experimental results on ModelNet40 are presented in the appendix, with the main experiments focusing on ShapeNet.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to leverage diffusion models for 3D adversarial attacks.
- Effectiveness: ⭐⭐⭐⭐⭐ — A quantum leap in black-box ASR from 6% to 90%.
- Practicality: ⭐⭐⭐⭐ — Useful as a security evaluation tool.
- Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)

</div>

<!-- RELATED:END -->
