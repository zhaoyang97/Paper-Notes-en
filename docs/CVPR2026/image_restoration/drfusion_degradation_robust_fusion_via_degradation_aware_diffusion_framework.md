---
title: >-
  [Paper Note] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework
description: >-
  [CVPR 2026][Image Restoration][multimodal image fusion] This paper proposes DRFusion, a degradation-aware diffusion framework that achieves multimodal image fusion under arbitrary degradation scenarios within a small num…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "multimodal image fusion"
  - "diffusion model"
  - "degradation-aware"
  - "joint observation model"
date: 2026-05-08
content_hash: c872f79483ec6ce5
---

# DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework

**Conference**: CVPR 2026
**arXiv**: [2604.08922](https://arxiv.org/abs/2604.08922)  
**Code**: [https://github.com/YShi-cool/DRFusion](https://github.com/YShi-cool/DRFusion)  
**Area**: Image Fusion / Image Restoration
**Keywords**: multimodal image fusion, diffusion model, degradation-aware, joint observation model, image restoration

## TL;DR

This paper proposes DRFusion, a degradation-aware diffusion framework that achieves multimodal image fusion under arbitrary degradation scenarios within a small number of diffusion steps, via direct regression of the fused image (rather than explicit noise prediction) and a joint observation model correction mechanism.

## Background & Motivation

Real-world image fusion faces degradation challenges including noise, blur, and low resolution. Conventional "restore-then-fuse" pipelines suffer from error accumulation and deployment complexity. End-to-end neural network methods are simple and efficient but lack interpretability. Diffusion models offer strong theoretical foundations yet exhibit three inherent limitations: (1) they require training data from the target distribution, whereas fusion lacks natural ground-truth fused images; (2) standard diffusion models handle single-domain distributions, while fusion requires modeling complementary information from multiple sources; and (3) iterative sampling incurs high computational cost.

Existing diffusion-based fusion methods either handle only specific degradation types or rely on independently pretrained restoration models, lacking a flexible and unified framework.

## Method

### Overall Architecture

The framework discards the explicit noise prediction step of standard diffusion models and retains only the reverse process, directly mapping multi-source degraded inputs to fused outputs through a limited number of diffusion iterations. A joint observation correction step is inserted at each diffusion iteration.

### Key Designs

1. **Fusion-oriented diffusion framework**: Rather than predicting noise, the model directly regresses the fused image, with denoising implicitly embedded in the intermediate representations. This endows the framework with flexibility comparable to end-to-end networks, enabling self-supervised fusion training (without fusion labels) while requiring only a small number of diffusion steps to achieve high-quality results.

2. **Joint observation model**: The degradation constraints of both source images and the fusion constraint are unified in matrix form. The key innovation lies in replacing the position of the fused image with a zero matrix (eliminating the need for a pre-obtained fused image) and deriving a closed-form solution for the pseudo-inverse of the joint degradation matrix (by solving each sub-equation separately to avoid direct computation of the high-dimensional pseudo-inverse).

3. **Joint observation correction mechanism**: Following each DDIM sampling step, degradation constraints and fusion constraints are simultaneously injected to force the intermediate sample to align with the degradation model while preserving cross-modal complementary information. A scaling factor $\Sigma_t$ is introduced under noisy conditions to control the correction strength.

### Loss & Training

Fusion weights are learned in a data-driven manner via a multi-task architecture that simultaneously predicts noise and weight maps, subject to the constraint $W_1 + W_2 = 1$. The framework supports unified handling of multiple degradation types (noise, blur, low resolution, and their combinations).

## Key Experimental Results

### Main Results

| Fusion Task | Degradation Type | Ours | Competing Methods | Notes |
|---|---|---|---|---|
| Infrared–visible fusion | Noise + blur | Best | DeFusion, DDFM, etc. | Strong degradation robustness |
| Medical image fusion | Low resolution | Best | Multiple methods | Integrated restoration + fusion |
| Multi-focus fusion | Defocus blur | Competitive | Multiple methods | Flexible adaptation |

### Key Findings

- Significantly outperforms existing methods under complex degradation scenarios
- Competitive results are achieved with a small number of diffusion steps (e.g., 5–10)
- Joint observation correction is critical for maintaining restoration accuracy
- Data-driven fusion weight learning outperforms fixed weights

## Highlights & Insights

- The joint observation model unifies degradation restoration and multimodal fusion into a single constrained optimization problem
- The closed-form pseudo-inverse solution elegantly avoids high-dimensional matrix computation
- Removing explicit noise prediction enables efficient performance under few-step sampling
- Unified handling of noise, blur, low resolution, and arbitrary combinations thereof

## Limitations & Future Work

- The degradation model must be known or estimable (the degradation operator $A$ must be explicitly provided)
- Reducing the number of diffusion steps may affect quality under extreme degradation conditions
- Learning of fusion weights depends on the representativeness of the training data

## Related Work & Insights

- Shares conceptual similarity with the pseudo-inverse constraint approach of DDNM, extended here to multi-input fusion scenarios
- Provides a general degradation-aware diffusion paradigm for other multi-input image processing tasks

## Rating

- Novelty: ⭐⭐⭐⭐ — Degradation-aware diffusion fusion via joint observation model
- Technical Depth: ⭐⭐⭐⭐⭐ — Rigorous mathematical derivation with elegant pseudo-inverse solution
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple tasks and degradation types
- Value: ⭐⭐⭐⭐ — Unified framework for handling arbitrary degradations

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](../../ICML2026/image_restoration/degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)
- [\[CVPR 2026\] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](../../ICCV2025/image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)

</div>

<!-- RELATED:END -->
