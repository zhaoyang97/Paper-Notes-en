---
title: >-
  [Paper Note] Efficient Diffusion as Low Light Enhancer (ReDDiT)
description: >-
  [CVPR 2025][Image Restoration][Low-light enhancement] ReDDiT is proposed to distill diffusion-based low-light enhancement from 10+ steps down to 2-4 steps. By correcting fitting errors via linear extrapolation and refining trajectories with Retinex-decomposed reflectance to bridge the inference gap, it achieves state-of-the-art (SOTA) performance across 10 benchmarks in just 4 steps.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Low-light enhancement"
  - "diffusion distillation"
  - "reflectance prior"
  - "trajectory refinement"
  - "fast inference"
date: 2026-05-08
content_hash: 446a5ee3e23fdead
---

# Efficient Diffusion as Low Light Enhancer (ReDDiT)

**Conference**: CVPR 2025  
**arXiv**: [2410.12346](https://arxiv.org/abs/2410.12346)  
**Code**: Yes (Project Page)  
**Area**: Image Restoration  
**Keywords**: Low-light enhancement, diffusion distillation, reflectance prior, trajectory refinement, fast inference

## TL;DR
ReDDiT is proposed to distill diffusion-based low-light enhancement from 10+ steps down to 2-4 steps. By correcting fitting errors via linear extrapolation and refining trajectories with Retinex-decomposed reflectance to bridge the inference gap, it achieves state-of-the-art (SOTA) performance across 10 benchmarks in just 4 steps.

## Background & Motivation

**Background**: Diffusion models have achieved excellent performance in low-light image enhancement (LLIE) (e.g., GSAD), but require 10-1000 inference steps, making them too slow for real-time applications.

**Limitations of Prior Work**: (1) Directly applying diffusion distillation methods (e.g., Progressive Distillation/Consistency Models) to LLIE performs poorly because low-light images have unique degradation patterns, rendering standard distillation assumptions inapplicable. (2) The trajectory of the teacher model inherently contains fitting errors, meaning the student learns an imperfect trajectory. (3) There is an inference gap during distillation between the diffusion process starting from Gaussian noise and the restoration task starting from low-light images.

**Key Challenge**: Distillation requires the student to faithfully replicate the teacher's trajectory, but the teacher's trajectory itself has errors. Furthermore, the diffusion path from pure noise to a clean image does not align well with the low-light to normal restoration path.

**Goal**: Design a distillation scheme specifically tailored for low-light enhancement that corrects teacher trajectory errors and utilizes reflectance priors to bridge the inference gap.

**Key Insight**: (1) Apply linear extrapolation to the teacher's score function to correct fitting errors without retraining. (2) Estimate reflectance using Retinex decomposition—since reflectance is a shared property between low-light and normal-light images—as a deterministic prior to refine diffusion trajectories.

**Core Idea**: Correct teacher trajectory fitting errors with linear extrapolation and use Retinex reflectance as a deterministic anchor for trajectory refinement, achieving high-quality low-light enhancement in 2-4 steps.

## Method

### Overall Architecture
Teacher diffusion model (multi-step) $\to$ Linear extrapolation to correct the score function $\to$ Reflectance-Aware Trajectory Refinement (RATR) $\to$ Student model learning corrected 2nd-order trajectory endpoints $\to$ Auxiliary pixel/perceptual loss.

### Key Designs

1. **Linear Extrapolation for Fitting Error Correction**:

    - Function: Corrects the systematic bias of the teacher's score function without retraining.
    - Mechanism: The teacher score function has systematic fitting errors. Systematic bias is compensated by performing linear extrapolation based on the difference between predictions at two time steps.
    - Design Motivation: Since the teacher model is already fully trained, retraining is costly. Linear extrapolation serves as a cost-free approximate correction.

2. **Reflectance-Aware Trajectory Refinement (RATR)**:

    - Function: Pulls the diffusion trajectory from the Gaussian noise space towards the residual restoration space using a reflectance prior.
    - Mechanism: Extract reflectance $\tilde{x}_s$ using Retinex decomposition (max-channel illumination estimation + non-learning denoising). Refine the trajectory as $\tilde{x}^{\eta}_{s,u,t} = \omega x^{\eta}_{s,u,t} + (1-\omega)\tilde{x}_s$—performing linear interpolation between the teacher's prediction and the reflectance. Reflectance is an ideal intermediate quantity because it shares information with the clean image (same reflectance) and is connected to the low-light input (estimable from the low-light image).
    - Design Motivation: Standard diffusion starts from pure Gaussian noise, which mismatches the actual path of low-light restoration. Reflectance acts as a deterministic anchor to bridge this gap.

3. **Auxiliary Pixel/Perceptual Loss**:

    - Function: Provides pixel-level and perceptual-level supervision in addition to the distillation loss.
    - Mechanism: L2 pixel loss + perceptual loss are used to supplement the distillation loss.
    - Design Motivation: Pure distillation can yield blurry outputs; auxiliary losses help enhance sharpness.

### Loss & Training
Distillation loss (matching the endpoints of the teacher's corrected trajectory) + L2 + perceptual loss. The non-learning reflectance estimation requires no additional neural networks.

## Key Experimental Results

### Main Results

| Method | Steps | LOLv1 PSNR | LOLv2-real PSNR | LOLv2-synth PSNR |
|------|------|-----------|----------------|-----------------|
| GSAD | 10 | 27.84 | 28.82 | 28.67 |
| Retinexformer | 1 | 27.18 | 27.71 | 29.04 |
| **ReDDiT-4** | **4** | **27.98** | **31.25** | **30.03** |
| **ReDDiT-2** | **2** | **27.40** | **30.61** | **29.35** |

### Ablation Study

| Component | Effect |
|------|------|
| W/o linear extrapolation | PSNR drops by ~0.5 |
| W/o RATR | PSNR drops by ~1.0 |
| W/o auxiliary losses | Output is slightly blurry |

### Key Findings
- **Matching 10-step diffusion in 2 steps**: ReDDiT-2 achieves a Lolv2-real PSNR of 30.61, surpassing the 10-step GSAD (28.82).
- **New SOTA across 10 benchmarks in 4 steps**: ReDDiT-4 sets new state-of-the-art results across all tested datasets.
- **Reflectance prior is key**: RATR yields the largest performance benefit (~1.0 PSNR gain), validating the effectiveness of reflectance as an illumination-invariant prior.

## Highlights & Insights
- **Retinex reflectance as a diffusion anchor** is an elegant cross-disciplinary fusion, where classical image processing theory (Retinex) guides modern deep diffusion model distillation.
- **Correcting fitting errors via linear extrapolation** is cost-free and generalizable, and can be applied to any diffusion distillation scenario.
- **2-step low-light enhancement** paves the way for real-time applications.

## Limitations & Future Work
- The quality of Retinex reflectance estimation relies on the max-channel method, which may be inaccurate under extreme low-light conditions.
- It only targets low-light enhancement; its applicability to other image restoration tasks (e.g., deraining, dehazing) is unverified.
- Utilizing non-learning denoising for reflectance estimation might limit the upper bound of accuracy.
- Although the 2-step model is near real-time, it is still slower than non-generative methods (e.g., Retinexformer, which requires only a single forward pass).
- The choice of the $\omega$ parameter affects performance, yet the paper does not provide an automated selection strategy.

## Related Work & Insights
- **vs GSAD**: Uses 10 diffusion steps to achieve LOLv1 PSNR of 27.84. ReDDiT-4 achieves 27.98, and ReDDiT-8 reaches 28.09, balancing both performance and efficiency.
- **vs Retinexformer**: A single-step deterministic method that does not require iteration. However, ReDDiT's reflectance prior combined with diffusion flexibility yields better results—surpassing it by 0.88 dB PSNR on SID.
- **vs PyDiff**: Accelerates to 4 steps using pyramid conditioning + DDIM, achieving LOLv2-real PSNR of 29.63. ReDDiT reaches 30.61 in just 2 steps, demonstrating that task-tailored distillation is more effective than generic acceleration.
- **vs Consistency Distillation**: As generic methods fail to consider the deterministic restoration requirements of LLIE, the inference gap remains unresolved, resulting in inferior performance on LLIE compared to ReDDiT.
- **vs WCDM**: Conducts diffusion in the wavelet space, reaching LOLv2-real PSNR of 30.46 in 10 steps. ReDDiT-4 surpasses this (31.25) without requiring extra wavelet transform computations.

## Rating
- Novelty: ⭐⭐⭐⭐ Both reflectance-aware trajectory refinement and linear extrapolation correction are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 benchmarks, multiple step-count configurations, and component ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The analysis of degradation factors is clear.
- Value: ⭐⭐⭐⭐ Highly significant for the real-time deployment of low-light enhancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DarkIR: Robust Low-Light Image Restoration](darkir_robust_low-light_image_restoration.md)
- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](hvi_a_new_color_space_for_low-light_image_enhancement.md)
- [\[CVPR 2026\] Bi-Bridge: Bidirectional Diffusion Bridges for Low-Light Image Enhancement](../../CVPR2026/image_restoration/bi-bridge_bidirectional_diffusion_bridges_for_low-light_image_enhancement.md)
- [\[CVPR 2025\] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration](urwkv_unified_rwkv_model_with_multi-state_perspective_for_low-light_image_restor.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)

</div>

<!-- RELATED:END -->
