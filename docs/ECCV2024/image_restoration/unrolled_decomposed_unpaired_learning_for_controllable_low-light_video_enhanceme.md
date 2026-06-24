---
title: >-
  [Paper Note] Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement
description: >-
  [ECCV 2024][Image Restoration][Low-light video enhancement] This paper proposes UDU-Net, which models low-light video enhancement as a MAP optimization problem and unrolls it into a deep network. It processes spatial (illumination) and temporal (consistency) degradations through Intra/Inter sub-networks respectively, supporting unpaired training and controllable enhancement guided by human perceptual feedback.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Low-light video enhancement"
  - "unpaired learning"
  - "deep unrolling"
  - "temporal consistency"
  - "human perceptual feedback"
date: 2026-05-08
content_hash: 7e7093454adfc79b
---

# Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement

**Conference**: ECCV 2024  
**arXiv**: [2408.12316](https://arxiv.org/abs/2408.12316)  
**Code**: [Yes](https://github.com/lingyzhu0101/UDU.git)  
**Area**: Image Restoration  
**Keywords**: Low-light video enhancement, unpaired learning, deep unrolling, temporal consistency, human perceptual feedback

## TL;DR

This paper proposes UDU-Net, which models low-light video enhancement as a MAP optimization problem and unrolls it into a deep network. It processes spatial (illumination) and temporal (consistency) degradations through Intra/Inter sub-networks respectively, supporting unpaired training and controllable enhancement guided by human perceptual feedback.

## Background & Motivation

Low-light video enhancement faces three challenges:

**Difficulty in paired data acquisition**: Compared to static images, obtaining paired low-light/normal-light videos in dynamic scenes is much more difficult, making unpaired learning a necessary technological direction.

**Intertwined spatio-temporal degradation**: Noise, underexposure, and non-uniform contrast are intertwined with temporal consistency requirements in the spatial domain.

**Over/underexposure issues**: Distribution alignment-only methods lack pixel-level constraints and human perceptual feedback, which easily leads to abnormal exposure.

Limitations of prior work:
- **Image enhancement methods** directly applied to videos ignore inter-frame temporal context, leading to inconsistent enhancement results.
- **Deep unrolling methods** have been explored in low-light enhancement (e.g., Retinex-inspired), but suffer from issues such as sub-optimal local convergence caused by separate module training, and failure to correctly model the reflectance-illumination relationship.
- No prior work has simultaneously addressed both spatial and temporal degradations of low-light videos within a deep unrolling architecture.

## Method

### Overall Architecture

UDU-Net is based on the MAP estimation framework, modeling low-light video enhancement as:

$$\hat{x} = \arg\min_x \frac{1}{2}\|y - Ax\|^2 + \lambda_s J_s(x) + \lambda_t J_t(x)$$

This formulation is decomposed into three sub-problems via ADMM and unrolled into cascaded Intra sub-networks (single-frame enhancers) and Inter sub-networks (multi-frame enhancers) to progressively optimize from both spatial and temporal perspectives in a multi-stage manner.

| Stage | Sub-network | Function | Key Technology |
|------|------|------|----------|
| Stage $k$ | Intra (a) | Coarse illumination estimation | Learning illumination distribution from unpaired expert-retouched data |
| Stage $k$ | Inter (b) | Temporal smoothness learning | 3D convolution + optical flow alignment + masking mechanism |
| Stage $k+1$ | Intra (c) | Fine illumination optimization | Controllable enhancement guided by human perceptual feedback |
| Stage $k+1$ | Inter (d) | Temporal detail compensation | Same as Inter (b), with further temporal refinement |

### Key Designs

**1. Intra Sub-network — Spatial Prior: Unpaired Retouched Illumination**

- Stage $k$: Uses the MIT-Adobe FiveK dataset (Expert C retouched version) as the unpaired data source, matching the enhanced results to the illumination distribution of expert retouches through adversarial learning.
- Losses include: semantic self-supervised loss (VGG feature matching), content self-supervised loss (multi-scale L1), and a Relativistic Average HingeGAN discriminator.

**2. Human Perceptual Feedback Mechanism**

Stage $k+1$ introduces controllable illumination adjustment:
- Generates target frames through gamma correction and linear scaling: $m_i^{k+1} = \beta \times (\alpha \times \tilde{x}_i^{k+1})^\gamma$
- Parameters are sampled from a uniform distribution $U(1.00, 1.10)$.
- Uses the BRISQUE no-reference quality assessment model to simulate human visual system feedback.
- Automatically selects the version with the lowest BRISQUE score (best quality) as the optimization target.

**3. Inter Sub-network — Temporal Prior: Temporal Cue Exploration**

- Uses a pre-trained RAFT optical flow model to estimate inter-frame motion, with fine-tuning during training.
- First stage: 3D convolution aggregates 5 aligned neighboring frames to estimate the noise-free structural signal $s_t$.
- Second stage: Estimates compensated detail residuals based on the structural signal $s_t$ and neighboring frames.
- Introduces a soft mask $M_t$ to balance noise suppression and texture preservation.

**4. 3D Noise Suppression Strategy**

- **Spatial-domain adversarial learning**: Uses unpaired data to guide the model in reducing noise in generator frames.
- **Temporal consistency learning**: Aligns and fuses neighboring frames to further suppress noise.
- **Masking mechanism**: Uses an exponential-function-based mask derived from structural signal discrepancies, with parameter $\omega$ controlling the texture-noise balance.

### Loss & Training

**Intra Sub-network Loss (Stage $k$)**:
- Semantic self-supervised loss $L_{\text{semantic}-G}$: L2 distance of VGG features.
- Content self-supervised loss $L_{\text{content}-G}$: multi-scale L1 loss.
- Adversarial loss: Relativistic Average HingeGAN.

**Intra Sub-network Loss (Stage $k+1$)**:
- Content self-supervised loss: L1 distance to the human perceptual feedback target.

**Inter Sub-network Loss**:
- Optical flow loss $L_{\text{flow}-R}$: L1 distance between aligned frames and the current frame.
- Temporal content loss $L_{\text{content}-T}$: structural signal constraint + mask-weighted inter-frame consistency L1 loss.

## Key Experimental Results

### Main Results

Quantitative results on the SDSD dataset (compared with no-reference methods):

| Method | Outdoor PSNR↑ | Outdoor SSIM↑ | Outdoor warp↓ | Indoor PSNR↑ | Indoor SSIM↑ |
|------|---------------|---------------|---------------|--------------|--------------|
| EnlightenGAN | 18.63 | 0.5399 | 4.49 | 19.59 | 0.5874 |
| CLIP-LIT | 20.88 | 0.5872 | 3.36 | 19.08 | 0.4582 |
| SRIE | 21.89 | 0.6288 | 2.74 | 15.78 | 0.6294 |
| **UDU-Net (Ours)** | **23.94** | **0.7446** | **0.24** | **22.41** | **0.7368** |
| SDSDNet (Supervised)* | 24.30 | 0.7445 | 0.95 | 27.03 | 0.7788 |

### Ablation Study

| Configuration | PSNR | SSIM | warp |
|------|------|------|------|
| Ours-v1 (Intra-a only) | Baseline | Baseline | Baseline |
| Ours-v2 (+Inter-b) | Gain | Gain | Significantly decreased |
| Ours-v3 (+Intra-c) | Further gain | Further gain | Decreased |
| Default (+Inter-d + Human perception) | Best | Best | Best |
| Default w/o H (w/o Human perception) | Slightly lower | Slightly lower | Slightly lower |

### Key Findings

- UDU-Net achieves a PSNR of 23.94 dB on the SDSD outdoor scenes, which is a 2.05 dB gain over the second-best no-reference method SRIE.
- The temporal quality index (warp error) is only 0.24, significantly lower than the second-best method (SRIE's 2.74), demonstrating superb temporal consistency.
- It achieves a comparable or even slightly superior SSIM compared to the supervised method SDSDNet on outdoor scenes (0.7446 vs 0.7445).
- The human perceptual feedback mechanism effectively suppresses over/underexposure, with BRISQUE score guidance resulting in enhanced outcomes more aligned with human visual preferences.
- Every component (Intra-a, Inter-b, Intra-c, Inter-d) contributes positively to the final performance.

## Highlights & Insights

1. **Innovative Application of Unrolling Methods**: First to apply a MAP-optimized unrolling framework to low-light video enhancement, modeling both spatial and temporal constraints concurrently.
2. **Unpaired + Controllable**: Achieves controllable enhancement through human perceptual feedback without relying on paired data, representing a highly practical design.
3. **Systematic Noise Suppression**: Cooperatively suppresses noise across three dimensions: the spatial domain (adversarial learning), the temporal domain (frame fusion), and a masking mechanism.
4. **BRISQUE as Proxy for Human Vision**: Employs a no-reference quality assessment model to simulate human perception, automatically adjusting the target illumination level.
5. **End-to-End Training**: Circumnavigates local optima issues typically caused by separately trained modules.

## Limitations & Future Work

- Optical flow estimation may be inaccurate in scenes with severe camera movement or heavy occlusions.
- The human perceptual feedback mechanism relies heavily on the BRISQUE model, which has its own limitations.
- The expert-retouched style from the MIT-Adobe FiveK dataset may not generalize to all application scenarios.
- The parameter range $U(1.00, 1.10)$ for gamma correction is empirically set and may require scene-specific tuning.
- Comparisons with recent diffusion model-based methods are not included.

## Related Work & Insights

- **Deep Unrolling Methods**: PnP and ADMM frameworks inspired the idea of unrolling optimization problems into trainable deep networks.
- **EnlightenGAN**: A pioneer in unpaired enhancement using dual discriminators, but lacks temporal consideration.
- **StableLLVE**: Utilizes optical flow to simulate dynamic scenes for enhancing temporal stability.
- **Zero-DCE**: Formulates enhancement as a curve estimation problem.
- Limitations of **Retinex Decomposition** in unrolling frameworks motivated the design of UDU-Net.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 4 |
| Practicality | 4 |
| Writing Quality | 3.5 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)
- [\[ECCV 2024\] Learning to Robustly Reconstruct Dynamic Scenes from Low-Light Spike Streams](learning_to_robustly_reconstruct_dynamic_scenes_from_low-light_spike_streams.md)
- [\[ICML 2026\] AnyMod-LLVE: Low-Light Video Enhancement with Modality-Agnostic Inference](../../ICML2026/image_restoration/anymod-llve_low-light_video_enhancement_with_modality-agnostic_inference.md)
- [\[CVPR 2026\] BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement](../../CVPR2026/image_restoration/bievlight_bi-level_learning_of_task-aware_event_refinement_for_low-light_image_e.md)
- [\[CVPR 2025\] Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](../../CVPR2025/image_restoration/classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)

</div>

<!-- RELATED:END -->
