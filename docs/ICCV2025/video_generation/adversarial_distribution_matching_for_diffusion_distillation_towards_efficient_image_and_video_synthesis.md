---
title: >-
  [Paper Note] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis
description: >-
  [ICCV 2025][Video Generation][diffusion distillation] This paper proposes an Adversarial Distribution Matching (ADM) framework that replaces the predefined KL divergence in DMD with an implicit…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "diffusion distillation"
  - "adversarial training"
  - "distribution matching"
  - "one-step generation"
  - "score distillation"
date: 2026-05-08
content_hash: ba00090d3da3df30
---

# Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis

**Conference**: ICCV 2025
**Code**: N/A  
**Area**: Image/Video Generation / Diffusion Model Distillation
**Keywords**: diffusion distillation, adversarial training, distribution matching, one-step generation, score distillation

## TL;DR

This paper proposes an Adversarial Distribution Matching (ADM) framework that replaces the predefined KL divergence in DMD with an implicit, data-driven measure of distributional discrepancy. A diffusion-model-based discriminator aligns the latent predictions of real and fake score estimators along the PF-ODE. Combined with Adversarial Distillation Pre-training (ADP), the resulting DMDX pipeline surpasses DMD2 on one-step SDXL generation and extends naturally to SD3 and CogVideoX video synthesis.

## Background & Motivation

- **Background**: Distribution Matching Distillation (DMD) is a powerful approach to compressing pretrained diffusion models into efficient few-step generators.
- **Limitations of Prior Work**: DMD relies on reverse KL divergence minimization, which carries a risk of mode collapse due to its zero-forcing property — the model tends to concentrate probability mass on a few dominant modes. DMD2 mitigates this with a GAN regularizer but does not address the root cause.
- **Key Challenge**: One-step distillation suffers from elevated risks of gradient explosion/vanishing, attributed not merely to approximation errors in the fake score estimator but primarily to insufficient support overlap between the student and teacher distributions.
- **Goal**: Develop a framework that bypasses the limitations of predefined divergences by learning an implicit, data-driven discrepancy measure; additionally, improve initialization for one-step distillation to resolve the support overlap problem.

## Method

### Overall Architecture

The DMDX pipeline consists of two stages: (1) **Adversarial Distillation Pre-training (ADP)** — the generator is pre-trained using synthetic ODE pairs and a hybrid discriminator (latent space + pixel space), providing a stable initialization; (2) **ADM fine-tuning** — a diffusion-model-based discriminator adversarially aligns the PF-ODE predictions of real and fake score estimators, replacing the DMD loss for distribution matching.

### Key Designs

1. **Adversarial Distribution Matching (ADM)**: The discriminator $D_\tau$ consists of a frozen teacher diffusion model plus multiple trainable heads. Given a noised sample $x_t$ derived from the generator output $\hat{x}_0$, rather than solving to the endpoint $x_0$ as in DMD, the PF-ODE is integrated to $(t - \Delta t)$, yielding real/fake samples $x_{t-\Delta t}^{\text{real/fake}}$ as score predictions, which are then fed into the discriminator. A Hinge loss is used for alternating training of the generator and discriminator. A key innovation is that temporal step information is preserved (consistent with score distillation) while the discriminator adaptively adjusts the discrepancy measure during training — focusing on global differences early on and fine-grained local differences later.

2. **Adversarial Distillation Pre-training (ADP)**: ODE pairs are collected offline from the teacher model; noised samples are constructed via linear interpolation, and the prediction target is set to the velocity of the ODE pair. A hybrid discriminator is employed — a latent-space discriminator (initialized from the teacher model) and a pixel-space discriminator (initialized from the SAM visual encoder), with $\lambda_1 = 0.85$ and $\lambda_2 = 0.15$. A cubic timestep schedule biased toward high noise levels encourages exploration of new modes, while the discriminator uses a uniform timestep schedule to capture features across different scales.

3. **Essential Distinction between ADM and ADP**: ADM is a form of score distillation — supervising the matching of the full probability flow across different noise levels. ADP is a form of adversarial distillation — concerned only with the clean data distribution at $t = 0$. ADP artificially creates support overlap via isotropic Gaussian noise, making discrimination harder and gradients smoother, thereby providing a stable initialization for ADM fine-tuning.

### Loss & Training

ADM employs a Hinge GAN loss to align PF-ODE predictions; ADP uses a hybrid Hinge loss over latent and pixel spaces. ADP adopts a cubic generator timestep schedule combined with a uniform discriminator timestep schedule. For one-step SDXL distillation, ADP pre-training is followed by ADM fine-tuning, with total GPU time less than that of DMD2.

## Key Experimental Results

### Main Results

| Model | Method | NFE | FID↓ | CLIP Score↑ | GenEval |
|-------|--------|-----|------|-------------|---------|
| SDXL | Baseline (50 steps) | 50 | — | — | — |
| SDXL | DMD2 | 1 | reference | reference | reference |
| SDXL | **DMDX** | **1** | **surpasses DMD2** | **surpasses DMD2** | **surpasses DMD2** |
| SD3-Medium | ADM | 8 | new benchmark | — | — |
| SD3.5-Large | ADM | multi-step | new benchmark | — | — |
| CogVideoX-2B | ADM | 8 | VBench 79.86 | — | — |
| CogVideoX-5B | ADM | 8 | VBench 82.06 | — | — |

DMDX achieves competitive fidelity to the 50-step baseline on one-step SDXL generation (50× speedup).

### Ablation Study

- The DMD loss naturally decreases during ADM training, validating the hypothesis that the discriminator implicitly subsumes the reverse KL divergence.
- The hybrid discriminator (latent + pixel space) outperforms a single discriminator.
- The cubic timestep schedule effectively improves mode diversity.
- ADP pre-training is critical for one-step distillation, providing better support overlap.

### Key Findings

- The primary bottleneck in one-step distillation is insufficient support overlap between student and teacher distributions, not fake score estimator approximation error.
- A learnable discriminator can implicitly approximate arbitrary nonlinear functions to measure distributional discrepancy, offering greater flexibility than predefined divergences.
- ADM extends directly to video generation (CogVideoX), demonstrating the generality of the approach.
- Given good initialization, the two-timescale update rule (TTUR) has limited impact on final performance.

## Highlights & Insights

- Replacing predefined divergence with adversarial training for score distillation is conceptually clean and theoretically grounded.
- The two-stage ADP + ADM pipeline is well-motivated and resolves the initialization challenge inherent to one-step distillation.
- This work represents the first extension of score distillation to video diffusion models (CogVideoX).
- The hybrid discriminator design (latent + pixel space) is transferable to other GAN training scenarios.

## Limitations & Future Work

- The method still requires two-stage training (pre-training + fine-tuning), resulting in a relatively complex pipeline.
- The discriminator uses the frozen teacher model as a backbone, incurring significant memory overhead.
- Validation is limited to the Stable Diffusion family and CogVideoX; other architectures have not been tested.
- The pixel-space discriminator relies on the SAM encoder, introducing an additional dependency.

## Related Work & Insights

- DMD/DMD2 serve as the most direct baselines and targets for improvement.
- Adversarial methods such as ADD (Adversarial Diffusion Distillation) and SDXL-Lightning provide important reference points.
- The ODE pair collection strategy from Rectified Flow is adopted in ADP.
- The unified perspective bridging score distillation and adversarial distillation constitutes a theoretical contribution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of adversarial score distillation is original.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Theoretical analysis is rigorous; the distinction between ADM and ADP is thoroughly discussed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across image and video generation on multiple model architectures.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated with complete theoretical derivations.
- **Value**: ⭐⭐⭐⭐ — 50× speedup with maintained quality yields high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)

</div>

<!-- RELATED:END -->
