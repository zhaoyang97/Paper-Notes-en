---
title: >-
  [Paper Note] LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal
description: >-
  [ICCV 2025][Autonomous Driving][Lens Flare Removal] This paper proposes LightsOut, a diffusion-based image outpainting framework that enhances existing single-image flare removal (SIFR) methods by predicting and reconstr…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Lens Flare Removal"
  - "Diffusion Model"
  - "Image Outpainting"
  - "LoRA Fine-tuning"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 51fe9f94fd4cfdb0
---

# LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal

**Conference**: ICCV 2025
**arXiv**: [2510.15868](https://arxiv.org/abs/2510.15868)
**Code**: [https://ray-1026.github.io/lightsout/](https://ray-1026.github.io/lightsout/)
**Area**: Autonomous Driving / Image Restoration
**Keywords**: Lens Flare Removal, Diffusion Model, Image Outpainting, LoRA Fine-tuning, Plug-and-Play

## TL;DR

This paper proposes LightsOut, a diffusion-based image outpainting framework that enhances existing single-image flare removal (SIFR) methods by predicting and reconstructing out-of-frame light sources. It serves as a plug-and-play preprocessing module that improves arbitrary SIFR models without requiring additional training.

## Background & Motivation

Lens flare severely degrades image quality and adversely affects computer vision tasks such as object detection and autonomous driving. Existing deep learning-based SIFR methods perform well when the complete light source is visible, but suffer significant performance degradation when the light source is partially or entirely outside the frame. A key observation is that **complete light source information is critical for effective flare removal**. When the light source is cropped beyond the image boundary, existing methods lack sufficient contextual information to understand and remove flare artifacts.

The authors empirically validate this core motivation: under both "no light source" and "incomplete light source" scenarios, metrics such as PSNR and LPIPS degrade substantially, confirming the critical role of light source completeness in flare removal.

## Method

### Overall Architecture

LightsOut adopts a **three-stage pipeline**:
1. **Light Source Prediction and Conditional Generation**: Predicts out-of-frame light source parameters and generates a light source mask.
2. **Light Source Outpainting**: Uses a LoRA fine-tuned diffusion model to outpaint and complete the missing light source regions.
3. **SIFR Enhancement**: Feeds the outpainted image into an existing SIFR model for flare removal.

### Key Designs

1. **Multitask Regression Module**:

    - Predicts a parameterized representation of out-of-frame light sources, modeling each source as a circular entity $(x, y, r)$.
    - Simultaneously predicts $N$ sets of physical light source parameters $\mathbf{P} \in \mathbb{R}^{N \times 3}$ and confidence scores $\mathbf{c} \in [0,1]^{N \times 1}$.
    - Employs a CNN feature extractor with dual MLP heads (one for position parameters, one for confidence scores).
    - Uses a bipartite matching strategy to handle permutation invariance between predictions and ground truth.
    - A rendering function generates the final light source mask $M_L$ via Sigmoid activation and confidence thresholding:
    $M_L(x,y) = \sum_{i=1}^{N} \tilde{c}_i \cdot \sigma(r_i - \sqrt{(x-x_i)^2 + (y-y_i)^2})$

2. **LoRA Fine-tuned Diffusion Outpainting Model**:

    - Built upon Stable Diffusion v2 Inpainting with injected LoRA weights for efficient fine-tuning.
    - Training loss: $\mathcal{L} = \mathbb{E}_{x,t,\epsilon,m} \|\epsilon_\theta(x_t, t, p, M, I_M) - \epsilon\|_2^2$
    - Uses BLIP-2 to automatically generate text prompts as conditioning signals.
    - Performs Alpha compositing in RGB space during inference (rather than latent space blending) to avoid distortion in preserved regions.

3. **Noise Reinjection**:

    - Addresses inconsistencies arising from independent processing of masked/unmasked regions during denoising.
    - Reintroduces noise at intermediate steps, allowing the model to re-denoise for better distribution alignment.
    - The noise reinjection operation is repeated $R$ times to ensure visual coherence between generated and original regions.

4. **Light Source Condition Module**:

    - Leverages the predicted light source mask $M_L$ to guide the outpainting process.
    - Conditions the generation process via a learnable mechanism, ensuring physically plausible light source placement.
    - Constrained by an L2 loss: $\mathcal{L}_{\text{light}} = \|\tilde{M}_L - M_L\|_2^2$

### Loss & Training

The multitask regression module employs an uncertainty-aware weighting mechanism to balance multiple losses:
$$\mathcal{L} = \frac{1}{2\sigma_1^2}\mathcal{L}_{\text{pos}} + \frac{1}{2\sigma_2^2}\mathcal{L}_{\text{conf}} + \log(1+\sigma_1^2) + \log(1+\sigma_2^2)$$

where $\mathcal{L}_{\text{pos}}$ supervises positional parameters via Smooth L1 loss and $\mathcal{L}_{\text{conf}}$ supervises confidence via binary cross-entropy. The three modules are trained independently:
- Multitask Regression Module: lr=1e-4, batch=32, 100 epochs, N=4
- Light Source Condition Module: lr=1e-5, batch=8, 20K steps
- LoRA Fine-tuned Diffusion Model: lr=1e-4, batch=8, 25K steps

## Key Experimental Results

### Main Results

Evaluated on the Flare7K dataset (100 real images + 100 synthetic images) under two scenarios: no light source / incomplete light source.

| Scenario | SIFR Model | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------|------------|--------|-------|-------|--------|
| No Light Source | Flare7k++ | Direct input | 26.29 | 0.8337 | 0.0442 |
| No Light Source | Flare7k++ | SD-Inpainting | 27.98 | 0.8938 | 0.0421 |
| No Light Source | Flare7k++ | PowerPaint | 27.10 | 0.8814 | 0.0839 |
| No Light Source | Flare7k++ | **Ours** | **28.41** | **0.8956** | **0.0397** |
| Incomplete Light Source | Flare7k++ | Direct input | 26.07 | 0.8333 | 0.0463 |
| Incomplete Light Source | Flare7k++ | SD-Inpainting | 28.02 | 0.8944 | 0.0431 |
| Incomplete Light Source | Flare7k++ | **Ours** | **28.15** | **0.8957** | **0.0409** |

Key finding: Under the "no light source" scenario, LightsOut improves the PSNR of Flare7k++ from 26.29 dB to 28.41 dB (+2.12 dB).

### Ablation Study

| Ablated Component | PSNR (Real) | SSIM | LPIPS |
|-------------------|-------------|------|-------|
| Without noise reinjection | 28.28 | 0.8949 | 0.0412 |
| With noise reinjection | **28.41** | **0.8956** | **0.0397** |
| Latent space blending | 26.91 | 0.8859 | 0.0434 |
| RGB space blending | **27.09** | **0.8856** | **0.0424** |

Component contribution ablation (SD-Inpainting → Ours):

| LoRA | Condition Module | PSNR↑ |
|------|-----------------|-------|
| ✗ | ✗ | 26.82 |
| ✓ | ✗ | 27.12 (+0.30) |
| ✗ | ✓ | 27.06 (+0.24) |
| ✓ | ✓ | **27.43** (+0.61) |

### Key Findings

- RGB space blending substantially outperforms latent space blending, particularly on synthetic data (PSNR 31.55 vs. 24.13).
- Noise reinjection significantly improves visual coherence, reducing LPIPS from 0.0412 to 0.0397.
- LoRA fine-tuning and the light source condition module offer complementary contributions, with their combination yielding the best results.
- The multitask regression approach outperforms both the UNet baseline and differentiable rendering methods (mIoU 0.6310 vs. 0.6216 vs. 0.5212).

## Highlights & Insights

- **Precise Problem Formulation**: Accurately identifies the root cause of SIFR degradation—missing out-of-frame light source information.
- **Plug-and-Play Design**: Requires no modification to existing SIFR model architectures and can directly serve as a preprocessing stage to enhance arbitrary methods.
- **Physical Prior Integration**: Models light sources as parameterized circular entities, introducing physical constraints to guide diffusion-based generation.
- Downstream task validation: Uses the YOLOv11 detector to verify the indirect improvement the method provides for object detection.

## Limitations & Future Work

- The three-stage pipeline introduces additional computational overhead; end-to-end optimization warrants further exploration.
- Performance is limited when overall image brightness is high or flare occupies an excessively large proportion of the image.
- Training and evaluation are conducted solely on the Flare7K dataset; generalization to real-world scenarios remains to be verified.
- The circular light source assumption may be insufficient for irregularly shaped sources.

## Related Work & Insights

- Complementary to, rather than competitive with, SIFR methods such as Difflare and MFDNet.
- The effectiveness of LoRA fine-tuning for task-specific diffusion model adaptation is worth generalizing.
- The noise reinjection technique can be extended to other image outpainting and inpainting tasks.
- The physically parameterized light source prediction approach may inspire other vision tasks requiring physical priors.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Addressing flare removal from the perspective of light source outpainting is a fresh angle; combining diffusion models with physical priors is a creative design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-baseline comparisons, multi-scenario evaluations, detailed ablations, and downstream task validation constitute a comprehensive experimental study.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, method descriptions are complete, and figures and tables are intuitive.
- **Value**: ⭐⭐⭐⭐ The plug-and-play design offers strong practical utility, with meaningful implications for autonomous driving and related applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[NeurIPS 2025\] Neurosymbolic Diffusion Models](../../NeurIPS2025/autonomous_driving/neurosymbolic_diffusion_models.md)
- [\[ICCV 2025\] Decoupled Diffusion Sparks Adaptive Scene Generation](decoupled_diffusion_sparks_adaptive_scene_generation.md)
- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)

</div>

<!-- RELATED:END -->
