---
title: >-
  [Paper Note] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model
description: >-
  [CVPR 2025][Image Generation][Image Super-Resolution] This work discovers that different regions of LR images (flat areas vs. textured edge regions) correspond to different timesteps in the diffusion process, and proposes an Uncertainty-guided Noise Weighting (UNW) strategy. UNW applies less noise to flat regions to preserve crucial LR information, achieving state-of-the-art (SOTA) super-resolution performance with a smaller model size and lower training cost.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Super-Resolution"
  - "Diffusion Model"
  - "Uncertainty Estimation"
  - "Anisotropic Noise"
  - "Region Adaptation"
date: 2026-05-08
content_hash: c694fb755fbb6db8
---

# Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2503.18512](https://arxiv.org/abs/2503.18512)  
**Code**: [https://github.com/LabShuHangGU/UPSR](https://github.com/LabShuHangGU/UPSR)  
**Area**: Image Generation  
**Keywords**: Image Super-Resolution, Diffusion Model, Uncertainty Estimation, Anisotropic Noise, Region Adaptation

## TL;DR
This work discovers that different regions of LR images (flat areas vs. textured edge regions) correspond to different timesteps in the diffusion process, and proposes an Uncertainty-guided Noise Weighting (UNW) strategy. UNW applies less noise to flat regions to preserve crucial LR information, achieving state-of-the-art (SOTA) super-resolution performance with a smaller model size and lower training cost.

## Background & Motivation
1. **Background**: Diffusion models have demonstrated superior perceptual quality over GANs in image super-resolution. ResShift simplifies the diffusion process by embedding the LR image into the initial noise map and only estimating the LR-HR residual.
2. **Limitations of Prior Work**: Even though ResShift embeds LR information into the initial state, isotropic noise still conceals useful details. All existing methods employ **isotropic** noise, applying the same noise intensity across all image regions regardless of varying restoration difficulties.
3. **Key Challenge**: The LR values in flat regions are already close to the HR targets, requiring only minor noise perturbations, whereas edge/textured regions differ significantly from HR targets and require higher noise levels to explore low-density areas. A uniform noise level cannot satisfy both requirements simultaneously.
4. **Goal**: Design a region-adaptive, anisotropic noise perturbation strategy.
5. **Key Insight**: Associate pixel-level LR-HR residuals with uncertainty: larger residuals represent higher uncertainty, indicating a need for more noise.
6. **Core Idea**: Approximate uncertainty using the predicted residual $|g(y) - y|$ from a pre-trained SR network to generate pixel-wise noise weighting coefficients $w_u(y)$. Lower noise levels are applied to low-uncertainty regions to preserve more LR information.

## Method

### Overall Architecture
Input LR image $y_0$ $\to$ predictions from auxiliary SR network $g(\cdot)$ $\to$ calculate uncertainty $\psi_{est}(y_0) = \frac{1}{2}|g(y_0) - y_0|$ $\to$ generate weighting coefficients $w_u(y_0)$ $\to$ modify the forward process of ResShift to adjust the noise variance to $\kappa^2 w_u(y_0)^2 \alpha_t I$ (anisotropic) $\to$ denoising network $f_\theta$ predicts $x_0$ conditioned on $y_0$ and $g(y_0)$.

### Key Designs

1. **Uncertainty-guided Noise Weighting (UNW)**

    - **Function**: Adaptively adjust noise intensity based on the restoration difficulty of individual image regions.
    - **Mechanism**: Analysis reveals that the LR-HR residual $|y-x|$ follows a long-tailed distribution, with over 95% of data lying within $[0.01, 0.16]$. Experiments show that the sensitivity of perceptual quality to noise levels increases sharply as the residual grows, while fidelity remains virtually unchanged. Therefore, noise levels can be safely reduced in low-residual regions (flat areas) without sacrificing perceptual quality, thereby preserving more LR details. The weighting coefficient $w_u(y) = u(\psi_{est}(y))$ is a monotonically increasing function of uncertainty and is multiplied into the noise variance term of the forward diffusion.
    - **Design Motivation**: Change the super-resolution diffusion process from isotropic (uniform noise on all pixels) to anisotropic (regionally adjusted), which aligns more naturally with the intrinsic characteristics of the SR task.

2. **Uncertainty Estimation Based on Pre-trained SR Network**

    - **Function**: Estimate pixel-wise restoration uncertainty without requiring ground truth (GT) images.
    - **Mechanism**: If $g(\cdot)$ is well-trained, then $g(y) \approx x$, which implies $|g(y) - y| \approx |x - y|$. Consequently, the predicted residuals of the SR network approximate the ground-truth residuals. Visualizations verify that the predicted residuals are highly consistent with the ground-truth residuals in edge and textured regions. The uncertainty estimation is defined as $\psi_{est}(y) = \frac{1}{2}|g(y) - y|$.
    - **Design Motivation**: Avoid the need for GT images to calculate the true residual during inference by leveraging a pre-trained SR network as a proxy.

3. **Dual Conditional Inputs**

    - **Function**: Provide more accurate conditional guidance to the denoising network.
    - **Mechanism**: Concatenate the auxiliary SR network's output $g(y_0)$ with the original LR input $y_0$ to serve as conditioning inputs for the denoiser. Since $g(y_0)$ is a closer estimate to $x_0$ than $y_0$, it provides more precise structural guidance.
    - **Design Motivation**: Given that the auxiliary SR network is already utilized for uncertainty estimation, it is practical to simultaneously leverage its predictions as conditional input.

### Loss & Training
$$\mathcal{L}(\theta) = \sum_t [\|f_\theta(x_t, y_0, g(y_0), t) - x_0\|_2^2 + \lambda L_{per}(f_\theta(\cdot), x_0)]$$

A hybrid objective consisting of pixel-level L2 loss and LPIPS perceptual loss to balance fidelity and perceptual quality.

## Key Experimental Results

### Main Results

| Method | RealSR PSNR↑ | CLIPIQA↑ | MUSIQ↑ | Model Size |
|------|-------------|----------|--------|---------|
| ResShift | ~27.5 | ~0.65 | ~68 | 118M |
| StableSR | ~27.0 | ~0.68 | ~70 | - |
| **UPSR** | **~28.0** | **~0.70** | **~72** | **~80M** |

### Ablation Study

| UNW | SR Condition | PSNR↑ | CLIPIQA↑ | Description |
|-----|--------|-------|----------|------|
| ✗ | ✗ | baseline | baseline | ResShift baseline |
| ✗ | ✓ | +0.2 | +0.02 | SR condition alone is effective |
| ✓ | ✗ | +0.15 | +0.03 | UNW alone is effective |
| ✓ | ✓ | **+0.5** | **+0.05** | Combine both for best performance |

### Key Findings
- UNW effectively reduces unnecessary noise perturbations in flat regions, preserving more LR structural details.
- The model reduces parameters by ~30% while achieving superior performance, demonstrating the high efficiency of a more specialized diffusion process.
- Anisotropic noise yields particularly significant improvements in perceptual quality metrics (e.g., CLIPIQA, MUSIQ).
- State-of-the-art results are achieved on both real-world SR datasets (RealSR, RealSet) and classic benchmark datasets.

## Highlights & Insights
- The insight that **"different regions correspond to different diffusion timesteps"** is highly accurate: flat regions map to $t \to 0$ (virtually no noise) and textured regions map to $t \to T$ (high noise), elegantly integrating SR prior knowledge into the diffusion framework.
- Approximating uncertainty via the residual of a pre-trained SR network is a zero-cost solution, requiring no extra training for an uncertainty model.
- The dual-conditional input design is highly pragmatic: since the predictions from the auxiliary network are already available, utilizing them as conditions comes with no additional overhead.

## Limitations & Future Work
- The quality of the auxiliary SR network directly limits the accuracy of the uncertainty estimation.
- The specific formulation of $u(\cdot)$ (the monotonically increasing function) requires manual design.
- Performance has only been validated on 4x super-resolution; its efficacy under larger scaling factors remains to be investigated.
- Future work can explore end-to-end learning for uncertainty estimation to bypass the dependency on a pre-trained SR network.

## Related Work & Insights
- **vs ResShift**: Both embed LR into the initial state, but while ResShift uses isotropic noise, this work employs a more "specialized" anisotropic noise.
- **vs SR3**: SR3 starts from pure noise without leveraging LR information, whereas this work maximizes the utilization of LR prior knowledge.
- **vs LDM-SR**: LDM-SR performs diffusion in the latent space to enhance efficiency but still relies on isotropic noise; the proposed UNW framework is orthogonal to this approach.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing uncertainty and anisotropic noise to the super-resolution diffusion framework is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extremely thorough, with multi-dataset validation, detailed ablations, and visual analyses.
- Writing Quality: ⭐⭐⭐⭐ The writing style is highly experiment-driven, presenting data and figures with strong persuasive power.
- Value: ⭐⭐⭐⭐ Offers practical value with a smaller model size and better performance; the UNW concept can be transferred to other diffusion-based restoration tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)
- [\[CVPR 2025\] Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](../../CVPR2026/image_generation/vosr_a_vision_only_generative_model_for_image_super_resolution.md)

</div>

<!-- RELATED:END -->
