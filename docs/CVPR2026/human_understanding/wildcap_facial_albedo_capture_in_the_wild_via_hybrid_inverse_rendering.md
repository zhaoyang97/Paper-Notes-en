---
title: >-
  [Paper Note] WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering
description: >-
  [CVPR 2026][Human Understanding][facial albedo capture] This paper proposes WildCap, a hybrid inverse rendering framework that reconstructs high-quality 4K facial diffuse albedo maps from casual in-the-wild smartphone vi…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "facial albedo capture"
  - "inverse rendering"
  - "diffusion prior"
  - "texel grid lighting"
  - "in-the-wild"
date: 2026-05-08
content_hash: 67b31bcc5f6b9a9b
---

# WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering

**Conference**: CVPR 2026
**arXiv**: [2512.11237](https://arxiv.org/abs/2512.11237)
**Code**: [Released](https://arxiv.org/abs/2512.11237) (as declared in the paper)
**Area**: Human Understanding
**Keywords**: facial albedo capture, inverse rendering, diffusion prior, texel grid lighting, in-the-wild

## TL;DR

This paper proposes WildCap, a hybrid inverse rendering framework that reconstructs high-quality 4K facial diffuse albedo maps from casual in-the-wild smartphone videos. The approach combines data-driven relighting (SwitchLight), model-based texel grid lighting optimization, and diffusion prior sampling, substantially closing the quality gap between in-the-wild capture and controlled-illumination methods.

## Background & Motivation

1. **Facial albedo capture is central to digital human creation**: Cloning a real person into the digital world requires high-quality facial reflectance maps, a problem studied for over two decades.
2. **High-quality methods rely on controlled illumination**: From Light Stage equipment to smartphone flash lighting, existing approaches assume known scene illumination, increasing capture cost and limiting accessibility.
3. **Model-based inverse rendering is unstable under complex illumination**: Jointly optimizing illumination and reflectance to match observed images becomes highly ill-posed and unstable in the presence of complex light transport effects such as shadows.
4. **Data-driven methods are robust but suffer from baking artifacts**: Networks such as SwitchLight can directly predict reflectance components but inevitably bake illumination effects (e.g., shadows) into their predictions.
5. **The two paradigms are complementary**: Model-based methods yield physically plausible decompositions but lack robustness; data-driven methods are robust but imperfect. Combining them is a natural direction.
6. **In-the-wild capture has significant practical value**: Enabling high-quality facial capture from casually recorded smartphone videos would substantially lower the barrier to digital human production.

## Method

### Overall Architecture: Hybrid Inverse Rendering

The pipeline consists of three stages:
1. **Data preprocessing**: Uniformly sample 300 frames (960×720) from a smartphone capture session, calibrate camera parameters via COLMAP, reconstruct a fine mesh with 2DGS, register the ICT template using Wrap3D, and select $V=16$ frames for reflectance estimation.
2. **Data-driven relighting**: Apply SwitchLight to predict per-frame diffuse albedo images $\{I^i\}$, converting complex in-the-wild illumination into a more constrained condition.
3. **Model-based optimization**: Interpret SwitchLight's baking artifacts as illumination effects in UV space, jointly optimizing a texel grid lighting model and sampling a diffusion prior to obtain a clean albedo map $A$.

### Key Design 1: Texel Grid Lighting Model

SwitchLight predictions are not produced by physical light sources; conventional spherical harmonics (SH) environment lighting cannot explain the non-physical shadow baking artifacts in these predictions.

- **Design motivation**: Assign local SH illumination to facial regions exhibiting baking artifacts, so that artifacts can be interpreted as "clean albedo under dark local lighting."
- **Specific structure**:
    - Global SH illumination $\gamma^g \in \mathbb{R}^{N_c}$ models the base illumination over the entire face.
    - A 2D UV-space grid $V \in \mathbb{R}^{\frac{H}{g} \times \frac{W}{g} \times N_c}$ stores local SH parameters.
    - Modulated via a binary mask $M$: $\gamma = \gamma^g + \gamma^V \cdot M[u][v]$.
    - Grid size $g=96$, second-order SH ($N_c=27$), queried via bilinear interpolation.
- **Mask acquisition**: Supports both manual annotation (Photoshop polygon lasso) and automatic detection (DiFaReli shadow detection lifted into UV space).

### Key Design 2: Diffusion Prior and Posterior Sampling Optimization

Increasing the expressiveness of the lighting model exacerbates ill-posedness (scale ambiguity between illumination and albedo), necessitating prior constraints:

- **Patch-level diffusion prior training**: A patch-level diffusion model at 64×64 resolution is trained on 48 Light Stage scans, modeling a 7-channel signal (3ch diffuse albedo + 3ch normal + 1ch specular albedo).
- **Initialization strategy**: The scan with the closest skin tone $x_0^{ref}$ is selected from the training set; $T_{init}=0.6T$ steps of noise are added before sampling begins (rather than starting from pure noise), reducing the number of sampling steps required.
- **Joint optimization**: At each diffusion timestep, both the reflectance map $x_t$ (diffusion denoising + photometric gradient guidance) and the lighting parameters $\theta_t$ (gradient descent + regularization) are updated simultaneously.

### Loss & Training

- **Photometric loss**: $\mathcal{L}_{pho} = \|I_{UV} - \Gamma_\theta(A, N_c)\|_2^2$
- **Lighting regularization**: $\mathcal{L}_{reg} = 0.1 \cdot \mathcal{L}_{TV} + \mathcal{L}_{neg}$
    - TV regularization enforces spatial smoothness of the lighting field.
    - Negative shading regularization $\mathcal{L}_{neg}$ ensures local lighting produces dark shading (to explain shadow baking).
- **Texture map construction**: Minimizes LPIPS + gradient-space L1 loss.

### Post-processing: 4K Super-Resolution

An RCAN super-resolution network upsamples the 1K reflectance map to 4K. Compared to DoRA, which requires 508 minutes to directly sample a 4K map, WildCap requires only **8 minutes** (24 GB RTX 4090).

## Key Experimental Results

### Main Results (Facial Relighting, Average over 6 Subjects)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| DeFace* | 22.20 | 0.9279 | 0.1192 |
| FLARE* | 27.81 | 0.9411 | 0.0929 |
| **WildCap (Ours)** | **28.79** | **0.9520** | **0.0610** |

### Main Results (Synthetic Data — Digital Emily, Albedo Reconstruction)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| DeFace* | 28.43 | 0.9791 | 0.0826 |
| FLARE* | 22.48 | 0.9742 | 0.0571 |
| **WildCap (Ours)** | **28.71** | **0.9802** | **0.0388** |

### Ablation Study

- **w/o Hybrid** (optimizing directly on raw images): Fails to effectively separate specular highlights and shadows under complex illumination.
- **w/o TGL** (global SH only): Cannot explain non-physical baking artifacts; visible shadow residuals remain.
- **w/o Prior** (no diffusion prior; direct Adam optimization per texel): Produces severe artifacts and fails to converge to a plausible reflectance map.
- **Grid size ablation**: $g=1/24$ provides insufficient expressiveness; $g=384$ is overly smooth and loses fine detail; $g=96$ achieves the best balance.

## Highlights & Insights

1. **Elegant hybrid inverse rendering framework**: The approach organically combines the robustness of data-driven methods with the physical plausibility of model-based methods in a concise and well-motivated design.
2. **Novel and effective Texel Grid Lighting Model**: Transcends the limitations of physical illumination models by using a non-physical yet more expressive local SH grid to account for baking artifacts in network predictions.
3. **Diffusion prior elegantly resolves scale ambiguity**: Sampling albedo from a reasonable distribution while jointly optimizing illumination converts an ill-posed problem into a well-posed one.
4. **High efficiency**: Requires only 8 minutes versus 508 minutes for DoRA, while achieving comparable quality to controlled-illumination methods.
5. **Thorough experimentation**: Includes comprehensive ablations, quantitative evaluation on synthetic data, cross-setting comparison with DoRA, diverse scene demonstrations, and failure case analysis.

## Limitations & Future Work

1. **Dependency on SwitchLight preprocessing**: SwitchLight is a closed-source commercial model accessible only via API, limiting the reproducibility and extensibility of the method.
2. **Automatic shadow detection relies on DiFaReli**: Iterative diffusion sampling is slow and may miss ambient occlusion effects.
3. **Continuity constraints on lighting representation**: When SwitchLight predictions contain sharp shadow boundaries (e.g., under harsh noon sunlight), the continuous grid representation cannot fully remove them.
4. **Limited training data scale**: The diffusion prior is trained on only 48 Light Stage scans, with limited racial and skin-tone diversity (33 Caucasian / 9 African / 6 Asian subjects).
5. **Requires a target skin tone reference**: Although obtainable manually or automatically, this introduces an additional step in the pipeline.

## Related Work & Insights

- **vs. DeFace**: DeFace partitions the face into a limited number of regions (5–10), each corresponding to a trainable network, offering limited expressiveness; WildCap's texel grid provides finer granularity.
- **vs. FLARE**: FLARE employs a split-sum approximation to model illumination; the physical model cannot account for non-physical baking artifacts.
- **vs. DoRA** (controlled-illumination method): WildCap achieves quality comparable to DoRA under the more challenging in-the-wild setting, better preserves personal features (e.g., moles), and is approximately 63× faster.
- **vs. Rainer et al.**: Using a small MLP to model shading is difficult to optimize within a diffusion posterior sampling framework; WildCap's grid representation is more amenable to optimization.
- **vs. test scenarios in Xu et al. / Rainer et al.**: Prior methods are evaluated only on mildly shadowed scenes; WildCap addresses the more challenging case of strong cast shadows.

## Rating

- Novelty: ⭐⭐⭐⭐ — The hybrid inverse rendering framework and the texel grid lighting model are conceptually novel; the joint diffusion prior optimization is technically sophisticated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablations, thorough quantitative and qualitative comparisons, synthetic data evaluation, and failure case analysis.
- Writing Quality: ⭐⭐⭐⭐ — Overall clear presentation with well-motivated problem setup and detailed supplementary material.
- Value: ⭐⭐⭐⭐ — Substantially lowers the barrier to facial appearance capture with practical implications for digital human production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](../../ICCV2025/human_understanding/monocular_facial_appearance_capture_in_the_wild.md)
- [\[CVPR 2026\] RAM: Recover Any 3D Human Motion in-the-Wild](ram_recover_any_3d_human_motion_in-the-wild.md)
- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](a_two_stage_dual_modality_model_for_facial_expression_recognition.md)
- [\[CVPR 2026\] HUM4D: A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture](hum4d_markerless_motion_capture.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)

</div>

<!-- RELATED:END -->
