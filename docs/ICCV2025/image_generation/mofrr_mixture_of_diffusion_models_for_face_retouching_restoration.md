---
title: >-
  [Paper Note] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration
description: >-
  [ICCV 2025][Image Generation][Face retouching restoration] This paper introduces the Face Retouching Restoration (FRR) task for the first time and proposes the MoFRR framework—inspired by DeepSeek MoE—which activates retouching-type-specific experts (Wavelet DDIM) and a shared expert (general DDIM) via a router, achieving near-authentic restoration of retouched faces on the newly constructed million-scale RetouchingFFHQ++ dataset.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Face retouching restoration"
  - "mixture of experts"
  - "wavelet transform"
  - "diffusion models"
  - "frequency-domain decomposition"
date: 2026-05-08
content_hash: 8ab89fd7eeccd10c
---

# MoFRR: Mixture of Diffusion Models for Face Retouching Restoration

**Conference**: ICCV 2025
**arXiv**: [2507.19770](https://arxiv.org/abs/2507.19770)  
**Code**: Unavailable  
**Area**: Image Generation / Face Restoration
**Keywords**: Face retouching restoration, mixture of experts, wavelet transform, diffusion models, frequency-domain decomposition

## TL;DR

This paper introduces the Face Retouching Restoration (FRR) task for the first time and proposes the MoFRR framework—inspired by DeepSeek MoE—which activates retouching-type-specific experts (Wavelet DDIM) and a shared expert (general DDIM) via a router, achieving near-authentic restoration of retouched faces on the newly constructed million-scale RetouchingFFHQ++ dataset.

## Background & Motivation

**Social problem**: Face retouching operations (face slimming, eye enlargement, skin whitening, skin smoothing) are pervasive on social platforms, giving rise to aesthetic degradation, commercial fraud, and identity falsification. Norway, the United States, and Israel have enacted legislation requiring disclosure of retouched content.

**Research gap**: Existing work focuses on retouching detection, yet **how to restore authentic faces from retouched images** remains unanswered—a capability critical for tracing the true identity behind heavily retouched faces.

**FRR differs from conventional tasks**:
- Distinct from **Image Restoration (IR)**: IR targets high-frequency texture recovery, whereas FRR must recover low-frequency structural information (face shape, eye size).
- Distinct from **makeup removal**: Retouching involves facial structural changes (face slimming, eye enlargement), while makeup removal concerns only texture/color changes.
- Different retouching operations follow independent logic with distinct objectives, making single-model treatment inappropriate.

## Method

### Overall Architecture (MoE)

A divide-and-conquer strategy inspired by DeepSeek MoE:

1. **Router**: A ResNet-MAM multi-label classifier that outputs a 4-dimensional binary vector $[b_w, b_s, b_f, b_e]$, corresponding to whitening, smoothing, face slimming, and eye enlargement, respectively.
2. **Expert networks**: 4 specialized experts (WaveFRR) + 1 shared expert (standard DDIM).
3. **Merging module**: A lightweight UNet that fuses intermediate outputs from all activated experts along with the original image.

### Key Designs: WaveFRR Expert Model

Each WaveFRR adopts a dual-branch structure:

**Low-frequency branch (DDIM + IDEM)**:
- Applies Discrete Wavelet Transform (DWT) to the input image, yielding a low-frequency subband $x_{LL}$ and high-frequency subbands $x_H$.
- **Degree estimator**: ResNet50 predicts the degree $z$ of a specific retouching operation.
- **IDEM module**: Generates pixel-level conditions via a multi-scale channel attention network:

$$F = \text{MCA}(z+x_{LL}) \otimes x_{LL} + (1-\text{MCA}(z+x_{LL})) \otimes z$$
$$\hat{R} = \text{MCA}(F+\hat{y}_t) \otimes \hat{y}_t + (1-\text{MCA}(F+\hat{y}_t)) \otimes F$$
$$\tilde{x} = \text{Concat}(x_{LL}, \hat{R})$$

- A conditional DDIM samples the retouching-free low-frequency subband under this guidance.

**High-frequency branch (HFCAM)**:
$$\hat{y}_H = x_H + \text{Conv}(\text{CA}(\hat{y}_0, x_H))$$

The restored low-frequency subband $\hat{y}_0$ and the original high-frequency subbands $x_H$ are aligned via cross-attention to refine high-frequency details. The final output is reconstructed via inverse wavelet transform: $\hat{Y}_M = \text{IDWT}(\hat{y}_0, \hat{y}_H)$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{freq} + \mathcal{L}_{space} + \mathcal{L}_{class}$$

- $\mathcal{L}_{freq} = \mathcal{L}_{IDEM} + \mathcal{L}_{simple} + \mathcal{L}_{high}$: Frequency-domain loss (IDEM residual L2 + diffusion loss + high-frequency L2+TV).
- $\mathcal{L}_{space} = \mathcal{L}_{hyb}(Y,\hat{Y}) + \sum_M\mathcal{L}_{hyb}(Y,\hat{Y}_M)$: Spatial loss (L1 + 1-SSIM).
- $\mathcal{L}_{class}$: Cross-entropy loss for the router and degree estimator.

### Shared Expert Design

- Adopts a standard DDIM architecture (without wavelet decomposition); the architectural difference from WaveFRR encourages functional complementarity.
- Trained on a mixed-retouching data subset to capture universal patterns across retouching types.
- Always activated, independent of router gating.

## Key Experimental Results

### In-API Testing (Mixed Retouching)

| Method | PSNR↑ (Whitening) | PSNR↑ (Smoothing) | PSNR↑ (Face Slim) | PSNR↑ (Eye Enlarge) | PSNR↑ (Mixed) |
|--------|-------------------|-------------------|-------------------|----------------------|---------------|
| Input | 29.14 | 35.59 | 29.55 | 35.82 | 28.03 |
| Pix2pix | 27.72 | 28.55 | 27.34 | 28.41 | 28.73 |
| Restormer | 29.89 | - | - | - | - |
| **MoFRR** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Expert Visualization Analysis

| Expert | Function |
|--------|----------|
| Shared expert | Global retouching artifact handling |
| Face slimming expert | Restoring original face contour |
| Eye enlargement expert | Restoring original eye size |
| Smoothing expert | Recovering skin texture details |
| Whitening expert | Restoring original skin tone |

### Key Findings

- RetouchingFFHQ++ contains 1.07 million retouched images from 4 commercial APIs, far exceeding prior datasets in scale.
- Visualization of individual experts confirms that each expert learns targeted inverse operations for its corresponding retouching type.
- The cosine similarity distribution of facial feature embeddings demonstrates that restored results are significantly closer to the original identity.
- Cross-API testing validates the model's generalization capability.

## Highlights & Insights

1. **First FRR task formulation**: Extends retouching detection to retouching restoration, with broad application scenarios (forensic analysis, identity verification).
2. **MoE + diffusion model integration**: A shared-plus-specialized expert architecture inspired by DeepSeek.
3. **Wavelet-domain divide-and-conquer**: Low-frequency branch restores structure; high-frequency branch refines details—tailored to the characteristics of FRR.
4. **Million-scale dataset**: Extends RetouchingFFHQ and redefines the standard for retouching degree annotation.

## Limitations & Future Work

- Only 4 retouching types are supported, whereas real-world commercial beauty filters encompass a much richer set of operations.
- Classification errors in the router propagate and affect subsequent restoration quality.
- DDIM inference requires multi-step sampling, limiting real-time applicability.

## Related Work & Insights

- **Makeup removal**: PairedCycleGAN, PSGAN++, SSAT
- **Image restoration**: DR2, Restormer, ResDiff
- **Retouching detection**: ResNet-MAM, RetouchingFFHQ

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First FRR task definition + MoE diffusion experts
- Technical Depth: ⭐⭐⭐⭐ — Multi-level design of wavelet + DDIM + MoE
- Experimental Thoroughness: ⭐⭐⭐⭐ — In-API + cross-API evaluation
- Value: ⭐⭐⭐⭐ — Anti-retouching fraud, forensic evidence analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](../../CVPR2025/image_generation/svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](../../CVPR2025/image_generation/osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ICML 2025\] Gaussian Mixture Flow Matching Models](../../ICML2025/image_generation/gaussian_mixture_flow_matching_models.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)

</div>

<!-- RELATED:END -->
