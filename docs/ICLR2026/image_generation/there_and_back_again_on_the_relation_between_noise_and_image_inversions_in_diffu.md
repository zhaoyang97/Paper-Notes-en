---
title: >-
  [Paper Note] There and Back Again: On the Relation between Noise and Image Inversions in Diffusion Models
description: >-
  [ICLR 2026][Image Generation][DDIM inversion] This paper conducts an in-depth analysis of the error mechanisms in DDIM inversion, revealing that latent encodings exhibit low diversity and high correlation in smooth image regions (e.g., sky), traces this phenomenon to inaccurate noise predictions in the early inversion steps, and proposes a simple fix that replaces the first few inversion steps with forward diffusion.
tags:
  - ICLR 2026
  - Image Generation
  - DDIM inversion
  - latent encoding
  - noise correlation
  - smooth regions
  - forward diffusion fix
date: 2026-05-08
content_hash: 3efd1f9d7337905b
---

# There and Back Again: On the Relation between Noise and Image Inversions in Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2410.23530](https://arxiv.org/abs/2410.23530)
**Code**: [GitHub](https://github.com/luk-st/taba)
**Area**: Diffusion Models / Inversion Analysis / Image Editing
**Keywords**: DDIM inversion, latent encoding, noise correlation, smooth regions, forward diffusion fix

## TL;DR

This paper conducts an in-depth analysis of the error mechanisms in DDIM inversion, revealing that latent encodings exhibit low diversity and high correlation in smooth image regions (e.g., sky), traces this phenomenon to inaccurate noise predictions in the early inversion steps, and proposes a simple fix that replaces the first few inversion steps with forward diffusion.

## Background & Motivation

Diffusion models lack an explicit low-dimensional latent space encoding editable features of the data. DDIM inversion partially addresses this by reversing the denoising trajectory to map an image to its approximate initial noise. However:

**Unclear sources of inversion error**: Although it is known that DDIM inversion produces latents that are not perfectly Gaussian, the root cause and manifestations have not been systematically studied.

**Limitations of existing improvements**: Methods such as Null-text inversion and Renoise improve reconstruction quality but do not truly resolve the non-Gaussianity of the latent space.

**Poor manipulability of latent encodings**: Interpolation and editing in the inverted latent space yield lower quality compared to the original noise space.

## Method

### Analysis Framework

The study examines the relationships among three variables:
- **Noise** $\mathbf{x_T}$: the Gaussian input used to generate images
- **Sample** $\mathbf{x_0}$: the image produced by the diffusion model
- **Latent encoding** $\hat{\mathbf{x}}_T$: the result of DDIM inversion

### Key Finding 1: Latent Encodings Deviate from the Gaussian Distribution

Pearson correlation coefficients computed within 8×8 pixel patches:

| Model | Noise Corr. | Latent Corr. | Sample Corr. |
|-------|-------------|--------------|--------------|
| ADM-32 | 0.039 | **0.382** | 0.964 |
| ADM-64 | 0.039 | **0.126** | 0.925 |
| IF | 0.039 | **0.498** | 0.936 |
| LDM | 0.039 | **0.045** | 0.645 |
| DiT | 0.041 | **0.103** | 0.748 |
| SDXL | 0.036 | **0.155** | 0.637 |

Latent encodings exhibit substantially higher correlation than noise, and image structure patterns are visually observable in them.

### Key Finding 2: Smooth Regions Are the Primary Source of Error

Images are partitioned into "smooth regions" (e.g., sky, background) and "non-smooth regions":

| Model | Smooth Error | Non-Smooth Error | Smooth Std. | Non-Smooth Std. |
|-------|-------------|-----------------|-------------|-----------------|
| ADM-32 | **0.49** | 0.43 | **0.34** | 0.46 |
| IF | **0.56** | 0.40 | **0.46** | 0.72 |
| LDM | **0.13** | 0.03 | **0.45** | 0.59 |
| DiT | **0.12** | 0.06 | **0.43** | 0.54 |

Smooth regions exhibit higher inversion error and lower latent diversity.

### Key Finding 3: The Problem Originates from the Initial Inversion Steps

Analysis of the denoising trajectory reveals:
- Intermediate states $x_t$ begin converging toward the inverted latent $\hat{\mathbf{x}}_T$ at approximately 50–70% of the generation trajectory.
- Latent encodings retain certain properties of the original sample.
- **Crucially**, noise predictions in the first few inversion steps are particularly inaccurate and exhibit low diversity in smooth regions.

Linear interpolation path analysis $\|(1-\lambda)\mathbf{x_T} + \lambda \hat{\mathbf{x}}_T - x_t\|_2$ demonstrates that the generation trajectory progressively converges toward the inverted latent.

### Key Finding 4: DDIM Latent Space Exhibits Poor Manipulability

**Interpolation experiments**: Spherical interpolation in the DDIM latent space yields lower quality than in the original noise space.
**Editing experiments**: Edits applied to smooth image regions are particularly limited in quality.

### Key Finding 5: Existing Improvements Do Not Address the Root Problem

| Method | Reconstruction Improvement | Gaussianity Preserved |
|--------|---------------------------|----------------------|
| Null-text inversion | ✓ | ✗ |
| Renoise | ✓ | ✗ |
| DPM-Solver inversion | ✓ | ✗ |
| Regularization methods | Partial | Partial |

### Proposed Fix

**Simple fix**: Replace the first $k$ steps of DDIM inversion with forward diffusion.

$$\text{First } k \text{ steps: } x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t$$
$$\text{Remaining steps: standard DDIM inversion}$$

**Effects**:
- Successfully decorrelates latent encodings
- Does not degrade reconstruction quality
- Improves interpolation and editing quality
- Particularly effective in smooth regions

## Experiments

### Models Evaluated
Seven diffusion models spanning pixel-space / latent-space, conditional / unconditional, and U-Net / DiT architectures.

### Fix Validation

| Strategy | Correlation↓ | Reconstruction↑ | Interpolation↑ | Editing↑ |
|----------|-------------|----------------|----------------|---------|
| Standard DDIM inversion | High | Baseline | Low | Low |
| + Regularization | Medium | Baseline | Medium | Medium |
| + Forward diffusion (first k steps) | **Low** | Maintained | **High** | **High** |

### Extension to Flow Matching
The same correlation and low-diversity issues are observed in Flow Matching models.

### Ablation Study

| Parameter | Effect |
|-----------|--------|
| Increasing replacement steps $k$ | Diversity improves but reconstruction may degrade |
| Increasing DDIM steps | Error decreases but the problem persists |
| Different model architectures | Trends remain consistent |

## Highlights & Insights

1. **Depth of systematic analysis**: Inversion error patterns are comprehensively validated across 7 models.
2. **Discovery of the smooth-region problem**: The spatial distribution of errors is precisely localized.
3. **Simple yet effective fix**: Replacing only the first few steps yields significant improvements.
4. **Cross-architecture generalization**: Applicable to U-Net, DiT, pixel-space, and latent-space models alike.
5. **Correction of community understanding**: Existing improvement methods do not truly resolve the underlying non-Gaussianity of latent encodings.

## Limitations & Future Work

1. The forward diffusion steps introduced by the fix add stochasticity, which may affect deterministic reconstruction.
2. The optimal number of replacement steps $k$ requires tuning per model.
3. The analysis is primarily conducted with the DDIM sampler; generalization to other samplers warrants further investigation.
4. The definition of smooth regions relies on a manually set threshold $\tau = 0.025$.
5. No theoretical explanation is provided for why noise predictions in the early steps are particularly inaccurate in smooth regions.

## Related Work & Insights

- **DDIM inversion**: Song et al. (2021), Dhariwal & Nichol (2021)
- **Inversion improvements**: Null-text inversion (Mokady 2023), Renoise (Garibi 2024)
- **Image editing**: P2P (Hertz 2022), SDEdit (Meng 2021)
- **Flow Matching**: Lipman et al. (2023)

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematic analytical perspective is novel with findings of substantial depth
- **Value**: ⭐⭐⭐⭐ — The simple fix is directly applicable in practice
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparison and validation across 7 models
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Analysis builds progressively with clear logic

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICLR 2026\] Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

<!-- RELATED:END -->
