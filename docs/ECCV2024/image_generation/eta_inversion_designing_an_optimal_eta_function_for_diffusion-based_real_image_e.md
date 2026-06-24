---
title: >-
  [Paper Note] Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing
description: >-
  [ECCV 2024][Image Generation][Diffusion Inversion] By theoretically analyzing the role of the $\eta$ parameter in the DDIM sampling equation, this work designs time- and region-dependent $\eta$ functions to achieve more flexible and precise real image editing.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion Inversion"
  - "Image Editing"
  - "DDIM Sampling"
  - "Eta Function"
  - "Real Image Editing"
date: 2026-05-08
content_hash: 35cefbb3412d13b0
---

# Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing

**Conference**: ECCV 2024  
**arXiv**: [2403.09468](https://arxiv.org/abs/2403.09468)  
**Code**: Available  
**Area**: Image Generation  
**Keywords**: Diffusion Inversion, Image Editing, DDIM Sampling, Eta Function, Real Image Editing

## TL;DR

By theoretically analyzing the role of the $\eta$ parameter in the DDIM sampling equation, this work designs time- and region-dependent $\eta$ functions to achieve more flexible and precise real image editing.

## Background & Motivation

Image editing based on diffusion models has become an important research direction in the field of image generation. A typical editing pipeline is: first performing inversion on the real image to map it into the noise space, and then injecting new text guidance during the denoising process to achieve editing. However, existing methods face a key challenge in editing quality—**the trade-off between editing faithfulness and source image fidelity**.

Specifically: (1) Although DDIM inversion is theoretically deterministic, discretization errors in practical applications lead to imperfect reconstruction; (2) Methods such as Null-text inversion improve reconstruction by optimizing the null-text embedding, but suffer from limited editing flexibility; (3) Most existing methods ignore the role of the $\eta$ parameter in the DDIM sampling equation, simply setting it to 0 (deterministic sampling) or 1 (fully stochastic sampling).

The Core Idea of this paper is: the $\eta$ parameter is actually a key lever controlling the degree of editing. Through an in-depth theoretical analysis of the mathematical role of $\eta$ in DDIM sampling, the authors discover that $\eta$ can be flexibly adjusted in both the temporal and spatial (regional) dimensions, thereby achieving precise control over the editing scope and intensity. This finding opens up a brand new design space.

## Method

### Overall Architecture

The overall pipeline of Eta Inversion consists of three steps: (1) performing standard DDIM inversion on the input image to obtain the noise representation; (2) designing a time- and region-dependent $\eta(t, r)$ function; (3) performing edit-guided sampling using the designed $\eta$ function during the denoising sampling process.

### Key Designs

1. **Theoretical Analysis of the $\eta$ Parameter**:
    - Function: Reveal the mechanism of how $\eta$ affects editing capability in DDIM sampling.
    - Mechanism: Starting from the DDIM sampling equation, derive the impact of $\eta$ on the signal-to-noise ratio during the denoising process. It is theoretically proven that: larger $\eta$ values introduce more randomness, causing the sampling trajectory to deviate more from the inversion trajectory, thereby allowing a greater degree of editing; smaller $\eta$ values maintain better fidelity of the source image.
    - Design Motivation: Existing methods either completely ignore $\eta$ or simply treat it as a hyperparameter to tune, lacking a systematic understanding of its role.

2. **Time-dependent $\eta$ Function**:
    - Function: Fine-tune editing intensity by applying different $\eta$ values at different timesteps.
    - Mechanism: Design $\eta(t)$ as a function of time, using larger $\eta$ in the early denoising steps (high noise levels) to allow structural editing, and smaller $\eta$ in the later steps (low noise levels) to preserve detail fidelity. This aligns with the "coarse-to-fine" generation nature of diffusion models.
    - Design Motivation: Different timesteps affect representations at different semantic levels of the image; a uniform $\eta$ value cannot accommodate this hierarchical structure.

3. **Region-dependent $\eta$ Function**:
    - Function: Apply different $\eta$ values across different spatial regions of the image.
    - Mechanism: Identify the edit region through cross-attention maps or user-specified masks, using a larger $\eta$ value in regions that require editing and a smaller $\eta$ value in regions that should remain unchanged. This achieves the objective of "editing what should be edited, and preserving what should be preserved."
    - Design Motivation: Real image editing typically requires modifying only local areas of an image, whereas a globally uniform $\eta$ value leads to unwanted changes in non-edited regions.

### Loss & Training

Eta Inversion is a training-free, inference-time method. The core lies in the design of the $\eta$ function rather than model parameter optimization. The $\eta(t, r)$ function is determined through the following strategies:
- Temporal Dimension: Employing a piecewise function or smooth decay function, gradually transitioning from a larger $\eta$ at higher timesteps to a smaller $\eta$ at lower timesteps.
- Spatial Dimension: Utilizing the softmax distribution of the cross-attention map as regional weights to automatically identify editing areas.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| PIE-Bench | Structure Dist ↓ | Best | Null-text Inv | Significant reduction |
| PIE-Bench | CLIP Sim ↑ | Best | P2P+NTI | +2-5% |
| PIE-Bench | LPIPS ↓ | Best | DirectInverse | Significant reduction |
| User Study | Preference Rate | >60% | All baselines | Obvious lead |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $\eta=0$ (Standard DDIM) | Weak editing capability | Overly conservative, difficult to achieve meaningful edits |
| $\eta=1$ (Fully stochastic) | Low fidelity | Introduces too much randomness, destroying the source image |
| Time-dependent $\eta$ only | Moderate performance | Better than fixed $\eta$, but non-editing regions still change |
| Time + Region dependent $\eta$ | Best | Precise editing and high fidelity |

### Key Findings

- The $\eta$ parameter is an important control lever in DDIM sampling that has long been overlooked.
- The time-dependent $\eta$ function exploits the hierarchical generation characteristics of diffusion models.
- The region-dependent $\eta$ function achieves precise local editing control.
- The method requires no additional training or optimization, making it plug-and-play during inference.

## Highlights & Insights

- The theoretical analysis is deep and practical, translating mathematical derivations into concrete design schemes.
- The method is simple and elegant: it does not modify the model architecture, requires no training, and achieves better editing simply by adjusting sampling parameters.
- The analyses of $\eta$ in temporal and spatial dimensions complement each other well.
- It provides a new perspective on controllable generation in diffusion models.

## Limitations & Future Work

- The specific form of the $\eta$ function still requires manual design or heuristic selection.
- For scenarios requiring major structural editing (e.g., pose changes), the method's effectiveness may be limited.
- The region-dependent $\eta$ relies on the quality of the cross-attention maps, which may not be precise enough for complex scenes.
- Future work could explore learning the optimal $\eta$ function for further automation.

## Related Work & Insights

- **DDIM Inversion**: The deterministic sampling method proposed by Song et al., which serves as the theoretical foundation of this work.
- **Null-text Inversion**: Optimizes null-text to improve inversion quality, but incurs large computational overhead.
- **Prompt-to-Prompt**: Achieves editing by manipulating cross-attention, complementing the region-dependent $\eta$ of this work.
- Insight: There are still many underutilized control variables in the diffusion model sampling process that warrant deeper theoretical analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ Approaches from the neglected $\eta$ parameter, backed by solid theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative and qualitative evaluations with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and intuitive experimental presentation.
- Value: ⭐⭐⭐⭐ Highly practical, providing a new training-free method for diffusion model editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)

</div>

<!-- RELATED:END -->
