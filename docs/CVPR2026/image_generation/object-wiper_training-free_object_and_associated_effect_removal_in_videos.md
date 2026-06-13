---
title: >-
  [Paper Note] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos
description: >-
  [CVPR 2026][Image Generation][Video object removal] This paper presents Object-WIPER, the first training-free framework for removing objects and their associated visual effects (shadows, reflections, mirror images…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Video object removal"
  - "associated effects"
  - "training-free"
  - "attention mechanism"
  - "diffusion models"
date: 2026-05-08
content_hash: 2327c5ab5513b125
---

# Object-WIPER: Training-Free Object and Associated Effect Removal in Videos

**Conference**: CVPR 2026
**arXiv**: [2601.06391](https://arxiv.org/abs/2601.06391)  
**Code**: Coming soon  
**Area**: Image Generation / Video Editing
**Keywords**: Video object removal, associated effects, training-free, attention mechanism, diffusion models

## TL;DR
This paper presents Object-WIPER, the first training-free framework for removing objects and their associated visual effects (shadows, reflections, mirror images, etc.) in videos. It leverages text-visual cross-attention and visual self-attention within DiT to localize associated effect regions, achieves clean removal via foreground re-initialization and attention scaling, and introduces the TokSim metric along with WIPER-Bench, a real-world benchmark.

## Background & Motivation

**Background**: Video object removal is a critical technique in film production and privacy protection. Classical methods (PatchMatch / graph cut) and learning-based methods (Propainter) focus on inpainting the object region while entirely ignoring associated effects (shadows/reflections). Recent diffusion-based approaches (VACE/Videopainter) also preserve associated effects.

**Limitations of Prior Work**: (a) Nearly all existing methods retain shadows/reflections, producing visual artifacts; (b) ROSE handles associated effects but requires training on large amounts of synthetic data; (c) Omnimatte-Zero extends user masks to cover associated regions but relies on an external point-tracking model (TAP-Net), which fails under fast motion or transparent objects, and its expansion strategy is suboptimal.

**Key Challenge**: Object removal is not equivalent to region inpainting — a truly clean removal must simultaneously eliminate all "visual traces" of the object (shadows, reflections, mirror images, etc.).

**Goal**: Simultaneously remove objects and all their associated visual effects without any training.

**Key Insight**: Exploit the shared text-visual embedding space in MMDiT to directly localize associated effects, without relying on external models.

**Core Idea**: Cross-attention localizes associated effect seeds → self-attention refines → foreground re-initialization + attention scaling → adaptive temporal masking.

## Method

### Overall Architecture
The inputs are an RGB video $\mathcal{I}_k$, an object mask $\mathbf{M}^{obj}$, and text prompts $\{P_s, P_T\}$ describing the object and its effects. Processing proceeds in three steps: (1) associated effect localization; (2) inversion to obtain structured noise while saving background values; (3) foreground re-initialization followed by denoising to generate a clean video.

### Key Designs

1. **Associated Effect Localization**:

    - Function: Identify the spatial locations of object-associated effects (shadows, reflections, etc.) in the video.
    - Mechanism: **Two-step approach** — Step 1: Extract visual tokens highly correlated with object/effect text tokens from $T\to I$ cross-attention: $\bar{\mathbf{A}}^{\tilde{T}\to I} = \text{Mean}(\text{Softmax}(\frac{\mathbf{Q}_{\tilde{T}}\cdot\mathbf{K}_I^\top}{\sqrt{d}}))$, then apply Otsu thresholding to obtain a proposal mask $m^{PRO}$. Step 2: Use visual self-attention $\mathbf{A}^{I\to I}$ to compute each token's response ratio to $m^{PRO}$, then threshold to obtain the final mask $\mathbf{M}^{AE}$.
    - Design Motivation: (a) Expanding only from the object mask (as in Omnimatte-Zero) misses weakly activated regions; (b) Cross-attention provides semantic localization but is incomplete (with internal holes); (c) Self-attention refinement fills these holes — tokens belonging to the same object necessarily exhibit high self-attention.
    - Difference from Omnimatte-Zero: Does not rely on an external point-tracking model; leverages DiT's intrinsic attention for greater robustness.

2. **Timestep-Adaptive Masking**:

    - Function: Address the insufficient coverage of fixed masks in noise space.
    - Mechanism: During inversion, compute an object response score $RS_p(j) = \frac{\sum_{y\in\mathbf{M}^{obj}(j)}A_{p,y}^{I\to I}}{\sum_{x\in\mathcal{I}(j)}A_{p,x}^{I\to I}}$; as the timestep increases, the object's "presence" diffuses via self-attention, and thresholding yields an adaptive mask $\hat{M}_t^{obj}$.
    - Design Motivation: During inversion toward the noise distribution, self-attention causes the object's influence to spread progressively; a fixed mask cannot fully cover this spread.

3. **Attention Scaling**:

    - During inversion: Suppress background attention to the foreground: $\tilde{\mathbf{A}}^{bg\to obj} = \text{Softmax}(\frac{\mathbf{Q}_I^{bg}\cdot(c\mathbf{K}_I^{obj})^\top}{\sqrt{d}})$, where $c<1$.
    - During denoising: Amplify foreground attention to the background: $\tilde{\mathbf{A}}^{obj\to bg} = \text{Softmax}(\frac{\mathbf{Q}_I^{obj}\cdot(b\mathbf{K}_I^{bg})^\top}{\sqrt{d}})$, where $b>1$.
    - Design Motivation: During inversion, reduce background "contamination" by the foreground; during denoising, enable the re-initialized foreground to actively acquire semantics from the background.

4. **Foreground Re-initialization**:

    - Function: Replace the foreground region in the inverted latent with Gaussian noise.
    - Mechanism: $\tilde{\mathbf{Z}}_1 = \mathbf{Z}_1\odot(1-\mathbf{M}^{obj}\cup\mathbf{M}^{AE}) + \varepsilon\odot(\mathbf{M}^{obj}\cup\mathbf{M}^{AE})$
    - Design Motivation: Eliminate any residual prior from the object and its associated effects.

5. **TokSim Metric**:

    - Mechanism: $\text{TokSim} = 100\cdot\frac{1}{F}\sum_z\sum_i \lambda_z^k\cdot(1-\eta_z^k)\cdot\tau_z^k$, where $\lambda$ rewards temporal consistency, $\eta$ penalizes object residuals, and $\tau$ rewards foreground-background blending.

### Loss & Training
- Entirely training-free; built upon a pretrained T2V DiT.
- At inference, only attention manipulation and value copying are required.

## Key Experimental Results

### Main Results

| Method | Training | DAVIS TokSim↑ | WIPER TokSim↑ | DAVIS BG-PSNR↑ | DAVIS Text-align↑ |
|------|------|--------------|---------------|----------------|-------------------|
| Propainter | ✓ | 28.24 | 20.99 | 34.01 | 26.18 |
| ROSE | ✓ | 29.36 | 30.02 | 26.97 | 26.13 |
| VACE | ✓ | 15.86 | 11.53 | 24.48 | 24.01 |
| Gen-Prop | ✓ | 30.52 | - | 24.27 | 25.89 |
| KV-Edit-Video | ✗ | 28.68 | 23.26 | 25.78 | 25.21 |
| Attentive-Eraser | ✗ | 30.82 | 25.28 | 28.07 | 26.31 |
| **Object-WIPER** | **✗** | **32.80** | **33.09** | 23.02 | **26.63** |

### Ablation Study

| Configuration | TokSim↑ | BG-PSNR↑ | Text-align↑ |
|------|---------|----------|-------------|
| Full Object-WIPER | 32.80 | 23.02 | 26.63 |
| w/o attention scaling | 32.97 | 21.92 | 26.42 |
| w/o adaptive mask | 32.10 | 22.73 | 26.44 |
| w/o re-initialization | 30.36 | 23.47 | 25.92 |
| w/o $\mathbf{M}^{AE}$ | 32.18 | 23.10 | 26.17 |

### Key Findings
- Without any training, Object-WIPER surpasses all trained methods on TokSim, including ROSE, which is specifically trained for associated effects.
- TokSim is far more discriminative than BG-PSNR: VAE reconstruction (without object removal) achieves BG-PSNR of 34.05 but TokSim of only 0.32.
- Re-initialization is the most critical component (removing it causes TokSim to drop by 2.44).
- The associated effect mask $\mathbf{M}^{AE}$ is essential for WIPER-Bench — shadows and reflections can only be removed when it is included.
- Adaptive masking is indispensable in fast-motion scenarios (e.g., high-speed vehicles).

## Highlights & Insights
- **Intrinsic MMDiT attention for associated effect localization**: The approach requires no external models and exploits semantic associations in the shared text-visual space for precise localization. This technique is transferable to any MMDiT-based editing task.
- **TokSim metric design** is elegant: it simultaneously measures removal completeness, temporal consistency, and background blending, exposing fundamental flaws in existing metrics.
- **WIPER-Bench** is the first object removal benchmark covering real-world scenarios including mirrors, transparent objects, and multiple associated effects.

## Limitations & Future Work
- BG-PSNR is inferior to trained methods, as the background is also regenerated by the diffusion model.
- The approach depends on text descriptions of the object and effect types, limiting automation.
- Video resolution is constrained by the pretrained model.
- Only dynamic objects are addressed; static object removal is not discussed.

## Related Work & Insights
- **vs. Omnimatte-Zero**: Does not rely on TAP-Net point tracking; the localization strategy is more complete.
- **vs. ROSE/Gen-Prop**: Training-based methods require large amounts of synthetic data, whereas Object-WIPER incurs zero training cost.
- **vs. KV-Edit**: KV-Edit is designed for images; naive extension to video yields poor results.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to address associated effect localization and removal within DiT; TokSim metric is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + new benchmark + new metric + complete ablation.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation and methodology are presented in a coherent, layered manner.
- Value: ⭐⭐⭐⭐⭐ WIPER-Bench and TokSim offer lasting value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](../../ICML2026/image_generation/adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)

</div>

<!-- RELATED:END -->
