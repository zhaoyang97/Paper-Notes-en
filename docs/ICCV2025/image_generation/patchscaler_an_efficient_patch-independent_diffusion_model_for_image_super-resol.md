---
title: >-
  [Paper Note] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution
description: >-
  [ICCV 2025][Image Generation][Super-Resolution] This paper proposes PatchScaler, a patch-level independent diffusion super-resolution pipeline that employs a Global Restoration Module to generate confidence maps quantifying per-region reconstruction difficulty, partitions patches into easy/medium/hard groups with different sampling step budgets, and incorporates a texture prompt retrieval mechanism — achieving superior quality on RealSR at only 0.23× the inference time of Res…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Super-Resolution"
  - "Patch-Adaptive Sampling"
  - "Diffusion Acceleration"
  - "Texture Prompt"
  - "DiT"
date: 2026-05-08
content_hash: 21a46ec8bca782bc
---

# PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution

**Conference**: ICCV 2025
**arXiv**: [2405.17158](https://arxiv.org/abs/2405.17158)  
**Code**: [https://github.com/yongliuy/PatchScaler](https://github.com/yongliuy/PatchScaler)  
**Area**: Image Generation
**Keywords**: Super-Resolution, Patch-Adaptive Sampling, Diffusion Acceleration, Texture Prompt, DiT

## TL;DR

This paper proposes PatchScaler, a patch-level independent diffusion super-resolution pipeline that employs a Global Restoration Module to generate confidence maps quantifying per-region reconstruction difficulty, partitions patches into easy/medium/hard groups with different sampling step budgets, and incorporates a texture prompt retrieval mechanism — achieving superior quality on RealSR at only 0.23× the inference time of ResShift.

## Background & Motivation

**Efficiency of diffusion-based SR**: Diffusion models have substantially improved perceptual quality in super-resolution, but their iterative sampling procedure leads to poor inference efficiency, especially for high-resolution images with prohibitive computational cost.

**Suboptimality of uniform sampling**: Existing acceleration methods (conditional distillation, redefined diffusion processes) uniformly reduce sampling steps across all regions, ignoring the heterogeneous reconstruction difficulty — structurally simple regions can be reconstructed in a few steps, whereas texture-rich regions require significantly more.

**Limitations of text prompts**: In SR tasks, the alignment between text prompts and image content is far weaker than in text-to-image generation; local texture restoration benefits more from visual-level conditioning than textual descriptions.

**Core observation**: As shown in Figure 1(a), simple patches can be reconstructed with high quality in just 2 steps, while complex patches require up to 15 steps.

## Method

### Overall Architecture

PatchScaler operates in three stages:

1. **Global Restoration Module (GRM)**: Removes degradation and produces coarse HR features along with a confidence map.
2. **Patch-adaptive Grouped Sampling (PGS)**: Groups patches by confidence score and assigns different sampling configurations accordingly.
3. **Patch-DiT**: Refines each group of patches conditioned on texture prompts.

### Key Design 1: Global Restoration Module and Confidence Map

The GRM jointly outputs coarse HR features $\mathbf{y}_{HR}$ and a confidence map $C$, trained with the objective:

$$L(\theta) = \|\mathbf{y}_{HR} - \mathbf{x}_{HR}\|_1^2 + \lambda(C\|\mathbf{y}_{HR} - \mathbf{x}_{HR}\|_2^2 - \eta\log(C))$$

Low-confidence regions indicate areas where GRM struggles to reconstruct (requiring more diffusion steps), whereas high-confidence regions are already well recovered.

### Key Design 2: Patch-Adaptive Grouped Sampling (PGS)

Coarse HR features are divided into patches and grouped by average confidence:

$$Qmap_{\mathbf{y}_{0,i}} = \begin{cases}\text{Simple}, & Avg(C\langle\mathbf{y}_{0,i}\rangle) \in (\gamma_1, 1] \\\text{Medium}, & Avg(C\langle\mathbf{y}_{0,i}\rangle) \in (\gamma_2, \gamma_1] \\\text{Hard}, & Avg(C\langle\mathbf{y}_{0,i}\rangle) \in [0, \gamma_2]\end{cases}$$

**Shortcut path derivation**: Let $\mathbf{x}_0 = \mathbf{y}_0 + \triangle\mathbf{x}_0$; after degradation removal by GRM, $\triangle\mathbf{x}_0$ is small. An appropriate intermediate timestep $\tau$ is identified such that $\sqrt{\bar{\alpha}_\tau}\triangle\mathbf{x}_0 \to 0$:

$$q(\mathbf{x}_\tau|\mathbf{y}_0) \approx \mathcal{N}(\mathbf{x}_\tau; \sqrt{\bar{\alpha}_\tau}\mathbf{y}_0, (1-\bar{\alpha}_\tau)\mathbf{I})$$

Different groups are assigned different $(T_i, N_i)$ configurations:
- Simple patches start from a closer intermediate point and complete sampling in fewer steps.
- $T_1 < T_2 < T_3$, $N_1 < N_2 < N_3$ across simple, medium, and hard groups.

### Key Design 3: Texture Prompts

A general-purpose **Reference Texture Memory (RTM)** is constructed:
- Diverse high-quality texture patches are collected as RTM-values.
- A texture classifier extracts semantic features as RTM-keys.
- At inference, a query is extracted from the target patch, and the most similar texture patch is retrieved via inner-product similarity to serve as conditioning.

Texture prompts supply local texture priors to Patch-DiT, replacing text prompts that are insufficiently precise for SR tasks.

### Patch-DiT Architecture

Built upon DiT, which is naturally suited for patch-level features represented as token sequences. Compared to U-Net, DiT performs more effectively on low-resolution patches.

## Key Experimental Results

### RealSR 4× Quantitative Comparison

| Method | CLIPIQA↑ | MUSIQ↑ | NIQE↓ | Runtime (s) |
|--------|---------|--------|-------|------------|
| Real-ESRGAN | Baseline | Baseline | Baseline | Fast |
| StableSR | High | High | — | Slow |
| DiffBIR | High | High | — | Slow |
| ResShift | Relatively high | Relatively high | — | Moderate |
| **PatchScaler** | **Best** | **Best** | **Best** | **0.23× ResShift** |

### Ablation Study

| Configuration | Outcome |
|--------------|---------|
| Uniform sampling vs. PGS | PGS achieves comparable quality with significantly faster speed |
| Without texture prompts | Detail degradation in texture-rich regions |
| Text prompts vs. texture prompts | Texture prompts are more effective for SR tasks |
| 3 groups vs. 2 groups vs. 1 group | 3 groups yield the best quality-efficiency trade-off |

### Key Findings

- PatchScaler achieves 0.23× the runtime of ResShift on 512→2048 SR tasks.
- Confidence maps accurately reflect regional difficulty: complex textures map to the hard group; flat regions to the simple group.
- Texture prompts outperform text prompts in SR — text prompts are inherently poorly aligned with local texture details.
- Simple patches can skip the majority of diffusion steps without quality loss, validating the rationale for adaptive sampling.
- Acceleration is more pronounced for higher-resolution images, where the proportion of simple patches tends to be larger.

## Highlights & Insights

1. **Patch-level adaptive sampling** is introduced to diffusion-based SR for the first time, fundamentally eliminating the efficiency waste caused by uniform sampling.
2. **Confidence-driven grouping** is theoretically grounded — when $\triangle\mathbf{x}_0$ is small, diffusion can start from a closer intermediate timestep.
3. **Texture prompts** elegantly replace text prompts, which are insufficiently precise in SR scenarios.
4. The **patch-independent pipeline** naturally supports parallel computation and scales gracefully to high resolutions.

## Limitations & Future Work

- Pretraining the GRM and constructing the RTM introduce additional training overhead.
- Patch boundary artifacts (e.g., stitching artifacts) require careful handling.
- Retrieval quality depends on the coverage and diversity of the RTM.

## Related Work & Insights

- **Diffusion-based SR**: StableSR, DiffBIR, ResShift
- **Classical SR**: Real-ESRGAN, BSRGAN, SwinIR
- **Diffusion acceleration**: Conditional distillation, DDIM, DPM-Solver

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Dual innovations in patch-adaptive sampling and texture prompts
- Technical Depth: ⭐⭐⭐⭐ — Complete theoretical derivation of the shortcut path
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset evaluation with detailed speed comparisons
- Value: ⭐⭐⭐⭐⭐ — 0.23× runtime, high-resolution friendly

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](../../CVPR2025/image_generation/uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](../../CVPR2026/image_generation/vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[NeurIPS 2025\] Image Super-Resolution with Guarantees via Conformalized Generative Models](../../NeurIPS2025/image_generation/image_super-resolution_with_guarantees_via_conformalized_generative_models.md)

</div>

<!-- RELATED:END -->
