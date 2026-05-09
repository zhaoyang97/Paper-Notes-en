---
title: >-
  [Paper Note] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models
description: >-
  [NeurIPS 2025][3D Vision][depth estimation] Merge proposes a plug-and-play framework that inserts lightweight learnable Converters before each frozen pretrained T2I diffusion block, enabling depth estimation with only ~12% additional parameters while perfectly preserving the original image generation capability. It achieves state-of-the-art performance among unified models on multiple zero-shot depth estimation benchmarks.
tags:
  - NeurIPS 2025
  - 3D Vision
  - depth estimation
  - text-to-image diffusion models
  - parameter-efficient fine-tuning
  - unified model
  - generative depth estimation
date: 2026-05-08
content_hash: 4b5dbc2477fa1620
---

# More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.23574](https://arxiv.org/abs/2510.23574)
**Code**: [GitHub](https://github.com/H-EmbodVis/MERGE)
**Area**: 3D Vision
**Keywords**: depth estimation, text-to-image diffusion models, parameter-efficient fine-tuning, unified model, generative depth estimation

## TL;DR

Merge proposes a plug-and-play framework that inserts lightweight learnable Converters before each frozen pretrained T2I diffusion block, enabling depth estimation with only ~12% additional parameters while perfectly preserving the original image generation capability. It achieves state-of-the-art performance among unified models on multiple zero-shot depth estimation benchmarks.

## Background & Motivation

- **Background**: Generative depth estimation methods (e.g., Marigold) leverage rich visual priors in pretrained T2I diffusion models, demonstrating impressive zero-shot depth estimation capability. However, **full-parameter fine-tuning catastrophically destroys the model's original image generation ability**, reducing it to a single-purpose depth estimator.
- **Limitations of Prior Work**: Existing methods for unifying generation and depth estimation exhibit clear shortcomings:
  - **JointNet/UniCon**: Adopt dual-branch parallel architectures requiring two diffusion models to run simultaneously, incurring high computational costs; feature interaction also degrades the original T2I capability.
  - **OneDiffusion**: Trains a unified model from scratch, requiring 100M data samples and enormous resource consumption.
- **Key Challenge**: The core challenge is: **Can the latent depth estimation capability embedded in a pretrained model be unlocked at minimal cost without degrading its T2I generation ability?**
- **Goal**: Merge's key insight is that the vast data distribution learned by T2I models already contains latent information highly correlated with depth. A simple Converter suffices to guide and release this capability without modifying the original parameters.

## Method

### Overall Architecture

Merge builds on a frozen DiT-based T2I model (e.g., PixArt or FLUX) by inserting a learnable Converter before each pretrained T2I block. The design is remarkably concise:
- **Depth estimation mode**: Input features pass through the Converter before entering the T2I block.
- **Image generation mode**: The Converter is bypassed; the original T2I block is used directly.

This plug-and-play design allows the model to **seamlessly switch** between the two modes without any structural modification.

### Key Designs

1. **Play-and-Plug Framework**: Unlike full fine-tuning approaches, Merge inserts a learnable block—structurally identical to the corresponding T2I block—before each pretrained block as a Converter. The Converter transforms the latent features from those suited for image generation into those suited for depth estimation. Since pretrained model parameters are entirely frozen, image generation capability is completely unaffected. Each Converter is initialized with the pretrained weights of the first T2I block in its corresponding group, enabling smooth feature transition.

2. **Group Reuse Mechanism (GRE)**: The authors observe that **adjacent layers in pretrained T2I models produce highly similar output features** (verified via cosine similarity analysis). Based on this, all T2I blocks are evenly divided into groups, with blocks within a group sharing a single Converter. For example, PixArt's 28 blocks are divided into 14 groups of 2, each sharing one Converter. This reduces additional parameters from 596M to 110M (~18%) with only marginal performance degradation (A.Rel from 7.0 to 7.5). Reducing to 7 groups further lowers parameters to 56M with still acceptable performance (A.Rel 7.8).

3. **Converter Simplification**: Experiments reveal that the Cross-Attention module in the Converter is redundant, as depth estimation uses empty text prompts and Cross-Attention cannot extract useful information. Removing it reduces parameters by 25% with negligible performance impact. Additionally, reducing the FFN expansion ratio from 4 to 1 further decreases parameters by ~36% with almost no performance change. The final simplified Converter contains only Self-Attention and a compact FFN.

### Loss & Training

Training follows the Marigold paradigm using the standard diffusion denoising loss:

$$\mathcal{L}_{LDM} = \mathbb{E}_{\varepsilon(x), y, \epsilon \sim \mathcal{N}(0,1), t} [\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2]$$

- Only Converter parameters are trained; the pretrained model is fully frozen.
- The patchify layer's input channels are doubled to handle image conditioning.
- Training data: Hypersim (indoor) + Virtual KITTI (outdoor), totaling 74K samples.
- Training: 30K iterations, batch size 32, 8× H20 GPUs.
- Learning rate: 1e-4 for PixArt, 3e-4 for FLUX.

## Key Experimental Results

### Main Results

**Comparison with unified models (supporting both generation and depth estimation):**

| Method | Training Data | Extra Params | NYUv2 A.Rel↓ | NYUv2 δ1↑ | ScanNet A.Rel↓ | DIODE A.Rel↓ |
|--------|--------------|-------------|-------------|----------|--------------|-------------|
| JointNet | 65M | 889M (100%) | 13.7 | 81.9 | 14.7 | - |
| UniCon | 16K | 125M (15%) | 7.9 | 93.9 | 9.2 | - |
| OneDiffusion | 100M | 2.8B (100%) | 6.8 | 95.2 | - | 29.4 |
| **Merge-B** | 74K | 110M (18%) | 7.5 | 94.2 | 9.9 | 32.5 |
| **Merge-L** | 74K | 1.4B (12%) | **5.9** | **95.4** | **7.1** | 31.4 |

**Comparison with parameter-efficient fine-tuning methods (same base T2I model):**

| Method | Extra Params | NYUv2 A.Rel↓ | NYUv2 δ1↑ |
|--------|-------------|-------------|----------|
| LoRA (r=128) | 110M | 8.7 | 92.3 |
| DoRA (r=128) | 110M | 8.6 | 92.4 |
| **Merge-B** | 110M | **7.5** | **94.2** |

**Comparison with full fine-tuning:**

| Method | Supports Generation | Params | NYUv2 A.Rel↓ |
|--------|-------------------|--------|-------------|
| Marigold-P (full fine-tune) | No | 596M | 7.4 |
| Merge-B-28 | Yes | 224M (37%) | 7.0 |
| Merge-B | Yes | 110M (18%) | 7.5 |

### Ablation Study

**Effect of GRE grouping:**

| Groups | GRE | Params | A.Rel↓ | δ1↑ | Note |
|--------|-----|--------|--------|-----|------|
| 28 | No | 224M | 7.0 | 94.7 | One Converter per layer |
| 14 | No | 110M | 15.6 | 78.8 | Fixed-position insertion |
| 14 | Yes | 110M | 7.5 | 94.2 | Intra-group sharing |
| 7 | Yes | 56M | 7.8 | 93.5 | Fewer groups |
| 4 | Yes | 32M | 9.3 | 91.0 | Very few parameters |

**Converter component ablation:**

| Config | SA | CA | FFN | Params | A.Rel↓ | Note |
|--------|----|----|-----|--------|--------|------|
| A | ✓ | ✓ | ✓(×4) | 596M | 6.9 | Full block |
| B | ✓ | - | ✓(×4) | 447M | 6.9 | Removing CA is lossless |
| D | ✓ | - | - | 149M | 7.4 | Removing FFN hurts |
| E | ✓ | - | ✓(×1) | 224M | 7.0 | Smaller FFN has no impact |

### Key Findings

- GRE is critical: without intra-group sharing, 14 Converters cause A.Rel to collapse from 7.0 to 15.6; sharing recovers it to 7.5.
- Cross-Attention is entirely redundant for depth estimation under empty prompts and can be removed without loss.
- Merge generalizes to surface normal estimation; Merge-L matches specialized methods such as Lotus on NYUv2 normal estimation.
- Richer text prompts (dense captions) yield a slight performance gain (A.Rel 7.5→7.3).

## Highlights & Insights

- **Elegant simplicity**: The entire method relies solely on Converters and group sharing with no complex components, yet achieves strong performance.
- **Zero degradation to the pretrained model**: Original model parameters are entirely frozen, preserving 100% of the generation capability—something prior methods cannot achieve.
- **Remarkable parameter efficiency**: With only 12% additional parameters and less than one-thousandth of OneDiffusion's training data, Merge surpasses it on depth estimation.
- **The empirical GRE finding is significant**: It reveals high redundancy between adjacent layer features in DiT models, offering useful insights for model compression and efficient adaptation.

## Limitations & Future Work

- A performance gap relative to discriminative methods (e.g., DepthAnything v2) remains on benchmarks with extensive outdoor scenes such as DIODE.
- Semantic segmentation is difficult to handle with this approach, as the stochasticity of the denoising process destabilizes ID-based label mapping.
- Only empty prompts are used; the potential of text conditioning is not fully exploited.
- The Converter design is largely intuition-driven, lacking theoretical analysis.

## Related Work & Insights

- **Marigold**: The pioneering generative depth estimation method; full fine-tuning destroys generation capability.
- **Comparison with LoRA/DoRA** is compelling: at equal parameter budgets, block-level Converters outperform layer-level low-rank adaptation.
- The proposed paradigm is extensible to other tasks (normals, segmentation), representing a general "capability expansion without forgetting" approach.
- The framework also offers reference value for unlocking latent capabilities in other large pretrained models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The plug-and-play + group reuse design is simple yet highly effective, with a fresh perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple benchmarks, diverse comparisons (unified models / LoRA / full fine-tuning / discriminative methods), and complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and well-organized experimental presentation.
- **Value**: ⭐⭐⭐⭐ Provides a practical, low-cost paradigm for extending the capabilities of pretrained models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)
- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[NeurIPS 2025\] Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[NeurIPS 2025\] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
