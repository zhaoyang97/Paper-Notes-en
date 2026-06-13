---
title: >-
  [Paper Note] SyncHuman: Synchronizing 2D and 3D Generative Models for Single-View Human Reconstruction
description: >-
  [NeurIPS 2025][3D Vision][Single-view human reconstruction] SyncHuman is the first framework to unify 2D multi-view generative models and 3D native generative models within a single pipeline. Through pixel-aligned 2D-3D…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Single-view human reconstruction"
  - "2D-3D generative model synchronization"
  - "multi-view generation"
  - "3D Gaussian splatting"
  - "textured mesh"
date: 2026-05-08
content_hash: 989bb375201c69f7
---

# SyncHuman: Synchronizing 2D and 3D Generative Models for Single-View Human Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2510.07723](https://arxiv.org/abs/2510.07723)  
**Code**: [GitHub](https://xishuxishu.github.io/SyncHuman.github.io)  
**Area**: 3D Vision / Human Reconstruction
**Keywords**: Single-view human reconstruction, 2D-3D generative model synchronization, multi-view generation, 3D Gaussian splatting, textured mesh

## TL;DR

SyncHuman is the first framework to unify 2D multi-view generative models and 3D native generative models within a single pipeline. Through pixel-aligned 2D-3D synchronized attention, the two branches mutually enhance each other, achieving high-fidelity textured mesh reconstruction under complex human poses and surpassing existing methods in both geometric accuracy and visual quality.

## Background & Motivation

Reconstructing a 3D clothed human from a single RGB image is a critical task in AR/VR, virtual try-on, gaming, and film production. Existing methods primarily rely on SMPL body estimation for structural information and leverage conditional multi-view generative models to complete unseen views. However, this paradigm is subject to two fundamental tensions:

**Inaccurate SMPL estimation**: Single-view pose estimation is insufficiently precise under occlusion and complex poses, and SMPL meshes represent only the naked body, failing to model loose clothing accurately.

**Complementary weaknesses of 2D and 3D generation**: Multi-view generative models excel at capturing fine 2D details but suffer from poor cross-view consistency; 3D native generative models provide good structural consistency but yield coarse, low-fidelity results.

- **Core Idea**: Allow 2D and 3D generative models to supervise each other — the 3D model provides structural guidance to the 2D branch to improve multi-view consistency, while the 2D model injects fine-grained texture into the 3D branch to enhance fidelity.

## Method

### Overall Architecture

Given a single full-body human image, SyncHuman operates in two stages:
- **Stage 1**: A 2D-3D cross-space generative model simultaneously produces multi-view color and normal maps alongside an aligned sparse 3D voxel mesh.
- **Stage 2**: A multi-view guided decoder decodes the structured latents into high-quality textured meshes.

### Key Designs

1. **2D-3D Cross-Space Generative Model**:

    - **Multi-view generation branch**: Built upon the PSHuman architecture, this branch takes the input image as the front view and generates color and normal maps for four orthogonal viewpoints (front/back/left/right), employing row-wise multi-view attention to enhance cross-view consistency.
    - **3D structure generation branch**: Built upon the Trellis architecture, a DiT-based flow transformer generates sparse structural latents from a 3D noise mesh, which are then decoded into a volumetric occupancy grid.
    - **2D-3D synchronized attention**: The core contribution, comprising bidirectional cross-attention:
        - *2D→3D*: Each 3D voxel feature is orthogonally projected onto four view planes to retrieve corresponding 2D features; cross-attention is performed with voxel features as queries and 2D features as keys/values, followed by a zero-initialized MLP with residual connection.
        - *3D→2D*: Each 2D multi-view feature is projected into 3D space to query the corresponding voxel column features; cross-attention is performed with 2D features as queries and voxel features as keys/values.

2. **Joint Training Strategy**:

    - SD 2.1 is re-adapted to the flow matching framework used by Trellis.
    - Both 2D and 3D branches are jointly trained under a flow matching objective; the loss is the sum of velocity prediction errors from both branches.

3. **Multi-View Guided Decoder (MVGD)**:

    - Following structured latent generation by Trellis, a multi-view feature injection mechanism is introduced.
    - DINOv2 features are extracted from the generated multi-view images and processed by a trainable MLP.
    - For each 3D voxel, the corresponding features from the four views are queried, concatenated with the structured latents, and fed into the original decoder.
    - The mesh branch and 3DGS branch are trained separately; the rendered colors from 3DGS are ultimately baked onto the mesh.

### Loss & Training

- **Generation stage**: Joint optimization via the flow matching loss of each branch independently.
- **3DGS decoding**: L1 + SSIM + LPIPS + opacity regularization.
- **Mesh decoding**: L1/Huber losses on foreground masks, depth maps, and normal maps.
- Training data: 3D human scan datasets including THuman2.1, CustomHumans, THuman3.0, and 2K2K.
- Rendering resolution: 768×768 with 8 orthogonal cameras distributed at equal intervals.

## Key Experimental Results

### Main Results

| Dataset | Metric | SyncHuman | PSHuman (Prev. SOTA) | Gain |
|--------|------|-----------|-------------------|------|
| X-Humans | Chamfer Dist↓ | **0.835** | 1.438 | −42% |
| X-Humans | P2S↓ | **0.759** | 1.138 | −33% |
| X-Humans | NC↑ | **0.887** | 0.839 | +5.7% |
| X-Humans | PSNR↑ | **21.84** | 20.84 | +1.0 |
| X-Humans | LPIPS↓ | **0.079** | 0.098 | −19% |
| CAPE-NFP | Chamfer Dist↓ | **0.913** | 1.373 | −33% |
| CAPE-FP | Chamfer Dist↓ | **0.641** | 0.776 | −17% |

### Ablation Study

| Configuration | PSNR↑ | Chamfer↓ | NC↑ | Note |
|------|-------|---------|-----|------|
| Trellis (original) | 17.08 | 2.004 | 0.772 | Poor detail in 3D native model |
| Trellis (fine-tuned) | 20.34 | 1.135 | 0.848 | Significant improvement after fine-tuning |
| PSHuman | 20.84 | 1.438 | 0.839 | Weak geometry in 2D model |
| SyncHuman (full) | **21.84** | **0.835** | **0.887** | 2D+3D complementarity yields best results |

| Decoder Configuration | PSNR↑ | Chamfer↓ | Note |
|-----------|-------|---------|------|
| Original Trellis decoder | 21.08 | 0.895 | Lacks detail |
| Fine-tuned Trellis decoder | 21.36 | 0.887 | Marginal improvement |
| MVGD (multi-view guided) | **21.84** | **0.835** | Significant improvement |

### Key Findings

- The 2D-3D synchronized attention simultaneously improves geometric accuracy and texture quality, validating the core hypothesis of 2D-3D complementarity.
- The multi-view feature injection in MVGD, though straightforward, effectively "carves" high-resolution details into the 3D shape.
- Compared to SMPL estimation, the 3D native generative model is more robust under complex poses, avoiding artifacts such as self-intersections.

## Highlights & Insights

- **2D-3D complementarity paradigm**: This work is the first to demonstrate that unifying pixel-level 2D generation and voxel-level 3D generation within the same denoising process is both feasible and effective, offering a new direction for future 3D generative models.
- **Zero-initialized attention**: The synchronized attention layers employ zero-initialized MLPs, enabling smooth transition from pretrained models and more stable training.
- **SMPL-free design**: By eliminating reliance on human pose priors and directly leveraging the 3D generative model for structural information, robustness under complex poses is substantially improved.
- **Unified flow matching training**: SD 2.1 and Trellis are unified under the same flow matching framework for joint training.

## Limitations & Future Work

- Training data comprises only approximately 5,000 scans; the limited scale leads to degraded texture quality under extreme lighting conditions.
- Multi-view generation is built on a fine-tuned SD 2.1, whose generation quality remains constrained by the base model capacity.
- Future directions include scaling with video generative models or large-scale multi-view human datasets.

## Related Work & Insights

- **vs. PSHuman**: PSHuman relies solely on 2D multi-view diffusion generation followed by remeshing, resulting in poor geometric consistency; SyncHuman adds a 3D branch for collaborative improvement.
- **vs. Trellis**: As a 3D native model, Trellis lacks fine detail; SyncHuman recovers detail through 2D multi-view feature injection.
- **vs. Human3Diff**: Human3Diff attempts to introduce 3D constraints during denoising but remains fundamentally a 2D denoising model; SyncHuman performs genuine 2D-3D joint generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to synchronize 2D multi-view and 3D native generative models in joint training; the idea is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across three datasets with sufficient ablations, though comparisons against a broader set of 3D generative baselines are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, high-quality figures, and detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ — Provides a valuable 2D-3D joint modeling paradigm for both 3D human reconstruction and general 3D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Orientation Matters: Making 3D Generative Models Orientation-Aligned](orientation_matters_making_3d_generative_models_orientation-aligned.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[NeurIPS 2025\] ROGR: Relightable 3D Objects using Generative Relighting](rogr_relightable_3d_objects_using_generative_relighting.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](../../ICCV2025/3d_vision/spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)

</div>

<!-- RELATED:END -->
