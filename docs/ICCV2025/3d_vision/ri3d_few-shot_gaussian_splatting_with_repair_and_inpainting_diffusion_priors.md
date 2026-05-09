---
title: >-
  [Paper Note] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors
description: >-
  [3D Vision] Proposes RI3D, decomposing sparse-view synthesis into two subtasks—"repairing visible regions" and "completing missing regions"—leveraging two personalized diffusion models (repair + inpainting) with a two-stage optimization strategy to achieve high-quality 3DGS reconstruction under extremely sparse inputs.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: b30c97bd9a9d24b0
---
## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.10860](https://arxiv.org/abs/2503.10860)
- **Authors**: Avinash Paliwal, Xilong Zhou, Wei Ye, Jinhui Xiong, Rakesh Ranjan, Nima Khademi Kalantari
- **Institutions**: Texas A&M University, Meta Reality Labs, Max Planck Institute for Informatics
- **Code**: [Project Page](https://people.engr.tamu.edu/nimak/Papers/RI3D)
- **Area**: 3D Vision / Sparse-View Synthesis
- **Keywords**: 3D Gaussian Splatting, diffusion model priors, sparse-view reconstruction, inpainting, few-shot

## TL;DR
RI3D decomposes sparse-view synthesis into two sub-tasks — repairing visible regions and completing missing regions — and introduces two personalized diffusion models (repair + inpainting) combined with a two-stage optimization strategy to achieve high-quality 3DGS reconstruction under extremely sparse inputs.

## Background & Motivation

### Problem Definition
Novel view synthesis from sparse inputs (e.g., only 3 images) is an extremely challenging task. Existing methods face two core problems:

**Overfitting on visible regions**: Optimization constrained by only a few input images leads to severe artifacts in rendered novel views.

**Failure to recover missing regions**: Occluded or uncovered regions typically yield only blurry or dark results.

### Limitations of Prior Work
- **Regularization-based methods** (DNGaussian, FSGS, CoR-GS, etc.): Apply depth supervision, densification strategies, and other regularization constraints, but still fail to hallucinate fine details in missing regions under extremely sparse settings.
- **Diffusion model-based methods** (ReconFusion, CAT3D): Train view-synthesis diffusion models to generate novel views, but the generated results lack 3D consistency, causing over-blurred optimization outcomes; additionally, these methods rely on NeRF representations with slow rendering speeds.

### Core Idea
The key insight is to **decouple the view synthesis process into two independent sub-tasks** — repairing visible regions and completing missing regions — each handled by a dedicated diffusion model, thereby avoiding the difficulty of a single model simultaneously addressing both objectives.

## Method

### Overall Architecture
RI3D consists of three core components:
1. **High-quality depth initialization**: Fuses DUSt3R and monocular depth to obtain per-view depth maps.
2. **Two personalized diffusion models**: A repair model for correcting rendering artifacts, and an inpainting model for filling missing regions.
3. **Two-stage optimization**: Stage 1 reconstructs visible regions; Stage 2 completes missing regions.

### 1. 3D Gaussian Initialization
The core idea is to exploit the complementary strengths of DUSt3R depth (3D-consistent but smooth) and monocular depth (detail-rich but relative and inconsistent):

$$\mathbf{d}^{*} = \arg\min_{\mathbf{d}} \left[ \mathbf{M} \odot \|\mathbf{d} - \mathbf{d}^{D}\|_2 + \lambda \|\nabla \mathbf{d} - \nabla \mathbf{d}^{M}\|_2 \right]$$

- First term: enforces depth consistency in high-confidence DUSt3R regions $\mathbf{M}$.
- Second term: preserves monocular depth gradients (edge details) globally, with $\lambda=10$.
- Analogous to Poisson Blending but with gradient constraints applied across all regions.
- Solved efficiently via a sparse matrix solver, followed by bilateral filtering to sharpen boundaries.
- Each pixel is assigned one Gaussian and projected into 3D space.

### 2. Repair Diffusion Model
- Fine-tuned from a pretrained ControlNet.
- **Training data generation**: A leave-one-out strategy constructs $N$ subsets (each omitting one image), trains $N$ 3DGS models — optimized for 6,000 steps first, then the excluded image is reintroduced and optimization continues to 10,000 steps — generating progressive "corrupted"–"clean" image pairs.
- Fine-tuned for 1,800 steps on the target scene for scene personalization.

### 3. Inpainting Diffusion Model
- Based on the Stable Diffusion Inpainting model.
- Fine-tuned on input images with randomly generated masks to produce input–output pairs (similar to RealFill).
- Fine-tuned for 2,000 steps for scene personalization.
- Both models operate at $512\times512$ resolution.

### 4. Two-Stage Optimization

**Stage 1: Reconstructing Visible Regions**
$$\mathcal{L}_{\text{stage1}} = \sum_{i=1}^{N} \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_i^{\text{ref}}, \mathbf{I}_i^{\text{ref}}) + \sum_{j=1}^{M} \lambda_j \mathbf{M}_j^{\alpha} \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_j^{\text{nov}}, \mathbf{G}_j^{\text{nov}}) + \sum_{j=1}^{M} \|\mathbf{A}_j \odot (1-\mathbf{M}_j^{\alpha}) \odot \mathbf{M}_j^{b}\|_1$$

- First term: reconstruction loss on input views (L1 + SSIM + LPIPS + depth correlation).
- Second term: supervision of $M$ novel views using pseudo ground truth from the repair model, applied only in visible regions ($\mathbf{M}^{\alpha}$).
- Third term: encourages opacity in missing regions to approach zero, preventing visible Gaussians from being placed there.
- Repair results are refreshed every 400 steps; total optimization: 4,000 steps.

**Stage 2: Completing Missing Regions**
$$\mathcal{L}_{\text{stage2}} = \sum_{i=1}^{N} \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_i^{\text{ref}}, \mathbf{I}_i^{\text{ref}}) + \sum_{j=1}^{M} \lambda_j \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_j^{\text{nov}}, \mathbf{G}_j^{\text{nov}}) + \sum_{k=1}^{K} (1-\mathbf{M}_k^{\alpha}) \odot \mathbf{M}_k^{b} \odot L_p(\hat{\mathbf{I}}_k^{\text{nov}}, \hat{\mathbf{L}}_k^{\text{nov}})$$

- $K<M$ non-overlapping novel views are selected for inpainting to avoid inconsistencies from independently completing overlapping content.
- Completed regions are projected into 3D space via monocular depth (using Eq. 2 to fuse depth ranges).
- Third term constrains consistency between rendered results and inpainted results in missing regions.
- One round of inpainting + repair is performed every 200 steps, iterating until all missing regions are filled; total optimization: 4,000 steps.

## Key Experimental Results

### Main Results: Mip-NeRF 360 Dataset

| Method | 3-view PSNR↑ | 3-view SSIM↑ | 3-view LPIPS↓ | 9-view PSNR↑ | 9-view LPIPS↓ |
|------|-------------|-------------|--------------|-------------|--------------|
| DNGaussian | 12.02 | 0.226 | 0.665 | 12.97 | 0.637 |
| FSGS | 13.14 | 0.288 | 0.578 | 16.00 | 0.470 |
| CoR-GS | 13.51 | 0.314 | 0.633 | 15.48 | 0.574 |
| ReconFusion | 15.50 | 0.358 | 0.585 | 18.19 | 0.511 |
| CAT3D | 16.62 | 0.377 | 0.515 | 18.67 | 0.460 |
| **RI3D (Ours)** | **15.74** | **0.342** | **0.505** | **17.48** | **0.415** |

- RI3D achieves the best LPIPS scores across all settings, indicating the richest synthesized texture details.
- PSNR/SSIM are slightly lower than ReconFusion and CAT3D, as the blurry missing regions produced by those methods paradoxically benefit pixel-level metrics.

### Ablation Study: 3-view Mip-NeRF 360

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| w/o depth enhancement | 15.61 | 0.336 | 0.513 |
| Single-stage (repair only) | 14.92 | 0.309 | 0.527 |
| w/o fine-tuned inpainting | 15.64 | 0.341 | 0.508 |
| **RI3D (Full)** | **15.74** | **0.342** | **0.505** |

### Key Findings
1. **Depth fusion is critical**: Removing depth enhancement degrades LPIPS from 0.505 to 0.513; inaccurate DUSt3R depth in low-confidence regions causes floating artifacts.
2. **Two-stage optimization is necessary**: Using only the repair model in a single stage reduces PSNR by ~0.8 and LPIPS by 0.022, as the repair model is not designed to hallucinate high-quality textures in missing regions.
3. **Personalized fine-tuning improves consistency**: Without fine-tuning the inpainting model, generated textures may be visually plausible but stylistically inconsistent with the surrounding scene (e.g., gray vegetation, leaf-textured floors).
4. RI3D also achieves the best LPIPS on the CO3D dataset, particularly outperforming ReconFusion in the 9-view setting.

## Highlights & Insights

1. **Task decomposition design**: Decomposing the challenges of sparse-view synthesis into two independent sub-tasks, each handled by a specialized diffusion model, proves more effective than a single all-purpose model.
2. **Depth fusion strategy**: The Poisson-blending-style fusion of DUSt3R and monocular depth is an elegant design that simultaneously ensures 3D consistency and preserves fine details.
3. **Personalized diffusion models**: Fine-tuning diffusion models on the target scene (repair: 1,800 steps; inpainting: 2,000 steps) ensures that generated results are stylistically consistent with the scene.
4. **Fast rendering**: Adopting 3DGS rather than NeRF as the 3D representation balances quality and efficiency.

## Limitations & Future Work

- Strong dependence on DUSt3R depth quality; severely inaccurate depth estimates lead to ghosting artifacts.
- Cannot handle single-image inputs, as the leave-one-out strategy requires at least 2 images.
- Fine-tuning two diffusion models increases computational cost.
- Training requires multiple GPUs (Stage 2 requires two A5000 GPUs).

## Related Work & Insights

- **vs. ReconFusion/CAT3D**: The key improvement of RI3D is decoupling "repair" from "completion," rather than using a single diffusion model for view synthesis.
- **vs. GaussianObject**: Borrows its ControlNet-based repair strategy but extends it from object-level to scene-level reconstruction.
- **Depth fusion**: Inspired by Poisson Image Editing; future work could replace DUSt3R with higher-quality MVS methods.
- **Broader implication**: The paradigm of task decomposition with specialized models has wide transfer value and is applicable to other 3D reconstruction problems with sparse inputs.

## Rating ⭐⭐⭐⭐
The method is elegantly designed, with original contributions in depth fusion and two-stage optimization. RI3D achieves consistently superior LPIPS scores backed by thorough experiments. However, dependence on DUSt3R and multi-GPU requirements remain practical bottlenecks.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)

<!-- RELATED:END -->
