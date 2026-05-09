---
title: >-
  [Paper Note] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes a framework for training 3D diffusion models with 2D image supervision: a pretrained deterministic 3D reconstruction model serves as a "noisy teacher" to generate noisy 3D samples, and a multi-step denoising strategy combined with rendering losses enables cross-modal training (3D denoising + 2D supervision). The approach surpasses the teacher model by 0.5–0.85 PSNR while using a smaller model.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Diffusion Models
  - 2D Supervision
  - Single-view 3D Reconstruction
  - Teacher Guidance
date: 2026-05-08
content_hash: d5dc963878cf9c29
---

# A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision

**Conference**: ICCV 2025
**arXiv**: [2412.00623](https://arxiv.org/abs/2412.00623)
**Code**: [https://lesson-in-splats.github.io/](https://lesson-in-splats.github.io/) (Project Page)
**Area**: 3D Vision / Diffusion Models / 3DGS
**Keywords**: 3D Gaussian Splatting, Diffusion Models, 2D Supervision, Single-view 3D Reconstruction, Teacher Guidance

## TL;DR
This paper proposes a framework for training 3D diffusion models with 2D image supervision: a pretrained deterministic 3D reconstruction model serves as a "noisy teacher" to generate noisy 3D samples, and a multi-step denoising strategy combined with rendering losses enables cross-modal training (3D denoising + 2D supervision). The approach surpasses the teacher model by 0.5–0.85 PSNR while using a smaller model.

## Background & Motivation

### Limitations of Prior Work

**Background**: In 3D reconstruction, deterministic feed-forward models (e.g., Splatter Image, Flash3D) are constrained by the inherent ambiguity of 2D-to-3D mappings, leading to blurry predictions. While 3D diffusion models can capture distributional diversity, conventional training requires denoising and supervision to reside in the same modality—necessitating large amounts of 3D data, which are extremely scarce in practice. Training a 3D diffusion model using only 2D images remains a critically underexplored problem.

### Root Cause

**Goal**: How can a diffusion model operating in 3D space be trained without 3D ground truth? The core challenge is that standard diffusion training requires noisy samples and supervision signals to share the same modality, whereas here denoising occurs in 3D while supervision is available only as 2D images.

## Method

### Overall Architecture
Input: a single image → pretrained deterministic model (teacher) generates an initial 3DGS prediction → the prediction is noised and fed into a 3D diffusion denoiser → the denoised 3DGS is rendered via differentiable rendering to obtain a 2D image → loss is computed against the GT image. Training proceeds in two stages: Stage 1 Bootstrapping and Stage 2 Multi-step Denoising Fine-tuning.

### Key Designs
1. **Noise-Teacher Decoupling Strategy**: Decouples the source of noisy samples (the teacher model's 3D predictions) from the supervision signal (2D images). The key insight draws from SDEdit—at sufficiently large noise levels $t \geq t^*$, the distribution of noised teacher predictions approximately overlaps with that of noised ground-truth 3D data. This allows the teacher's imperfect 3D predictions to serve as valid denoising starting points.
2. **Multi-step Denoising Training**: Standard single-step denoising can only supervise large noise levels ($t > t^*$) and fails to recover fine details. The proposed approach innovatively executes multi-step DDIM denoising (10 steps) during training, progressively denoising from $t > t^*$ to $t = 0$, rendering the final clean 3D prediction, and supervising it with 2D images. Gradients are back-propagated through all denoising steps, enabling the model to learn fine-grained denoising at low noise levels.
3. **Cycle-Consistency Regularization**: The denoised result is rendered to a target viewpoint to produce an image, which then drives a second 3D prediction; the second prediction is rendered back to the source viewpoint and compared against the source image. This ensures that the generated novel view not only appears visually consistent but also retains sufficient information to reconstruct the source viewpoint.

### Loss & Training
- Stage 1 (Bootstrapping): $L_{\text{bootstrap}} = L_{\text{3DGS}} + L_{\text{image}} + L_{\text{cyc}}$, using the teacher's 3D predictions as direct supervision alongside rendering losses for rapid initialization of the diffusion model.
- Stage 2 (Multi-step Fine-tuning): $L_{\text{mlt-stp}} + L_{\text{cyc}}$; the 3D supervision is removed (as the teacher becomes a bottleneck), and only rendering losses are used, allowing the model to surpass the teacher.
- Rendering losses at different timesteps are weighted by $\lambda_t$, with weights varying across timesteps.
- $t^* = 20$ (out of 100 total steps); DDIM sampling with 10 steps.

## Key Experimental Results

### ShapeNet-SRN (Single-view Reconstruction)

| Dataset | Metric | Ours (Medium) | Splatter Image (Large) | Gain |
|---------|--------|---------------|------------------------|------|
| Cars | PSNR | 24.84 | 24.00 | +0.84 |
| Cars | SSIM | 0.93 | 0.92 | +0.01 |
| Cars | LPIPS | 0.077 | 0.078 | −0.001 |
| Chairs | PSNR | 25.21 | 24.43 | +0.78 |

### RealEstate10K (Novel View Synthesis)

| Setting | Metric | Ours | Flash3D | Gain |
|---------|--------|------|---------|------|
| 5 frames | PSNR | 29.12 | 28.46 | +0.66 |
| 10 frames | PSNR | 26.54 | 25.94 | +0.60 |
| 30 frames | PSNR | 25.40 | 24.93 | +0.47 |

### Co3D Hydrant

| Metric | Ours | Splatter Image | ViewSet Diffusion |
|--------|------|----------------|-------------------|
| PSNR | 22.34 | 21.77 | 21.24 |
| LPIPS | 0.149 | 0.154 | 0.201 |

### Model Efficiency

| Model | VRAM (GB) | Parameters (MB) |
|-------|-----------|-----------------|
| Ours (Medium) | 1.15 | 295 |
| Splatter Image (Large) | 1.71 | 646 |
| VisionNeRF | 6.42 | 1390 |

### Ablation Study
- Removing teacher guidance and training directly with rendering losses: PSNR drops sharply to 16.73 (vs. 24.49), demonstrating the indispensability of teacher guidance.
- Stage 1 with rendering loss only (no diffusion loss): 18.82; adding diffusion loss: 22.61.
- Retaining diffusion loss in Stage 2 constrains the model to the teacher's performance ceiling: 23.13 vs. 24.49 with rendering loss only.
- Cycle-consistency loss yields improvements in both stages.
- Weighted vs. unweighted loss: 24.49 vs. 22.88, confirming the importance of timestep-dependent weighting.
- A deterministic feed-forward model with the same architecture achieves only 19.99 PSNR, demonstrating that the gains stem from the diffusion framework rather than the architecture.

## Highlights & Insights
- **Cross-modal diffusion training is feasible**: This work breaks the constraint that diffusion models must be trained within the same modality, a principle transferable to other tasks lacking direct 3D supervision.
- **The "noisy teacher" concept is elegant**: Imperfect deterministic predictions, once noised, serve as effective inputs for a diffusion model; the noise-coverage insight from SDEdit is cleverly repurposed.
- **Smaller model surpasses larger model**: A 295 MB medium model outperforms a 646 MB large teacher model, demonstrating the superior representational capacity of the diffusion framework over deterministic approaches.
- **Multi-step denoising during training**: Unrolling multi-step inference and back-propagating gradients through all steps incurs high memory overhead, but a two-stage strategy effectively controls computational cost.

## Limitations & Future Work
- The method relies on pixel-aligned 3DGS representations; Gaussians are concentrated in visible regions, leading to insufficient coverage of occluded areas and over-smoothed novel views.
- In Stage 2, the batch size is reduced from 100 to 10 due to multi-step denoising, and computational cost remains non-trivial.
- Validation is limited to 3DGS representations; extension to other 3D representations (mesh, NeRF, etc.) has not been explored.
- The choice of teacher model imposes an upper bound on achievable quality.

## Related Work & Insights
- **vs. Splatter Image / Flash3D (deterministic models)**: Used as teachers in this work, surpassed by a smaller model. Deterministic models cannot capture distributional diversity and produce blurry outputs in ambiguous regions; the proposed diffusion model captures the full distribution.
- **vs. HoloDiffusion**: Also trains a 3D diffusion model with 2D supervision, but bridges the distributional gap via an additional denoising pass; the proposed method more elegantly addresses this by selecting an appropriate noise level combined with multi-step denoising.
- **vs. ViewsetDiffusion**: Denoises across multiple-view images and aggregates into 3D, but is limited by the number of views and inconsistencies from independent noising; denoising directly in 3D space is more coherent.

## Relevance to My Research
- The cross-modal distillation paradigm (3D operation, 2D supervision) may be applicable to other tasks lacking direct supervision signals.
- The "noisy teacher" strategy is generalizable: any imperfect pretrained model can serve as a noisy teacher to bootstrap diffusion training.

## Rating
- Novelty: ⭐⭐⭐⭐ — Cross-modal diffusion training is a novel idea, though the core inspiration derives from SDEdit.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both synthetic and real-world datasets with thorough ablations, though evaluation across more object categories is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Logically clear, with detailed method descriptions and well-explained two-stage training.
- Value: ⭐⭐⭐⭐ — The cross-modal training idea is inspiring, and the noisy teacher concept is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)

</div>

<!-- RELATED:END -->
