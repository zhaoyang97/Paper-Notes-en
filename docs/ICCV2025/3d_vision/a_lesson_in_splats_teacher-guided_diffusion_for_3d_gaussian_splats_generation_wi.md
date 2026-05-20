---
title: >-
  [Paper Note] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes a novel framework for training 3D diffusion models using only 2D image supervision. By employing a deterministic 3D reconstruction model as a "noisy teach…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Diffusion Models"
  - "2D Supervision"
  - "Teacher Guidance"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: b0243e10733ac8e4
---

# A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision

**Conference**: ICCV 2025
**arXiv**: [2412.00623](https://arxiv.org/abs/2412.00623)  
**Code**: [https://lesson-in-splats.github.io/](https://lesson-in-splats.github.io/)  
**Area**: 3D Vision / 3D Generation
**Keywords**: 3D Gaussian Splatting, Diffusion Models, 2D Supervision, Teacher Guidance, Novel View Synthesis

## TL;DR
This paper proposes a novel framework for training 3D diffusion models using only 2D image supervision. By employing a deterministic 3D reconstruction model as a "noisy teacher" to generate 3D noisy samples, and combining a multi-step denoising strategy with cycle-consistency regularization, the proposed method achieves 3D Gaussian Splatting generation quality that surpasses the teacher model (PSNR gain of 0.5–0.85).

## Background & Motivation
- **Background**: Recovering 3D structure from 2D images is an inherently ill-posed problem. Generative models such as diffusion models can capture the distribution over plausible 3D structures. However, existing 3D diffusion models almost universally require complete 3D ground truth as supervision signals.
- **Limitations of Prior Work**: Feed-forward 3D reconstruction methods such as Splatter Image and Flash3D can be trained with sparse 2D views, but as deterministic models they cannot capture the diversity of plausible reconstructions and produce blurry predictions in uncertain regions.
- **Key Challenge**: Standard diffusion training requires the denoising process and supervision signal to share the same modality — noise is added in 3D space, and supervision also requires 3D ground truth — yet large-scale 3D data is extremely scarce.
- **Goal**: Decouple the denoising modality (3D) from the supervision modality (2D). The imperfect 3D predictions of a deterministic reconstruction model serve as a "noisy teacher" to produce noisy inputs, while differentiable rendering maps the 3D outputs to 2D images for supervision.
- **Core Idea**: Inspired by SDEdit, at sufficiently large noise levels $t > t^*$, the noisy 3D samples produced by the teacher model and those derived from true 3D ground truth follow nearly identical distributions.

## Method

### Overall Architecture
Training proceeds in two stages. Stage 1 (Bootstrapping) uses the teacher model's 3D predictions as direct supervision to efficiently initialize the diffusion model. Stage 2 (Multi-step Denoising Fine-tuning) applies image-level supervision by rendering multi-step denoising outputs to 2D, enabling the model to surpass the teacher. Cycle-consistency regularization is applied in both stages.

### Key Designs

1. **Noisy Teacher and Modality Decoupling**:

    - Function: A pretrained deterministic 3D reconstruction model (e.g., Splatter Image or Flash3D) serves as the "noisy teacher" $T_\phi$, whose output $s_0^{\text{teacher}}$, though imperfect, provides a source of noisy 3D samples.
    - Mechanism: Noise is added to the teacher output as $s_t = \sqrt{\alpha_t}s_0^{\text{teacher}} + \sqrt{1-\alpha_t}\epsilon$. The key insight is choosing a critical timestep $t^*$ such that for $t \geq t^*$, the teacher noise distribution is sufficiently close to the true GT noise distribution.
    - Design Motivation: Conventional diffusion training requires noise addition and supervision to occur within the same modality. This decoupling strategy breaks that constraint, enabling 3D diffusion models to be trained using only 2D images.

2. **Multi-step Denoising Training Strategy**:

    - Function: During training, instead of single-step denoising (sampling only at $t > t^*$), the model performs iterative multi-step denoising starting from $t > t^*$ down to $t=0$ to obtain the final 3D prediction.
    - Core Formula: $\hat{s}_0 = D_\theta(\hat{s}_1, 1, x_{\text{src}}) \circ \cdots \circ D_\theta(s_t, t, x_{\text{src}})$
    - Rendering Supervision Loss: $\mathcal{L}_{\text{mlt-stp}} = \mathbb{E}[\lambda_t \|x_{\text{tgt}}^v - \mathcal{R}(\hat{s}_0, v)\|_2^2]$
    - Design Motivation: Training exclusively at high noise levels ($t > t^*$) prevents the model from learning fine-detail recovery at low noise levels. Multi-step denoising "unrolls" the full denoising chain, enabling gradients to back-propagate through all timesteps and allowing the model to produce high-quality results at low noise levels as well.

3. **Bootstrap Warm-up Stage**:

    - Function: Prior to multi-step denoising, the model is initialized with single-step denoising training supervised by the teacher's 3D outputs.
    - Core Formula: $\mathcal{L}_{\text{bootstrap}} = \mathbb{E}[\ell_{\text{3DGS}} + \ell_{\text{image}}]$, where $\ell_{\text{3DGS}} = \|s_0^{\text{teacher}} - D_\theta(s_t,t,x_{\text{src}})\|^2$ and $\ell_{\text{image}} = \|x_{\text{tgt}}^v - \mathcal{R}(D_\theta(s_t,t,x_{\text{src}}),v)\|_2^2$
    - Design Motivation: Training from scratch with multi-step denoising is prohibitively expensive due to the need to maintain gradients across multiple steps. The Bootstrap stage efficiently initializes the model to the teacher's performance level using single-step denoising with 3D teacher supervision.

4. **Cycle-Consistency Regularization**:

    - Function: The predicted 3DGS is rendered to a target viewpoint $\hat{x}_{\text{tgt}}$, which then drives a second 3D prediction $\tilde{s}_0$, which is subsequently rendered back to the source viewpoint for comparison.
    - Core Formula: $\mathcal{L}_{\text{cyc}} = \|x_{\text{src}} - \mathcal{R}(\tilde{s}_0, v_{\text{src}})\|_2^2$
    - Design Motivation: Inspired by CycleGAN, cycle-consistency constrains predictions to not only match the target image in appearance but also be sufficiently reliable to drive a reverse reconstruction.

### Loss & Training
- Stage 1: Bootstrap loss = 3D L2 loss + 2D rendering loss + cycle-consistency loss; batch size = 100/GPU; full timestep sampling.
- Stage 2: Multi-step denoising rendering loss + cycle-consistency loss; DDIM sampler with 10 steps; batch size = 10/GPU.
- Training is conducted on 4 NVIDIA A6000 GPUs.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------|--------|--------|---------|
| ShapeNet Cars (1-view) | Splatter Image (Large) | 24.00 | 0.92 | 0.078 |
| ShapeNet Cars (1-view) | **SplatDiffusion (Medium)** | **24.84** | **0.93** | **0.077** |
| ShapeNet Chairs (1-view) | Splatter Image (Large) | 24.43 | 0.93 | 0.067 |
| ShapeNet Chairs (1-view) | **SplatDiffusion (Medium)** | **25.21** | **0.93** | **0.066** |
| RealEstate10k (5 frames) | Flash3D | 28.46 | 0.899 | 0.100 |
| RealEstate10k (5 frames) | **SplatDiffusion** | **29.12** | **0.932** | **0.087** |
| RealEstate10k (10 frames) | Flash3D | 25.94 | 0.857 | 0.133 |
| RealEstate10k (10 frames) | **SplatDiffusion** | **26.54** | **0.887** | **0.122** |

### Ablation Study

| Configuration | Novel View PSNR ↑ | Source View PSNR ↑ | Notes |
|------|-------------------|-------------------|------|
| Splatter Image (Large) Teacher | 24.20 | 31.12 | Teacher model baseline |
| Stage I (rendering loss only) | 18.82 | 20.98 | Poor results without 3D supervision |
| Stage I (diffusion + rendering loss) | 22.61 | 28.20 | Bootstrap is effective |
| Stage II (rendering loss only) | 24.49 | 31.98 | Already surpasses teacher |
| Stage I+II without cycle-consistency | 24.69 | 33.06 | Without regularization |
| **Full model** | **24.91** | **33.71** | All components combined |

### Key Findings
- A smaller Medium model (295 MB) surpasses the larger Large teacher model (646 MB) while requiring less GPU memory (1.15 GB vs. 1.71 GB).
- Multi-step denoising training is the core innovation — Stage II rendering loss training improves PSNR from 22.61 to 24.49.
- Cycle-consistency yields consistent gains in both stages.
- The framework is flexible and can be adapted to both Splatter Image (object-level) and Flash3D (scene-level) teachers.

## Highlights & Insights
- **Paradigm Shift**: This work is the first to systematically address the technical challenge of "denoising in 3D space with only 2D supervision," breaking the same-modality constraint of diffusion model training.
- **Smaller Surpassing Larger**: The Medium model outperforms the Large teacher model, demonstrating that the generative capacity of diffusion models can compensate for reduced model size.
- **General Framework**: The teacher model is interchangeable, with effectiveness validated on both object-level and scene-level datasets.
- The insight regarding the critical noise threshold $t^*$ from SDEdit is particularly elegant — sufficiently large noise levels can bridge the distributional gap between imperfect teacher outputs and true ground truth.

## Limitations & Future Work
- Computational cost: Multi-step denoising training remains expensive, forcing a reduction in batch size to 10.
- Teacher model quality floor: If the teacher model is too weak, the noise distributions may remain misaligned even at high noise levels.
- The framework has only been validated with 3DGS representations; generalizability to other 3D representations such as NeRF or meshes remains unexplored.
- Diversity evaluation is insufficient — while the framework inherently supports diverse generation, experiments primarily focus on reconstruction quality.

## Related Work & Insights
- **vs. Holodiffusion**: Holodiffusion also explores 2D-supervised 3D diffusion but addresses distributional mismatch through an additional denoising pass; the proposed method resolves this more elegantly through noise level selection and multi-step denoising.
- **vs. Score Distillation (SDS)**: SDS "lifts" 2D diffusion models to 3D but suffers from view-consistency issues (the Janus problem); the proposed method avoids this by denoising directly in 3D space.
- **vs. ViewsetDiffusion**: ViewsetDiffusion relies on a bijective relationship between multi-view images and 3D, which limits its effectiveness when the number of views is small.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of modality decoupling combined with multi-step denoising training is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of both object-level and scene-level datasets with complete ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear and the method is presented in a well-structured, progressive manner.
- Value: ⭐⭐⭐⭐ Opens a new direction for scalable training of 3D generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)
- [\[ICCV 2025\] Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction](baking_gaussian_splatting_into_diffusion_denoiser_for_fast_and_scalable_single-s.md)
- [\[ICCV 2025\] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization](cl-splats_continual_learning_of_gaussian_splatting_with_local_optimization.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)

</div>

<!-- RELATED:END -->
