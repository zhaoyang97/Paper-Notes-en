---
title: >-
  [Paper Note] Decompositional Neural Scene Reconstruction with Generative Diffusion Prior
description: >-
  [CVPR 2025][3D Vision][Decompositional Scene Reconstruction] Proposes DP-Recon, which introduces generative diffusion priors (SDS) into decompositional neural scene reconstruction. By dynamically adjusting pixel-wise SDS weights using visibility guidance, it resolves conflicts between reconstruction objectives and generation guidance, achieving complete object geometry and appearance recovery under sparse views.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Decompositional Scene Reconstruction"
  - "Diffusion Prior"
  - "SDS Loss"
  - "Visibility Guidance"
  - "Sparse-View Reconstruction"
date: 2026-05-08
content_hash: 0dc335e7a4a24b74
---

# Decompositional Neural Scene Reconstruction with Generative Diffusion Prior

**Conference**: CVPR 2025  
**arXiv**: [2503.14830](https://arxiv.org/abs/2503.14830)  
**Code**: [Project Page](https://dp-recon.github.io/)  
**Area**: 3D Vision / Scene Reconstruction  
**Keywords**: Decompositional Scene Reconstruction, Diffusion Prior, SDS Loss, Visibility Guidance, Sparse-View Reconstruction

## TL;DR

Proposes DP-Recon, which introduces generative diffusion priors (SDS) into decompositional neural scene reconstruction. By dynamically adjusting pixel-wise SDS weights using visibility guidance, it resolves conflicts between reconstruction objectives and generation guidance, achieving complete object geometry and appearance recovery under sparse views.

## Background & Motivation

- Decompositional 3D scene reconstruction aims to separate a scene into individual objects, which is crucial for embodied AI, robotics, and scene editing.
- Existing methods (RICO, ObjectSDF++) perform poorly in **sparse views and heavily occluded** regions, leading to severe degradation in geometry and appearance recovery.
- Semantic/geometric regularization methods (FreeNeRF, RegNeRF) cannot provide **new information** for under-constrained regions.
- **Core Argument**: The key to solving this problem lies in **supplementing missing information** for unobserved regions — generative priors from diffusion models serve as an ideal source.
- **Key Challenge**: Direct introduction of SDS into the reconstruction pipeline causes conflicts in observed regions, necessitating a balance between reconstruction guidance and generation guidance.

## Method

### Overall Architecture

DP-Recon operates in three stages: (1) decompositional neural implicit surface reconstruction using reconstruction losses; (2) visibility-guided geometric SDS optimization applied to each object; (3) visibility-guided appearance SDS optimization after exporting meshes. Reconstruction and generation guidance are coordinated through a learnable visibility grid.

### Key Designs

1. **Visibility-Guided SDS Optimization**:
    - Function: Dynamically adjusts the SDS loss weight for each pixel, reducing generation guidance in high-visibility regions and enhancing it in low-visibility regions.
    - Mechanism: Introduces a learnable visibility grid $G$, optimized using the accumulated transmittance $T$ in volume rendering: $\mathcal{L}_v = \sum_{i=0}^{n} \max(T_i - G(p_i), 0)$; then renders a visibility map $V(r) = \sum T_i \alpha_i v_i$ under novel views.
    - Visibility Weighting Function: A piecewise linear function that assigns higher weights to SDS in low-visibility regions and suppresses SDS weights in high-visibility regions.
    - Design Motivation: SDS exhibits artifacts such as over-saturation and over-smoothing. Observed regions with reconstruction guidance should rely primarily on reconstruction, whereas unobserved/occluded regions require generative priors to supplement information.

2. **Decompositional Prior-Guided Geometric Optimization**:
    - Function: Applies SDS independently to each object to improve geometry.
    - Mechanism: Renders the normal map and mask map for the $j$-th object to construct the input $\tilde{n}_j$ for Stable Diffusion; the gradient is formulated as $\nabla_\theta \mathcal{L}_{\text{SDS}}^{g-v} = \mathbb{E}[w^v(z)w(t)(\hat{\epsilon}_\phi(z_t;y,t) - \epsilon)\frac{\partial z}{\partial \tilde{n}_j}\frac{\partial \tilde{n}_j}{\partial \theta}]$.
    - Utilizes OccGrid sampling to accelerate rendering, requiring only 0.01 seconds for a $128 \times 128$ resolution.
    - Design Motivation: Unlike whole-scene SDS, object-wise SDS ensures 3D consistency across views and recovers objects behind occlusions.

3. **Mesh-Level Appearance Optimization and Background Inpainting**:
    - Function: Optimizes UV textures using SDS after exporting individual object meshes.
    - Mechanism: Employs NVDiffrast for differentiable rendering, utilizes a small network $\psi$ to predict surface point colors, and performs joint optimization with appearance SDS and color rendering losses.
    - The background is supervised using depth-guided inpainting to generate a panoramic color map.
    - Design Motivation: Optimizing appearance directly on meshes generates detailed UV mapping, which is compatible with lighting rendering and VFX editing in standard 3D software.

### Loss & Training

- Stage 1: Reconstruction loss $\mathcal{L}_{recon}$ (color, depth, normal, SDF regularization, etc.)
- Stage 2: $\mathcal{L}_{recon} + \mathcal{L}_{\text{SDS}}^{g-v}$ (visibility-guided geometric SDS)
- Stage 3: Color rendering loss + $\mathcal{L}_{\text{SDS}}^{a-v}$ (visibility-guided appearance SDS)
- Uses a pretrained Stable Diffusion (without fine-tuning), guided by text descriptions.
- The visibility grid is optimized after Stage 1 and frozen during Stages 2 and 3.

## Key Experimental Results

### Main Results (Replica, 10 views)

| Method | CD↓ | F-Score↑ | NC↑ | PSNR↑ | MUSIQ↑ |
|------|-----|---------|-----|-------|--------|
| MonoSDF | 12.57 | 43.25 | 83.14 | 22.44 | 36.02 |
| ObjectSDF++ | 8.57 | 50.11 | 85.44 | 24.66 | 41.42 |
| **Ours (geo)** | **7.91** | **50.99** | **89.36** | **25.08** | 43.33 |
| **Ours (full)** | **7.91** | **50.99** | **89.36** | 24.52 | **49.22** |

### Decompositional Object Reconstruction (Replica)

| Method | Object CD↓ | Object F-Score↑ | Object NC↑ | mIoU↑ |
|------|---------|------------|---------|-------|
| RICO | 10.32 | 49.26 | 61.27 | 71.21 |
| ObjectSDF++ | 7.49 | 56.69 | 64.75 | 71.72 |
| **Ours** | **5.54** | **67.71** | **73.50** | **88.21** |

### Different Number of Views (Replica, Scene CD↓ / NC↑)

| Method | 5 views | 10 views | 15 views |
|------|---------|----------|----------|
| ObjectSDF++ | – | 8.57 / 85.44 | – |
| **Ours** | – | **7.91 / 89.36** | – |

### Key Findings

- **Reconstruction quality with 10 views outperforms baseline results obtained with 100 views** (in heavily occluded scenes).
- Object reconstruction mIoU increases by 16.5 pp (71.72 → 88.21), demonstrating that generative priors significantly improve object segmentation and completeness.
- Object CD on the ScanNet++ real-world dataset decreases from 14.52 to 5.03, representing a 65% improvement.
- The MUSIQ perceptual quality metric improves from 41.42 to 49.22, indicating that appearance SDS significantly enhances visual quality.

## Highlights & Insights

- **Core Innovation**: Integrates SDS into decompositional scene reconstruction for the first time, applying generative priors to each object independently rather than to the entire scene.
- **Elegant Visibility Guidance Design**: Leverages transmittance information naturally occurring in volume rendering without requiring external visibility priors, leading to negligible computational cost.
- **Practical Value**: The generated decoupled UV meshes can be directly imported into 3D software like Blender for VFX editing.
- **10 views > 100 views**: Demonstrates the immense potential of generative priors under extremely sparse views.

## Limitations & Future Work

- The inherent over-saturation and over-smoothing issues of SDS are alleviated by visibility guidance but not fully eliminated.
- The background is processed using panoramic inpainting, which may suffer from quality degradation in complex outdoor environments.
- Dependence on Stable Diffusion means out-of-domain objects (rare/unusual objects) may exhibit poor generation results.
- The training process involves multiple stages, leading to a relatively high overall training time cost.
- Evaluation is limited to indoor scenes (Replica, ScanNet++), and generalization to outdoor scenes remains to be verified.

## Related Work & Insights

- Unlike DreamFusion's SDS, DP-Recon represents a **hybrid paradigm of reconstruction and generation** rather than pure generation.
- The visibility guidance concept can be transferred to other SDS application scenarios (e.g., balancing known/unknown regions in single-image 3D generation).
- Inspires future work to replace 2D Stable Diffusion with more advanced 3D-aware diffusion models (e.g., video diffusion models).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneering work integrating SDS with decompositional reconstruction, featuring an exquisitely designed visibility guidance strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evaluations on Replica and ScanNet++, comparisons with various baselines, experiments with different view counts, comprehensive ablation study, and demonstrations of multiple downstream applications.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition with a logical progression of the methodology.
- Value: ⭐⭐⭐⭐⭐ Groundbreaking demonstration of the immense value of generative priors in sparse-view decompositional reconstruction; achieving better results with 10 views than 100 views holds landmark significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron CT Data](regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)
- [\[CVPR 2025\] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](nerfprior_learning_neural_radiance_field_as_a_prior_for_indoor_scene_reconstruct.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)
- [\[ICCV 2025\] MoGA: 3D Generative Avatar Prior for Monocular Gaussian Avatar Reconstruction](../../ICCV2025/3d_vision/moga_3d_generative_avatar_prior_for_monocular_gaussian_avatar_reconstruction.md)
- [\[CVPR 2025\] SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
