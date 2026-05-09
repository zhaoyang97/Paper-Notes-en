---
title: >-
  [Paper Note] Human Interaction-Aware 3D Reconstruction from a Single Image
description: >-
  [CVPR 2026][3D Vision][Multi-person 3D reconstruction] This paper proposes HUG3D, a framework that achieves high-fidelity textured 3D reconstruction of interacting multiple persons from a single image via perspective-to-orthographic view transformation, a group-instance multi-view diffusion model, and physics-aware geometry reconstruction, outperforming existing methods across CD/P2S/NC and other metrics.
tags:
  - CVPR 2026
  - 3D Vision
  - Multi-person 3D reconstruction
  - human interaction
  - multi-view diffusion
  - physical constraints
  - occlusion completion
date: 2026-05-08
content_hash: 2cbf97fdc927d141
---

# Human Interaction-Aware 3D Reconstruction from a Single Image

**Conference**: CVPR 2026
**arXiv**: [2604.05436](https://arxiv.org/abs/2604.05436)
**Code**: None (Project page: jongheean11.github.io/HUG3D_project)
**Area**: 3D Vision
**Keywords**: Multi-person 3D reconstruction, human interaction, multi-view diffusion, physical constraints, occlusion completion

## TL;DR
This paper proposes HUG3D, a framework that achieves high-fidelity textured 3D reconstruction of interacting multiple persons from a single image via perspective-to-orthographic view transformation, a group-instance multi-view diffusion model, and physics-aware geometry reconstruction, outperforming existing methods across CD/P2S/NC and other metrics.

## Background & Motivation
1. **Background**: Single-person 3D reconstruction has seen significant progress (SIFU/SiTH/PSHuman, etc.), but these methods focus exclusively on individual subjects and cannot handle multi-person interaction scenarios.
2. **Limitations of Prior Work**: Multi-person scenes present three core challenges: (1) geometric complexity and perspective distortion—large depth variation in multi-person scenes invalidates the orthographic assumption; (2) lack of interaction awareness—independently reconstructing each person leads to physically implausible limb penetration and unnatural proximity; (3) missing geometry and texture in occluded regions—inter-person occlusion causes loss of critical body part information.
3. **Key Challenge**: Existing methods treat each person independently, entirely ignoring group context and interaction priors, whereas physical plausibility in multi-person interaction (contact, penetration avoidance) requires global information.
4. **Goal**: To reconstruct high-fidelity textured 3D models of multi-person interaction scenes from a single image while guaranteeing physical plausibility.
5. **Key Insight**: Simultaneously exploiting group-level and instance-level information—implicitly learning interaction priors via diffusion models and explicitly enforcing contact and penetration avoidance via physical constraints.
6. **Core Idea**: A dual-level group/instance multi-view diffusion model for occlusion completion, combined with physics-based geometry optimization to ensure interaction plausibility.

## Method

### Overall Architecture
HUG3D consists of three stages: (1) the Pers2Ortho module converts the input perspective image into a canonical orthographic multi-view representation; (2) the HUG-MVD diffusion model jointly completes geometry and texture in occluded regions; (3) HUG-GR performs physics-aware geometry reconstruction and texture fusion. The input is a single RGB image; the output is a textured 3D mesh of interacting persons.

### Key Designs
1. **Perspective-to-Orthographic View Transformation (Pers2Ortho)**:
    - Function: Eliminates perspective distortion and converts the input into a canonical orthographic representation compatible with multi-view diffusion models.
    - Mechanism: SMPL-X meshes and camera parameters are first estimated using BUDDI, then the initial mesh is refined using depth/normals predicted by Sapiens to obtain partial 3D geometry $\mathcal{M}$. Six orthographic cameras are placed around a normalized bounding box (azimuths 0°/45°/90°/180°/270°/315°). RGB information from the input image is transferred to each orthographic view via point cloud reprojection, yielding partial RGB conditional inputs $x_{pcd}^{(i)}$.
    - Design Motivation: Training multi-view diffusion models on perspective images is difficult (Fig. 3 provides comparisons). Orthographic representations encode geometric relationships more compactly, and the standardized camera layout makes it easier for the diffusion model to learn interaction patterns.

2. **Group-Instance Multi-View Diffusion (HUG-MVD)**:
    - Function: Jointly completes RGB images and normal maps for 6 orthographic views, simultaneously modeling group interaction and individual detail.
    - Mechanism: Built upon PSHuman/SD 2.1; takes masked RGB and SMPL-X normals (via ControlNet) as input and jointly predicts RGB and normals. The key innovation lies in joint training and dual-level inference: during training, single-person data (THuman2.0/CustomHumans, for diversity) and multi-person data (Hi4D, for interaction patterns) are mixed; during inference, instance-to-group latent composition is applied—at each step, each individual's latent $z_{t,inst(k)}^{(i)}$ is injected into the corresponding region of the group latent $z_{t,group}^{(i)}$, with blending factor $\alpha=0.8$.
    - Design Motivation: Neither data source alone is sufficient—single-person data provides diversity but lacks interaction, while multi-person data provides interaction but limited identity variety. The dual-level latent composition enables group consistency and individual detail to mutually reinforce each other.

3. **Physics-Aware Geometry Reconstruction (HUG-GR)**:
    - Function: Optimizes SMPL-X meshes to geometrically match diffusion-predicted normals while satisfying physical constraints.
    - Mechanism: The total loss is $\mathcal{L}_{total} = \mathcal{L}_{normal} + \lambda_{vis}\mathcal{L}_{vis} + \lambda_{pen}\mathcal{L}_{pen}$. Normal supervision is applied at both group and instance levels. The penetration loss $\mathcal{L}_{pen}$ imposes minimum distance constraints on body part pairs in contact regions (using softplus for smooth penalization). The visibility loss $\mathcal{L}_{vis}$ ensures that rendered occlusion relationships are consistent with ground truth.
    - Design Motivation: Diffusion models provide appearance priors but do not guarantee physical plausibility; explicit penetration/contact constraints are therefore necessary. Higher learning rates are applied to high-frequency semantic regions (hands, face).

### Loss & Training
Diffusion model training: a joint DDPM denoising objective (Eq. 3) is applied to both RGB and normals, with a two-stage curriculum—no occlusion mask for the first 1,000 steps, followed by simulated occlusion for the subsequent 1,000 steps. Training takes approximately two days on a single A100 (80 GB) GPU using the Adam optimizer (lr=$5\times10^{-6}$, $\beta_1=0.9$, $\beta_2=0.999$, batch=16, gradient accumulation over 8 steps). A DDPM scheduler (1,000 steps) is used for training and DDIM (40 steps, $\eta=1.0$) for inference. HUG-GR geometry optimization runs for 200 steps with Adam (lr=0.01), $\lambda_{group}=1.0$, $\lambda_{inst}=0.2$, $\lambda_{pen}=2.0$, $\lambda_{vis}=1.0$. Finer learning rates are applied to high-frequency regions such as hands and face. Texture is produced by fusing multi-view RGB projections; occluded regions are blended using view-aware confidence masks, and high-fidelity face restoration is applied to lateral viewpoints.

## Key Experimental Results

### Main Results (MultiHuman Dataset)

| Method | CD↓ | P2S↓ | NC↑ | F-score↑ | CP↑ |
|------|-----|------|-----|----------|-----|
| SIFU | 5.644 | 2.284 | 0.754 | 29.244 | 0.089 |
| SiTH | 9.251 | 3.185 | 0.709 | 21.037 | 0.135 |
| PSHuman | 15.579 | 6.088 | 0.617 | 9.749 | 0.027 |
| DeepMultiCap | 13.719 | 2.555 | 0.749 | 18.125 | 0.083 |
| **HUG3D** | **3.631** | **1.752** | **0.811** | **41.504** | **0.240** |

Texture quality: PSNR 16.456 (vs. SIFU 15.202), SSIM 0.809, LPIPS 0.168

### Ablation Study

| Configuration | CD↓ | Occ.Norm L2↓ | Occ.PSNR↑ |
|------|-----|-------------|-----------|
| Group-only training | 4.564 | 0.157 | 7.423 |
| Instance-only training | 4.645 | 0.156 | 7.726 |
| w/o instance-to-group latent composition | 4.646 | 0.159 | 7.916 |
| Instance-only normal supervision | 4.642 | 0.156 | 7.902 |
| Group-only normal supervision | 4.620 | 0.159 | 7.678 |
| **Full HUG3D** | **4.316** | **0.153** | **8.082** |

### Key Findings
- HUG3D achieves substantial gains across all geometric metrics: CD is reduced by 35.7% (vs. SIFU) and F-score improves by 41.8%.
- The contact plausibility metric CP improves from the best baseline of 0.135 to 0.240, demonstrating that interaction modeling significantly enhances physical plausibility.
- Normal estimation and PSNR in occluded regions are also notably improved, validating the effectiveness of the diffusion model for occlusion completion.
- Dual-level training (group + instance) outperforms either strategy alone; latent composition contributes most to PSNR in occluded regions.

## Highlights & Insights
- **Practical value of Pers2Ortho**: The perspective-to-orthographic transformation is critical for multi-person scenes; training diffusion models directly on perspective images yields poor results with severe distortions (Fig. 3). This transformation paradigm generalizes to any 3D generation task requiring a canonical space and constitutes a reusable module.
- **Dual-level latent composition as an inference strategy**: A single diffusion model performs both group-level and instance-level inference simultaneously; latent injection via spatial region assignment achieves complementarity between the two levels more efficiently than training two separate models. Group-level latents ensure global consistency (e.g., occlusion relationships), while instance-level latents preserve local detail (e.g., fingers, face), balanced by a blending factor of $\alpha=0.8$.
- **Introduction and optimization of the CP metric**: The CP metric quantifies the contact plausibility between persons in reconstructed results. HUG-GR's penetration loss directly optimizes this criterion; HUG3D's CP (0.240) exceeds all baselines (maximum 0.135) by 78%, providing a reliable evaluation dimension for future multi-person interaction reconstruction.
- **Point cloud reprojection outperforms mesh vertex coloring**: Point cloud reprojection preserves dense appearance details, whereas mesh vertex coloring is typically sparse and of low quality. This design choice makes a non-negligible contribution to texture quality.

## Limitations & Future Work
- The current method handles only two-person interaction scenarios; scalability to three or more persons remains unvalidated—how the group-level representation scales with the number of individuals is a key open challenge.
- The pipeline depends on the quality of initial SMPL-X estimates; RoBUDDI in particular has limited robustness under extreme poses and heavy occlusion, and failures in initial estimation propagate through the entire pipeline.
- Partial 3D construction in Pers2Ortho may be insufficient under extreme occlusion, with only the 45° and 315° reprojection views available.
- Training data primarily consists of laboratory-captured scans (Hi4D, THuman2.0); generalization to in-the-wild scenes (complex lighting, cluttered backgrounds) requires further validation.
- Person-object occlusion (e.g., a table occluding the lower body) is not handled, despite being common in real-world scenarios.
- HUG-GR optimization requires 200 iterations; combined with the 40-step diffusion denoising, total processing time per image remains considerable.

## Related Work & Insights
- **vs. SIFU/PSHuman**: These single-person methods independently process each individual, leading to penetration and inconsistency. HUG3D's group-aware design demonstrates a clear advantage: SIFU (CD 5.644) vs. HUG3D (CD 3.631), a 36% reduction.
- **vs. BUDDI**: BUDDI performs interaction modeling only at the SMPL-X level (coarse geometry); HUG3D extends this to full textured mesh reconstruction. The proposed RoBUDDI is an improved variant of BUDDI used for initial pose estimation.
- **vs. DeepMultiCap**: DeepMultiCap achieves the best P2S among multi-person methods, but its NC and F-score are far inferior to HUG3D. Moreover, DeepMultiCap is designed for multi-view input and performs poorly in the single-image setting.
- **vs. Multiply (video method)**: Designed for video input, it is equally unsuitable for the single-frame setting. HUG3D is specifically designed for single-image scenarios, filling this gap.
- **Implications for future work**: HUG3D's three-stage framework can be extended to human-scene interaction reconstruction, applying Pers2Ortho to scene-level multi-object reconstruction and using interaction-aware diffusion priors to handle human-object occlusion.
- **Contribution of the evaluation protocol**: The paper defines a comprehensive evaluation suite for multi-person reconstruction (geometry + texture + occluded regions + contact accuracy), providing a unified benchmark for subsequent research.
- **Insights from the training data strategy**: The strategy of combining single-person datasets (for diversity) and multi-person datasets (for interaction knowledge) generalizes to other tasks that require exploiting complementary data sources.

## Rating
- Novelty: ⭐⭐⭐⭐ — First end-to-end framework for single-image multi-person textured 3D reconstruction; the group-instance design is highly inventive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage via quantitative, qualitative, ablation, and in-the-wild evaluations; the CP metric is novel and useful.
- Writing Quality: ⭐⭐⭐⭐ — Clear analysis of three challenges, systematic method description, and excellent figures.
- Value: ⭐⭐⭐⭐ — Opens a new direction for multi-person 3D reconstruction with broad practical applications (AR/VR, telepresence, digital humans).

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction](cari4d_category_agnostic_4d_reconstruction_of_human_object_interaction.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](coherent_humanscene_reconstruction_from_multiperso.md)
- [\[CVPR 2026\] 3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image](3d-fixer_coarse-to-fine_in-place_completion_for_3d_scenes_from_a_single_image.md)
- [\[CVPR 2026\] Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image](zero-shot_reconstruction_of_animatable_3d_avatars_with_cloth_dynamics_from_a_sin.md)

<!-- RELATED:END -->
