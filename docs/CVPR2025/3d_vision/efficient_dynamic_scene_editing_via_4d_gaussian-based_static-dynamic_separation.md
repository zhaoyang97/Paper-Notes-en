---
title: >-
  [Paper Note] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation
description: >-
  [CVPR 2025][3D Vision][4D Gaussian] Instruct-4DGS is proposed, leveraging the inherent separability of static 3D Gaussians and Hexplane deformation fields in 4D Gaussian Splatting (4DGS) to achieve efficient dynamic scene editing by focusing solely on editing static canonical Gaussians. Temporal alignment is refined via Coherent-IP2P-driven score distillation to eliminate motion artifacts, reducing the editing time by more than half while requiring only a single GPU.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D Gaussian"
  - "Dynamic Scene Editing"
  - "Static-Dynamic Separation"
  - "InstructPix2Pix"
  - "Score Distillation"
  - "Hexplane"
date: 2026-05-08
content_hash: 8b22ec9fc8283165
---

# Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation

**Conference**: CVPR 2025  
**arXiv**: [2502.02091](https://arxiv.org/abs/2502.02091)  
**Code**: [https://hanbyelcho.info/instruct-4dgs/](https://hanbyelcho.info/instruct-4dgs/)  
**Area**: 3D Vision / Scene Editing  
**Keywords**: 4D Gaussian, Dynamic Scene Editing, Static-Dynamic Separation, InstructPix2Pix, Score Distillation, Hexplane

## TL;DR

Instruct-4DGS is proposed, leveraging the inherent separability of static 3D Gaussians and Hexplane deformation fields in 4D Gaussian Splatting (4DGS) to achieve efficient dynamic scene editing by focusing solely on editing static canonical Gaussians. Temporal alignment is refined via Coherent-IP2P-driven score distillation to eliminate motion artifacts, reducing the editing time by more than half while requiring only a single GPU.

## Background & Motivation

**Background**: Instruction-guided 3D scene editing has achieved significant progress. InstructPix2Pix (IP2P) enables instruction-based 2D image editing, and its integration with NeRF/3DGS enables spatially consistent 3D editing. However, 4D dynamic scene editing remains under-explored.

**Limitations of Prior Work**: Existing methods (such as Instruct 4D-to-4D) require editing "all 2D images" ($T \times \mathcal{M}$ frames, where $T$ is the number of timesteps and $\mathcal{M}$ is the number of cameras) used for dynamic scene synthesis, followed by updating the entire scene via iterative dataset updates and additional training loops. A single edit takes hours of processing and requires 2 parallel GPUs. Crucially, the editing time of such approaches scales linearly with the temporal dimension, hindering scalability to long videos.

**Key Challenge**: The goal of dynamic scene editing is to modify the appearance while preserving motion. However, existing methods couple appearance and motion to update them jointly, which is highly inefficient and unnecessary.

**Goal**: How to efficiently edit the appearance of 4D dynamic scenes such that the editing time does not scale linearly with the number of timesteps?

**Key Insight**: 4DGS inherently separates the scene into static canonical 3D Gaussians (appearance) and a Hexplane deformation field (motion). Since only the appearance needs to be edited, why not modify only the static Gaussians while keeping the deformation field frozen?

**Core Idea**: Leverage the static-dynamic separability in 4DGS by using only multi-view images of the first frame to perform rapid coarse editing of the static Gaussians, followed by score distillation for temporal refinement without requiring additional image editing.

## Method

### Overall Architecture

Instruct-4DGS consists of three steps: (1) training the target 4DGS scene $\{\mathcal{G}_{canon}^{opt}, \mathcal{E}^{opt}, \mathcal{D}^{opt}\}$ with multi-camera video data; (2) Stage 1—editing only the multi-view images at the first timestep (using IP2P) to update the SH colors and positions of the static canonical Gaussians $\mathcal{G}_{canon}$ supervised by L1 RGB loss, yielding a pseudo-edited scene; (3) Stage 2—using Coherent-IP2P-driven score distillation (SDS) to refine the alignment between static Gaussians and the deformation field, eliminating motion artifacts. Throughout the process, only the static Gaussian parameters are updated, while the deformation field remains frozen.

### Key Designs

1. **Static Gaussian Editing Strategy (Stage 1)**:

    - **Function**: Highly efficient editing of dynamic scene appearance by modifying only the minimal yet sufficient components.
    - **Mechanism**: Multi-view images at the first timestep $t=0$ are extracted, and IP2P is used to edit these images based on user instructions to serve as supervision targets. Then, all parameters of the Hexplane deformation field are frozen, and only the SH color and position attributes of the static canonical Gaussians $\mathcal{G}_{canon}$ are optimized using L1 RGB loss, which takes about 800-1000 iterations. The key is that the appearance information of the entire dynamic scene is fully captured by the static Gaussians (the deformation field only handles changes in position, scale, and rotation); thus, editing only the static Gaussians is sufficient.
    - **Design Motivation**: Contrast to baselines that edit $T \times \mathcal{M}$ images, this method only edits $\mathcal{M}$ images (the first-frame multi-view images). The editing workload is completely decoupled from the number of timesteps $T$, achieving scalability in the temporal dimension.

2. **Score Distillation Temporal Refinement (Stage 2)**:

    - **Function**: Eliminate motion artifacts caused by editing static Gaussians in Stage 1, realigning the edited Gaussians with the original deformation field.
    - **Mechanism**: Stage 1 editing introduces two types of artifacts: (a) slight shifts in Gaussian positions can lead to offset voxel features queried from the Hexplane, causing deformation distortion; (b) only the SH colors of the surfaces visible in the first frame are updated, meaning unedited SH values are exposed when Gaussians rotate in subsequent timesteps. To solve these issues, SDS is utilized to distill 2D priors from IP2P into the 4D space: in each iteration, cameras and timesteps are randomly sampled to render pseudo-edited scene images, and static Gaussian parameters are updated using $\nabla \mathcal{L}_{SDS} = \mathbb{E}[(\epsilon_\theta(\tilde{I}, c_I, c_T) - \epsilon) \frac{\partial \tilde{I}}{\partial \mathcal{G}_{canon}^{edit}}]$ for about 800 iterations.
    - **Design Motivation**: SDS does not require generating additional edited 2D images, directly utilizing the editing priors of diffusion models to guide consistent optimization across time. Since this is refinement rather than generation from scratch, fewer iterations are needed, and the inherent Janus problem of SDS is not severe.

3. **Coherent-IP2P Consistency Editing**:

    - **Function**: Ensure spatio-temporal consistency for multi-view/multi-timestep editing, avoiding blurring caused by inconsistent guidance in SDS.
    - **Mechanism**: Inspired by MVDream and Tune-a-Video, the 2D self-attention in the U-Net of IP2P is replaced with 3D cross-attention—multiple images within the same batch share attention weights, maintaining consistent editing guidance across images. This is used for spatial consistency in multi-view editing (Stage 1) and temporal consistency in multi-timestep refinement (Stage 2).
    - **Design Motivation**: Original IP2P independently generates inconsistent editing guidance for different images, leading to blurry outputs when accumulated in the SDS loss. Coherent-IP2P realizes synergetic editing across images via shared attention, significantly improving detail preservation and semantic consistency.

### Loss & Training

- **4DGS training loss**: $\mathcal{L}_{4DGS} = |\hat{I}_{M,t} - I_{M,t}| + \mathcal{L}_{TV}$, combining L1 rendering reconstruction with total variation smoothing regularization of the deformation field.
- **Stage 1 editing loss**: Standard L1 RGB loss, supervising the rendered first-frame of the edited scene against the IP2P-edited images.
- **Stage 2 SDS loss**: Dual-conditioned (image condition $s_I$ + text condition $s_T$) score distillation using Classifier-Free Guidance.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS_VGG↓ | CLIP sim↑ | Editing Time | GPU Count |
|:--|:--|:--|:--|:--|:--|:--|
| Instruct 4D-to-4D (avg) | 20.40 | 0.736 | 0.491 | 0.230 | ~2h | 2 GPUs |
| **Instruct-4DGS (avg)** | 19.25 | **0.783** | **0.303** | **0.249** | **~40min** | **1 GPU** |

### Ablation Study

| Method Variant | User Preference (1st Ratio) | Key Issues |
|:--|:--|:--|
| Fully SDS (w/o Stage 1) | Lowest | Smooth motion but poor instruction alignment, low fidelity |
| Refine w/ original IP2P | Lower | Severe visual artifacts |
| Ours w/ refine {E,D} | Medium | Refining deformation fields introduces temporal inconsistency instead |
| **Ours w/o refine {E,D}** | **Highest** | Highest editing fidelity + temporal consistency |

### Key Findings

- **Advantage in LPIPS Consistency**: Although PSNR is slightly lower than the baseline (since all edited images are not directly optimized as targets), LPIPS consistently and significantly outperforms the baseline across all scenes (0.303 vs 0.491), indicating better perceptual quality.
- **Significant Efficiency Advantage**: Editing time is reduced from 2h (2 GPUs) to 40min (1 GPU), achieving a 2-3x speedup.
- **Deformation Fields Should Not Be Refined**: Ablation studies clearly show that refining deformation field parameters introduces more temporal inconsistencies and motion artifacts.
- **Coherent-IP2P is Crucial**: Using original IP2P for SDS refinement leads to severe visual artifacts and blurriness.

## Highlights & Insights

1. **Profound utilization of static-dynamic separation**: The core insight is that the decoupling of appearance and motion in 4DGS is key to editing efficiency. By editing only the appearance carrier (static Gaussians) and leaving the motion carrier (deformation field) untouched, the editing task is decoupled from the temporal dimension.
2. **Complementary two-stage design**: Stage 1 provides high-fidelity but temporally-limited coarse editing, while Stage 2 provides consistent cross-temporal refinement despite its fidelity depending on Stage 1. Both stages are indispensable.
3. **Counter-intuitive ablation finding**: Refining the deformation field is actually harmful, indicating that the root cause of the alignment issue resides in the static Gaussians rather than the deformation field.
4. **High practicality**: Single GPU, 40 minutes, text-driven, and supports various editing styles, rendering it a highly practical tool-like form.

## Limitations & Future Work

1. **Dependency on IP2P performance**: Editing quality is limited by the generation quality and instruction comprehension capability of IP2P.
2. **Inability to directly edit motion**: Only appearance can be edited; the motion patterns of objects (e.g., changing velocity or trajectory) cannot be modified.
3. **Global instead of local editing**: Inability to selectively edit specific objects in a scene (requiring an additional segmentation step).
4. **Inherent limitations of 4D representation**: When the 4DGS reconstruction quality is suboptimal in certain scenes, motion artifacts may persist even after temporal refinement.
5. **Tested only on forward-facing views**: DyNeRF and Technicolor are both forward-facing datasets; generalization to 360° scenes remains unverified.

## Related Work & Insights

- **Instruct 4D-to-4D**: The only prior work that edits 4D scenes by iteratively updating all 2D images, serving as the direct baseline for comparison.
- **DreamFusion (SDS)**: The pioneer of Score Distillation Sampling, which this work adopts for refinement rather than generation.
- **MVDream / Tune-a-Video**: The concept of multi-view/video consistent editing, which inspired the design of Coherent-IP2P.
- **4D Gaussian Splatting (4DGS)**: The 4D representation of static 3D Gaussians + Hexplane deformation fields. This work fully exploits its structural characteristics.
- **Insight**: For 4D/video editing tasks, one should leverage the structural properties of representation to decompose the problem as much as possible, rather than resorting to brute-force end-to-end processing.

## Rating

⭐⭐⭐⭐ — The core idea is simple and elegant; the utilization of static-dynamic separation is highly insightful. The efficiency improvement is significant, and the experiment is comprehensive. However, the method relies heavily on the specific structure of 4DGS, and the inability to edit motion properties remains an inherent limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)
- [\[ICCV 2025\] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](../../ICCV2025/3d_vision/mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[CVPR 2025\] Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis](dynamic_neural_surfaces_for_elastic_4d_shape_representation_and_analysis.md)
- [\[CVPR 2026\] Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](../../CVPR2026/3d_vision/catalyst4d_highfidelity_3dto4d_scene_editing_via_d.md)

</div>

<!-- RELATED:END -->
