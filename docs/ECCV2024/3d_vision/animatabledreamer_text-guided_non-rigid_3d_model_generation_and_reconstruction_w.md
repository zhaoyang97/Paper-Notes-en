---
title: >-
  [Paper Note] AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation
description: >-
  [ECCV 2024][3D Vision][Text-to-4D] This work proposes AnimatableDreamer, which extracts skeletons and motion from monocular videos and generates text-guided animatable 3D non-rigid models via Canonical Score Distillation (CSD), comprehensively outperforming existing methods in both generation quality and temporal consistency.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Text-to-4D"
  - "Non-rigid 3D"
  - "Score Distillation"
  - "Skeleton Animation"
  - "Canonical Space"
date: 2026-05-08
content_hash: f7e5d02f5123983e
---

# AnimatableDreamer: Text-Guided Non-rigid 3D Model Generation and Reconstruction with Canonical Score Distillation

**Conference**: ECCV 2024  
**arXiv**: [2312.03795](https://arxiv.org/abs/2312.03795)  
**Code**: [https://zz7379.github.io/AnimatableDreamer/](https://zz7379.github.io/AnimatableDreamer/)  
**Area**: Model Compression  
**Keywords**: Text-to-4D, Non-rigid 3D, Score Distillation, Skeleton Animation, Canonical Space

## TL;DR
This work proposes AnimatableDreamer, which extracts skeletons and motion from monocular videos and generates text-guided animatable 3D non-rigid models via Canonical Score Distillation (CSD), comprehensively outperforming existing methods in both generation quality and temporal consistency.

## Background & Motivation
**Background**: Text-to-3D generation (SDS/DreamFusion) is capable of generating high-quality static 3D objects, but generating deformable/non-rigid objects remains challenging. Existing methods are either limited to specific categories (e.g., humans) or fail to guarantee morphological consistency across different poses.

**Limitations of Prior Work**:
   - Applying vanilla SDS directly to animatable objects disrupts motion consistency, leading to incoherent surfaces generated under different poses.
   - Reconstructing deformable objects from monocular videos yields poor geometric quality in unobserved regions (e.g., the other side of an animal).
   - Existing methods (e.g., BANMo) require multiple video sequences to obtain high-quality 3D reconstructions.

**Key Challenge**: Generating animatable non-rigid 3D models requires SDS supervision, but there is a lack of a consistency bridge between the canonical space and the observation space.

**Goal**: (a) How to extract reusable skeletons/skinning from monocular videos; (b) How to generate new animatable 3D models guided by text under skeletal constraints.

**Key Insight**: The warping field is utilized as a bridge between the canonical space and the observation space, allowing gradients from the diffusion model to propagate back to the canonical model through warping.

**Core Idea**: Canonical Score Distillation: calculating diffusion prior gradients on observed deformed poses and backpropagating them to the canonical model via differentiable warping, ensuring consistency across all poses.

## Method

### Overall Architecture
Two stages: (1) Extraction stage—learning an implicit articulate model (NeuS + skeletal warping field) from a monocular video and enhancing unobserved regions with CSD; (2) Generation stage—generating a text-guided new 3D model on the extracted skeleton using CSD + MVDream.

### Key Designs

1. **Implicit Articulate Model**:

    - **Function**: Representing the canonical model using NeuS (SDF + color + feature descriptor), and performing deformation via linear blend skinning (LBS).
    - **Mechanism**: Skinning weights for $B$ bones are modeled using Gaussian distributions and calculated via Mahalanobis distance. Bone transformations are learned using an MLP with Fourier temporal embeddings.
    - **Design Motivation**: The canonical space guarantees temporal consistency, while skeletal skinning provides controllable animation capabilities.

2. **Skeleton Construction**:

    - **Function**: Constructing a structured skeleton from the learned bone model to constrain the generation process.
    - **Mechanism**: DINOv2 features are used to compute semantic relationships between bones, and skinning weights are used to compute morphological relationships, yielding a joint connectivity strength $\mathcal{T}_{j,k}$. Translation and angular constraints are applied to prevent implausible deformations.
    - **Design Motivation**: Skeleton constraints ensure that the generated new objects maintain plausible motion patterns.

3. **Canonical Score Distillation (CSD)** $\leftarrow$ Core Contribution:

    - **Function**: Allowing supervision signals from the diffusion model to propagate back to the canonical model via the warping mechanism.
    - **Mechanism**: $$\nabla_\phi \mathcal{L}_{CSD} = \mathbb{E}[\underbrace{(\epsilon_\theta - \epsilon)}_{\text{Diffusion Prior}} \cdot \underbrace{\frac{\partial \mathcal{R}(\mathbf{X}_*)}{\partial \mathbf{X}_*}}_{\text{Canonical Rendering}} \cdot \underbrace{\frac{\partial W(\mathbf{X}^t)}{\partial \phi_w}}_{\text{Warp Refinement}}]$$
    - Three gradient terms: diffusion prior provides appearance guidance $\rightarrow$ canonical rendering guarantees consistency $\rightarrow$ warp refinement optimizes deformation parameters.
    - Using MVDream (a multi-view diffusion model) to render 4 orthogonal views simultaneously, ensuring 3D consistency.
    - **Design Motivation**: Standard SDS only computes gradients in the observation space, failing to guarantee canonical space consistency. CSD ensures gradient propagation to the canonical model via the chain rule of warping.

### Loss & Training
- Extraction stage: $\mathcal{L}_{Ext} = \mathcal{L}_{recon}(\text{RGB+Contour+Optical Flow}) + \mathcal{L}_{CSD} + \mathcal{L}_{reg}(\text{Feature Matching+Cycle Consistency})$
- Generation stage: $\mathcal{L}_{Gen} = \mathcal{L}_{skel} + \mathcal{L}_{bone} + \mathcal{L}_{CSD} + \mathcal{L}_{reg}$
- Rendering with a resolution of 200×200 and 4 orthogonal views.
- Training takes approximately 5 hours for 12,000 iterations on a single A800 GPU.

## Key Experimental Results

### Main Results

| Method | CLIP↑ | CLIP-T↑ | R-Precision@10↑ | GPT Eval3D↑ |
|------|-------|---------|-----------------|-------------|
| ProlificDreamer | 33.1 | 95.9 | 56.3 | 959 |
| MVDream | 34.8 | 94.4 | 31.2 | 979 |
| **Ours** | **38.2** | **96.6** | **87.5** | **1098** |

### Monocular Reconstruction Task (Chamfer Distance ↓ / F-score@2% ↑)

| Method | Videos | Cat-Coco | Cat-Pikachu | Penguin | Shiba |
|------|--------|----------|-------------|---------|-------|
| BANMo | 1 | 10.7/15.3 | 3.71/57.3 | 6.47/43.9 | 6.81/36.6 |
| RAC | 1 | 6.25/42.2 | 3.60/60.2 | 4.68/53.7 | 7.94/30.1 |
| **Ours** | **1** | **3.65/63.3** | **2.0/88.9** | **3.7/64.0** | **4.54/53.9** |

### Ablation Study

| Configuration | CLIP↑ | R-Precision↑ | Description |
|------|-------|-------------|------|
| w/o bone+skel | 27.1 | 35.6 | No bone constraints, complete failure |
| w/o skel | 28.4 | 40.1 | No skeleton constraints, motion collapse |
| w/o bone | 37.8 | 81.7 | No bone surface constraints, slight degradation in quality |
| **Full model** | **38.2** | **87.5** | Full model is optimal |

### Key Findings
- **CSD contributes significantly to reconstruction**: Without CSD on Cat-Coco, CD increases from 3.65 to 8.34 (+128%), and F-score drops from 63.3 to 32.6 (-48%).
- **R-Precision undergoes the most significant improvement**: 87.5% vs. MVDream 31.2%, showing that skeleton constraints greatly improve text-model consistency.
- **Monocular reconstruction outperforms multi-video methods**: It achieves 3.65 on Cat-Coco, outperforming BANMo (4.66) using 4 videos, which demonstrates that CSD effectively compensates for the missing information in single-view setups.

## Highlights & Insights
- **Key insight of CSD**: Utilizing warping as a differentiable bridge between the canonical space and the observation space. This concept can be generalized to any task that requires performing SDS on a transformed space, such as cloth simulation and fluids.
- **Two-level design of skeleton constraints**: Semantic correlation (DINOv2 feature similarity) + morphological correlation (skinning weight overlap) jointly determine the strength of joint connections, which is more robust than purely geometric or semantic alternatives.
- **Unified framework for generation and reconstruction**: The same CSD framework can be used for both reconstruction from videos (enhancing unobserved regions) and generation from text (ensuring animation consistency).

## Limitations & Future Work
- **High GPU memory consumption**: CSD requires a long gradient chain from the camera space to the canonical space, and MVDream processes 4 views simultaneously, leading to high VRAM demands.
- **Limited resolution**: It can only render at 200×200 resolution, limiting detail quality.
- **Restricted to skeleton-driven deformation**: It cannot handle topological changes (e.g., object splitting) or non-skeleton-driven deformations such as cloth.
- **Reliance on video quality**: Skeleton extraction depends on the quality of HOI detection and optical flow, which might fail under severe video occlusions.

## Related Work & Insights
- **vs. DreamFusion/ProlificDreamer**: While they generate static 3D models, this work generates animatable 3D models. CSD is a natural extension of SDS.
- **vs. BANMo/RAC**: These pure reconstruction methods rely on multiple video sequences. In contrast, this work utilizes the diffusion prior of CSD to compensate for information missing from a single video.
- **vs. 4D generation (such as MAV3D)**: MAV3D directly performs SDS on 4D NeRF with no canonical space to guarantee consistency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of CSD is novel, presenting a significant contribution by using warping as a differentiable bridge for SDS.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both generation and reconstruction tasks with detailed ablation studies, though the dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear method illustrations and well-derived CSD equations.
- Value: ⭐⭐⭐⭐⭐ Opens a new paradigm for animatable 3D generation, and the core concept of CSD is widely reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation](jointdreamer_ensuring_geometry_consistency_and_text_congruence_in_text-to-3d_gen.md)
- [\[ICCV 2025\] Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](../../ICCV2025/3d_vision/advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)
- [\[ECCV 2024\] WordRobe: Text-Guided Generation of Textured 3D Garments](wordrobe_text-guided_generation_of_textured_3d_garments.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
