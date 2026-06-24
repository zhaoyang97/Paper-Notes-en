---
title: >-
  [Paper Note] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation
description: >-
  [ECCV 2024][3D Vision] Proposes UniDream, which achieves relightable text-to-3D generation with clean albedo textures and PBR materials by training an albedo-normal aligned multi-view diffusion model (AN-MVM), integrated with a Transformer reconstruction model and stage-wise SDS optimization.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 8f95cb01c0963bda
---

# UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation

**Conference**: ECCV 2024  
**arXiv**: [2312.08754](https://arxiv.org/abs/2312.08754)  
**Code**: [Project Page](https://UniDream.github.io)  
**Area**: 3D Vision

## TL;DR

Proposes UniDream, which achieves relightable text-to-3D generation with clean albedo textures and PBR materials by training an albedo-normal aligned multi-view diffusion model (AN-MVM), integrated with a Transformer reconstruction model and stage-wise SDS optimization.

## Background & Motivation

### Background

**Background**: Existing text-to-3D methods (e.g., DreamFusion, Magic3D) utilize RGB diffusion models, which "bake" illumination and shadows into the textures of the generated 3D objects.

### Limitations of Prior Work

**Limitations of Prior Work**: Such baked-in lighting limits the realism and reusability of 3D objects under diverse illumination conditions.

### Key Challenge

**Key Challenge**: While Fantasia3D attempts to disentangle lighting and texture, it often mixes albedo and specular reflections.

**Core Problem**: How to generate 3D objects with clean PBR materials (albedo, normal, roughness, metallic) from text that can be re-rendered under arbitrary lighting.

## Method

### Overall Architecture

Three-stage pipeline:
1. AN-MVM generates multi-view albedo and normal maps.
2. The Transformer Reconstruction Model (TRM) reconstructs a coarse 3D model from the albedo maps, which is then refined using SDS via AN-MVM.
3. With albedo and normals fixed, Stable Diffusion is utilized to optimize roughness and metallic properties to generate PBR materials.

### Key Designs

**AN-MVM (Albedo-Normal Aligned Multi-View Diffusion)**:
- Jointly trains both albedo and normal domains on top of Stable Diffusion.
- Multi-view self-attention: Concatenates multi-view data before the self-attention layer of the UNet to enforce cross-view constraints.
- Multi-domain self-attention: Applies self-attention between corresponding views of albedo and normal domains to ensure cross-domain consistency.
- Uses class label $L$ to distinguish the normal domain; jointly trained on 70% 3D data + 30% LAION-Aesthetics 2D data to preserve semantic generalization.

**TRM (Transformer-Based Reconstruction)**:
- Extracts four-view albedo image features using DINO-v2, and encodes camera parameters via learnable camera modulation MLPs.
- A Transformer decoder performs cross-attention between learnable tokens and image features to output triplane representations.
- Trains the reconstruction model using albedo instead of RGB to avoid the negative impact of lighting and shadows on triplane-NeRF reconstruction.

**PBR Material Generation**:
- After fixing albedo and normals, an additional hash grid and MLP are introduced to predict roughness and metallic properties.
- Uses SDS supervision from Stable Diffusion, allowing simultaneous optimization of environmental illumination (restricted to a single channel to avoid color shifting).

### Loss & Training

The SDS loss uses the noise prediction difference on both albedo and normal domains by AN-MVM, with weights of 0.8 and 0.2, respectively. TRM is trained jointly using LPIPS + L2 + normal supervision.

## Key Experimental Results

### Main Results

| Method | User Study(%)↑ | CLIP Score↑ | R@1(%)↑ | R@5(%)↑ | R@10(%)↑ |
|------|----------------|-------------|---------|---------|----------|
| DreamFusion | 7.1 | 71.0 | 54.2 | 82.2 | 91.5 |
| Magic3D | 10.5 | 75.1 | 75.9 | 93.5 | 96.6 |
| MVDream | 32.1 | 75.7 | 76.8 | 94.3 | 96.9 |
| **UniDream** | **50.3** | **77.9** | **80.3** | **97.4** | **98.5** |

### Ablation Study

Comparison of multi-view diffusion models: The 2D images output by UniDream's AN-MVM successfully achieve lighting-texture disentanglement, and the generated normal maps show better cross-view consistency than MVDream's RGB outputs.

The progressive process of TRM reconstruction $\rightarrow$ SDS refinement $\rightarrow$ PBR material demonstrates step-by-step quality improvement: coarse reconstruction maintains clear texture and geometric boundaries, SDS refinement achieves high-quality 3D models, and the PBR stage adds realistic material properties.

PBR comparison: While Fantasia3D mixes lighting and shadow information into the albedo, UniDream effectively achieves disentanglement, allowing relighting under different environmental illuminations.

### Key Findings

- The preference rate of 50.3% in the user study significantly outperforms MVDream (32.1%), validating the practical value of the relighting capability.
- Normal supervision significantly accelerates geometric convergence, yielding cleaner and smoother surfaces.
- The 3D prior provided by TRM effectively avoids the "Janus problem" in SDS optimization.
- The single-channel environmental illumination constraint effectively prevents color shifting introduced by Stable Diffusion.

## Highlights & Insights

- The stage-wise strategy of "albedo first, then PBR" breaks down the complex problem progressively, which is more stable than direct joint optimization.
- Consistency in the normal domain is naturally easier to guarantee (as world-space normals of the same 3D point remain consistent across different views), which helps drive the convergence of multi-view consistency in the albedo domain.
- The diffusion model trained jointly on albedo and normal domains is a key technical innovation for understanding and disentangling the appearance of 3D objects.
- 3D prior from the reconstruction model + details from the diffusion model = complementary strengths.

## Training and Generation Details

AN-MVM training: 32 A800 GPUs, 256×256 resolution, with per-GPU batch_size=128 (16 objects × 2 domains × 4 views), taking about 19 hours for 50K iterations. TRM training: 32 A800 GPUs, batch_size=96, taking about 3 days for 70K steps. SDS refinement: 5000 steps for the NeRF stage, and 2000 steps for the DMTet stage. PBR material stage: 512×512 rendering, 2000 iterations.

The 3D dataset uses approximately 300K Objaverse objects after strict filtering (excluding untextured, non-single objects, low-quality, and caption-less models) to render multi-view albedo and normal data for training. AN-MVM is jointly trained using 70% 3D data and 30% LAION-Aesthetics 2D data, with ", 3D asset" appended to 3D captions to distinguish them.

## Limitations & Future Work

- The training data is limited to approximately 300K Objaverse objects, which limits semantic and material generalization.
- Special material types, such as transparent materials, are not supported.
- The rendering pipeline is not integrated with path tracing, resulting in simplified relighting effects.
- The multi-stage pipeline is relatively time-consuming.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first relightable text-to-3D framework
- Effectiveness: ⭐⭐⭐⭐ — Leads significantly in both quantitative and user studies
- Practicality: ⭐⭐⭐⭐ — PBR outputs are directly usable in games and AR/VR
- Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)

</div>

<!-- RELATED:END -->
