---
title: >-
  [Paper Note] CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization
description: >-
  [CVPR 2026][3D Vision][indoor scene texturing] CustomTex is a framework that achieves high-fidelity, instance-controllable texture generation for 3D indoor scenes through instance-level multi-reference image conditioning…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "indoor scene texturing"
  - "multi-reference image customization"
  - "dual distillation"
  - "VSD optimization"
  - "instance-level control"
date: 2026-05-08
content_hash: 959e34bc0be54f07
---

# CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization

**Conference**: CVPR 2026
**arXiv**: [2603.19121](https://arxiv.org/abs/2603.19121)
**Code**: [https://chenweilinx.github.io/CustomTex/](https://chenweilinx.github.io/CustomTex/)
**Area**: 3D Vision
**Keywords**: indoor scene texturing, multi-reference image customization, dual distillation, VSD optimization, instance-level control

## TL;DR
CustomTex is a framework that achieves high-fidelity, instance-controllable texture generation for 3D indoor scenes through instance-level multi-reference image conditioning and a dual distillation training strategy (semantic-level VSD distillation + pixel-level super-resolution distillation), surpassing existing methods in semantic consistency, texture sharpness, and reduction of baked-in shading.

## Background & Motivation
Generating realistic textures for 3D indoor scenes is fundamental to VR/AR, architectural visualization, and film production. **Limitations of prior work**: (1) **Text-driven methods** (SceneTex, TEXture, etc.) suffer from semantic ambiguity and cannot convey precise visual characteristics (e.g., fabric patterns, wood grain, wallpaper designs); (2) single-reference-image conditioning provides only coarse global control; (3) texture quality is insufficient — results are blurry and artifact-prone, and diffusion models tend to absorb lighting information from training data, producing **baked-in shading** that is incompatible with relighting under different illumination conditions.

**Key Challenge**: Semantic control and pixel quality are entangled within the diffusion process — InstanceTex supports multi-text instance-level control but remains constrained by text precision and output quality. **Key Insight**: The paper replaces text with multiple reference images (one per instance) and decouples "semantic generation" from "pixel enhancement" into two independent distillation processes, unified under the VSD framework.

## Method

### Overall Architecture
Given an untextured 3D indoor scene mesh (with UV unwrapping) and a reference image per object instance, each iteration proceeds as follows: (1) render RGB, depth, and instance mask images from a random viewpoint; (2) semantic-level distillation computes VSD gradients via depth-to-image diffusion with Instance Cross-Attention and LoRA; (3) pixel-level distillation computes SR gradients via a pretrained super-resolution model; (4) both gradients jointly update the implicit texture field.

### Key Designs
1. **Instance Cross-Attention + InsVSD (Semantic-Level Distillation)**:

    - Function: Ensures that each instance's texture is semantically consistent with its reference image.
    - Mechanism: IP-Adapter extracts reference image features $f^{ref}_i$; instance masks $m_i$ modulate cross-attention at the feature level:
    $Z' = \frac{1}{N}\sum_{i=1}^N m_i \cdot \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}_i^\top}{\sqrt{d_k}}\right)\mathbf{V}_i$
    - VSD alternating optimization: freeze LoRA to update texture $\theta$ (VSD gradient $\nabla_\theta\mathcal{L}_{\text{VSD}} = \mathbb{E}[\omega(t)(\epsilon_{\phi_d} - \epsilon_{\phi_{\text{LoRA}}})\frac{\partial\mathcal{T}}{\partial\theta}]$), then freeze $\theta$ to update LoRA $\phi$.
    - Design Motivation: Feature-level masking is more stable than noise-level masking (confirmed by ablation), enabling precise alignment of each reference feature to its corresponding instance region.

2. **Pixel-Level Distillation**:

    - Function: Enhances texture sharpness and high-frequency detail.
    - Mechanism: A pretrained SR model $\phi_{SR}$ computes SR gradients: $\nabla_\theta\mathcal{L}_{\text{SR}} = \mathbb{E}[\omega(t)(\epsilon_{\phi_{SR}} - \epsilon_{\phi_{\text{LoRA}}})\frac{\partial\mathcal{T}}{\partial\theta}]$
    - Final gradient: $\nabla_\theta\mathcal{L} = \nabla_\theta\mathcal{L}_{\text{VSD}} + \lambda_{SR}\nabla_\theta\mathcal{L}_{\text{SR}}$
    - Training strategy: $\lambda_{SR}=0$ for the first 5,000 iterations (semantic distillation only), then $\lambda_{SR}=1.2$ to engage pixel enhancement.
    - Design Motivation: Integrating SR into the distillation process substantially outperforms post-processing SR — UV textures lack the natural image semantic structure that SR models rely on for direct super-resolution.

3. **Multi-Resolution Hash Grid Texture Representation**:

    - Function: Implicitly represents texture and supports arbitrary-resolution output.
    - Mechanism: Based on Instant-NGP's multi-resolution hash grid: UV coordinates → multi-scale grids → hash mapping → feature concatenation → Cross-Attention decoder → RGB.
    - Inference efficiency: ~2.4 seconds for 4K textures, ~22 seconds for 12K.
    - Design Motivation: More flexible than fixed-resolution texture maps and enables more efficient optimization.

### Loss & Training
- VSD gradient (semantic) + SR gradient (pixel), with alternating optimization of texture $\theta$ and LoRA $\phi$.
- Time annealing: $t\sim U(0.02,0.98)$ for the first 5,000 iterations, then $t\sim U(0.02,0.5)$.
- 30,000 iterations, 5,000 spherically distributed viewpoints; LR: 0.001 for texture, 0.0001 for LoRA.
- Approximately 48 hours on a single RTX A800.

## Key Experimental Results

### Main Results
Image-to-texture (10 3D-FRONT scenes):

| Method | CLIP-I↑ | CLIP-FID↓ | Q-Align IQA↑ | Q-Align IAA↑ |
|------|---------|-----------|-------------|-------------|
| **CustomTex** | **0.797** | **106.229** | **4.469** | **3.629** |
| SceneTex-IPA | 0.741 | 121.118 | 4.009 | 3.594 |
| Paint3D | 0.694 | 130.138 | 2.896 | 2.401 |
| HY3D-2.1 | 0.682 | 134.680 | 2.187 | 1.838 |

Text-to-texture:

| Method | CLIP-T↑ | IS↑ | Q-Align IQA↑ |
|------|---------|-----|-------------|
| **CustomTex** | **0.766** | **3.311** | **4.252** |
| SceneTex | 0.639 | 3.009 | 3.824 |
| HY3D-2.1 | 0.734 | 2.381 | 2.774 |

### Ablation Study

| Configuration | CLIP-I↑ | CLIP-FID↓ | Q-Align IQA↑ | Note |
|------|---------|-----------|-------------|------|
| post-SR | 0.746 | 114.612 | 2.959 | Post-processing SR yields poor quality |
| w/o $\mathcal{L}_{SR}$ | 0.736 | 118.247 | 3.330 | Lacks high-frequency detail |
| w/o multi-ref | 0.757 | 109.243 | 4.053 | Reduced instance consistency + baked-in shading |
| w/o f-mask | 0.743 | 111.205 | 3.689 | Unstable illumination at object boundaries |
| **Full model** | **0.797** | **106.229** | **4.469** | Best overall |

### Key Findings
- **Integrated SR distillation >> post-processing SR**: post-SR IQA is only 2.959 vs. 4.469 for the full model.
- Feature-level masking yields more stable illumination than noise-level masking.
- Multi-reference input is critical: concatenating reference images prevents the model from distinguishing individual instances.
- Instance mask decomposition from global to local generation is the **key factor in reducing baked-in shading**.
- In a user study (60 participants), CustomTex achieves the highest scores for both visual quality and consistency.

## Highlights & Insights
- **Dual distillation decoupling paradigm**: semantic distillation governs *what* is generated; pixel distillation governs *how well* it is generated.
- **Instance Cross-Attention for precise alignment**: mask-modulated attention enables accurate mapping from reference images to their corresponding instance regions.
- **Insightful treatment of baked-in shading**: instance mask decomposition breaks global generation into local sub-problems, preventing the diffusion model from forming a unified lighting pattern across the full image.
- Supports both photorealistic and artistic styles (Van Gogh, Cyberpunk).
- Efficient inference: 4K textures in only 2.4 seconds.

## Limitations & Future Work
- Training requires 48 hours on a single GPU.
- Only diffuse albedo textures are generated; full PBR material maps (normal/roughness/metallic) are not produced.
- Relies on high-quality UV unwrapping.
- Future directions: accelerating training and extending to full PBR material generation.

## Related Work & Insights
- The dual distillation paradigm is generalizable to other 3D generation tasks that require simultaneous semantic correctness and visual quality.
- The Instance Cross-Attention design is applicable to other multi-instance or multi-region conditioned generation tasks.
- The finding that "SR integrated into distillation outperforms post-processing SR" is a valuable reference for the SDS/VSD community.
- The text→image→texture pipeline using GPT-4v for reference image generation offers a novel interaction paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of dual distillation and Instance Cross-Attention is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes quantitative, qualitative, user study, 5 ablation groups, and comparisons with closed-source methods.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, in-depth ablation analysis, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Establishes a new benchmark for instance-level scene texture customization with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](catalyst4d_highfidelity_3dto4d_scene_editing_via_d.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2026\] TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification](topomesh_high-fidelity_mesh_autoencoding_via_topological_unification.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

</div>

<!-- RELATED:END -->
