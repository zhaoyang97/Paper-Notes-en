---
title: >-
  [Paper Note] SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation
description: >-
  [CVPR 2025][3D Vision][3D scene generation] This paper proposes SceneFactor, which achieves text-guided large-scale 3D indoor scene generation through factored latent space diffusion (generating coarse semantic box layouts first, followed by fine geometry details). It also supports intuitive local editing via semantic box manipulation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D scene generation"
  - "chunk-based diffusion"
  - "semantic guidance"
  - "editable generation"
  - "VQ-VAE"
date: 2026-05-08
content_hash: 4c015f89fd8990a2
---

# SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.01801](https://arxiv.org/abs/2412.01801)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D scene generation, chunk-based diffusion, semantic guidance, editable generation, VQ-VAE

## TL;DR

This paper proposes SceneFactor, which achieves text-guided large-scale 3D indoor scene generation through factored latent space diffusion (generating coarse semantic box layouts first, followed by fine geometry details). It also supports intuitive local editing via semantic box manipulation.

## Background & Motivation

The editable generation of 3D scenes is crucial for AR/VR, games, and architectural design. Content creation is naturally an iterative process, requiring users to control and edit local spaces. However, existing methods suffer from significant limitations:

- **2D lifting methods** (e.g., Score Distillation): Lack 3D reasoning, leading to incoherent global structures.
- **Object retrieval methods**: Constrained by object databases, with fixed and unalterable geometry.
- **Direct scene diffusion** (e.g., BlockFusion/XCube): Yield good generation quality but do not support local editing.
- **Conditional generation methods**: Require synthesizing the entire scene from scratch when input conditions are modified.

Key Challenge: A method that can generate 3D scenes with high fidelity while allowing easy local editing resembling "building blocks".

## Method

### Overall Architecture

SceneFactor adopts a two-stage factored generation pipeline:
1. Text $\rightarrow$ Semantic box layout $S$: Generates a coarse 3D semantic layout via text-conditioned latent space diffusion.
2. Semantic layout $S$ $\rightarrow$ Geometry $G$: Generates high-fidelity unsigned distance field geometry via semantic-conditioned latent space diffusion.
3. Large-scale scenes are generated through chunk-by-chunk outpainting.
4. Editing is achieved by manipulating boxes in the semantic space, requiring only the regeneration of geometry in the edited regions.

### Key Designs

#### Key Design 1: Dual VQ-VAE Latent Spaces

- **Function**: To learn highly compressed latent space representations for semantics and geometry respectively.
- **Mechanism**: The geometry chunk $G \in \mathbb{R}^{128 \times 64 \times 128}$ is compressed $4\times$ to $f_G \in \mathbb{R}^{32 \times 16 \times 32}$ via a 3D VQ-VAE encoder; the semantic chunk $S \in \mathbb{Z}^{c \times 32 \times 16 \times 32}$ ($c=10$ classes) is compressed $4\times$ to $f_S \in \mathbb{R}^{8 \times 4 \times 8}$. The feature dimensions of both VQ-VAEs are 1D.
- **Design Motivation**: Direct generation of high-dimensional 3D data is intractable. VQ-VAE not only achieves an exceptionally high spatial compression rate but also generates a smooth manifold, which is beneficial for subsequent diffusion modeling. Crucially, maintaining the 3D grid structure ensures that the latent space precisely aligns with the physical space, thereby supporting local editing. Extremely high compression of the semantic space ($128 \rightarrow 8$) simplifies diffusion learning, while the geometry space retains sufficient resolution to reconstruct details.

#### Key Design 2: Factored Diffusion Generation

- **Function**: Decoupling complex 3D scene generation into two simpler tasks: high-level structure and low-level geometry.
- **Mechanism**: In the first stage, the text-to-semantic diffusion $\Psi_S$ uses a 3D OpenAI LDM + BERT text attention, with the objective $\mathcal{L}_{LDM,sem} = \|\Psi_S(f_{S,t}, t, \tau_i) - v_{S,t}\|_1$. In the second stage, the semantic-to-geometry diffusion $\Psi_G$ utilizes spatial cross-attention (conv-based, window size 3), taking the semantic map as value and the geometry latent as query/key, with the objective $\mathcal{L}_{LDM,geo} = \|\Psi_G(f_{G,t}, t, f_S) - v_{G,t}\|_2$.
- **Design Motivation**: Directly generating fine geometry from text is a highly ill-posed problem. With factorization, the first stage only needs to learn the mapping from text to coarse boxes (extremely small semantic space), and the second stage only fills in geometric details under explicit spatial constraints (greatly simplifying the task). Conv-based attention increases the receptive field of semantic conditions, effectively capturing local semantic-geometry correlations.

#### Key Design 3: Semantic Box-based Local Editing

- **Function**: Supports 5 intuitive scene editing operations requiring only 2 mouse clicks each.
- **Mechanism**: Editing is performed using box operations (add/delete/replace/scale/move) on the semantic map $S$. The corresponding geometry region $\mathcal{R}$ is filled with Gaussian noise and then regenerated using inpainting. The remaining regions remain unchanged.
- **Design Motivation**: Since $f_G$ and $S$ have the same resolution and are precisely aligned spatially, local modifications in the semantic space can seamlessly propagate to the geometry space. Users only need to specify two diagonal vertices of the box, avoiding the need for precise region boundary segmentation or complex prompt engineering.

### Loss & Training

- VQ-VAE Geometry: $\mathcal{L}^{geo} = \|G - \mathcal{D}^G(\mathcal{E}^G(G))\|_1 + \mathcal{L}^{quant}(f_G)$
- VQ-VAE Semantics: $\mathcal{L}^{sem} = \mathcal{L}^{NLL}(S, \mathcal{D}^S(\mathcal{E}^S(S))) + \mathcal{L}^{quant}(f_S)$
- Both diffusion models utilize $v$-parameterization.

## Key Experimental Results

### Main Results: 3D-FRONT Scene Generation Quality

| Method | FID ↓ | Scene-FID ↓ | KID(×100) ↓ | Coverage ↑ |
|------|-------|-------------|-------------|------------|
| SDFusion | High | High | High | Low |
| BlockFusion* | Medium | Medium | Medium | Medium |
| **SceneFactor** | **Lowest** | **Lowest** | **Lowest** | **Highest** |

*Note: BlockFusion is for unconditional generation. SceneFactor outperforms baselines on both single chunk and full-scene generation.

### Ablation Study

| Configuration | Effect |
|------|------|
| No semantic factorization (direct text-to-geometry) | Significant decrease in geometry quality, poor text alignment |
| No conv attention (using standard attention) | Weakened local correspondence between semantics and geometry |
| Single-stage VQ-VAE | Limited compression rate or poor reconstruction quality |

### Key Findings

- Factored generation significantly improves text-scene alignment and geometric fidelity compared to direct end-to-end approaches.
- Local editing maintains global consistency: the scene structure outside the edited region remains intact.
- Chunk-based outpainting enables the generation of scenes of arbitrary sizes.

## Highlights & Insights

1. **Simplification via Factorization**: The strategy of decomposing an intractable high-dimensional generation problem into two low-dimensional subproblems is highly generalizable.
2. **Edit-friendly Representation**: Using semantic boxes as an intermediate proxy layer gracefully bridges user intent with complex 3D geometry.
3. **Spatial Alignment Design**: The equal-resolution design of $f_G$ and $S$ is a crucial prerequisite for successful local editing.

## Limitations & Future Work

- Only 10 semantic classes are supported, which limits scene diversity.
- Trained on 3D-FRONT, and generalization to real-world scanned scenes is unverified.
- Editing granularity is limited to the semantic class level rather than the instance level.
- No appearance/texture generation; only outputs geometric distance fields.
- Future work could extend this to more classes and instance-level editing.

## Related Work & Insights

- **BlockFusion**: Generates scenes via a sliding window, but is unconditional and does not support editing.
- **XCube**: Hierarchical coarse-to-fine generation, which is also uneditable.
- **RePaint**: Source of inspiration for the inpainting strategy, used for consistent outpainting between chunks.

## Rating

⭐⭐⭐⭐ — The factored generation approach is elegant, and the local editing capability is a key breakthrough in practicality. The main limitations lie in its restriction to indoor scenes, fixed classes, and absence of texture, but the core methodology holds broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LT3SD: Latent Trees for 3D Scene Diffusion](lt3sd_latent_trees_for_3d_scene_diffusion.md)
- [\[CVPR 2025\] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)
- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)
- [\[NeurIPS 2025\] From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries](../../NeurIPS2025/3d_vision/from_programs_to_poses_factored_real-world_scene_generation_via_learned_program_.md)
- [\[CVPR 2025\] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion](ouroboros3d_image-to-3d_generation_via_3d-aware_recursive_diffusion.md)

</div>

<!-- RELATED:END -->
