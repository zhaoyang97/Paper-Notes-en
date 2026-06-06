---
title: >-
  [Paper Note] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers
description: >-
  [3D Vision] JointDiT builds an RGB-Depth joint distribution model upon the Flux diffusion Transformer. Through adaptive scheduling weights and an unbalanced timestep sampling strategy…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 9d7a4894fe064fdd
---

# JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2505.00482](https://arxiv.org/abs/2505.00482)
- **Code**: [Project Page](https://byungki-k.github.io/JointDiT/)
- **Area**: 3D Vision
- **Keywords**: Diffusion Transformer, RGB-Depth Joint Generation, Depth Estimation, Joint Distribution Modeling, Flow Matching

## TL;DR
JointDiT builds an RGB-Depth joint distribution model upon the Flux diffusion Transformer. Through adaptive scheduling weights and an unbalanced timestep sampling strategy, a single model can flexibly perform three tasks—joint generation, depth estimation, and depth-conditioned image generation—by controlling the timestep of each modality.

## Background & Motivation

Diffusion models have achieved remarkable progress in image generation and conditional generation (depth estimation, depth-guided generation, etc.). Recent work has explored **joint distribution modeling** of RGB and depth, finding that it not only enables joint generation but also serves as a unified substitute for conditional generation. However, two core problems remain:

**Limited generation quality**: Existing joint models (LDM3D, JointNet) are built on the relatively weak Stable Diffusion architecture, yielding suboptimal image fidelity and depth accuracy.

**Challenges of decoupled timestep training**: Achieving "one model, multiple tasks" requires training with independent noise levels for each modality, yet how to train this effectively has not been thoroughly explored.

Key insight: Advanced diffusion Transformers such as Flux possess superior image priors and global receptive fields (Transformer architecture), while Transformers have also been proven effective for depth estimation (DPT, Depth Anything).

## Method

### Overall Architecture

JointDiT constructs a parallel Depth branch alongside the RGB branch of Flux, exchanging features via a Joint Connection Module to achieve joint distribution modeling. The pretrained backbone is frozen; only LoRA and the Joint Connection Module are trained.

### Joint Conditional Flow Matching (JCFM)

The flow matching framework is extended to learn a joint vector field $v_{t_x,t_y}(x,y|x_1,y_1)$, with independent timesteps $t_x, t_y$ for each modality:

$$\mathcal{L}_{\text{JCFM}}(\theta) = \mathbb{E}_{t_x,t_y}\left[\|v_{t_x,t_y,\theta}(x,y) - v_{t_x,t_y}(x,y|x_1,y_1)\|\right]$$

Tasks are switched by controlling the initial timesteps:
- **Joint generation**: $t_x=0, t_y=0$
- **Depth estimation**: $t_x=1, t_y=0$ (image is clean; depth starts from noise)
- **Depth-conditioned generation**: $t_x=0, t_y=1$

### Key Design 1: Adaptive Scheduling Weights

In joint cross-attention, information-passing weights are dynamically adjusted according to the relative noise levels of the two modalities:

$$w_x(t_x, t_y) = \text{sigmoid}\left(\alpha\left(\frac{t_y}{t_x+t_y} - \frac{1}{2}\right)\right)$$

$$w_y(t_x, t_y) = \text{sigmoid}\left(\alpha\left(\frac{t_x}{t_x+t_y} - \frac{1}{2}\right)\right)$$

where $\alpha=3$. The intuition is that the noisier branch should draw more structural information from the cleaner branch.

### Key Design 2: Unbalanced Timestep Sampling

To ensure sufficient coverage of the joint timestep combination space for both joint and conditional generation:
- 50% probability: $t_x$ and $t_y$ are sampled independently from two distinct distributions $f(t)$ and $g(t)$
- 50% probability: $t_x = t_y$, sampled from $f(t)$

This ensures the model receives adequate training across all $(t_x, t_y)$ combinations.

### Loss & Training

The final output combines self-attention and joint cross-attention:

$$\mathbf{G}_x = \text{Attn}(\mathbf{S}_x) + w_x \cdot \text{JointAttn}(\mathbf{S}_x, \mathbf{S}_y)$$

## Key Experimental Results

### Main Results: Zero-Shot Depth Estimation Generalization

| Type | Method | NYUv2 AbsRel↓ | KITTI AbsRel↓ | ETH3D AbsRel↓ |
|------|------|-------------|-------------|-------------|
| Discriminative | Depth-Anything-V2 | 4.4 | 7.5 | 13.2 |
| Diffusion-based | Marigold | 5.5 | 9.6 | 6.5 |
| Diffusion-based | GeoWizard | 5.2 | 10.1 | 6.4 |
| **Joint** | **JointDiT** | **4.9** | **9.4** | **5.6** |

As a joint model, JointDiT achieves depth estimation performance comparable to dedicated depth estimation models.

### Ablation Study: Contribution of Key Components

| Adaptive Scheduling Weights | Unbalanced Sampling | Joint Gen. FID↓ | Depth Est. AbsRel↓ |
|-------------|---------|-----------|-------------|
| ✗ | ✗ | High | High |
| ✓ | ✗ | Improved | Improved |
| ✗ | ✓ | Improved | Improved |
| ✓ | ✓ | **Lowest** | **Lowest** |

Both techniques contribute significantly and are complementary to each other.

### Key Findings
- JointDiT's 3D lifting results substantially outperform LDM3D and JointNet, producing geometrically accurate 3D point clouds.
- The RGB and Depth branches exhibit complementary behavior during generation: the Depth branch captures structural information while the RGB branch focuses on texture and appearance.
- In challenging domains (cartoons, pixel art), JointDiT's depth estimation surpasses dedicated methods, benefiting from the complementary advantages of joint modeling.

## Highlights & Insights

1. **Joint distribution as a substitute for conditional generation**: A single model covers multiple tasks through timestep control alone.
2. **Leveraging advanced diffusion Transformers**: Flux's image priors combined with the Transformer's global receptive field are key to successful joint modeling.
3. **Lightweight adaptation**: Only LoRA and the Joint Connection Module are trained, preserving pretrained knowledge.
4. **Discovery of complementary behavior**: The RGB and Depth branches naturally specialize during generation.

## Limitations & Future Work
- Training data consists of only 50k pairs, potentially limiting generalization.
- The model relies on pseudo depth labels generated by Depth-Anything-V2, and may inherit its biases.
- Although joint generation FID improves substantially, absolute values still leave room for improvement.
- Only $512 \times 512$ resolution is supported.

## Related Work & Insights
- LDM3D / JointNet: SD-based RGB-Depth joint generation
- Marigold / GeoWizard: Diffusion-based depth estimation
- Flux: Advanced Flow Matching diffusion Transformer
- ControlNet: Depth-conditioned image generation

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Adaptive scheduling weights and unbalanced sampling strategy are original contributions.
- **Practicality**: ⭐⭐⭐⭐ — A single model handles multiple tasks with flexible applicability.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across joint generation, depth estimation, and conditional generation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated with complete technical detail.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)

</div>

<!-- RELATED:END -->
