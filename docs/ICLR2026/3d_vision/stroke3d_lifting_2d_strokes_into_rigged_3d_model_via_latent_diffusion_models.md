---
title: >-
  [Paper Note] Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models
description: >-
  [ICLR 2026][3D Vision][3D Generation] Stroke3D is the first method to generate rigged 3D mesh models directly from user-drawn 2D strokes and text prompts. It employs a skeleton-first two-stage pipeline: a graph VAE and g…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "3D Generation"
  - "Skeleton Generation"
  - "Graph Diffusion"
  - "Rigging"
  - "DPO"
date: 2026-05-08
content_hash: 2a6a8e92e6c977e1
---

# Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2602.09713](https://arxiv.org/abs/2602.09713)  
**Code**: [https://whalesong-zrs.github.io/Stroke3D_project_page/](https://whalesong-zrs.github.io/Stroke3D_project_page/)  
**Area**: 3D Vision
**Keywords**: 3D Generation, Skeleton Generation, Graph Diffusion, Rigging, DPO

## TL;DR

Stroke3D is the first method to generate rigged 3D mesh models directly from user-drawn 2D strokes and text prompts. It employs a skeleton-first two-stage pipeline: a graph VAE and graph DiT are used to generate controllable 3D skeletons, followed by TextuRig dataset augmentation and SKA-DPO optimization to synthesize high-quality meshes.

## Background & Motivation

Rigged 3D assets are fundamental to 3D deformation and animation, with broad applications in AR/VR, robotic simulation, and the film industry. Existing methods face two key limitations:

**Difficulty generating animatable geometry**: A large body of 3D generation methods (e.g., MVDream, CLAY) produce only static geometry, lacking the skeletal hierarchies required for animation. Skeleton-conditioned methods such as SKDream are constrained by the scarcity of high-quality paired datasets.

**Lack of structural control in skeleton creation**: Existing skeleton generation methods (e.g., MagicArticulate, UniRig) adopt an end-to-end mesh-to-skeleton paradigm without explicit structural constraints, resulting in bones appearing in undesired locations while missing critical joints.

The core innovation lies in the **skeleton-driven workflow**: unlike conventional approaches that first generate a mesh and then rig it, Stroke3D first generates a skeleton from 2D strokes, then conditions mesh generation on the skeleton.

## Method

### Overall Architecture

Stroke3D consists of two main stages: (1) **Controllable Skeleton Generation** — Sk-VAE encodes the skeleton graph structure into a latent space, and Sk-DiT generates skeleton embeddings within that space; (2) **Enhanced Mesh Synthesis** — training data is augmented with the TextuRig dataset, followed by SKA-DPO optimization to improve skeleton–mesh alignment.

### Key Designs

1. **Skeleton Graph VAE (Sk-VAE)**

   The 3D skeleton is represented as an undirected graph $\mathcal{G} = (\mathbf{X}, \mathbf{E})$, where $\mathbf{X} \in \mathbb{R}^{N \times 3}$ denotes joint coordinates and $\mathbf{E}$ denotes topological edges. The encoder comprises GCN and TransformerConv layers and maps the graph structure into a continuous latent space. Training uses an $L_2$ reconstruction loss with lightweight KL divergence regularization ($kl\_\beta = 1 \times 10^{-8}$) to ensure a smooth latent space.

2. **Skeleton Graph DiT (Sk-DiT)**

   Built on the DiT architecture, standard self-attention is replaced with TransformerConv to accommodate graph-structured data, and cross-attention is introduced to incorporate CLIP-encoded text embeddings. 2D strokes are projected through a feature mapping and concatenated with the noisy latent representation to provide structural guidance. During training, hand-drawn strokes are simulated by applying perturbations to 2D projections of 3D skeletons:

   $$\mathcal{L}_{\text{Sk-DiT}} = \mathbb{E}_{\mathbf{z}_0, t, \epsilon, \mathbf{J}_{xy}, \mathbf{E}, \mathbf{c}_{\text{text}}} \left[\|\epsilon_\phi(\mathbf{z}_t, t, \mathbf{J}_{xy}, \mathbf{E}, \mathbf{c}_{\text{text}}) - \epsilon\|_2^2\right]$$

3. **TextuRig Dataset**

   To address the lack of textures in rigged models from Objaverse-XL, a dedicated processing pipeline is developed: models with texture maps or vertex colors are filtered, and descriptive annotations are regenerated using Gemini. This yields 6,800 high-quality samples added to SKDream's 24,000 training examples.

4. **SKA-DPO (Skeleton–Mesh Alignment Direct Preference Optimization)**

   A reference model generates a pair of candidate multi-view images for each skeleton–text pair. The SKA Score evaluates skeleton–mesh alignment quality to select winning and losing samples, forming a preference dataset for fine-tuning with the DiffusionDPO objective:

   $$\mathcal{L}(\theta) = -\mathbb{E} \log\sigma\big(-\beta(\|\epsilon^{win} - \epsilon_\theta(x_t^{win}, t)\|_2^2 - \|\epsilon^{win} - \epsilon_{\text{ref}}(x_t^{win}, t)\|_2^2 - (\text{lose terms}))\big)$$

### Loss & Training

- Sk-VAE: $L_2$ reconstruction + extremely lightweight KL regularization ($10^{-8}$)
- Sk-DiT: standard diffusion denoising loss + classifier-free guidance (CFG)
- Mesh generation: supervised fine-tuning augmentation (TextuRig) followed by SKA-DPO alignment optimization
- Sk-VAE and Sk-DiT are each trained for 500K iterations; SKDream SFT for 9K steps; DPO for 1K steps

## Key Experimental Results

### Main Results

| Dataset / Metric | Ours (Stroke3D) | MagicArticulate | UniRig | SKDream |
|---|---|---|---|---|
| CD-J2J (All)↓ | **0.048** | 0.052 | 0.063 | 0.111 |
| CD-J2B (All)↓ | **0.039** | 0.041 | 0.051 | 0.092 |
| CD-B2B (All)↓ | **0.034** | 0.034 | 0.041 | 0.083 |
| SKA MeanInst.↑ | **87.83** | - | - | 80.43 |
| SKA MeanClass↑ | **84.36** | - | - | 74.38 |

### Ablation Study

| Configuration | MeanInst.↑ | MeanClass↑ | Notes |
|---|---|---|---|
| SKDream baseline | 80.43 | 74.38 | Original baseline |
| +TextuRig (SFT) | 82.37 | 76.84 | Data augmentation +1.9 |
| +SKA-DPO | 85.57 | 81.12 | DPO +5.1 |
| +TextuRig & SKA-DPO | **87.83** | **84.36** | Complementary gains +7.4 |

### Key Findings

- Structural conditioning (2D strokes) is critical to model convergence; training without structural conditioning fails to converge on large-scale data.
- Skeleton generation is robust to input sparsity: CD scores remain stable when fewer than 5 joints are removed.
- An SKA-DPO preference score margin of 0.1 achieves the optimal balance.
- Generated skeleton–mesh pairs can be directly animated via Blender auto-skinning with good structural integrity.

## Highlights & Insights

1. **Skeleton-first paradigm**: This inverts the conventional mesh-then-rig workflow, granting users direct structural control.
2. **Elegant 2D-to-3D bridging**: A canvas-based tool allows users to create topologically isomorphic 2D inputs via click-and-connect interactions, elegantly resolving the 2D–3D domain gap.
3. **RL introduced into 3D generation**: DPO is adapted from language/image models to 3D mesh generation, using skeleton–mesh alignment as the reward signal.
4. **Modular design**: Skeleton generation and mesh synthesis are decoupled, allowing each component to be improved independently.

## Limitations & Future Work

- The number of skeleton joints is limited to 0–30, which may constrain complex skeletal structures.
- Input is restricted to orthographic 2D projections; multi-view guidance could improve quality.
- The TextuRig dataset (6.8K samples) remains relatively small; larger-scale data may yield further gains.
- Auto-skinning quality depends on Blender tooling; end-to-end skinning is identified as a future direction.

## Related Work & Insights

- MagicArticulate and UniRig represent the trend toward autoregressive skeleton generation but lack explicit structural control.
- SKDream's MCF skeleton and conditional generation framework provides the foundation upon which Stroke3D introduces significant improvements at both the data and optimization levels.
- The preference optimization approach of DiffusionDPO (Wallace et al., 2024) is effectively adapted to the 3D domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first method to generate rigged 3D meshes from 2D strokes; the skeleton-first pipeline is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Skeletons and meshes are evaluated separately on standard benchmarks with thorough ablations, though a user study is absent.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clearly presented with informative figures, though certain sections are somewhat verbose.
- **Value**: ⭐⭐⭐⭐ Lowers the barrier to creating animatable 3D assets, though adoption by professional artists remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 2D-LFM: Lifting Foundation Model without 3D Supervision](../../CVPR2026/3d_vision/2d-lfm_lifting_foundation_model_without_3d_supervision.md)
- [\[ICLR 2026\] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](../../ICCV2025/3d_vision/representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)
- [\[CVPR 2026\] Circular-DPO: Aligning Multi-Stage 3D Generative Models via Preference Feedback Loop](../../CVPR2026/3d_vision/circular-dpo_aligning_multi-stage_3d_generative_models_via_preference_feedback_l.md)

</div>

<!-- RELATED:END -->
