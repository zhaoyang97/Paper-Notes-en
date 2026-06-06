---
title: >-
  [Paper Note] AnimateAnyMesh: A Feed-Forward 4D Foundation Model for Text-Driven Universal Mesh Animation
description: >-
  [ICCV 2025][3D Vision][4D Generation] This paper proposes AnimateAnyMesh, the first feed-forward text-driven universal mesh animation framework. It introduces DyMeshVAE to decompose dynamic meshes into initial positions…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "4D Generation"
  - "Text-Driven Animation"
  - "Dynamic Mesh"
  - "VAE"
  - "Rectified Flow"
date: 2026-05-08
content_hash: 4ca6eaaa8cabad1a
---

# AnimateAnyMesh: A Feed-Forward 4D Foundation Model for Text-Driven Universal Mesh Animation

**Conference**: ICCV 2025
**arXiv**: [2506.09982](https://arxiv.org/abs/2506.09982)  
**Code**: [Project Page](https://animateanymesh.github.io/AnimateAnyMesh/) (available; code, data, and models promised to be open-sourced)  
**Area**: 3D Vision / 4D Generation / Mesh Animation
**Keywords**: 4D Generation, Text-Driven Animation, Dynamic Mesh, VAE, Rectified Flow

## TL;DR
This paper proposes AnimateAnyMesh, the first feed-forward text-driven universal mesh animation framework. It introduces DyMeshVAE to decompose dynamic meshes into initial positions and relative trajectories, compressing them into a latent space. A Rectified Flow-based MMDiT model then learns the trajectory distribution conditioned on text. Trained on the 4M+ DyMesh dataset, the framework generates high-quality animations for meshes of arbitrary topology within 6 seconds, comprehensively outperforming DG4D, L4GM, and Animate3D.

## Background & Motivation
4D content generation (dynamic 3D) is an essential requirement in VR/AR and gaming, yet existing methods suffer from two major limitations:
1. **Per-scene optimization methods** (e.g., DG4D): rely on SDS distillation from video diffusion models to optimize 3DGS/NeRF, incurring high computational cost (~10 min/scene) and spatial inconsistency due to the absence of real 4D data supervision.
2. **Multi-view dynamic video methods** (e.g., Animate3D, L4GM): generate multi-view videos before reconstructing 4D representations, requiring post-processing with errors accumulating across stages, and still depending on per-scene reconstruction.

The core insight of this paper is that, rather than directly generating 4D end-to-end, one can leverage the abundance of high-quality 3D mesh assets and decompose the problem into two steps: *geometry creation* and *motion modeling*. Dynamic meshes, as the standard representation in graphics pipelines, are rendering-efficient and naturally decouple geometry from motion. The authors therefore focus on **text-driven mesh animation** as the foundational task.

## Core Problem
How to build a feed-forward framework that generates semantically aligned and temporally coherent animation sequences for 3D meshes of arbitrary topology from text descriptions within seconds? The core challenges are: (1) efficient compression and reconstruction of dynamic meshes, (2) modeling the motion distribution conditioned on text, and (3) the scarcity of large-scale 4D training data.

## Method

### Overall Architecture
AnimateAnyMesh consists of three components:
- **DyMeshVAE**: compresses dynamic mesh sequences into a structured latent space and reconstructs them with high fidelity.
- **Shape-Guided Text-to-Trajectory Model**: learns the trajectory distribution conditioned on text and mesh shape within the compressed latent space.
- **DyMesh Dataset**: a 4M+ dynamic mesh sequence dataset supporting large-scale training.

Inference pipeline: static mesh + text description → DyMeshVAE encoder extracts mesh shape features → Rectified Flow sampling generates trajectory latent codes → DyMeshVAE decoder reconstructs vertex trajectories → outputs dynamic mesh sequence.

### Key Designs
1. **DyMeshVAE Encoder — Trajectory Decomposition + Topology-Aware Attention**:

    - Vertex sequences $V$ are decomposed into initial frame positions $V_0$ and relative trajectories $V_T$ (i.e., $V_t = V_0 + V_T^t$), making the motion distribution closer to a zero-mean Gaussian.
    - Separate positional encodings (PE) are applied to $V_0$ and $V_T$, mapping low-dimensional information to a higher-dimensional space to improve discriminability.
    - Core design: an adjacency matrix $\text{Adj}$ is constructed from face information $F$ and used as a self-attention mask, allowing each vertex to aggregate information from its topological neighbors. This resolves the confusion between spatially proximate but topologically unrelated vertices (e.g., hands and waist).
    - FPS (Farthest Point Sampling) is applied on the topology-enriched features to sample $n$ tokens (default 512), followed by cross-attention to aggregate global information.

2. **DyMeshVAE Decoder — Shape-Feature-Guided Trajectory Reconstruction**:

    - Self-attention is applied to the sampled topology-aware features $V_0^n$, while the resulting attention map is used to project $Z_T^n$.
    - $K$ identical blocks are stacked to progressively enhance features.
    - During decoding, the full vertex features $V_0$ of the initial mesh serve as queries, reconstructing the complete relative trajectories from the compressed latent via cross-attention.
    - This design allows training with a fixed 512 tokens while dynamically adjusting the token count at inference to accommodate meshes of varying complexity.

3. **Shape-Guided Text-to-Trajectory Model — MMDiT + Rectified Flow**:

    - Based on the MMDiT (Multimodal DiT) architecture with 12 Transformer blocks and 8-head attention.
    - Text is encoded by CLIP ViT-L/14 (up to 77 tokens).
    - AdaLN (Adaptive Layer Normalization) is applied separately to trajectory and text embeddings to eliminate cross-modal distribution discrepancy.
    - The two modalities are concatenated for self-attention computation, then split and restored to their original scales.

### Loss & Training
- **DyMeshVAE loss**: $\mathcal{L}_{dvae} = \mathcal{L}_{rec} + \gamma \cdot \mathcal{L}_{kl}$, where $\mathcal{L}_{rec}$ is MSE reconstruction loss, $\mathcal{L}_{kl}$ is KL divergence regularization, and $\gamma = 0.001$.
- **Rectified Flow training**: noise is added to trajectory latents $Z_T^n$; the model learns to predict the velocity field using a tangent timestep schedule; CFG scale $\gamma = 3.0$.
- **DyMeshVAE**: Adam optimizer, lr = 1e-4, 1000 epochs, 8× H20 GPUs.
- **Flow Model**: Adam optimizer, lr = 2e-4, 1000 epochs, 32× H20 GPUs.
- Inference uses 64-step uniform ODE solving.
- **DyMesh Dataset**: 66k complete animations (Objaverse ~55k + AMASS ~8k + DT4D ~2k), yielding ~2.6M 16-frame sequences and ~1.6M 32-frame sequences (~4M+ total) via slicing and augmentation; text annotations are generated by Qwen-2.5-VL.

## Key Experimental Results

| Method | I2V↑ | M.Sm↑ | Aest.Q↑ | User.Ta↑ | User.Mn↑ | User.Sp↑ | Time |
|--------|------|-------|---------|----------|----------|----------|------|
| DG4D | 0.811 | 0.926 | 0.476 | 2.130 | 2.460 | 2.755 | 10 min |
| L4GM | 0.844 | 0.992 | 0.464 | 2.885 | 2.865 | 2.835 | 30 s |
| Animate3D | 0.936 | 0.992 | 0.526 | 2.850 | 3.195 | 3.405 | 14 min |
| **Ours** | **0.954** | **0.995** | **0.539** | **4.505** | **4.700** | **4.790** | **6 s** |

The proposed method achieves comprehensive improvements on VBench metrics and user studies: I2V (shape preservation) +1.9%, user text-alignment score +58% (vs. Animate3D), and inference speed 140× faster (vs. DG4D).

| Inference Time | 5k vertices | 10k vertices | 20k vertices | 50k vertices |
|----------------|------------|--------------|--------------|--------------|
| Time (s) | 3.95 | 5.99 | 10.68 | 21.86 |

### Ablation Study

| Adj | PE₀ | PE_T | SepAttn | EmbFPS | Rec Error↓ |
|-----|-----|------|---------|--------|------------|
| ✗ | ✓ | ✓ | ✓ | ✓ | 0.500 |
| ✓ | ✗ | ✓ | ✓ | ✓ | 0.443 |
| ✓ | ✓ | ✗ | ✓ | ✓ | 0.441 |
| ✓ | ✓ | ✓ | ✗ | ✓ | 0.478 |
| ✓ | ✓ | ✓ | ✓ | ✗ | 0.291 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **0.223** |

- **Adjacency matrix (Adj)** has the largest impact (removal increases error from 0.223 to 0.500), demonstrating that topological information is critical for distinguishing vertices in different semantic regions.
- **EmbFPS** (performing FPS on enriched features rather than raw coordinates) contributes substantially (0.291 → 0.223).
- Scaling experiments show that increasing the number of vertices, frames, and model parameters consistently improves performance, confirming good scalability (I2V improves from 0.954 to 0.968 as parameters scale from 200M to 740M).

## Highlights & Insights
- **First feed-forward universal mesh animation framework**: no per-scene optimization required; 6-second generation makes it practically viable.
- **Trajectory decomposition**: decomposing $V$ into $V_0 + V_T$ naturally decouples shape and motion; the resulting motion distribution approximates a Gaussian, facilitating generative model learning.
- **Topology-aware attention**: cleverly leverages mesh topology to construct attention masks, resolving the confusion between spatially close but semantically distinct vertices (e.g., hand–waist entanglement).
- **4M+ scale dataset**: systematically constructed from multi-source 4D assets, representing the largest dynamic mesh dataset to date.
- **Flexible token count**: fixed at 512 tokens during training, dynamically adjustable at inference to accommodate meshes of varying complexity.

## Limitations & Future Work
- **Limited dataset diversity**: despite 4M+ sequences, fewer than 100k unique mesh identities exist, limiting generalization to rare categories.
- **Text annotation quality**: captions are generated by Qwen-2.5-VL from rendered videos, but VLMs perform poorly on background-free 3D renderings, resulting in coarse motion descriptions.
- **Sequence length constraint**: only 16/32-frame generation is currently supported; long-duration animations cannot be produced.
- **No physical constraints**: generated motions lack guarantees of physical plausibility (e.g., penetration, unrealistic non-rigid deformations).
- **Limited evaluation scale**: quantitative comparisons are conducted on only 10 test samples, and only three baselines are included.

## Related Work & Insights
- **vs. DG4D** (SDS distillation): DG4D relies on SDS-based optimization of 3DGS using video diffusion models, requiring 10 minutes per scene with poor shape preservation and severe object drift. AnimateAnyMesh directly predicts trajectories in the mesh vertex space, completing generation in 6 seconds with superior geometric detail.
- **vs. Animate3D** (multi-view video + ARAP optimization): Animate3D involves a complex pipeline (Mesh → GS → multi-view video → ARAP optimization) taking 14 minutes per scene, with errors accumulating at each stage. AnimateAnyMesh is end-to-end feed-forward, outperforming in both quality and efficiency.
- **vs. L4GM** (single-view video reconstruction): L4GM is constrained by the quality of the video generator and performs poorly on background-free 3D renderings; though requiring only 30 seconds, its user scores are low. AnimateAnyMesh directly models the motion distribution in latent space without relying on video generation.

- **Extension of the feed-forward paradigm**: this work demonstrates the feasibility of feed-forward 4D generation; the topology-aware compression design in DyMeshVAE is transferable to other tasks involving irregular topological structures (e.g., molecular dynamics, cloth simulation).

## Rating
- Novelty: ⭐⭐⭐⭐ First feed-forward universal mesh animation framework; the trajectory decomposition and topology-aware attention in DyMeshVAE are novel contributions; however, the Rectified Flow + MMDiT combination is already mature in the generative modeling literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies are comprehensive (5 components + FPS ratio + scaling), but quantitative comparisons use only 10 test samples and include only 3 baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Structure is clear, motivation is well-articulated, technical details are complete, and figures are informative.
- Value: ⭐⭐⭐⭐⭐ The 4M+ dataset and the first practical feed-forward 4D framework represent a landmark contribution to 4D content creation; the commitment to full open-source release further enhances its impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)
- [\[CVPR 2026\] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](../../CVPR2026/3d_vision/physgm_large_physical_gaussian_4d_synthesis.md)
- [\[ICCV 2025\] FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation](fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)
- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)

</div>

<!-- RELATED:END -->
