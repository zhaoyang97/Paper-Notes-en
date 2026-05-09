---
title: >-
  [Paper Note] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction
description: >-
  [ICCV2025][3D Vision][Mamba] MeshMamba introduces a Mamba state space model-based approach for articulated 3D mesh generation and reconstruction. By designing vertex serialization strategies based on body-part UV maps and template mesh coordinates, the method achieves efficient generation and reconstruction of meshes with tens of thousands of vertices, running 6–9× faster than Transformer-based counterparts.
tags:
  - ICCV2025
  - 3D Vision
  - Mamba
  - state space model
  - 3D mesh generation
  - human mesh recovery
  - vertex serialization
date: 2026-05-08
content_hash: e345d7b598a361da
---

# MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction

**Conference**: ICCV2025  
**arXiv**: [2507.15212](https://arxiv.org/abs/2507.15212)  
**Code**: -  
**Area**: 3D Vision  
**Keywords**: Mamba, state space model, 3D mesh generation, human mesh recovery, vertex serialization  

## TL;DR

MeshMamba introduces a Mamba state space model-based approach for articulated 3D mesh generation and reconstruction. By designing vertex serialization strategies based on body-part UV maps and template mesh coordinates, the method achieves efficient generation and reconstruction of meshes with tens of thousands of vertices, running 6–9× faster than Transformer-based counterparts.

## Background & Motivation

- **Problem Definition**: Generating and reconstructing 3D articulated body meshes with diverse body shapes and poses is a fundamental problem in computer vision and graphics.
- **Limitations of Prior Work**:
    - **Parametric methods** (SMPL/SMPL-X): Compact representations but constrained by predefined body models, making it difficult to capture complex deformations such as clothing.
    - **Vertex-based methods**: More flexible through direct manipulation of mesh vertices, but Transformer architectures face an $O(n^2)$ complexity bottleneck.
    - Existing vertex-level Transformer methods typically handle only coarse-resolution meshes of ~500 vertices, requiring additional upsampling that loses local geometric detail.
    - Consequently, current methods are limited to body-only pose reconstruction, unable to incorporate hand poses and facial expressions.
- **Key Challenge**: Mamba SSMs offer near-linear complexity and efficient inference, making them ideal for processing large numbers of vertex tokens. The core challenge lies in serializing mesh vertices into a 1D sequence amenable to Mamba.

## Method

### Overall Architecture

MeshMamba consists of three core components:
1. **Vertex Serialization**: Serializes mesh vertices into an ordered 1D sequence.
2. **MambaDiff3D**: A denoising diffusion model based on MeshMamba for 3D articulated mesh generation.
3. **Mamba-HMR**: A MeshMamba-based human mesh recovery model that reconstructs 3D human meshes from a single image.

### Key Design 1: Vertex Serialization

Unlike Transformers, Mamba requires ordered input sequences. Existing z-order and Hilbert curve orderings for 3D point clouds are unsuitable for mesh generation tasks that begin from random noise or images.

**Two serialization strategies**:
1. **DensePose IUV map-based**: Sorted by the I segmentation map (24 body parts), then by U and V maps within each part.
2. **Template mesh 3D coordinate-based**: Sorted by coordinate axes of the T-pose template (e.g., first by $x$, then $y$, then $z$).

**Multi-strategy combination**: Six serialization variants are generated (e.g., "xyz", "-xyz", "yzx"), alternated across different Mamba layers. Experiments show that a **combination of both strategies** achieves the best balance between efficiency and quality: all but one layer use one strategy, and the remaining layer uses the other.

### Key Design 2: MambaDiff3D

**Network architecture**: Inspired by U-ViT, the model consists of $L+1$ Mamba blocks with input/output MLPs. The first $L/2$ blocks (shallow), one middle block, and the last $L/2$ blocks (deep) are connected with skip connections between shallow and deep layers. Time embeddings are injected into each Mamba block via addition.

**Training loss** (v-prediction parameterization with cosine variance schedule):

$$L = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \; w_t \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|_2^2, \quad w_t = e^{-\lambda_t / 2}$$

**Sampling**: DDIM sampler with $T=1000$ diffusion steps; 50/100/250 sampling steps available.

**Joint vertex and normal generation**: To address local noise in vertex-only generation and global distortion in Jacobian-only generation, the method jointly generates vertex positions and surface normals, fusing them via a Poisson system to achieve smooth reconstruction while preserving surface detail.

### Key Design 3: Mamba-HMR

Replaces Transformer blocks in Mesh Transformer with MeshMamba blocks. CNN image features are used as input, with joint queries and vertex queries combined with positional encodings.

**Key advantage**: Directly outputs full-resolution meshes (10,475 vertices) without an upsampler, substantially reducing model parameters.

**Training loss**:

$$L = \lambda_{3D}^V L^V + \lambda_{3D}^J (L_{3D}^J + L_{reg3D}^J) + \lambda_{2D}^J (L_{2D}^J + L_{reg2D}^J) + \lambda_{edge} L_{edge} + \lambda_{lap} L_{lap} + \lambda_{normal} L_{normal}$$

where $L_{edge}$, $L_{lap}$, and $L_{normal}$ are local geometric regularization losses on edge lengths, Laplacian, and surface normals, respectively, which are critical for preserving local shape in dense meshes.

## Key Experimental Results

### Main Results: 3D Human Body Generation

| Method | Training Set | 1-NNA [%] ↓ | FID ↓ | APD ↑ |
|--------|-------------|-------------|-------|-------|
| Pose-NDF | AMASS | 92.0 | 3.92 | 37.81 |
| NRDF | AMASS | 81.6 | 0.64 | 23.12 |
| VPoser | AMASS | 60.7 | 0.05 | 14.68 |
| DiffSurf | SURREAL | 54.4 | - | - |
| **MambaDiff3D** | SURREAL | **53.1** | 0.32 | 23.01 |
| **MambaDiff3D** | AMASS | 55.1 | **0.22** | **23.8** |

**Whole-body Human Mesh Recovery (UBody dataset)**:

| Method | PA-MVE All ↓ | PA-MVE Hands ↓ | PA-MVE Face ↓ | MVE All ↓ | FPS |
|--------|-------------|----------------|---------------|-----------|-----|
| SMPLer-X-L† | 31.9 | 10.3 | 2.8 | 57.4 | 24 |
| AiOS | 32.5 | 7.3 | 2.8 | 58.6 | - |
| Multi-HMR-B | 31.4 | 9.8 | 6.1 | 65.1 | 23 |
| **Mamba-HMR†** | **25.9** | **9.7** | **2.1** | **51.7** | 22 |

### Ablation Study

**Comparison of block types** (1-NNA ↓):

| Block Type | 1-NNA [%] ↓ |
|------------|-------------|
| MLP | 73.7 |
| GNN | 74.2 |
| Transformer | 53.6 |
| **Mamba** | **53.1** |

**Comparison of serialization strategies**:

| Serialization | 1-NNA [%] ↓ |
|---------------|-------------|
| SMPL connectivity ×1 | 60.0 |
| part-IUV ×1 | 54.4 |
| part-IUV ×2 | 53.7 |
| SMPL ×1 + XYZ ×1 | **53.1** |

### Key Findings

1. **Inference speed advantage**: Generating a 10,475-vertex mesh on an A100 GPU takes 4.5s (250 steps) with Mamba vs. 28.1s with Transformer — **6× faster**; 9× faster on V100.
2. Training time: Mamba ~18 min/epoch vs. Transformer ~100 min/epoch (6,890 vertices).
3. Random-order serialization completely fails to train, confirming the necessity of structured ordering.
4. Combining two serialization strategies achieves the best balance; adding more strategies yields marginal improvement at increased inference cost.
5. Joint vertex and normal generation outperforms vertex-only or Jacobian-only generation.

## Highlights & Insights

1. **First application of Mamba to 3D mesh generation and reconstruction**: Elegantly leverages SSM near-linear complexity to overcome the efficiency bottleneck of Transformers on large vertex sets.
2. **Well-designed serialization**: Exploits prior knowledge of body-part structure and template mesh geometry to ensure serialization respects the structural topology of articulated bodies.
3. **Scalability breakthrough**: Enables dense mesh generation with 10,000+ vertices for the first time, capturing clothing deformations and grasping gestures.
4. **End-to-end full-resolution output**: Mamba-HMR eliminates the downsample–upsample pipeline, reducing information loss.
5. **Gradient-domain mesh representation innovation**: Combines position and normal generation in a non-end-to-end framework, achieving smooth reconstruction via a Poisson system.

## Limitations & Future Work

1. Restricted to close-fitting clothing with fixed topology; cannot handle topological changes from loose garments.
2. Limited generalization to out-of-distribution datasets not seen during training.
3. Non-end-to-end generation pipeline (generation and Poisson reconstruction are decoupled), potentially constraining holistic optimization.
4. Mamba-HMR still has room for improvement in hand and facial reconstruction accuracy.

## Related Work & Insights

- **Mamba in vision**: ViM (images) → DiS (image generation) → 3D point cloud analysis → **this work: first use for 3D meshes**.
- **Transformer vs. Mamba trade-off**: When the token count is large (>5,000), Mamba's efficiency advantage becomes substantial, offering inspiration for tasks requiring high-resolution outputs.
- **General serialization principle**: Using domain-specific prior knowledge (body-part structure) for ordering is generalizable to other scenarios where structured data must be fed into sequential models.
- **Gradient-domain methods in generation**: Combining Poisson reconstruction with generative models is a direction worthy of further exploration.

## Rating ⭐⭐⭐⭐

This work is the first to introduce Mamba into 3D mesh generation and reconstruction, presenting a novel method with well-motivated serialization design. The efficiency gains are substantial (6–9× speedup), and state-of-the-art results are achieved on both generation and reconstruction tasks. Experiments are thorough with detailed ablation analysis. Limitations include fixed topology and generalization constraints, but as a pioneering work the overall contribution is well-executed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[ICCV 2025\] Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation](nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)

</div>

<!-- RELATED:END -->
