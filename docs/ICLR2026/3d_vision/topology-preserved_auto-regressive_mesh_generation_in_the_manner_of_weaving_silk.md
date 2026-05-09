---
title: >-
  [Paper Note] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk
description: >-
  [3D Vision] This paper proposes a "silk-weaving"-inspired mesh tokenization algorithm that provides a canonical topological framework through vertex layering and ordering, guaranteeing manifoldness, watertightness, normal consistency, and part-awareness in generated meshes while achieving state-of-the-art compression efficiency.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: c67b4bd3acf9e57b
---

# Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk

- **Conference**: ICLR 2026
- **arXiv**: [2507.02477](https://arxiv.org/abs/2507.02477)
- **Code**: [Project Page](https://gaochao-s.github.io/pages/MeshSilksong/)
- **Area**: 3D Vision / Mesh Generation
- **Keywords**: Mesh Generation, Auto-regressive, Topology Preservation, Mesh Tokenization, Manifold, Watertight

## TL;DR

This paper proposes a "silk-weaving"-inspired mesh tokenization algorithm that provides a canonical topological framework through vertex layering and ordering, guaranteeing manifoldness, watertightness, normal consistency, and part-awareness in generated meshes while achieving state-of-the-art compression efficiency.

## Background & Motivation

Auto-regressive mesh generation has emerged as an important direction in 3D content creation. However, existing methods (MeshAnything v2, EdgeRunner, BPT, DeepMesh) treat meshes as simple collections of triangles without awareness of global topological structure, leading to:

**Inability to guarantee watertightness**: Generated meshes contain holes, hindering downstream applications such as 3D printing and physical simulation.

**Lack of part-awareness**: Without the concept of connected components, small yet important parts (e.g., eyes) cannot be captured.

**Inconsistent normal orientation**: Flipped face normals produce rendering artifacts.

**Non-manifold topology**: Local patch-based methods (BPT, DeepMesh) cannot guarantee manifold topology.

**Core Contribution**: The first mesh tokenization algorithm that simultaneously guarantees manifoldness, watertightness detection, normal consistency, and part-awareness.

## Method

### Overall Architecture

Three sequential steps:
1. **Vertex Layering and Ordering** → 2. **Inter-layer Adjacency Matrix Compression** → 3. **Token Packing**

### 1. Vertex Layering and Ordering

For each connected component:
- A starting half-edge $j$-$m$ is determined via y-z-x coordinate sorting.
- BFS assigns a layer index $L$ based on shortest graph distance from the starting vertex $j$.
- Vertices within each layer are ordered through half-edge traversal (analogous to a mathematical induction proof).
- Vertices are labeled $\mathcal{V}_i^L$, where $L$ denotes the layer index and $i$ denotes the intra-layer order.

### 2. Inter-layer Adjacency Matrix Compression

Vertex connections are divided into two categories:

**Self-layer matrix $\mathcal{S}_L$** (blue edges): connections between vertices within the same layer.
- Symmetric 0-1 matrix of size $M \times M$.
- Binary compression with a fixed window size of $W=8$.
- Covers 99.1% of cases; extreme cases use COO format.

**Inter-layer matrix $\mathcal{B}_L$** (red edges): connections between vertices in adjacent layers.
- 0-1 matrix of size $M \times N$.
- RLE-like compression: consecutive "1"s are encoded as $(x, y)$ (starting column index, length).
- Further reformulated as an equivalent "Stars and Bars" problem, requiring only one token per row.

**Token Packing**: Each vertex $\mathcal{V}_i^L$ produces 4 tokens:
- 2 position tokens $V_{(L,i)}$ (compressed using block-offset representation)
- 1 self-layer topology token $S_{(L,i)}$
- 1 inter-layer topology token $B_{(L,i)}$

### 3. Geometric Property Guarantees

**(1) Manifold Topology**: During generation, edge connections exist only within the same layer or between adjacent layers; triangles are filled layer by layer, strictly guaranteeing manifold topology.

**(2) Watertightness Detection/Repair**: Holes necessarily arise from zeros in the self-layer/inter-layer matrices being incorrectly predicted, enabling direct detection and correction.

**(3) Normal Consistency**: The half-edge direction in layer $L$ is defined as ascending and in layer $L-1$ as descending, ensuring counter-clockwise vertex traversal for every triangle face and thus consistent normals.

**(4) Part-Awareness**: A special token $C$ marks the beginning of each connected component, supporting the generation of small parts.

### Non-manifold Handling Algorithm

For non-manifold edges (shared by more than 2 faces):
- Unlike Libigl's degree-priority merging strategy, the proposed method additionally detects the "edge graph" structure $\mathcal{G}$ around non-manifold vertices.
- This ensures the edge graph forms pure cycles to preserve surface integrity.

### Loss & Training

**Progressive Balanced Sampling**:

$$p_j^{PB}(t) = (1 - t/T) p_j^{IB} + (t/T) p_j^{CB}$$

Early training favors instance-balanced sampling (learning simpler cases); later training shifts toward category-balanced sampling (learning complex cases), with categories defined per 100 faces.

**Loss Function** — Standard cross-entropy loss:

$$\mathcal{L}_{ce} = -\sum_{t=1}^{T-1} S_{t+1} \log \hat{S}_t$$

## Key Experimental Results

### Dataset and Evaluation Metrics

- **Training**: gObjaverse (~280k) + ShapeNetV2 + 3D-FUTURE + Toys4K (~100k), without manual filtering.
- **Evaluation**: 500 held-out samples from gObjaverse.
- **Metrics**: CD (Chamfer Distance), HD (Hausdorff Distance), NC (Normal Consistency), |NC|, Bits-per-face, Compression Ratio.

### Main Results

| Method | CD↓ | HD↓ | NC↑ | \|NC\|↑ | Bits/face↓ | Comp. Ratio↓ |
|--------|-----|-----|-----|---------|------------|--------------|
| EdgeRunner* | 0.053 | 0.144 | 0.418 | 0.789 | 29.61 | 0.47 |
| TreeMeshGPT | 0.030 | 0.103 | 0.706 | 0.892 | 42.00 | 0.22 |
| BPT | 0.027 | 0.094 | 0.770 | 0.909 | 28.48 | 0.26 |
| **Ours** | **0.025** | **0.087** | **0.792** | **0.924** | **26.65** | **0.22** |

### Ablation Study

| Ablation | CD↓ | HD↓ | NC↑ | \|NC\|↑ |
|----------|-----|-----|-----|---------|
| w/ resampling | 0.025 | 0.087 | 0.792 | 0.924 |
| w/o resampling | 0.032 | 0.103 | 0.700 | 0.880 |
| Manifold + non-manifold data | 0.022 | 0.080 | 0.801 | 0.932 |
| Manifold data only | 0.027 | 0.090 | 0.688 | 0.871 |

### Geometric Property Comparison

| Method | Lossless | Manifold | Watertight | Normal Consistent | Part-Aware |
|--------|----------|----------|------------|-------------------|------------|
| Ours | ✓ | ✓ | ✓ | ✓ | ✓ |
| MeshAnything v2 | ✓ | ✗ | ✗ | ✗ | ✗ |
| EdgeRunner | ✓ | ✓ | ✗ | ✗ | ✗ |
| BPT | ✓ | ✗ | ✗ | ✗ | ✗ |

## Highlights & Insights

1. **Elegant silk-weaving analogy**: The inter-layer weaving paradigm naturally guarantees topological properties, reflecting a clever design philosophy.
2. **Only method to simultaneously guarantee all 5 geometric properties** in tokenization.
3. **State-of-the-art compression efficiency**: 26.65 Bits/face, 0.22 compression ratio.
4. **Strong practical applicability**: Supports downstream applications including 3D printing, physical simulation, and animation rigging.
5. **Online non-manifold handling**: Effectively expands the scale of trainable data.

## Limitations & Future Work

1. Vocabulary size is relatively large (up to 10,267), though bits-per-face remains optimal.
2. A maximum per-layer vertex count $m$ must be predefined, limiting support for extremely complex meshes.
3. Currently restricted to triangle meshes; supporting mixed polygonal meshes is a future direction.
4. The model has approximately 500M parameters; training requires 16×H800 GPUs for approximately 15 days.

## Related Work & Insights

- **VQ-VAE methods**: MeshGPT (Siddiqui et al., 2024) employs an encoder-decoder architecture for mesh tokenization.
- **Direct quantization**: MeshXL and BPT directly quantize triangle vertex coordinates.
- **Tree traversal**: EdgeRunner and TreeMeshGPT guarantee manifoldness via tree structures but at lower compression efficiency.
- **Reinforcement learning**: DeepMesh fine-tunes with human feedback to improve aesthetic quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The inter-layer weaving concept is original and elegant.
- **Practicality**: ⭐⭐⭐⭐⭐ — Geometric property guarantees directly serve industrial applications.
- **Clarity**: ⭐⭐⭐⭐ — Algorithmic descriptions are clear and illustrations are intuitive.
- **Significance**: ⭐⭐⭐⭐⭐ — Addresses a critical topological challenge in auto-regressive mesh generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](../../ICCV2025/3d_vision/meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[ICLR 2026\] Universal Beta Splatting](universal_beta_splatting.md)
- [\[ICLR 2026\] UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)

</div>

<!-- RELATED:END -->
