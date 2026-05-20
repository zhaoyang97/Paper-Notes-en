---
title: >-
  [Paper Note] Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation
description: >-
  [ICCV2025][3D Vision][mesh generation] Nautilus proposes a locality-aware autoencoder for scalable artist-like mesh generation. By introducing a nautilus-shell-structured mesh tokenization algorithm that reduces sequence…
tags:
  - "ICCV2025"
  - "3D Vision"
  - "mesh generation"
  - "autoregressive"
  - "tokenization"
  - "locality-aware"
  - "point cloud conditioning"
date: 2026-05-08
content_hash: 288b3f077e4d242d
---

# Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation

**Conference**: ICCV2025
**arXiv**: [2501.14317](https://arxiv.org/abs/2501.14317)  
**Code**: -  
**Area**: 3D Vision
**Keywords**: mesh generation, autoregressive, tokenization, locality-aware, point cloud conditioning

## TL;DR

Nautilus proposes a locality-aware autoencoder for scalable artist-like mesh generation. By introducing a nautilus-shell-structured mesh tokenization algorithm that reduces sequence length to 1/4 of the naive baseline, and combining it with a dual-stream point cloud conditioner to improve local structural fidelity, Nautilus achieves for the first time direct high-quality mesh generation with up to 5,000 faces.

## Background & Motivation

- **Problem Definition**: Automatically generating high-quality triangular meshes that exhibit artist-like compactness, structural regularity, and topological correctness.
- **Limitations of Prior Work**:
    - **Intermediate representation methods** (NeRF, 3DGS, SDF, etc.): Convert to meshes via marching cubes or Poisson reconstruction, yielding overly dense and suboptimal meshes lacking continuous surface quality.
    - **Direct mesh generation** (MeshGPT, MeshAnything): Autoregressively model vertices and faces, but suffer from two fundamental problems:
    1. **Poor local structural fidelity**: Manifold defects such as surface holes, overlapping faces, and missing parts, especially under complex topology.
    2. **Limited face count**: Excessively long sequences constrain the maximum number of faces, making it difficult to capture fine topological details.
- **Core Insight**: The locality of manifold meshes—adjacent faces share edges, and each face cluster converges to a central vertex—provides two key inspirations:
  1. Prioritizing local dependencies among neighbors ensures precise face interconnection and manifold validity.
  2. Explicitly modeling locally shared edges and vertices significantly compresses sequence length.

## Method

### Overall Architecture

1. **Nautilus-shell mesh tokenization**: Serializes artist meshes into compact token sequences.
2. **Dual-stream point cloud conditioner**: Provides geometric guidance for both global consistency and local structural fidelity.
3. **Autoregressive sequence decoder**: Transformer-based decoding supporting point cloud or single-view image conditioning.

### Key Design 1: Nautilus-Shell Mesh Tokenization

**Problem with the naive approach**: Flattening the 3 vertices of each face into $9N$ coordinates fails to preserve spatial adjacency and yields excessively long sequences.

**Nautilus-shell representation**: Mesh faces are organized into multiple shells, each centered on a vertex $O$ and containing an ordered sequence of surrounding vertices $P$. Each face is composed of $O$ and two adjacent $P$ vertices. This allows $N$ faces to be represented with only $N+2$ vertices instead of $3N$:

$$S(\mathbf{f}_{OP_1P_2}, \mathbf{f}_{OP_2P_3}, \ldots) = \{O, P_1, P_2, P_3, \ldots\}$$

Adding a new face requires extending the sequence by only one vertex, greatly compressing the sequence.

**Consecutive shell traversal**: Upon completing a shell, the neighbor with the highest degree among the last traversed vertex's neighbors is selected as the center of the next shell. This ensures continuity and spatial proximity between shells.

**Coordinate compression**: 3D coordinates $(x, y, z)$ are mapped to 2D space $(u, v)$:

$$x \cdot \alpha^2 + y \cdot \alpha + z = u \cdot \beta + v$$

With resolution 128 and multiplier 2048, separate codebooks are built for $u$ (size 1024) and $v$ (size 2048). The $u$ coordinate of the center vertex $O$ uses an independent codebook of size 1024 to mark the start of each shell, eliminating the need for special separator tokens.

**Compression efficiency**: The resulting sequence length is only **1/4** of the naive approach (compression ratio 0.275, vs. 0.462 for AMT and 0.474 for EdgeRunner).

### Key Design 2: Dual-Stream Point Cloud Conditioner

**Global point cloud encoder**: Uses the Michelangelo encoder to extract global features $f_{glb}$, which serve as keys and values in the decoder's cross-attention layers, providing overall shape information.

**Local point cloud encoder**: Uses a PointConv module $f_{loc}(\cdot)$ to capture local geometric information. For each shell's center vertex $O_k$, KNN sampling selects 100 nearest points to extract local features $f_{loc}(O_k)$, which are injected into the token features of each vertex $P_{k,i}$ within the shell (inspired by ControlAR).

**Tight integration with the shell structure**: Feature injection from the local encoder is synchronized step-by-step with shell generation, enabling progressive local geometric constraints.

### Autoregressive Decoding and Training

**Generation**: Follows the next-token prediction paradigm:

$$p(\mathbf{M}) = \prod_{i=1}^{L} p(c_i | c_{<i}), \quad c_i \in \{0, 1, \ldots, \alpha - 1\}$$

**Training loss**: Cross-entropy loss

$$L_{CE} = \text{CrossEntropy}(\hat{S}, S(\mathbf{M})_{>0})$$

**Image conditioning**: Leveraging Michelangelo's multimodal-aligned feature space, the model is trained with a frozen point cloud encoder and at inference time the encoder is replaced with an image encoder for single-image conditioned generation.

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | C.Dist. ↓ | H.Dist. ↓ | User Preference ↑ |
|--------|-----------|-----------|-------------------|
| MeshAnything | 0.133 | 0.293 | 10.27% |
| MeshAnythingV2 | 0.106 | 0.248 | 13.17% |
| **Nautilus** | **0.087** | **0.176** | **88.68%** |

### Tokenization Algorithm Comparison

| Metric | AMT | EdgeRunner | **Nautilus** |
|--------|-----|------------|--------------|
| Compression Ratio ↓ | 0.462 | 0.474 | **0.275** |
| Local Ratio ↑ | 0.378 | 0.461 | **0.554** |

### Key Findings

1. **Superior compression ratio**: Nautilus tokenization achieves a compression ratio of 0.275, significantly outperforming AMT (0.462) and EdgeRunner (0.474), enabling generation of meshes with up to 5,000 faces.
2. **Strong locality preservation**: A local ratio of 0.554 indicates that adjacent tokens in the sequence are also highly likely to be geometrically adjacent in the mesh, facilitating the autoregressive model's learning of local dependencies.
3. **Overwhelming user preference**: An 88.68% user preference rate far surpasses MeshAnything (10.27%) and V2 (13.17%).
4. Using coordinate compression or the shell structure alone is insufficient; their combination is necessary to achieve optimal quality and scalability.
5. The local point cloud encoder provides critical improvements in regions with complex topology, such as localized holes.

## Highlights & Insights

1. **Precise problem identification**: The method accurately targets the fundamental bottleneck of current direct mesh generation approaches—neglect of locality inherent in manifold meshes.
2. **Elegant shell-based tokenization**: Mimics the way artists construct meshes (fan-like expansion around a central vertex), simultaneously preserving locality and achieving high compression.
3. **Dual-stream conditioning mechanism**: Global stream preserves overall shape; local stream maintains topological details—perfectly synergistic with the shell-based generation paradigm.
4. **First-ever direct generation of 5,000-face meshes**: A substantial leap compared to MeshGPT (800 faces) and MeshAnything (1,600 faces).
5. **New evaluation metric—Local Ratio**: Quantifies the preservation of local dependencies after serialization, offering broad utility for evaluating mesh tokenization algorithms.

## Limitations & Future Work

1. Inference remains slow: generating a 5,000-face mesh requires approximately 4 minutes.
2. Heavy reliance on test samples generated by downstream image-to-3D methods, which may not align with the distribution of real artist meshes.
3. The tokenization algorithm's handling of non-manifold meshes is not discussed.
4. Training requires 311K high-quality artist meshes, imposing substantial data acquisition costs.
5. The greedy traversal strategy for shell construction (selecting the highest-degree neighbor) may not be globally optimal.

## Related Work & Insights

- **Evolution of mesh tokenization**: MeshGPT (VQ-VAE) → MeshAnythingV2 (AMT) → EdgeRunner (half-edge) → **Nautilus (shell structure)**—the trend is a shift from general-purpose tokenization toward specialized tokenization that exploits mesh geometric structure.
- **Importance of locality in sequence modeling**: Analogous to the role of positional encodings in NLP, preservation of spatial locality in 3D mesh tokenization is critical for generation quality.
- **Hierarchical conditioning design**: The dual-stream global+local conditioning paradigm is generalizable to other generation tasks requiring multi-scale control.
- **Balance between compression and quality**: Higher compression ratios allow models to train on more complex samples, serving as a key lever for raising the performance ceiling.

## Rating ⭐⭐⭐⭐⭐

The paper precisely identifies locality as the key bottleneck, proposes an elegant solution via shell-structured tokenization and dual-stream conditioning, and delivers strong results: 4× compression, 88.68% user preference, and first-ever 5,000-face direct generation. The new Local Ratio metric has broad applicability. The work is highly complete and represents an important advance in direct mesh generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[ICCV 2025\] VertexRegen: Mesh Generation with Continuous Level of Detail](vertexregen_mesh_generation_with_continuous_level_of_detail.md)
- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](../../NeurIPS2025/3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)

</div>

<!-- RELATED:END -->
