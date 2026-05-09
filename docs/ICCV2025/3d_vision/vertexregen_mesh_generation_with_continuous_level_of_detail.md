---
title: >-
  [Paper Note] VertexRegen: Mesh Generation with Continuous Level of Detail
description: >-
  [ICCV 2025][3D Vision][Mesh Generation] VertexRegen is proposed to reframe mesh generation—inspired by progressive meshes—as learning the inverse of edge collapse, i.e., vertex split, enabling "anytime" mesh generation with continuous level of detail.
tags:
  - ICCV 2025
  - 3D Vision
  - Mesh Generation
  - Progressive Meshes
  - Continuous Level of Detail
  - Autoregressive
  - Vertex Split
date: 2026-05-08
content_hash: 94d0b2788b69e6b3
---

# VertexRegen: Mesh Generation with Continuous Level of Detail

**Conference**: ICCV 2025
**arXiv**: [2508.09062](https://arxiv.org/abs/2508.09062)
**Code**: [Project Page](https://vertexregen.github.io)
**Area**: 3D Vision
**Keywords**: Mesh Generation, Progressive Meshes, Continuous Level of Detail, Autoregressive, Vertex Split

## TL;DR

VertexRegen is proposed to reframe mesh generation—inspired by progressive meshes—as learning the inverse of edge collapse, i.e., vertex split, enabling "anytime" mesh generation with continuous level of detail.

## Background & Motivation

Fundamental limitations of existing autoregressive mesh generation methods (MeshGPT, MeshXL, etc.):
- **Part-to-complete** generation paradigm: intermediate steps yield incomplete meshes (missing faces)
- **No detail control**: the full sequence must be completed to obtain a valid mesh
- Early stopping = missing faces, not a lower-resolution mesh

VertexRegen's innovation: a **coarse-to-fine** generation paradigm where every intermediate step produces a valid mesh, differing only in level of detail.

## Method

### Progressive Meshes Review

Two invertible operations:
- **Edge collapse** $\text{ecol}(v_s, v_t)$: merges two adjacent vertices, removing two faces and one vertex
- **Vertex split** $\text{vsplit}(v_s, v_l, v_r, v_t)$: the inverse operation, restoring a vertex and two faces

Progressive mesh representation: $\text{PM}(\mathcal{M}) = (\mathcal{M}_0, \{\text{vsplit}_0, \cdots, \text{vsplit}_{n-1}\})$

The edge collapse ordering is determined by QEM (Quadric Error Metrics).

### VertexRegen Serialization

Complete sequence:
```
M: [<bos>, [M_0 sequence], <sep>, [vsplit_0], ..., [vsplit_n-1], <eos>]
```

**Base mesh $\mathcal{M}_0$ tokenization**: follows the MeshXL scheme (9 tokens per face), but encodes only an extremely coarse initial mesh (averaging 18 faces vs. 457 in the original).

**Vertex split tokenization**: leverages the half-edge data structure to resolve ambiguity; each operation requires recording only four vertices $v_s, v_l, v_r, v_t$ (12 tokens, or 10 tokens in boundary cases).

### Key Role of the Half-Edge Data Structure

Vertex split requires determining which half of the one-ring neighborhood of $v_s$ belongs to $v_t$. By traversing half-edges from $\mathcal{H}_{ls}^1$ to $\mathcal{H}_{\cdot s}^r$, the partition is precisely determined:

$$\{v_k | \mathcal{H}_{\cdot s}^k, k \neq r\} = N_{k+1}(v_s) - \{v_l, v_r, v_t\}$$

### Guided Decoding

A state machine is maintained to execute vertex splits at runtime. Enforced constraints:
- $v_s$ must be a vertex in the current mesh
- $(v_s, v_l)$ and $(v_s, v_r)$ must be valid edges
- Only one of $v_l$ or $v_r$ is permitted to be `<nil>`

## Key Experimental Results

### Unconditional Mesh Generation

| Method | Tokenization | COV↑ | MMD↓ | 1-NNA (%) | JSD↓ |
|--------|-------------|------|------|-----------|------|
| MeshXL | Flattened coords | 51.76 | 8.30 | 50.84 | 3.81 |
| MeshAnything V2 | AMT | 50.33 | 8.50 | 52.25 | 4.84 |
| EdgeRunner | EdgeBreaker | 51.39 | 7.81 | 49.44 | 3.22 |
| **VertexRegen** | **Progressive** | 51.03 | 8.29 | 50.22 | **2.89** |

VertexRegen achieves generation quality comparable to SOTA while uniquely supporting continuous level of detail. It achieves the best JSD, indicating that the generated distribution more closely matches the reference distribution.

### Face-Constrained Generation (@400 Faces)

| Method | COV↑ | MMD↓ | 1-NNA |
|--------|------|------|-------|
| MeshXL (w/ FCC) | 41.20 | 10.03 | 59.06 |
| **VertexRegen** | **50.92** | **8.31** | **51.03** |

VertexRegen substantially outperforms baselines under low face-count constraints, as its coarse-to-fine generation preserves complete structure throughout.

### Compression Efficiency

| Method | Compression Ratio |
|--------|------------------|
| MeshXL | 1.0 |
| MeshAnything V2 | 0.46 |
| EdgeRunner | 0.47 |
| VertexRegen (w/ HE) | 0.73 |
| VertexRegen (w/o HE) | 0.89 |

The half-edge data structure reduces sequence length by 22%.

## Highlights & Insights

1. **Paradigm shift**: from "part-to-complete" to "coarse-to-fine," fundamentally transforming the properties of mesh generation
2. **Elegant use of the half-edge data structure**: resolves neighborhood assignment ambiguity in vertex splits
3. **Anytime generation**: stopping at any point yields a valid mesh
4. **Inherited theoretical guarantees from progressive meshes**: each vertex split is a deterministic inverse operation

## Limitations & Future Work

- The base mesh $\mathcal{M}_0$ still requires MeshXL-style generation, accounting for 5.68% of the total sequence length
- Compression efficiency (0.73) is inferior to AMT (0.46), as vertex splits require four vertices
- Predicting an invalid vertex split interrupts the chain of generation
- Guided decoding requires maintaining a state machine, increasing inference complexity

## Related Work & Insights

- MeshGPT, MeshXL: autoregressive mesh generation
- Progressive Meshes (Hoppe): theoretical foundation for progressive meshes
- EdgeRunner: EdgeBreaker compression-based tokenization

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (an elegant integration of progressive meshes and generative models)
- Technical Depth: ⭐⭐⭐⭐⭐ (sophisticated design of half-edge data structure and guided decoding)
- Experimental Thoroughness: ⭐⭐⭐⭐ (unconditional + conditional + ablation)
- Value: ⭐⭐⭐⭐⭐ (continuous LOD addresses a genuine practical need)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](../../NeurIPS2025/3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [\[ICCV 2025\] Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation](nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)

</div>

<!-- RELATED:END -->
