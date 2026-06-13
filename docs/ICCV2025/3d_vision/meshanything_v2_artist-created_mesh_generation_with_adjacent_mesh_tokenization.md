---
title: >-
  [Paper Note] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization
description: >-
  [ICCV 2025][3D Vision][Mesh Generation] MeshAnything V2 proposes Adjacent Mesh Tokenization (AMT), which represents adjacent faces using a single vertex rather than the conventional three…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Mesh Generation"
  - "Serialization"
  - "Tokenization"
  - "Autoregressive"
  - "Artist-Created Mesh"
date: 2026-05-08
content_hash: 57a928c53a7a983e
---

# MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization

**Conference**: ICCV 2025
**arXiv**: [2408.02555](https://arxiv.org/abs/2408.02555)  
**Code**: [https://buaacyw.github.io/meshanything-v2/](https://buaacyw.github.io/meshanything-v2/)  
**Area**: 3D Vision / Mesh Generation
**Keywords**: Mesh Generation, Serialization, Tokenization, Autoregressive, Artist-Created Mesh

## TL;DR
MeshAnything V2 proposes Adjacent Mesh Tokenization (AMT), which represents adjacent faces using a single vertex rather than the conventional three, reducing the average token sequence length by approximately half. This allows the maximum number of generated faces to scale from 800 to 1600 without additional computational cost, significantly improving the efficiency and quality of autoregressive mesh generation.

## Background & Motivation

**Background**: Autoregressive mesh generation has emerged as a prominent research direction, treating 3D meshes as sequences of faces and generating them vertex-by-vertex using LLM-style Transformers. Representative methods include PolyGen, MeshGPT, MeshXL, and MeshAnything. These methods learn from distributions of artist-created meshes (AMs) to produce efficient, aesthetically coherent, and production-ready geometry.

**Limitations of Prior Work**: Existing methods cannot generate complex meshes with large face counts. The fundamental bottleneck lies in **tokenization inefficiency**: each face is represented by three vertices, and each vertex requires three tokens (x/y/z coordinates), resulting in a token sequence length nine times the number of faces. This incurs substantial computational and memory overhead, and the high redundancy in the sequence degrades sequence learning performance.

**Key Challenge**: Meshes are graph-structured data admitting infinitely many serialization schemes. Effective tokenization must simultaneously satisfy two objectives: (a) **compactness**—shorter sequences reduce computational complexity; and (b) **regularity**—well-structured sequences are easier for Transformers to learn.

**Goal**: Design a more efficient mesh tokenization scheme that shortens the sequence while preserving its regularity and learnability.

**Key Insight**: The NLP community has extensively demonstrated the importance of tokenization for sequence learning (e.g., BPE vs. WordPiece). For graph-structured data such as meshes, the impact of tokenization is even more profound. A key observation is that redundancy in current methods stems primarily from repeatedly encoding already-visited vertices—if adjacent faces share an edge, only one new vertex is needed to represent the next face.

**Core Idea**: Adjacent faces share two vertices; therefore, only one new vertex is required to represent the next adjacent face. When no adjacent face is found, a special token "&" is inserted to restart the sequence.

## Method

### Overall Architecture
The input is a point cloud shape condition $\mathcal{S}$ (8192 points), encoded by a pretrained point cloud encoder into a token prefix $\mathcal{T}_S$. The mesh $\mathcal{M}$ is encoded via AMT into a compact token sequence $\mathcal{T}_M$. The two are concatenated and fed into an OPT-350M decoder-only Transformer, which learns the conditional distribution $p(\mathcal{M}|\mathcal{S})$ via cross-entropy loss. At inference time, $\mathcal{T}_M$ is autoregressively generated given $\mathcal{T}_S$ and decoded back into a mesh.

### Key Designs

1. **Adjacent Mesh Tokenization (AMT)**:

    - **Function**: Encodes meshes into more compact token sequences.
    - **Mechanism**: Conventional methods represent each face as three ordered vertices $f_i = (v_{i1}, v_{i2}, v_{i3})$, yielding a sequence of length $3N$ ($N$ = number of faces). AMT proceeds as follows:
        - The first face is still represented by three vertices $(v_1, v_2, v_3)$.
        - Subsequent faces that share an edge with the previous face (i.e., adjacent faces) are represented by appending only one new vertex.
        - When no adjacent face is found, a special token "&" marks the interruption, after which the next unencoded face begins a new strip with three vertices.
    - In the ideal case (infrequent "&" tokens), the sequence length reduces to $N+2$ (approximately one-third of the conventional approach). Empirically on Objaverse, the average reduction is approximately 50%.
    - **Design Motivation**: Eliminates redundant vertex repetition in the face sequence while exploiting topological adjacency to produce spatially contiguous sequences.

2. **Vertices Swap**:

    - **Function**: Expands the set of explorable adjacent faces.
    - **Mechanism**: Consider $f_1 = (v_1, v_2, v_3)$ and $f_2 = (v_1, v_3, v_4)$, adjacent via edge $(v_1, v_3)$. By default, AMT searches for adjacency using the last two vertices, i.e., $(v_2, v_3)$, failing to find $f_2$. A special token "\$" denotes a swap: the sequence $(v_1, v_2, v_3, \$, v_4)$ indicates that the next face is formed by the first and last vertex (rather than the last two).
    - **Design Motivation**: Reduces interruptions caused by edge mismatches, further compressing sequence length.

3. **Face Count Condition**:

    - **Function**: Allows users to specify the target face count of the generated mesh.
    - **Mechanism**: A face count embedding table (size = maximum face count) is initialized; the embedding corresponding to the target face count is retrieved and appended after the point cloud prefix. Random perturbation is applied during training to prevent overfitting to exact counts, and the condition is dropped with 10% probability to improve robustness.
    - **Design Motivation**: Different applications have varying face count requirements (low-poly for games, high-poly for film/VFX); prior methods provide no such control.

4. **Masking Invalid Predictions**:

    - **Function**: Constrains generation to produce only valid tokens at inference time.
    - **Mechanism**: Invalid logits are masked during inference—for example, "&" cannot immediately follow another "&" (a new strip requires at least three vertices), and vertices violating the coordinate sorting order are disallowed.
    - **Design Motivation**: Transformer outputs may violate the structural constraints of the sequence; explicit masking ensures that decoded outputs constitute valid meshes.

5. **AMT Positional Encoding**:

    - Distinct positional embeddings are assigned to different token types: each of the three positions in a new three-vertex face has a dedicated embedding, single-vertex adjacent faces have a separate embedding, and "&" tokens have their own embedding.
    - This enables the Transformer to distinguish the role of each token within the AMT sequence.

### Loss & Training
The Objaverse dataset is used, with the face count upper bound raised from 800 to 1600. Input point clouds are sampled at 8192 points (vs. 4096 in V1) to accommodate more complex meshes. The overall batch size is 256 (32 A800 GPUs × 8), with training lasting 4 days. Unlike V1, V2 updates the point cloud encoder weights during training to improve reconstruction fidelity.

## Key Experimental Results

### AMT Ablation (MeshAnything V2 vs. Variant without AMT)

| Method | CD↓ | ECD↓ | NC↑ | #V | #F | V_Ratio | F_Ratio | S_Ratio↓ |
|------|-----|------|-----|----|----|---------|---------|---------|
| V2 w/o AMT | 0.895 | 4.832 | 0.924 | 302.4 | 556.7 | 1.105 | 1.062 | 1.000 |
| **V2 (AMT)** | **0.874** | **4.721** | **0.933** | 308.6 | 571.8 | 1.127 | 1.097 | **0.497** |

AMT compresses the sequence length to 49.7% (approximately half) while reducing CD by 2.3% and improving NC. The variant without AMT requires nearly twice the GPU hours of V2.

### Tokenization Method Comparison (OPT-125M, ≤400 Faces)

| Method | CD↓ | ECD↓ | NC↑ | S_Ratio↓ | Perplexity↓ |
|------|-----|------|-----|---------|-------------|
| Baseline (3 vertices/face) | 2.478 | 18.21 | 0.893 | 1.000 | 1.150 |
| Unsort (no sorting) | 8.151 | 31.86 | 0.794 | 1.000 | 1.234 |
| PolyGen (AMT) | 3.226 | 22.97 | 0.872 | **0.372** | 1.589 |
| **AMT** | **2.348** | **19.33** | **0.904** | 0.492 | **1.363** |
| AMT (Swap) | 2.517 | 19.86 | 0.913 | 0.455 | 1.416 |

### Key Findings
- **Sorting is critical**: Unsort yields a CD 3.3× higher than Baseline (8.151 vs. 2.478), confirming that well-structured sequences are essential for learning.
- **PolyGen achieves the highest compression but the lowest quality**: S_Ratio = 0.372 is the shortest, yet perplexity is the highest (1.589), indicating that its sequence structure is ill-suited for autoregressive learning. The root cause is that generating a face requires referencing prior vertex indices, which increases sequence learning difficulty.
- **AMT achieves the best trade-off**: CD improves over Baseline (2.348 vs. 2.478) despite a 50% reduction in sequence length, and perplexity is also lower (1.363 vs. 1.150), demonstrating that AMT produces sequences that are simultaneously more compact and more learnable.
- **Swap provides marginal gains**: It further improves compression ratio (0.455 vs. 0.492) and yields the highest NC (0.913), though the additional special token slightly raises perplexity.
- **V2 doubles the maximum face count**: The upper limit increases from 800 to 1600, entirely attributable to AMT halving the sequence length.

## Highlights & Insights
- **Transferring NLP tokenization principles to 3D meshes** is the central insight: BPE shortens sequences by merging frequent subwords; AMT shortens face representations by exploiting topological adjacency. This cross-domain analogy is highly instructive.
- **The analysis of the compactness–regularity trade-off** is particularly valuable: PolyGen-style index-based methods yield shorter sequences but disrupt the predictability patterns of the sequence. The elegance of AMT lies in improving both dimensions simultaneously—shorter and more regular.
- **The "&" special token** provides a simple yet effective mechanism: it degrades gracefully when topological continuity breaks down, rather than distorting the sequence structure.
- **Masking Invalid Predictions** is a general technique for autoregressive generation of structured data, transferable to any generation task requiring syntactic constraints.

## Limitations & Future Work
- Only triangular meshes are handled; the paper mentions potential extension to polygonal meshes but provides no empirical validation.
- For topologically discontinuous meshes (e.g., those with many isolated patches), AMT may degrade due to excessive "&" interruptions.
- The 1600-face upper limit remains insufficient for high-fidelity industrial applications, which commonly require thousands to tens of thousands of faces.
- The representational capacity of the point cloud encoder may limit reconstruction quality for complex shapes.
- No comparison is made against recent diffusion-based mesh generation methods (e.g., PolyDiff).

## Related Work & Insights
- **vs. MeshAnything V1**: V2 doubles the face count upper bound and improves quality via AMT, with the sole architectural difference being the tokenization scheme. This demonstrates that improving the data representation alone—without changing the model architecture—can yield substantial gains.
- **vs. MeshGPT**: MeshGPT learns a mesh vocabulary via VQ-VAE; AMT directly discretizes coordinates. AMT and VQ-VAE are orthogonal and can be combined.
- **vs. PolyGen**: PolyGen generates vertices first and then faces via indexing, achieving higher compression but harder sequence learning. AMT opts for a design more amenable to learning.
- **Key Insight**: In autoregressive generation, the serialization scheme (tokenization) of data may be equally or more important than the model architecture itself.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The AMT concept is intuitive yet effective; Swap and Masking are valuable complementary contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic comparison across multiple tokenization methods, though direct comparison against other mesh generation methods is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Algorithm descriptions are clear, but inconsistent ablation settings (OPT-125M/400 faces vs. OPT-350M/1600 faces) limit comparability.
- **Value**: ⭐⭐⭐⭐⭐ AMT represents a foundational contribution to mesh generation; virtually all autoregressive mesh generation methods stand to benefit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VertexRegen: Mesh Generation with Continuous Level of Detail](vertexregen_mesh_generation_with_continuous_level_of_detail.md)
- [\[ICCV 2025\] Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation](nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning](deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)
- [\[CVPR 2026\] Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation](../../CVPR2026/3d_vision/mesh-pro_asynchronous_advantage-guided_ranking_preference_optimization_for_artis.md)

</div>

<!-- RELATED:END -->
