---
title: >-
  [Paper Note] Scaling Mesh Generation via Compressive Tokenization
description: >-
  [CVPR 2025][3D Vision][Mesh Generation] This paper proposes Blocked and Patchified Tokenization (BPT), an efficient representation method that compresses triangular mesh sequences by approximately 75%. This enables autoregressive Transformers to process high-fidelity meshes with over 8k faces for the first time, achieving production-grade quality in point cloud/image-conditioned generation and validating a positive correlation scaling law between the number of mesh faces and…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Mesh Generation"
  - "Compressive Representation"
  - "Autoregressive Transformer"
  - "High-polygon Mesh"
  - "Block Indexing"
date: 2026-05-08
content_hash: 8937a44b3364fb2c
---

# Scaling Mesh Generation via Compressive Tokenization

**Conference**: CVPR 2025  
**arXiv**: [2411.07025](https://arxiv.org/abs/2411.07025)  
**Code**: [https://whaohan.github.io/bpt](https://whaohan.github.io/bpt)  
**Area**: 3D Vision  
**Keywords**: Mesh Generation, Compressive Representation, Autoregressive Transformer, High-polygon Mesh, Block Indexing

## TL;DR

This paper proposes Blocked and Patchified Tokenization (BPT), an efficient representation method that compresses triangular mesh sequences by approximately 75%. This enables autoregressive Transformers to process high-fidelity meshes with over 8k faces for the first time, achieving production-grade quality in point cloud/image-conditioned generation and validating a positive correlation scaling law between the number of mesh faces and generation performance.

## Background & Motivation

**Background**: Native mesh generation is a core requirement for 3D content creation. Compared to neural representations like NeRF or 3D Gaussian Splatting, meshes possess explicit topological structures and can be directly used in games, movies, and simulations. Recently, works such as MeshGPT and MeshXL have directly generated vertex and face sequences using autoregressive Transformers, maintaining artist-level topology quality.

**Limitations of Prior Work**: Existing methods are limited by excessively long mesh sequences. A triangular face consists of 3 vertices, with each vertex having 3 coordinates, resulting in 9 tokens per face. MeshAnything can only handle meshes with up to 800 faces, and MeshAnythingV2 extends this to 1600 faces. These low-polygon meshes severely lack detail and fail to meet production-level requirements. Existing compression methods (e.g., AMT, EdgeRunner) achieve insufficient compression rates, with the best only reaching around 47%.

**Key Challenge**: Transformers have limited context windows (typically 4k-9.6k tokens), whereas high-polygon meshes (>4k faces) result in sequence lengths that far exceed these windows under current representations. Drastic compression is needed to expand the range of trainable mesh faces, thereby allowing the use of richer training data.

**Goal**: Design a mesh tokenization method with a 75% compression rate, enabling models to be trained on high-quality meshes with over 8k faces, significantly improving generation performance and robustness.

**Key Insight**: The authors compress the sequence from two orthogonal levels: at the vertex level, substituting Cartesian coordinates with block indexing to compress 3 tokens down to 1-2 tokens; at the face level, using patch aggregation to eliminate redundancies from shared vertices. Overlapping both achieves an approximate 75% compression.

**Core Idea**: Convert vertices from coordinate representations of $(x,y,z)$ into a binary index representation of $(block\_id, offset\_id)$ and aggregate adjacent faces into patches centered around high-degree vertices, simultaneously compressing sequence lengths at both the vertex and face levels.

## Method

### Overall Architecture

BPT converts a triangular mesh $\mathcal{M}$ into a compressed 1D token sequence to be modeled by a standard autoregressive Transformer. The input mesh is first sorted along the z-y-x axes, and then: (1) the 3D coordinates of each vertex are converted into block-wise indices, where consecutive vertices in the same block share the block index, achieving about 50% compression; (2) vertices with the highest number of unvisited faces are identified as patch centers, and all faces connected to them are aggregated into a patch, eliminating redundant occurrences of the center vertex and yielding an additional ~50% compression. The final compression rate is approximately 75%. During generation, the Transformer is conditioned on point clouds or images, injecting conditional information via cross-attention to generate BPT sequences autoregressively.

### Key Designs

1. **Block-wise Indexing**:

    - **Function**: Compresses 3D coordinates $(x,y,z)$ (requiring 3 tokens) into $(b_i, o_i)$ (at most 2 tokens).
    - **Mechanism**: Divides the quantized space into $B$ blocks along each axis, with each block containing $O$ steps. The block index is defined as $b_i = (x_i \mid O) \cdot B^2 + (y_i \mid O) \cdot B + z_i \mid O$, which marks the block containing the vertex. The offset index is defined as $o_i = (x_i \% O) \cdot O^2 + (y_i \% O) \cdot O + z_i \% O$, marking the offset within the block. Since vertices are sorted along z-y-x, adjacent vertices are highly likely to reside in the same block. Thus, consecutive vertices within the same block can share a single block index, further compressing the sequence. The vocabulary size is $B^3 + O^3$ (e.g., 512 + 4096 = 4608 when $B=8, O=16$), which is much smaller than the $128^3 \approx 200$ ten-thousands (2 million) of naive indexing.
    - **Design Motivation**: Naively mapping $(x,y,z)$ to a single index leads to an prohibitive vocabulary size of $r^3$. The block-wise approach splits the exponential vocabulary into two polynomial-sized sub-vocabularies, while leveraging spatial locality after sorting to eliminate redundant block indices.

2. **Patchified Aggregation**:

    - **Function**: Aggregates adjacent faces sharing vertices into patches to eliminate duplicate vertex occurrences.
    - **Mechanism**: Analogous to the patch concept in image generation. The algorithm proceeds as follows: find the first unvisited face, select the vertex connected to the most unvisited faces as the patch center $v_c$, and aggregate all faces connected to $v_c$ into a patch $P_c = (v_c, v_1, v_2, ..., v_n)$. Here, $v_c$ only needs to appear once rather than being repeated for every face (originally, $v_c$ would appear about 6 times on average). The visited faces are marked, and the process repeats. A dual-block vocabulary is used—distinct vocabularies are assigned to block indices of patch centers versus normal vertices. This implicitly marks boundaries and the start of a patch through vocabulary types, requiring no extra special tokens.
    - **Design Motivation**: In the original representation, each vertex appears as many times as its degree (average of 6 times), causing substantial redundancy. Patch aggregation reduces the occurrence of the center vertex from ~6 times to 1, and other vertices are also reduced from face counts to patch counts. This not only compresses the sequence but also enhances spatial locality, as vertices in the same patch are spatially adjacent, reducing the Transformer's reliance on capturing long-range dependencies.

3. **Conditional Mesh Generation Architecture**:

    - **Function**: Supports mesh generation conditioned on both point clouds and images.
    - **Mechanism**: A standard 24-layer autoregressive Transformer with a hidden size of 1024 is configured, with condition embedding injected via cross-attention. For point cloud conditioning: a Michelangelo-style pre-trained encoder extracts point cloud features, with 4096 points randomly sampled during training. For image conditioning: image features are first extracted using DINO, then a DiT-based diffusion model generates conditional point cloud features (bridging the image and point cloud spaces), which are subsequently fed into the point cloud conditional model. Training is split into two stages: pre-training on 1.5M large-scale data, followed by fine-tuning on 0.3M high-quality data.
    - **Design Motivation**: Point clouds are the most natural paired modality for meshes, as both are geometric representations. Image conditioning bridged by a diffusion model avoids the vast gap when mapping directly from 2D to 3D. Two-stage training balances generalization capability and quality.

### Loss & Training

The standard autoregressive cross-entropy loss is adopted: $L(\theta) = \prod_i p(p_i | p_{1:i-1}, c; \theta)$. The model is trained using the AdamW optimizer ($\beta_1=0.9, \beta_2=0.99$) with a learning rate of $10^{-4}$ on 4 machines with 8×L40 GPUs for approximately 7 days. The sampling temperature is set to 0.7, and the context window is 9600 tokens. Flash attention and bf16 mixed precision are employed for acceleration.

## Key Experimental Results

### Main Results

**Point Cloud-conditioned Generation**:

| Method | Hausdorff Distance↓ | Chamfer Distance↓ |
|------|---------------------|-------------------|
| MeshAnything | 0.301 | 0.136 |
| MeshAnythingV2 | 0.265 | 0.114 |
| **BPT (Ours)** | **0.166** | **0.094** |

### Ablation Study

**Block/Offset Size Selection** ($|B| \cdot |O| = 128$):

| (|B|, |O|) | Hausdorff↓ | Chamfer↓ |
|------------|-----------|---------|
| (4, 32) | 0.209 | 0.111 |
| **(8, 16)** | **0.166** | **0.094** |
| (16, 8) | 0.256 | 0.126 |

**Effect of Scaling Face Count**:

| Max Face Count | Hausdorff↓ | Trend |
|---------|-----------|------|
| 1600 faces | ~0.30 | Baseline |
| 3200 faces | ~0.23 | Significant improvement |
| 4800 faces | ~0.19 | Continued improvement |
| 8000 faces | ~0.166 | Optimal |

**Compression Rate Comparison**:

| Method | Compression Rate↓ |
|------|--------|
| MeshXL / MeshAnything | 1.00 |
| MeshGPT / PivotMesh | 0.67 |
| MeshAnythingV2 / EdgeRunner | 0.46-0.47 |
| **BPT** | **0.26** |

### Key Findings

- BPT achieves a SOTA compression rate of 26% (i.e., a 74% reduction), further compressing by about 44% compared to the second best method.
- Scaling the face count from 1600 to 8000 yields persistent performance improvements (Hausdorff distance drops from ~0.30 to 0.166), validating the scaling law that "more faces and richer data lead to better generation."
- Engineering tricks like truncated training + sliding window inference are unable to substitute actual long-sequence training—truncation compromises generation integrity and robustness.
- Meshes generated by BPT achieve the best performance across all context lengths on the AVD (Average Vertex Distance) metric, proving its superior spatial locality.

## Highlights & Insights

- The design philosophy of BPT is to "simultaneously compress across two orthogonal dimensions"—vertex level (coordinates to indices) and face level (faces to patches), each contributing independently around 50% compression rate and stacking effectively.
- The dual-block vocabulary design cleverly encodes patch boundaries implicitly—distinguishing the start of a patch without extra special tokens, incurring zero extra overhead.
- It validates the "scaling law" in the mesh generation domain: face count (data complexity) rather than model size holds the key to current performance bottlenecks. Once representation limitations are lifted, existing models exhibit marked improvements.
- The image $\rightarrow$ point cloud features $\rightarrow$ mesh two-stage bridging strategy successfully resolves the massive 2D-3D modal gap.

## Limitations & Future Work

- The current model only has 500M parameters; the authors believe that scaling up the model size could further improve performance.
- The fixed 7-bit quantization resolution (128 levels) limits the expressiveness of extremely fine geometric details.
- It is only validated on triangular meshes; its generalizability to quad or hybrid meshes remains unknown.
- Future directions: (1) scaling experiments with larger models and more data; (2) exploring other sequence modeling architectures (e.g., Mamba) to better leverage the inductive bias of meshes; (3) supporting joint generation of textures and materials.

## Related Work & Insights

- Relationship with MeshGPT: MeshGPT utilizes self-encoders to map faces to latent space tokens, whereas BPT directly compresses in the raw coordinate space, making it cleaner and lossless.
- Comparison with MeshAnythingV2 (AMT): AMT achieves a compression rate of approximately 46%, while BPT pushes it further to 26%. Crucially, BPT maintains stronger spatial locality, resulting in structurally more complete meshes.
- Insight: The block-wise indexing philosophy can be extended to other scenarios requiring representation of high-dimensional discrete spaces within limited vocabularies (e.g., voxel generation, molecular structure generation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of block-wise indexing and patch aggregation is elegant and effective. Though these are not entirely new concepts, they are systematically applied to the mesh domain for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Thoroughly validated through compression rate comparisons, scaling experiments, truncation comparisons, block size ablation, and point cloud/image-conditioned generation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with abundant and intuitive figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Successfully addresses the primary bottleneck in mesh generation, bringing production-grade mesh generation closer to reality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](../../ICCV2025/3d_vision/meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[CVPR 2025\] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing](treemeshgpt_artistic_mesh_generation_with_autoregressive_tree_sequencing.md)
- [\[ICML 2025\] FreeMesh: Boosting Mesh Generation with Coordinates Merging](../../ICML2025/3d_vision/freemesh_boosting_mesh_generation_with_coordinates_merging.md)
- [\[CVPR 2025\] DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh](dagsm_disentangled_avatar_generation_with_gs-enhanced_mesh.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)

</div>

<!-- RELATED:END -->
