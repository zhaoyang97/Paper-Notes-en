---
title: >-
  [Paper Note] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction
description: >-
  [NeurIPS 2025][3D Vision][3D mesh generation] This paper proposes to formulate 3D mesh generation as a coarse-to-fine, next-level-of-detail prediction process. By reversing a generalized mesh simplification algorithm (GSlim), a progressive refinement sequence is obtained, which is then learned autoregressively via a Transformer. Generation begins from a single point and incrementally adds geometric and topological detail to produce a complete mesh.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D mesh generation
  - autoregressive model
  - level-of-detail
  - mesh simplification
  - simplicial complex
date: 2026-05-08
content_hash: 0549009337036364
---

# ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2509.20824](https://arxiv.org/abs/2509.20824)
**Code**: [Project Page](https://jblei.site/proj/armesh)
**Area**: 3D Vision
**Keywords**: 3D mesh generation, autoregressive model, level-of-detail, mesh simplification, simplicial complex
**Institution**: The Chinese University of Hong Kong (Shenzhen), The University of Hong Kong, Tencent Hunyuan

## TL;DR

This paper proposes to formulate 3D mesh generation as a coarse-to-fine, next-level-of-detail prediction process. By reversing a generalized mesh simplification algorithm (GSlim), a progressive refinement sequence is obtained, which is then learned autoregressively via a Transformer. Generation begins from a single point and incrementally adds geometric and topological detail to produce a complete mesh.

## Background & Motivation

**Triangle meshes are the default representation in graphics**: Widely used in games and film production, triangle meshes are explicit, compact, and compatible with modern graphics pipelines. Directly generating meshes — rather than going through indirect representations — is therefore of significant importance.

**Existing AR methods generate faces one by one in a geometrically meaningless order**: Methods such as MeshGPT assign faces a lexicographic or traversal order, serializing meshes into 1D sequences for face-by-face generation. Such artificial orderings fail to capture global shape during generation and do not align with the human coarse-to-fine perceptual process.

**"Next-scale prediction" has been validated in 2D**: Works such as VAR generate images progressively from low to high resolution, outperforming raster-scan order. However, due to the irregular structure of 3D meshes, the LOD concept cannot be directly transferred.

**Mesh simplification algorithms naturally provide fine-to-coarse sequences**: Algorithms such as QSlim iteratively collapse edges to simplify complex meshes into coarser ones, naturally forming LOD hierarchies. Reversing this process yields a coarse-to-fine generation sequence.

**QSlim has critical limitations**: It cannot handle non-manifold meshes, cannot alter topology, and cannot reduce a mesh to a single point — yet an ideal generative process should start from a single point.

**Intermediate results are uncontrollable**: Prior face-by-face methods do not support early stopping for meshes of varying resolution; users have no flexible control over the quality-speed trade-off.

## Method

### Overall Architecture

The overall pipeline consists of three stages:
1. **GSlim simplification**: The input mesh (generalized as a simplicial complex) is progressively collapsed to a single point via a generalized simplification algorithm, producing a fine-to-coarse sequence.
2. **PSC inversion**: The simplification sequence is reversed into a progressive refinement sequence (Progressive Simplicial Complex, PSC), where each step corresponds to a vertex split operation.
3. **AR learning and generation**: The refinement sequence is tokenized and learned autoregressively by a Transformer. At generation time, the model starts from a single point and iteratively predicts refinement operations to reconstruct a complete mesh.

### Key Design 1: GSlim Generalized Mesh Simplification

- **Simplicial Complex (SC)**: The mesh representation is generalized to include isolated points and edge segments (not necessarily belonging to triangles), enabling any mesh to be reduced to a single point.
- **Generalized quadric definition**: Quadric coefficients are defined separately for each simplex dimension (point $d=0$, edge $d=1$, triangle $d=2$) as $\mathbf{A} = \mathbf{I} - \sum_{i=1}^{d} \mathbf{e}_i \mathbf{e}_i^\top$, providing a unified measure of geometric error.
- **Topology-changing edge collapse**: "Virtual edges" are introduced to bridge disconnected components (obtained via Delaunay tetrahedralization), enabling edge collapses that alter topology and resolving the topology-invariance bottleneck of QSlim.
- **Penalty factors for preference control**: Separate penalty weights are set for vertices, boundary edges, and faces (default VEF = 0, 1, 1), controlling simplification priority through weighted quadrics.
- A mesh with $n$ vertices requires exactly $n-1$ simplification steps to reduce to a single vertex.

### Key Design 2: PSC Refinement Sequence and Topology Labels

- Each vertex split must record four types of information: ① the index of the vertex to be split; ② whether the split position is the original point or the midpoint (boolean); ③ the offset vector of the new vertex; ④ a list of topology labels for adjacent simplices.
- **Topology labels**: Nine cases (0–8) are assigned to each adjacent simplex (2 for points, 4 for edges, 3 for faces), describing how local connectivity changes after the split.
- **Four constraint rules**: The authors formally derive hard constraints among topology labels to ensure the validity of predicted combinations; the constraints primarily apply to edge labels.

### Key Design 3: Tokenization and Constrained Decoding

- **Compact token layout**: The token sequence for each vertex split operation consists of: 2 bytes for vertex index (int16, range 0–255) → 6 bytes for offset (3×fp16, range 256–511) → 1 byte for vertex topology (512–513) → face topology labels (514–516) → edge topology labels (517–520) → 1 byte for midpoint flag.
- **BPE compression**: Byte Pair Encoding is applied to compress tokens, with a vocabulary size of 16,384, reducing sequence length by 2–3×.
- **Constrained decoding**: A boolean function $\phi(x_i | x_1, \ldots, x_{i-1})$ is designed to verify token validity; tree traversal via depth-first search combined with random sampling ensures diversity while guaranteeing topological consistency of all predicted operations.

### Loss & Training

Standard autoregressive next-token prediction cross-entropy loss is used, with no special design. The authors emphasize that the core contributions lie in the representation and serialization scheme; the learning objective adopts the simplest possible approach.

## Key Experimental Results

### Table 3: Unconditional Generation on ShapeNet (Four Categories, Compared with MeshGPT et al.)

| Category | Method | COV↑ | MMD↓ | 1-NNA | FID↓ | KID↓ |
|----------|--------|------|------|-------|------|------|
| Chair | MeshGPT | 43.28 | 3.29 | 75.51 | 18.46 | 0.010 |
| Chair | **ARMesh** | **36.67** | **2.44** | **67.40** | **1.54** | **0.0001** |
| Table | MeshGPT | 45.68 | 2.36 | 72.88 | 6.24 | 0.002 |
| Table | **ARMesh** | 32.94 | **1.94** | **69.46** | **2.05** | **0.0002** |
| Bench | MeshGPT | 55.23 | 1.44 | 68.24 | 8.72 | 0.001 |
| Bench | **ARMesh** | **56.29** | **0.75** | **39.81** | **2.63** | **0.0001** |
| Lamp | MeshGPT | 53.88 | 3.94 | 65.73 | 19.91 | 0.004 |
| Lamp | **ARMesh** | **70.68** | **1.54** | **41.20** | **6.60** | **0.0005** |

### Table 4: Comparison of Tokenization Schemes at Different AR Step Ratios (Bench Category)

| Method | COV@10% | COV@50% | COV@100% | MMD@10% | MMD@100% | Token/shape |
|--------|---------|---------|----------|---------|----------|-------------|
| EdgeRunner | 9.43 | 40.33 | 54.90 | 10.32 | 0.81 | 5840 |
| BPT | 15.15 | 34.11 | 56.35 | 3.02 | 0.76 | 3235 |
| **ARMesh** | **41.07** | **54.84** | 56.29 | **1.67** | **0.75** | **2556** |

### Table 2: Generation Quality at Different LOD Ratios (Lamp Category)

- 10% steps → 23 vertices / 36 faces, FID = 49.91
- 50% steps → 109 vertices / 209 faces, FID = 10.44
- 100% steps → 216 vertices / 418 faces, FID = 6.60

### Key Findings

1. **ARMesh achieves substantially lower FID/KID**: For the Lamp category, FID drops from 19.91 (MeshGPT) to 6.60, and KID is reduced by an order of magnitude.
2. **Intermediate results far surpass those of other methods**: At 10% of AR steps, ARMesh achieves COV = 41.07 versus EdgeRunner's 9.43, as the coarse-to-fine strategy naturally produces reasonable coarse meshes at intermediate stages.
3. **Highest token efficiency**: ARMesh averages 2,556 tokens per shape, 56% fewer than EdgeRunner (5,840).
4. **Decent quality at 50% simplification**: With an average generation time of approximately one minute, early stopping at 50% halves the runtime while still yielding usable results.

## Highlights & Insights

1. **The core insight is remarkably elegant**: Extending the 2D "next-scale prediction" paradigm to 3D mesh "next-LOD prediction" via reversal of simplification algorithms represents a compelling bridge between classical computer graphics and modern generative models.
2. **Unified handling of arbitrary topology**: The simplicial complex representation enables uniform treatment of non-manifold, non-watertight, and mixed-dimensional meshes — cases that would cause QSlim to fail or crash.
3. **Flexible early stopping**: Users can halt generation at any step to obtain a mesh of the corresponding resolution, a capability that is fundamentally impossible with prior face-by-face methods — providing an intuitive quality-speed trade-off.
4. **Complete derivation of topological constraints**: The nine topology cases for vertex splits are analyzed to yield four complete constraint rules, supporting constrained decoding that guarantees output validity.
5. **Broad application potential**: The framework supports shape refinement (from coarse sketches to detailed meshes) and skeleton editing (extracting coarse skeletons to manipulate fine-grained shapes), offering high practical utility.

## Limitations & Future Work

1. **Limited generalization**: Compared to diffusion models operating in continuous space, AR methods still exhibit a gap in cross-domain generalization and may require orders-of-magnitude more training data to mitigate this.
2. **Excessive topological flexibility may introduce noise**: PSC permits arbitrary topological changes, which may in some cases produce unnecessary topological structures. The authors suggest using PSC to establish initial topology and then switching to progressive meshes to restrict subsequent topological changes.
3. **Linear generation complexity**: Each vertex split depends serially on prior results; parallelism is not exploited. Future work could adopt parallel vertex splits to reduce complexity from linear to logarithmic.
4. **Non-differentiable**: The PSC representation does not support gradient-based updates, precluding integration with differentiable rendering or end-to-end optimization pipelines.
5. **Data scale**: Current experiments are conducted primarily on ShapeNet subsets (< 1,700 vertices / < 800 faces); performance on larger-scale, high-fidelity meshes remains to be validated.

## Related Work & Insights

- **Indirect mesh generation**: Meshes are obtained via implicit fields (SDF / NeRF) followed by marching cubes post-processing; this introduces post-processing errors and offers limited control over mesh quality.
- **Direct mesh generation (face-by-face)**: PolyGen → MeshGPT → PivotMesh → EdgeRunner → BPT — the dominant paradigm serializes mesh faces for AR generation. ARMesh fundamentally differs by generating per LOD rather than per face.
- **LOD methods**: Classical QSlim / Progressive Meshes provide multi-scale representations; in deep learning, EdgeRunner supports only 3 discrete LODs without intrinsic modeling. The concurrent work VertexRegen is based on progressive meshes but requires the initial and target meshes to be homeomorphic; ARMesh imposes no such constraint.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of reversing classical mesh simplification into a generation sequence is highly original; the GSlim → PSC → AR pipeline is both complete and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-category ShapeNet comparisons and ablation studies (penalty factors / LOD / manifold) are thorough, though experiments on larger-scale datasets and conditional generation are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The progressive narrative from QSlim to GSlim to PSC is clear and rigorous, with well-motivated background and rich illustrations.
- **Value**: ⭐⭐⭐⭐ — Opens a new LOD-based mesh generation paradigm; the flexibility of early stopping carries strong practical engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] VertexRegen: Mesh Generation with Continuous Level of Detail](../../ICCV2025/3d_vision/vertexregen_mesh_generation_with_continuous_level_of_detail.md)
- [\[NeurIPS 2025\] LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering](lodge_level-of-detail_large-scale_gaussian_splatting_with_efficient_rendering.md)
- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[ICLR 2026\] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models](../../ICLR2026/3d_vision/quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](../../ICCV2025/3d_vision/ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)

<!-- RELATED:END -->
