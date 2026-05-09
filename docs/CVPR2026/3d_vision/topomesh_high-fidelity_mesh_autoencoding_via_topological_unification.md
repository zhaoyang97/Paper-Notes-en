---
title: >-
  [Paper Note] TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification
description: >-
  [CVPR 2026][3D Vision][3D VAE] This paper proposes TopoMesh, which unifies both ground-truth and predicted meshes under the Dual Marching Cubes (DMC) topology framework, enabling explicit vertex- and face-level correspondence for the first time. This allows direct mesh-level supervision over topology, vertex positions, and face normals. The proposed method improves F1-Sharp by 5.9–7.1% over the current state of the art, with particularly notable advantages in sharp feature preservation.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D VAE
  - Mesh Autoencoding
  - Topological Unification
  - Dual Marching Cubes
  - Sharp Feature Preservation
date: 2026-05-08
content_hash: b4afab44540d8572
---

# TopoMesh: High-Fidelity Mesh Autoencoding via Topological Unification

**Conference**: CVPR 2026
**arXiv**: [2603.24278](https://arxiv.org/abs/2603.24278)
**Code**: [Project Page](https://logan0601.github.io/projects/topomesh/index.html)
**Area**: 3D Vision / 3D Generation
**Keywords**: 3D VAE, Mesh Autoencoding, Topological Unification, Dual Marching Cubes, Sharp Feature Preservation

## TL;DR

This paper proposes TopoMesh, which unifies both ground-truth and predicted meshes under the Dual Marching Cubes (DMC) topology framework, enabling explicit vertex- and face-level correspondence for the first time. This allows direct mesh-level supervision over topology, vertex positions, and face normals. The proposed method improves F1-Sharp by 5.9–7.1% over the current state of the art, with particularly notable advantages in sharp feature preservation.

## Background & Motivation

1. **Background**: The dominant paradigm for 3D generation is the VAE-Diffusion pipeline, where the reconstruction capacity of the VAE constitutes a hard upper bound on generation quality. Existing 3D VAEs encode meshes of arbitrary topology into a regular latent representation and reconstruct them via decoding.

2. **Limitations of Prior Work**: The core bottleneck is a **representation mismatch** — ground-truth meshes have arbitrary, variable topology (irregular connectivity, varying vertex counts), whereas VAEs typically predict fixed structures (e.g., SDFs on regular grids or rendered images), making it impossible to establish explicit mesh-level correspondences. This leads to two categories of indirect supervision, each with its own drawbacks:
    - **SDF supervision** (e.g., 3DShape2VecSet, TripoSG, Direct3D-S2): Meshes are extracted via Marching Cubes, but MC constrains vertices to lie on grid edges via linear interpolation, making it inherently incapable of representing sharp edges and corners.
    - **Rendering supervision** (e.g., Trellis, SparseFlex): More expressive decoding via FlexiCubes, but supervision is ambiguous — limited resolution, occlusion, and sparse viewpoints cause loss of detail gradients.

3. **Key Challenge**: Achieving high-fidelity reconstruction — particularly for sharp features — requires precise vertex/face correspondence between predicted and ground-truth meshes. Differing topological structures make this infeasible.

4. **Goal**: Design a VAE that is both capable of expressing sharp features and structurally aligned to enable precise correspondence for direct, unambiguous mesh-level supervision.

5. **Key Insight**: Unify both ends (ground truth and prediction) under the same DMC topology framework — ground-truth meshes are remeshed into DMC format, and the decoder directly outputs DMC-format meshes.

6. **Core Idea**: By unifying topology, predicted and ground-truth meshes share the same DMC structure, enabling direct vertex- and face-level supervision for the first time.

## Method

### Overall Architecture

TopoMesh comprises two core modules: **Topo-Remesh** (converting arbitrary input meshes into DMC-compatible representations while preserving sharp features) and **Topo-VAE** (a sparse voxel encoder combined with a decoupled FlexiCubes decoder that reconstructs meshes in the unified DMC format). The pipeline is as follows: vertex positions and normals of the input mesh are encoded into a compact latent representation via sparse voxel–point cross-attention, and the decoder outputs a mesh in the same DMC format. The topology-unified correspondence enables direct supervision over topology, vertex positions, and face normals.

### Key Designs

1. **Sparse Voxel–Point Cross-Attention Encoder**:

    - **Function**: Efficiently encodes million-scale point clouds into sparse voxel features.
    - **Mechanism**: The key observation is that each point resides in exactly one voxel, allowing global attention to be replaced by sparse local attention — each point interacts only with its containing voxel. This reduces the attention map from $O(N \times P)$ to $O(P)$ (from 74 GB to 3.8 MB). Point coordinates within each voxel are further normalized to local coordinates, enabling all voxels to share a single learnable query token: $O_i = \sum_{j=1}^{n_i} \text{Softmax}_i(QK_j^T / \sqrt{d}) \cdot V_j$.
    - **Design Motivation**: Global attention over million-scale 3D points is computationally infeasible. Sparsification exploits the spatial affiliation between points and voxels, while shared queries further reduce parameter count. The design is adaptive to multiple resolutions and supports progressive training.

2. **Topology–Geometry Decoupled Decoder**:

    - **Function**: Separates topology prediction from geometry refinement to avoid training instability caused by competing gradients.
    - **Mechanism**: Building on FlexiCubes, the SDF $s$ is decoupled into occupancy $o$ (sign, determining topology) and magnitude $u$ (determining geometry). Parameters are grouped into topology set $\text{Topo}=\{o, \gamma\}$ and geometry set $\text{Geom}=\{u, \alpha, \beta, \delta\}$. Faces are determined solely by topology parameters $F_o = \text{DMC}(o)$, while vertices depend on all parameters $V_o = \text{FlexiCubes}(o \times u, \alpha, \beta, \delta, \gamma)$.
    - **Design Motivation**: In standard FlexiCubes, the sign and magnitude of the SDF are coupled — topology changes can abruptly activate large geometry losses, whose gradients in turn destabilize topology prediction. Decoupling enables independent supervision and breaks this vicious cycle.

3. **Topo-Remesh and $L_\infty$ Distance Metric**:

    - **Function**: Converts arbitrary input meshes into DMC-compatible representations while preserving sharp features.
    - **Mechanism**: Conventional methods use $L_2$ distance for surface dilation, but $L_2$ is a point-to-point metric that rounds sharp corners. This work introduces the $L_\infty$ distance: $D_\infty(P,Q) = \max_{T_i \in \mathcal{T}(Q)} d(P, \Pi_i)$, i.e., the maximum planar distance from point $P$ to the planes of all triangles incident to its nearest surface point $Q$. During dilation, equidistant offsets along incident planes form a polyhedral envelope, and points on its boundary naturally preserve sharp angles.
    - **Design Motivation**: $L_2$ dilation produces rounded arcs at sharp corners, and post-processing (projection, rendering optimization) may introduce self-intersections and noise. $L_\infty$ mathematically guarantees angle preservation without post-processing. The entire pipeline is fully GPU-accelerated, requiring approximately 15 seconds at $1024^3$ resolution.

4. **Teacher Forcing Training Strategy**:

    - **Function**: Breaks the training instability caused by competing topology–geometry gradients.
    - **Mechanism**: During training, the ground-truth topology $o_{gt}$ is provided to the decoder (rather than the predicted topology), allowing geometry parameters to receive stable gradients under the correct topological configuration from the first iteration. At inference, the decoder independently predicts both. This is combined with GT-guided voxel pruning (to avoid holes caused by pruning based on early erroneous predictions) and progressive resolution training ($32^3 \to 64^3 \to 128^3$) to accelerate convergence.
    - **Design Motivation**: Preliminary experiments revealed severe instability — when topology is correct, its loss vanishes but a large geometry loss is suddenly activated, and the resulting gradients frequently flip topology back to incorrect states. Teacher Forcing decouples this conditional dependency.

### Loss & Training

The total loss comprises: topology loss (BCE on occupancy), vertex loss (L1 on vertex positions), normal loss (L1 on face normals — DMC quads are triangulated by the FlexiCubes $\gamma$ parameter, with four sub-triangles during training and two at inference, with GT triangles duplicated for supervision), FlexiCubes regularization, consistency loss, voxel pruning loss, and KL divergence loss. Training uses AdamW with lr=0.0001, a dataset of 320K samples, batch size 64, progressive resolution $32^3 \to 64^3 \to 128^3$ over 700K steps.

## Key Experimental Results

### Remesh Quality

| Method | Device | Thingi10K CD↓ | Thingi10K ANC↑ | Objaverse ANC↑ | Time↓ |
|--------|--------|--------------|----------------|----------------|-------|
| ManifoldPlus | CPU | **1.347** | 0.981 | 0.780 | 79.4s |
| Dora | GPU | 1.492 | 0.970 | 0.961 | 116.3s |
| **TopoMesh** | **GPU** | 1.479 | **0.984** | **0.964** | **18.5s** |

Topo-Remesh achieves the highest normal consistency (ANC) while being 4–9× faster than competing methods.

### VAE Autoencoding Reconstruction

| Method | #Latent | Topo-Bench F1-S↑ | Dora-Bench F1-S↑ | Dora-Bench CD↓ |
|--------|---------|------------------|------------------|----------------|
| TripoSG | 4096 | 0.715 | 0.717 | 1.697 |
| Dora | 4096 | 0.754 | 0.768 | 1.814 |
| SparseFlex | 244691 | 0.873 | 0.844 | 1.625 |
| **TopoMesh** | **56006** | **0.932** | **0.915** | **1.126** |

F1-Sharp (sharp feature preservation metric): TopoMesh improves over SparseFlex by 5.9% on Topo-Bench (0.873→0.932) and 7.1% on Dora-Bench (0.844→0.915), using only 1/4 of the latent tokens.

### Ablation Study

| Configuration | CD↓ | F1↑ | F1-S↑ | ANC↑ |
|---------------|-----|-----|-------|------|
| Rendering supervision (replacing mesh-level) | 1.731 | 0.776 | 0.711 | 0.932 |
| **Mesh-level supervision** | **0.150** | **0.975** | **0.991** | **0.999** |
| Resolution 32 | 1.812 | 0.933 | 0.790 | 0.968 |
| Resolution 128 | 1.126 | 0.973 | 0.915 | 0.995 |

Mesh-level vs. rendering supervision: In a single-shape overfitting experiment, mesh-level supervision achieves 1/11.5 the CD of rendering supervision, with F1-S rising from 0.711 to 0.991, demonstrating the decisive advantage of direct supervision.

### 3D Generation

| Method | Toys4K FID↓ | Toys4K KID (×10³)↓ |
|--------|-------------|---------------------|
| Hunyuan3D-2.1 | 59.43 | 5.97 |
| Trellis | 59.61 | 6.03 |
| Direct3D-S2 | 45.33 | 5.47 |
| **TopoMesh** | **42.48** | **4.63** |

### Key Findings

- Topological unification is the central breakthrough: the DMC framework enables ground-truth and predicted meshes to share an identical topological structure, achieving precise per-vertex and per-face correspondence for the first time.
- The $L_\infty$ distance metric is critical for sharp feature preservation: dihedral angle distribution visualizations show that $L_\infty$ faithfully preserves the sharp angle distribution of the original mesh, whereas $L_2$ collapses sharp angles toward near-planar configurations.
- Teacher Forcing effectively resolves topology–geometry training instability: despite the train–inference gap, the impact on reconstruction quality is negligible.
- Topo-VAE achieves superior reconstruction quality using only 56K latent tokens (1/4 of SparseFlex), attributed to efficient gradient propagation enabled by direct supervision.

## Highlights & Insights

- **The fundamental innovation of "topological unification"**: Rather than seeking better indirect supervision within a framework that tolerates representation mismatch, this work eliminates the mismatch at the root — making both ends speak "the same language." This abstract-level insight is transferable to any problem requiring structured output prediction where the prediction format and GT format are misaligned.
- **Angle-preserving property of $L_\infty$ distance**: Replacing point-to-point distance with max-over-incident-planes elegantly resolves angle degradation during dilation in a mathematically principled way, and is naturally robust to topological defects and incorrect normals.
- **Synergy of decoupling and Teacher Forcing**: Splitting FlexiCubes' coupled parameters into topology and geometry groups, then applying Teacher Forcing to break the conditional dependency — two independent techniques that produce a synergistic effect in combination.
- **DMC format compression scheme**: A single voxel's complete mesh information can be stored using only 3×10-bit coordinates + 8-bit occupancy + 3×10-bit offsets + 3-bit triangulation decisions, averaging only 28.7 MB per mesh at $1024^3$ resolution — two orders of magnitude faster than Draco codec.

## Limitations & Future Work

- Upsampling to high resolutions generates hundreds of millions of voxels, incurring significant computational and time overhead.
- The remeshing algorithm is constrained by the base resolution; extremely fine details smaller than the voxel size cannot be captured.
- Progressive training ($32^3 \to 128^3$) requires 700K total steps, leaving room for improved training efficiency.
- Teacher Forcing introduces a train–inference gap; while negligible in current experiments, this may become apparent in more extreme scenarios.
- Adaptive resolution strategies — using higher resolution in regions dense with sharp features — are a promising direction for future exploration.

## Related Work & Insights

- **vs. SparseFlex**: SparseFlex uses FlexiCubes with rendering supervision, achieving F1-Sharp of only 0.873. TopoMesh also builds on FlexiCubes but switches to direct mesh-level supervision, reaching F1-Sharp of 0.932 with only 1/4 the number of latent tokens.
- **vs. Trellis**: Trellis is also a sparse voxel VAE but is limited to $256^3$ resolution with rendering supervision; its reconstruction quality (F1 0.583) falls far short of TopoMesh (0.917).
- **vs. TripoSG/Dora**: VecSet-based VAEs represent shapes using global vector sets, making fine-grained geometry modeling difficult. TopoMesh's sparse voxel design is naturally suited for high-resolution local detail.
- **Insight**: The concept of "unifying prediction format with GT format" in 3D generation is generalizable to other 3D representations including point clouds and implicit fields.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The topological unification paradigm fundamentally resolves the core bottleneck of 3D VAEs; both the $L_\infty$ distance metric and the decoupled decoder are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of remeshing, autoencoding, generation, and ablation; introduces new metrics (F1-Sharp) and benchmarks (Topo-Bench).
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is precise (representation mismatch), methodological derivation is logically coherent, and coverage from mathematical principles to engineering implementation is complete.
- Value: ⭐⭐⭐⭐⭐ — As an infrastructure-level improvement to 3D VAEs, it directly raises the upper bound on downstream 3D generation quality, offering exceptionally high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2026\] CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization](customtex_high-fidelity_indoor_scene_texturing_via_multi-reference_customization.md)
- [\[CVPR 2026\] Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation](catalyst4d_high-fidelity_3d-to-4d_scene_editing_via_dynamic_propagation.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)

<!-- RELATED:END -->
