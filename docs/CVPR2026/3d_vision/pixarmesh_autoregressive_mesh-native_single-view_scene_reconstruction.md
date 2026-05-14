---
title: >-
  [Paper Note] PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction
description: >-
  [CVPR 2026][3D Vision][Single-view scene reconstruction] This paper proposes PixARMesh, the first autoregressive framework for single-view scene reconstruction that operates natively in mesh space (rather than SDF space)…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Single-view scene reconstruction"
  - "autoregressive mesh generation"
  - "native mesh"
  - "artist-ready"
  - "compositional 3D"
date: 2026-05-08
content_hash: 94c51a91cc777cd1
---

# PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.05888](https://arxiv.org/abs/2603.05888)
**Code**: [Project Page](https://mlpc-ucsd.github.io/PixARMesh)
**Area**: 3D Vision
**Keywords**: Single-view scene reconstruction, autoregressive mesh generation, native mesh, artist-ready, compositional 3D

## TL;DR
This paper proposes PixARMesh, the first autoregressive framework for single-view scene reconstruction that operates natively in mesh space (rather than SDF space). By enhancing a point cloud encoder with pixel-aligned image features and global scene context, the method jointly predicts object poses and meshes within a unified token sequence. PixARMesh achieves scene-level state-of-the-art on 3D-FRONT while producing compact, editable, artist-ready meshes.

## Background & Motivation

**Background**: Single-view 3D scene reconstruction is a long-standing ill-posed problem. The compositional generation paradigm has attracted growing attention, driven by advances in large-scale object-level reconstruction models (TRELLIS, CLAY, etc.).

**Limitations of Prior Work**:
- Holistic methods (Panoptic3D, Uni-3D) are constrained by voxel resolution and the limited expressiveness of feed-forward decoders.
- Compositional methods (Gen3DSR, DeepPriorAssembly) require inpainting occluded regions before generation, then estimate layout via optimization—prone to local optima.
- MIDI avoids layout optimization but still generates in normalized scene coordinates using SDF.
- **All existing methods rely on SDF representations**, requiring Marching Cubes for surface extraction, which produces over-triangulated, overly smooth, high-polygon meshes unsuitable for editing.

**Key Challenge**: Mesh generation models (MeshGPT, EdgeRunner, BPT) are limited to single-object outputs; no prior work has extended them to scene-level reconstruction.

**Key Insight**: Leverage a pretrained object-level autoregressive mesh generator (EdgeRunner/BPT), augment its point cloud encoder to incorporate appearance and global context, and jointly predict poses and meshes within a unified token sequence.

**Core Idea**: Jointly predict object poses (tokenized as bounding box corners) and native meshes (tokenized as vertices/faces) within a single autoregressive sequence, eliminating SDF extraction and post-hoc layout optimization.

## Method

### Overall Architecture
Input RGB image → depth estimation + instance segmentation + image feature extraction (all using off-the-shelf models) → depth unprojection to obtain global and per-object point clouds → pixel-aligned point cloud encoder fusing geometry and appearance → scene context aggregation → Transformer decoder autoregressively generating [pose tokens | mesh tokens].

### Key Designs

1. **Pixel-Aligned Point Cloud Encoder**

   - **Function**: Fuse image appearance features into the point cloud encoder.
   - **Mechanism**: For each 3D point $p$ in the instance point cloud $P_i$, project it onto the image plane via camera intrinsics $(u,v) = \text{Proj}(K, p)$, extract the DINOv2 feature $\mathbf{f}_p^{\text{img}}$ at the corresponding pixel, and concatenate it with the geometric feature $\mathbf{f}_p^{\text{pc}}$ before feeding into a Transformer fusion block. Learnable query vectors aggregate the fused features into a compact latent code $\mathbf{z}_i$.
   - **Design Motivation**: The original EdgeRunner/BPT point cloud encoders process only coordinates, ignoring rich appearance cues from images. In single-view scenes where objects are heavily occluded, appearance features are essential for inferring complete geometry.

2. **Scene Context Aggregation**

   - **Function**: Inject global scene context into each object's representation.
   - **Mechanism**: All point clouds are first normalized in a unified scene coordinate system (rather than independently per object) to preserve spatial consistency. A global scene point cloud is encoded to produce $\mathbf{z}_{\text{scene}}$, and each object latent code aggregates scene information via cross-attention: $\mathbf{z}_i^{\text{agg}} = \text{CrossAttn}(q=\mathbf{z}_i, k=\mathbf{z}_{\text{scene}}, v=\mathbf{z}_{\text{scene}})$.
   - **Design Motivation**: Local point cloud information from individual objects is insufficient for inferring complete geometry and precise poses. Contextual cues from nearby similar objects provide complementary information, especially under heavy occlusion.

3. **Unified Pose–Mesh Tokenization**

   - **Function**: Encode both object poses and meshes into a unified token sequence using the same vocabulary.
   - **Mechanism**: Poses are represented as gravity-aligned 7-DoF bounding boxes, encoded as the 3D coordinates of 8 corner points. The mesh generator's coordinate vocabulary is reused (EdgeRunner: 3 tokens per point `<x><y><z>`, 24 tokens total; BPT: 2 tokens per point `<block_id><offset_id>`, 16 tokens total). At inference, the local-to-global affine transformation $\mathbf{T}^\star$ is recovered from the 8 corner points to map the canonical-space mesh back to scene coordinates.
   - **Final sequence format**: `<bos> [pose_seq] <sep> [mesh_seq] <eos>`
   - **Design Motivation**: Avoids introducing new vocabulary types, enabling full vocabulary sharing. Pose sequences add only 16–24 tokens, negligible compared to the mesh sequence.

### Loss & Training
- A single next-token prediction cross-entropy loss: $\mathcal{L}_{\text{ce}} = -\sum_t \log p_\theta(s_t | s_{<t}, \mathbf{z}_{\text{agg}})$
- Depth jitter (±0.02) is applied during training to simulate monocular depth inaccuracies.
- Trained on 8×H100 GPUs: approximately 2 days for EdgeRunner and 18 hours for BPT.

## Key Experimental Results

### Main Results (3D-FRONT Dataset)

| Method | Scene CD↓ (×10⁻³) | Scene CD-S↓ | Scene F-Score↑ | Object CD↓ | Object F-Score↑ |
|---|---|---|---|---|---|
| InstPIFu | 213.4 | 124.9 | 13.72% | 44.74 | 29.63% |
| MIDI | 156.3 | 79.3 | 24.83% | 6.71 | 72.69% |
| DepR | 153.2 | 56.4 | 25.00% | **2.57** | **89.66%** |
| **PixARMesh-ER** | **98.8** | **49.1** | **33.55%** | 4.04 | 82.27% |
| **PixARMesh-BPT** | **98.4** | **47.6** | 32.26% | 4.57 | 80.30% |

### Ablation Study (Necessity of Joint Pose–Mesh Modeling)

| Configuration | Scene CD↓ | Scene F-Score↑ | Object CD↓ | Object F-Score↑ |
|---|---|---|---|---|
| EdgeRunner-FT (no layout) | 119.8 | 27.81% | 4.75 | 80.57% |
| Two-stage (separate models) | 99.8 | 33.32% | 4.75 | 80.85% |
| **PixARMesh (joint)** | **98.8** | **33.55%** | **4.04** | **82.27%** |

### Ablation Study (Module Contributions, Using GT Inputs)

| Image Features | Scene Context | Scene CD↓ | Scene F-Score↑ | Object CD↓ |
|---|---|---|---|---|
| ✗ | ✗ | 57.78 | 41.02% | 5.29 |
| ✓ | ✗ | 55.44 | 42.84% | 5.56 |
| ✗ | ✓ | 39.30 | 44.67% | 3.64 |
| ✓ | ✓ | 39.88 | 46.15% | 4.04 |

### Key Findings
- PixARMesh achieves scene-level SOTA across all metrics: scene CD drops from 153.2 (DepR) to 98.4 (−36%), and F-Score improves from 25% to 33.6%.
- DepR retains an advantage at the object level (CD 2.57 vs. 4.04), as diffusion-based SDF generation yields higher geometric fidelity. However, PixARMesh outputs compact artist-ready meshes (only a few thousand faces per object), whereas SDF-based methods produce densely triangulated high-polygon meshes.
- **Scene context aggregation is the most critical module**: adding it reduces scene CD from 57.78 to 39.30, contributing far more than image features alone.
- Joint modeling outperforms the two-stage baseline: object CD improves from 4.75 to 4.04, demonstrating that geometry generation benefits from pose inference.
- The EdgeRunner variant outperforms the BPT variant due to higher quantization resolution preserving more geometric detail.

## Highlights & Insights
- **First extension of autoregressive mesh generation to the scene level**: breaks the assumption that mesh generation models are limited to single objects. A clean token sequence design enables unified decoding of poses and meshes without post-hoc layout optimization.
- **Vocabulary-sharing pose tokenization is elegantly designed**: bounding box corner coordinates reuse the mesh vocabulary with zero additional token overhead, adding only 16–24 tokens. Layout is recovered at inference via least-squares affine fitting.
- **Emergent benefit of joint modeling**: pose prediction and mesh generation mutually reinforce each other—geometric information aids localization, and pose context aids geometry completion. This synergy is unattainable in two-stage pipelines.
- **Output meshes are directly usable in graphics applications** (editing, rendering, simulation), whereas Marching Cubes outputs from SDF-based methods require extensive post-processing.

## Limitations & Future Work
- Object-level geometric accuracy falls short of diffusion-based SDF methods such as DepR; autoregressive mesh generation has an inherent disadvantage on fine surface details.
- Currently trained only on 3D-FRONT indoor furniture scenes with a limited object category set.
- Relies on Grounded-SAM for segmentation and Depth Pro for depth estimation; errors in upstream models cascade through the pipeline.
- Autoregressive decoding slows down as the number of objects increases, since sequence length grows linearly.

## Related Work & Insights
- **vs. DepR**: DepR uses depth-guided diffusion to generate in SDF space, achieving finer object geometry (CD 2.57 vs. 4.04), but inferior scene layout (scene CD 153.2 vs. 98.8) and requiring Marching Cubes post-processing.
- **vs. MIDI**: MIDI directly generates SDF in normalized scene space, avoiding layout optimization, but still requires surface extraction and achieves lower scene-level accuracy than PixARMesh.
- **vs. original EdgeRunner / BPT**: these models support only single-object generation; PixARMesh extends them to the scene level by injecting pixel-aligned features and scene context.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First mesh-native scene reconstruction; the unified tokenization design for poses and meshes is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers both synthetic and real data with thorough ablations, but lacks comparison against more mesh generation baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear writing with a complete and coherent logical chain from motivation to design to experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes a new paradigm for mesh-native scene reconstruction with significant implications for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models](../../ICLR2026/3d_vision/quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](coherent_humanscene_reconstruction_from_multiperso.md)
- [\[CVPR 2026\] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)
- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)

</div>

<!-- RELATED:END -->
