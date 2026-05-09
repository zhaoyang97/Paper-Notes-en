---
title: >-
  [Paper Note] Global-Aware Monocular Semantic Scene Completion with State Space Models
description: >-
  [ICCV 2025][3D Vision][Semantic Scene Completion] This paper proposes GA-MonoSSC, a hybrid architecture combining Transformer (2D global context) and Mamba (3D long-range dependencies) for indoor monocular semantic scene completion. A novel Frustum Mamba Layer is introduced to address feature discontinuities in voxel serialization, achieving state-of-the-art performance on Occ-ScanNet and NYUv2.
tags:
  - ICCV 2025
  - 3D Vision
  - Semantic Scene Completion
  - Indoor Scene
  - State Space Models
  - Mamba
  - Monocular
  - Transformer
date: 2026-05-08
content_hash: 0dfbf4c1f0b4deea
---

# Global-Aware Monocular Semantic Scene Completion with State Space Models

**Conference**: ICCV 2025
**arXiv**: [2503.06569](https://arxiv.org/abs/2503.06569)
**Code**: Coming soon
**Area**: 3D Vision
**Keywords**: Semantic Scene Completion, Indoor Scene, State Space Models, Mamba, Monocular, Transformer

## TL;DR

This paper proposes GA-MonoSSC, a hybrid architecture combining Transformer (2D global context) and Mamba (3D long-range dependencies) for indoor monocular semantic scene completion. A novel Frustum Mamba Layer is introduced to address feature discontinuities in voxel serialization, achieving state-of-the-art performance on Occ-ScanNet and NYUv2.

## Background & Motivation

Semantic Scene Completion (SSC) reconstructs a complete 3D scene with semantic labels from input observations, which is critical for robot navigation, AR, and autonomous driving. **Indoor monocular SSC** is particularly challenging, as 3D geometry and semantics must be inferred from a single image.

**Two fundamental bottlenecks of CNN-based architectures**:

**Non-uniform distribution in 2D feature extraction**: Perspective projection causes nearby points to spread out and distant points to cluster together. CNN's fixed weights cannot adapt to this non-uniform distribution, leading to insufficient 2D feature representation.

**Lack of long-range dependencies in 3D space**: 3D-to-2D projection inherently loses depth information. Recovering this lost information requires global 3D dependency modeling, which CNNs cannot provide due to their limited receptive fields.

**Why not directly use Transformers?** In 3D space, the quadratic complexity of Transformers becomes computationally prohibitive when handling large numbers of voxels. A more efficient alternative is required.

**Why does Mamba struggle with 3D voxels directly?** The standard approach flattens voxels into sequences according to 3D coordinates. However, during 2D-to-3D back-projection, voxels along the same ray share similar features, yet these voxels may be separated to distant positions in the sequence when sorted by 3D coordinates, causing **feature discontinuities**.

## Method

### Overall Architecture

Input monocular image → **Dual-head Multimodal Encoder (DMenc)** extracts globally-aware 2D semantic/geometric features → FLoSP back-projects features to 3D → **Frustum Mamba Decoder (FMdec)** captures 3D long-range dependencies → Prediction head outputs semantic occupancy map.

### Dual-head Multimodal Encoder (DMenc)

A Transformer encoder is used to extract 2D features with global context, followed by two parallel decoding branches:

$$\mathcal{TK} = \mathbf{Enc}(\mathcal{X}^{2d})$$

- **Geometry decoder** $\mathbf{Dec}_{\text{geo}}$: Extracts multi-scale geometric features $\mathcal{F}^{\text{geo}}$, with additional supervision via depth-predicted 3D occupancy.
- **Semantic decoder** $\mathbf{Dec}_{\text{sem}}$: Extracts multi-scale semantic features $\mathcal{F}^{\text{sem}}$, fused with geometric information.

**Design Motivation**: Disentangling geometric and semantic learning enables the model to capture finer-grained representations. Geometric features are fused into semantic features at each scale:

$$\mathcal{F}_l^{\text{sem}} = \mathcal{F}_l^{\text{sem}} + \mathcal{F}_l^{\text{geo}}$$

### Frustum Mamba Decoder (FMdec)

#### Multi-scale Feature Fusion

Channel attention is used to adaptively weight 3D features across different scales:

$$W^{\text{geo}} = \mathbf{MLP}(\mathbf{AvgPool}(\mathcal{F}^{\text{geo},3d}))$$
$$\mathcal{F}^{\text{geo},3d} = \sum_{l=1}^{N_l} \mathcal{F}_l^{\text{geo},3d} \cdot w_l$$

Semantic and geometric features are fused via element-wise addition: $\mathcal{F}^{3d} = \mathcal{F}^{\text{sem},3d} + \mathcal{F}^{\text{geo},3d}$

#### Frustum Mamba Layer

**Core Innovation** — Voxels are serialized in frustum space rather than 3D Euclidean space.

**Problem**: Standard methods flatten voxels by 3D coordinates $(x,y,z)$. Due to 2D-to-3D back-projection, adjacent voxels in the sequence may originate from different image positions, causing discontinuous feature jumps.

**Solution**: Frustum Reordering Strategy — Voxels are reordered according to frustum space (image-plane coordinates + depth), such that:
- Voxels along the same ray are adjacent in the sequence (continuous in the depth direction)
- Voxels adjacent in the image plane share similar features
- The ordering better aligns with Mamba's sequential scanning mechanism

$$\mathcal{SFF}^{3d} = \mathbf{FReorder}(\mathcal{FF}^{3d})$$
$$\mathcal{SFF'}^{3d} = \mathbf{MambaLayers}(\mathcal{SFF}^{3d})$$
$$\mathcal{F'}^{3d} = \mathbf{LN}(\mathbf{Composit}(\mathcal{SFF'}^{3d})) + \mathcal{F}^{3d}$$

Each stage consists of a Frustum Mamba Layer followed by downsampling to enable multi-scale 3D feature extraction.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{rel}} + \mathcal{L}_{\text{scal}}^{\text{sem}} + \mathcal{L}_{\text{scal}}^{\text{geo}} + \mathcal{L}_{\text{fp}}$$

- $\mathcal{L}_{\text{CE}}$: Cross-entropy loss on final semantic predictions
- $\mathcal{L}_{\text{BCE}}$: Occupancy prediction loss from the geometry decoder
- $\mathcal{L}_{\text{scal}}^{\text{sem/geo}}$: Scene-class affinity loss
- $\mathcal{L}_{\text{fp}}$: Frustum proportion loss
- $\mathcal{L}_{\text{rel}}$: Contextual relation loss

## Key Experimental Results

### Main Results on Occ-ScanNet

| Method | Input | IoU | mIoU |
|:---|:---:|:---:|:---:|
| MonoScene | RGB | 25.41 | 10.66 |
| NDC-Scene | RGB | 30.16 | 14.13 |
| ISO | RGB | 31.96 | 15.43 |
| **GA-MonoSSC** | **RGB** | **33.21** | **17.82** |

### Main Results on NYUv2

| Method | IoU | mIoU |
|:---|:---:|:---:|
| MonoScene | 37.12 | 25.17 |
| NDC-Scene | 43.93 | 29.47 |
| ISO | 42.98 | 29.84 |
| **GA-MonoSSC** | **44.89** | **31.52** |

### Ablation Study

| Component | IoU | mIoU | Note |
|:---|:---:|:---:|:---|
| Baseline (CNN encoder + CNN decoder) | 30.16 | 14.13 | NDC-Scene |
| + Transformer encoder | 31.45 | 15.21 | Global 2D context is effective |
| + Dual-head disentanglement | 32.03 | 16.14 | Semantic/geometric separation improves detail |
| + Mamba (3D coordinate ordering) | 32.56 | 16.89 | Long-range dependencies effective but discontinuity remains |
| **+ Frustum Reordering** | **33.21** | **17.82** | **Frustum ordering resolves feature discontinuity** |

### Key Findings

1. **Transformer vs. CNN encoder**: Transformers significantly outperform CNNs in modeling global relationships in the 2D domain, particularly under the non-uniform distribution induced by perspective projection.
2. **Value of dual-head decoding**: Disentangled geometric and semantic learning is more effective than joint learning; geometric features serve as structural priors injected into semantic representations.
3. **Necessity of Frustum Reordering**: The difference between 3D coordinate ordering and frustum ordering amounts to approximately 1 mIoU, validating the existence of the feature discontinuity problem.
4. **Mamba vs. Transformer in 3D**: Mamba achieves comparable long-range modeling capability to Transformers at linear computational complexity.

## Highlights & Insights

1. **2D Transformer + 3D Mamba hybrid paradigm**: Achieves an optimal balance between computational efficiency and modeling capacity.
2. **Frustum space insight**: Recognizing that back-projected features are continuous along ray directions is a critical geometric prior.
3. **Dual-modality disentanglement**: Decoupled learning of geometry and semantics is demonstrated to be effective for the SSC task.

## Limitations & Future Work

1. Mamba's unidirectional scanning may miss dependencies along certain spatial directions.
2. Validation is limited to indoor scenes; generalization to large-scale outdoor scenarios remains unexplored.
3. Frustum reordering introduces additional preprocessing complexity.

## Related Work & Insights

- **Monocular SSC**: MonoScene, NDC-Scene, ISO
- **State Space Models**: Mamba, S4, VMamba
- **3D Point Cloud SSMs**: Hilbert curve ordering, octree ordering

## Rating

- Novelty: ⭐⭐⭐⭐ — The Frustum Mamba Layer is an insightful and well-motivated design
- Technical Depth: ⭐⭐⭐⭐ — The Transformer+Mamba hybrid architecture with dual-head encoding is well-rounded
- Experimental Thoroughness: ⭐⭐⭐⭐ — State-of-the-art on two datasets with comprehensive ablation studies
- Value: ⭐⭐⭐ — Clear application to indoor monocular SSC; real-time performance remains to be verified

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](monocular_semantic_scene_completion_via_masked_recurrent_networks.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)

</div>

<!-- RELATED:END -->
