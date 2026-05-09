---
title: >-
  [Paper Note] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion
description: >-
  [3D Vision] This paper proposes DISC, a category-aware dual-stream architecture for 3D semantic scene completion that disentangles instance and scene categories into separate query streams with dedicated decoding modules. Using only single-frame input, DISC surpasses multi-frame state-of-the-art methods on SemanticKITTI, achieving a 17.9% improvement in instance category mIoU.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: ddb5b2d4e4f0a91c
---

# Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion

## Paper Information

- **Conference**: ICCV 2025
- **arXiv**: 2507.08555
- **Code**: [https://github.com/Enyu-Liu/DISC](https://github.com/Enyu-Liu/DISC)
- **Area**: 3D Perception / Semantic Scene Completion
- **Keywords**: semantic scene completion, BEV, instance-scene disentanglement, dual-stream, autonomous driving

## TL;DR

This paper proposes DISC, a category-aware dual-stream architecture for 3D semantic scene completion that disentangles instance and scene categories into separate query streams with dedicated decoding modules. Using only single-frame input, DISC surpasses multi-frame state-of-the-art methods on SemanticKITTI, achieving a 17.9% improvement in instance category mIoU.

## Background & Motivation

3D Semantic Scene Completion (SSC) aims to jointly predict scene geometry and semantics from sparse inputs. Existing voxel-based methods (VoxFormer, CGFormer, Symphonies, etc.) suffer from three key limitations:

**Poor instance category prediction**: Occlusion and projection errors cause category misses and semantic ambiguity (e.g., pedestrians being misidentified).

**Incoherent scene category structure**: Out-of-view regions introduce topological errors (e.g., roads appearing in terrain areas).

**Inherent limitations of voxel-based approaches**: Using voxels as the basic interaction unit destroys category-level structural information, and unified modules struggle to address the divergent challenges of instances and scenes.

Core observation: Instance categories (car, person, bicycle) and scene categories (road, building, vegetation) face fundamentally different challenges and require differentiated processing strategies. DISC is the first to propose a category-level dual-stream paradigm to systematically address this problem.

## Method

### Overall Architecture

DISC operates in BEV space and consists of two core modules:

1. **Discriminative Query Generator (DQG)**: Generates queries with geometric and semantic priors separately for instances and scenes.
2. **Dual-Attention Category Decoder (DACD)**: Provides dedicated decoding layers tailored to the distinct challenges of instances and scenes.

### Key Design 1: Discriminative Query Generator (DQG)

**Coarse-to-fine BEV generation**: Coarse voxel features $V_{\text{coarse}}$ are produced via LSS lift, refined into $V_{\text{fine}}$ through depth-guided surface voxel refinement, and Z-axis max-pooling yields BEV features $C$.

**Instance queries**: Potential instances are detected in image space via segmentation and projected to BEV. A $k \times k$ neighborhood suppression strategy selects $N_{\text{ins}}$ reference locations, and BEV features are sampled at these locations to initialize instance queries:

$$X_{\text{ins}} = \{CT(\mathbf{g}_n) \mid \mathbf{g}_n \in \text{Top-}N(\{\text{Max}(B_{k \times k}^i)\}_{i=1}^s)\}$$

$$\mathbf{Q}_{\text{ins}} = C[X_{\text{ins}}]$$

**Scene queries**: A patch design captures continuous spatial distributions. BEV features are divided into equal-sized patches, with each patch center serving as a reference point; convolutional upsampling compresses each patch to $1 \times 1$ to initialize scene queries.

### Key Design 2: Adaptive Instance Layer (AIL)

Addresses the loss of height information and occlusion issues for instance categories:

1. **Adaptive height sampling**: For each instance query $q_{\text{ins}}$, $N$ most probable heights are predicted to form 3D reference points $P_j = (x_{\text{ins}}, h_j)$.
2. **Image cross-attention**: Reference points are projected to image space, and multi-scale features are sampled via deformable cross-attention:

$$q_{\text{ins}} = \sum_{j=1}^N w_j \text{DA}(q_{\text{ins}}, F^{2D}, \mathcal{T}^{WI}(x_{\text{ins}}, h_j))$$

3. **Scene context fusion**: Instance queries extract region-of-interest information from scene features (e.g., "a cylindrical object on a sidewalk is more likely a traffic sign than a tree trunk").
4. **UNet propagation**: Instance features are propagated across the entire BEV plane via a UNet network.

### Key Design 3: Global Scene Layer (GSL)

Addresses insufficient global reasoning for scene categories:

1. **Global semantic aggregation**: $\mathbf{Q}_{\text{img}}$ is constructed from the smallest-scale image features, and global information is aggregated via cross-attention.
2. **Random masking**: A subset of $\mathbf{Q}_{\text{img}}$ is dropped to simulate missing occlusion information, encouraging the network to reason about scene layout.
3. **Self-attention**: Expands the global receptive field, propagating visible-region features to distant and out-of-view areas.

### Feature Fusion and Loss Function

Category-disentangled height prediction fusion:

$$V = (C_{\text{ins}} \bigotimes H_{\text{ins}}) + (C_{\text{scn}} \bigotimes H_{\text{scn}})$$

The total loss comprises the SSC loss (Scene-Class Affinity + Cross-Entropy), BEV segmentation/height augmentation loss, and depth loss:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{ssc} + \lambda_2 \mathcal{L}_{aug} + \lambda_d \mathcal{L}_d$$

## Key Experimental Results

### Main Results: SemanticKITTI Test Set

| Method | Input | IoU | InsM | ScnM | mIoU |
|--------|-------|-----|------|------|------|
| VoxFormer-T | Multi-frame | 43.21 | 4.79 | 22.97 | 13.41 |
| HTCL | Multi-frame | 44.23 | 6.48 | 28.86 | 17.09 |
| VoxFormer-S | Single-frame | 42.95 | 4.39 | 20.89 | 12.20 |
| Symphonize | Single-frame | 42.19 | 6.14 | 24.93 | 15.04 |
| CGFormer | Single-frame | 44.41 | 6.15 | 28.24 | 16.63 |
| **DISC (ours)** | **Single-frame** | **45.32** | **7.25** | **28.56** | **17.35** |

- Single-frame DISC outperforms all multi-frame methods in mIoU (17.35 vs. HTCL 17.09).
- Instance mIoU (InsM) reaches 7.25, a 17.9% gain over single-frame SOTA and 11.9% over multi-frame SOTA.
- On SSCBench-KITTI-360, mIoU reaches 20.55, surpassing all camera-based and LiDAR-based methods.

### Ablation Study (SemanticKITTI val)

| Configuration | IoU | InsM | ScnM | mIoU |
|---------------|-----|------|------|------|
| Baseline (VoxFormer) | 43.13 | 3.59 | — | 12.18 |
| + DQG (instance queries) | 43.78 | 5.22 | — | 14.31 |
| + DQG (scene queries) | 44.01 | 3.98 | — | 14.76 |
| + DACD (AIL + GSL) | 44.85 | 6.15 | — | 16.10 |
| **Full DISC** | **45.32** | **7.25** | **28.56** | **17.35** |

**Key Findings**:
- Discriminative queries (replacing voxel queries) account for the majority of instance performance gains.
- The targeted design of the dual-attention decoder further improves InsM to 7.25.
- Training requires only 20 epochs, fewer than most existing methods, indicating faster convergence.

## Highlights & Insights

1. **First category-level paradigm**: Shifts from voxel-level to category-level interaction, fundamentally reframing the SSC processing paradigm.
2. **Feasibility of BEV-space disentanglement**: Instance-scene decoupling mitigates feature entanglement along the height axis in BEV (e.g., height ambiguity when pedestrians and roads share the same BEV cell).
3. **Single-frame surpassing multi-frame**: Demonstrates that fully exploiting single-frame category information holds greater potential than naive multi-frame fusion.
4. **Scene context aiding instance reasoning**: Validates the effectiveness of spatial priors such as "a cylindrical object on a sidewalk → traffic sign."

## Limitations & Future Work

- Performance depends on the quality of the pretrained MaskDINO instance segmentation; segmentation failures degrade instance query initialization.
- Height prediction in BEV space remains approximate; extreme height differences (e.g., tall buildings vs. ground) may not be captured accurately.
- Validation is limited to KITTI-series datasets; generalization to large-scale benchmarks such as nuScenes or Waymo has not been explored.

## Related Work & Insights

- Relation to Symphonies: Symphonies implicitly leverages instance priors, whereas DISC explicitly disentangles instance and scene streams.
- The instance query design in DQG (image-space detection → BEV projection → neighborhood suppression) is generalizable to other BEV perception tasks.
- The dual-stream architecture conceptually parallels the differentiated handling of target types in the DETR family of methods.

## Rating

⭐⭐⭐⭐⭐ — The work offers deep problem insight (instance vs. scene divergent challenges), an elegant solution (category-level dual-stream), outstanding experimental results (single-frame surpassing multi-frame), and a paradigm-level contribution to the SSC task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)

</div>

<!-- RELATED:END -->
