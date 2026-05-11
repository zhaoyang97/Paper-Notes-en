---
title: >-
  [Paper Note] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery
description: >-
  [CVPR 2026][3D Vision][Panoramic 3D reconstruction] This paper proposes PanoVGGT, a permutation-equivariant Transformer framework that jointly predicts camera poses, depth maps…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Panoramic 3D reconstruction"
  - "feed-forward multi-view reconstruction"
  - "spherical position encoding"
  - "SO(3) data augmentation"
  - "large-scale panoramic dataset"
date: 2026-05-08
content_hash: 3795b2b8504dddcb
---

# PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery

**Conference**: CVPR 2026
**arXiv**: [2603.17571](https://arxiv.org/abs/2603.17571)
**Code**: Available (coming soon)
**Area**: 3D Vision
**Keywords**: Panoramic 3D reconstruction, feed-forward multi-view reconstruction, spherical position encoding, SO(3) data augmentation, large-scale panoramic dataset

## TL;DR
This paper proposes PanoVGGT, a permutation-equivariant Transformer framework that jointly predicts camera poses, depth maps, and globally consistent 3D point clouds from one or more unordered panoramic images in a single feed-forward pass. The paper also contributes PanoCity — a large-scale dataset comprising over 120,000 outdoor panoramic images.

## Background & Motivation
**Background**: Feed-forward 3D reconstruction models such as DUSt3R, VGGT, and $\pi^3$ have achieved remarkable success on perspective images, jointly inferring depth, pose, and 3D structure in a single forward pass.

**Limitations of Prior Work**:
   - These models are fundamentally built on the pinhole projection assumption; applying them directly to equirectangular panoramic images introduces seam artifacts, inconsistent parallax, and geometric drift.
   - Decomposing panoramas into multiple perspective crops and stitching the results introduces additional artifacts.
   - Existing panoramic datasets suffer from insufficient scale, incomplete annotations, and inadequate viewpoint overlap.

**Key Challenge**: Panoramic images exhibit non-pinhole distortion and spherical geometry, rendering the position encodings, data augmentation strategies, and geometric reasoning of existing feed-forward models inapplicable.

**Goal**: (a) Extend the feed-forward 3D reconstruction paradigm to the panoramic image domain; (b) construct a sufficiently large panoramic dataset to support training.

**Key Insight**: Enable the Transformer to perform effective geometric reasoning in the spherical domain through spherical-aware position encoding and $SO(3)$ rotation augmentation.

**Core Idea**: Extend the feed-forward reconstruction paradigm of VGGT/$\pi^3$ to panoramic imagery via spherical position encoding, three-axis rotation augmentation, and a stochastic anchoring strategy.

## Method

### Overall Architecture
Given a set of unordered panoramic images $\{I_i\}_{i=1}^N$, the model passes them through an encode–aggregate–decode architecture to output camera poses $G = \{g_i\}$, depth maps $D = \{D_i\}$, and a world-coordinate 3D point cloud $P = \{P_i\}$. The model is permutation-equivariant — the input order does not affect the output.

### Key Designs

1. **Spherical-aware Position Embedding**

    - **Function**: Provides geometrically correct positional information for panoramic image patches under equirectangular projection.
    - **Mechanism**: For each patch, the spherical center coordinates $(\theta, \phi)$ are computed and encoded as a 4D cyclically symmetric vector $p_{\text{vec}} = [\sin\theta, \cos\theta, \sin\phi, \cos\phi]$, which is then mapped to a high-dimensional embedding $p_{\text{embed}} \in \mathbb{R}^C$ via an MLP.
    - **Design Motivation**: Standard ViT position encodings cannot handle the spatially varying sampling density of equirectangular projection. Trigonometric encodings naturally preserve wrap-around continuity at the boundary $\theta = \pm\pi$. More critically, under $SO(3)$ rotation augmentation, the decoupling of fixed position encodings from dynamically rotated content forces the network to disentangle distortion effects from semantic content.

2. **Geometry Aggregator**

    - **Function**: Performs local and global geometric reasoning across multi-view tokens.
    - **Mechanism**: $L$ alternating attention blocks, each comprising (a) intra-frame self-attention — tokens within the same panorama attend to each other to capture local structure and projection distortion; and (b) global self-attention — tokens from all panoramas are mixed to enable cross-view correspondence reasoning. The aggregated features are combined with spherical embeddings via three lightweight adapters and fed into three prediction heads: a camera pose head, a local point cloud head, and a global point cloud head.

3. **Stochastic Anchoring**

    - **Function**: Resolves global coordinate frame ambiguity arising from the permutation-equivariant design.
    - **Mechanism**: At each training iteration, a random panorama $k$ is selected as the anchor, and all poses and point clouds are aligned to a coordinate frame centered on frame $k$.
    - **Design Motivation**: Fixing the first frame as the origin (as in VGGT) introduces ordering bias and is unstable under unordered inputs. Stochastic anchoring establishes a stable "hub-and-spoke" geometric structure while maintaining full permutation equivariance.

4. **Panorama-specific Three-axis $SO(3)$ Data Augmentation**

    - **Function**: Exploits panoramas' natural support for spherical rotations to enable unbounded data augmentation.
    - **Mechanism**: For each RGB–depth–pose triplet, a random rotation $R_{\text{aug}} \in SO(3)$ is sampled; poses are updated as $g_i' = R_{\text{aug}} \cdot g_i$, and the panorama and depth map are projected onto the sphere, rotated, and resampled back to equirectangular format.
    - **Design Motivation**: Augmentation for perspective images is constrained by planar geometry, whereas panoramas support arbitrary three-axis rotations without compromising geometric validity, substantially alleviating data scarcity.

### Loss & Training
- **Scale-consistent local/global geometry losses**: $\mathcal{L}_{\text{lp}} + \mathcal{L}_{\text{gp}}$, with an optimal scale factor $s^*$ estimated in closed form to ensure metric consistency.
- **Normal consistency regularization**: $\mathcal{L}_{\text{nor}}$, penalizing angular differences between surface normals.
- **Relative pose supervision**: rotation loss $\mathcal{L}_{\text{rot}}$ (angular distance) + translation loss $\mathcal{L}_{\text{trans}}$ (L1).
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{lp}} + \mathcal{L}_{\text{gp}} + \mathcal{L}_{\text{nor}} + 0.1(100 \cdot \mathcal{L}_{\text{trans}} + \mathcal{L}_{\text{rot}})$
- Training takes approximately 10 days on 8× A100 GPUs.

## Key Experimental Results

### Main Results — Camera Pose Estimation

| Method | Matterport3D AUC@30↑ | PanoCity AUC@30↑ | PanoCity Rot. Err.↓ | PanoCity Trans. Err.↓ |
|------|---------------------|-----------------|------------------|------------------|
| BiFuse++ | 0.007 | 0.833 | 1.655° | 5.044° |
| VGGT | 0.034 | 0.205 | 7.659° | 35.867° |
| $\pi^3$ | 0.047 | 0.571 | 7.669° | 16.780° |
| $\pi^3$* (panoramic retrain) | 0.305 | 0.682 | — | — |
| **PanoVGGT** | **0.459** | **0.949** | **0.873°** | **2.168°** |

### Monocular Depth Estimation

| Method | Matterport3D Abs Rel↓ | Stanford2D3D Abs Rel↓ | PanoCity Abs Rel↓ |
|------|---------------------|----------------------|------------------|
| EGFormer | 0.0987 | 0.0929 | 0.0363 |
| BiFuse++ | 0.1076 | 0.1120 | 0.0200 |
| PanoVGGT (mono) | 0.0884 | 0.0711 | 0.0312 |
| PanoVGGT (multi-view) | **0.0840** | 0.0778 | **0.0196** |

### Key Findings
- PanoVGGT comprehensively outperforms all baselines on pose estimation: AUC@30 on Matterport3D rises from 0.305 (the second-best, $\pi^3$*) to 0.459, and reaches 0.949 on PanoCity.
- PanoVGGT surpasses dedicated depth estimation models on monocular depth estimation, while operating as a single unified model under a multi-task joint prediction setting.
- BiFuse++ performs poorly on Matterport3D/Stanford2D3D because its self-supervised training relies on ordered narrow-baseline frames, which is mismatched with these sparse unordered panoramas.
- The PanoCity dataset (120K frames) far exceeds the scale of existing panoramic datasets (Matterport3D: 10K; Structured3D: 12K) and provides complete multi-view overlap.

## Highlights & Insights
- **The synergistic design of spherical position encoding and $SO(3)$ augmentation is particularly elegant**: rotating content under fixed position encodings forces the network to learn to decouple projection distortion effects from semantic content, achieving an effect analogous to spherical convolutions without their complexity.
- **Stochastic anchoring** is a simple yet effective solution to the global coordinate frame ambiguity in permutation-equivariant models, and is more robust than fixing the first frame.
- **The PanoCity dataset is itself a significant contribution**: it provides the first large-scale outdoor panoramic dataset with continuous trajectories, complete 6-DoF poses, and high-precision depth, filling a critical gap in the field.
- **Overcomplete supervision**: jointly regressing both depth maps and 3D point clouds (which are theoretically mutually derivable) proves empirically that this redundant supervision significantly improves the accuracy of all predictions.

## Limitations & Future Work
- Training resolution is limited to $336 \times 672$ (far below the native $4096 \times 2048$), potentially losing high-frequency detail.
- Indoor datasets (Matterport3D, Stanford2D3D) contain few valid multi-view samples with poor overlap, limiting the thoroughness of indoor evaluation.
- The model is large (DINOv2 backbone); inference efficiency on edge devices is not discussed.
- PanoCity is a synthetic dataset; the domain gap with real-world panoramas is not sufficiently analyzed.

## Related Work & Insights
- **vs. VGGT / $\pi^3$**: Both models are built on the pinhole assumption and suffer drastic performance degradation when applied directly to panoramas. PanoVGGT effectively bridges this gap through spherical position encoding and rotation augmentation.
- **vs. BiFuse++**: BiFuse++ is designed for panoramas but relies on ordered narrow-baseline self-supervision, collapsing on sparse unordered viewpoints; PanoVGGT's permutation-equivariant design naturally accommodates unordered inputs.
- **vs. Traditional Panoramic Methods**: Most existing panoramic methods address only a single task (depth or pose); PanoVGGT is the first unified feed-forward model to jointly predict poses, depth, and point clouds in the panoramic domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The synergistic design of spherical position encoding and $SO(3)$ augmentation is creative, though the overall architecture follows VGGT/$\pi^3$.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation spans multiple datasets and tasks, including cross-domain generalization and complete ablations, with a high-quality dataset contribution.
- Writing Quality: ⭐⭐⭐⭐ Well-structured; both dataset construction and method design are described with sufficient clarity.
- Value: ⭐⭐⭐⭐⭐ The PanoCity dataset and the panoramic feed-forward reconstruction paradigm will have substantial impact on the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] Reliev3R: Relieving Feed-forward 3D Reconstruction from Multi-View Geometric Annotations](reliev3r_relieving_feed-forward_3d_reconstruction_from_multi-view_geometric_annot.md)

</div>

<!-- RELATED:END -->
