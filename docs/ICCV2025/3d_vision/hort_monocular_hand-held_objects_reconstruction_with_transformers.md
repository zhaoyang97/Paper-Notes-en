---
title: >-
  [Paper Note] HORT: Monocular Hand-held Objects Reconstruction with Transformers
description: >-
  [3D Vision] This paper proposes HORT, a coarse-to-fine Transformer-based framework that efficiently reconstructs dense 3D point clouds of hand-held objects from monocular images. By integrating image features with 3D hand geometry, HORT jointly predicts the object point cloud and its pose relative to the hand, achieving state-of-the-art performance in both reconstruction accuracy and inference speed.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: 3d9ec759ac0da248
---

# HORT: Monocular Hand-held Objects Reconstruction with Transformers

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.21313](https://arxiv.org/abs/2503.21313)
- **Code**: [https://zerchen.github.io/projects/hort.html](https://zerchen.github.io/projects/hort.html)
- **Area**: 3D Vision / Hand-held Object Reconstruction
- **Keywords**: Hand-Object Reconstruction, Point Cloud, Transformer, Coarse-to-Fine, Monocular 3D

## TL;DR
This paper proposes HORT, a coarse-to-fine Transformer-based framework that efficiently reconstructs dense 3D point clouds of hand-held objects from monocular images. By integrating image features with 3D hand geometry, HORT jointly predicts the object point cloud and its pose relative to the hand, achieving state-of-the-art performance in both reconstruction accuracy and inference speed.

## Background & Motivation

Reconstructing the 3D shape of hand-held objects from monocular images has broad applications in action recognition, human-computer interaction, and robotic manipulation. Existing methods face critical bottlenecks:

**Implicit representation methods** (SDF, etc.):
- Generate overly smooth 3D surfaces that lose geometric detail
- Require Marching Cubes post-processing to obtain explicit meshes, resulting in slow inference (~2 seconds)
- Cannot be flexibly applied to downstream tasks

**Explicit representation methods**:
- HO uses vertex representations but is limited in resolution
- D-SCO uses diffusion models to reconstruct high-resolution point clouds, but multi-step denoising leads to extremely slow inference (>13 seconds)

The root cause lies in the **trade-off between high-quality reconstruction and efficient inference**. Furthermore, hand geometry implicitly encodes cues about object geometry and location, yet existing methods fail to fully exploit this information.

## Method

### Overall Architecture

HORT adopts a coarse-to-fine two-stage strategy comprising four key modules:

1. **Image Encoder**: Extracts 256+1 visual feature tokens using DINOv2-Large
2. **Hand Encoder**: Encodes MANO hand geometry into rich 3D features
3. **Sparse Point Cloud Decoder**: Jointly predicts a sparse point cloud and hand-relative pose
4. **Dense Point Cloud Decoder**: Upsamples to a high-resolution point cloud using pixel-aligned features

### Fine-grained Hand Feature Encoding

The key innovation lies in encoding 3D hand geometry:

1. The MANO model reconstructs a hand mesh $v_h \in \mathbb{R}^{778\times3}$, driven by joints $j_h \in \mathbb{R}^{16\times3}$
2. Hand vertices are transformed into 22 local coordinate systems (16 joints + 5 fingertips + 1 palm center)
3. The transformed coordinates and absolute vertex indices are concatenated to obtain $e_h \in \mathbb{R}^{778\times67}$
4. A PointNet encodes this into hand features $f_h \in \mathbb{R}^{1024}$

This multi-coordinate-system representation captures pose- and shape-related geometric information of the hand, providing a strong structural prior for object reconstruction.

### Sparse Point Cloud Decoder

The reconstruction task is decomposed into two subtasks:
- **Canonical object point cloud generation**: Generated in the palm coordinate system
- **Hand-relative object pose estimation**: Predicts only the 3D translation $t_o \in \mathbb{R}^3$ relative to the palm (avoiding the ill-posed rotation prediction problem caused by object symmetry)

The decoder employs a unified multi-layer Transformer:
- Defines $1 + N_p^s$ learnable tokens (1 pose token + $N_p^s$ point cloud tokens)
- Applies self-attention followed by cross-attention separately over image features $f_v$ and hand features $f_h$
- A shared backbone enables mutual reinforcement between pose and point cloud predictions

### Dense Point Cloud Decoder

Upsampling strategy from sparse to dense:

1. **Pixel-aligned feature extraction**: Projects the sparse point cloud onto the image plane using predicted camera parameters and pose:
$$f_o = F(\pi(p_o^s + t_p + t_o, K_{cam}), f_v^r)$$
2. **Local self-attention**: Self-attention within KNN neighborhoods ($k=16$) to aggregate spatial and visual context
3. **Two-stage upsampling**: Upsamples at $2\times$ and $4\times$ respectively, yielding a final dense point cloud of $N_p^d = 16384$ points

### Loss & Training

End-to-end training with the following total loss:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{pose} + \lambda_2 \mathcal{L}_{cd}^s + \mathcal{L}_{cd}^d$$

- $\mathcal{L}_{pose}$: $\ell_1$ loss supervising the 3D translation of the object relative to the palm
- $\mathcal{L}_{cd}^s$: Chamfer Distance for the sparse point cloud
- $\mathcal{L}_{cd}^d$: Chamfer Distance for the dense point cloud
- Both hyperparameters are set to 2

## Key Experimental Results

### Main Results: Comparison on ObMan Dataset

| Method | FS@5 ↑ | FS@10 ↑ | CD ↓ |
|:---|:---:|:---:|:---:|
| HO | 0.23 | 0.56 | 6.4 |
| AlignSDF | 0.40 | 0.64 | 9.2 |
| gSDF | 0.44 | 0.66 | 8.8 |
| DDF-HO | 0.55 | 0.67 | 1.4 |
| D-SCO | 0.61 | 0.81 | 1.1 |
| **HORT (Ours)** | **0.66** | **0.88** | **1.0** |

HORT outperforms D-SCO on all metrics, with FS@5 improving by +0.05 and FS@10 by +0.07.

### HO3D and DexYCB Real-world Datasets

| Method | HO3D FS@5 ↑ | HO3D CD ↓ | DexYCB FS@5 ↑ | DexYCB CD ↓ |
|:---|:---:|:---:|:---:|:---:|
| D-SCO | 0.38 | 3.2 | 0.48 | 2.9 |
| **HORT** | **0.41** | **2.5** | **0.52** | **2.5** |

HORT maintains state-of-the-art performance on real-world datasets as well.

### Ablation Study: Encoder Design

| Config | Palm | Joints | Image Encoder | FS@5 ↑ | CD ↓ |
|:---|:---:|:---:|:---|:---:|:---:|
| R1 | × | × | Fine-tune | 0.45 | 3.1 |
| R2 | ✓ | × | Fine-tune | 0.53 | 2.4 |
| R3 | ✓ | ✓ | Fine-tune | 0.60 | 1.8 |
| R4 | ✓ | ✓ | Scratch | 0.51 | 2.6 |
| R5 | ✓ | ✓ | Frozen | 0.48 | 2.9 |

Key findings: (1) Multi-coordinate-system hand encoding is critical (R1→R3: FS@5 improves by 0.15); (2) The fine-tuning strategy for the image encoder has a significant impact.

### Inference Speed Comparison

HORT processes a single image in approximately **0.08 seconds** (including mesh extraction), compared to **13 seconds** for D-SCO and ~**2 seconds** for implicit methods — a **162× speedup**.

## Highlights & Insights
1. **Optimal speed-quality trade-off**: HORT achieves superior reconstruction quality while being two orders of magnitude faster than D-SCO
2. **Effective exploitation of hand priors**: Multi-coordinate-system transformation combined with PointNet-encoded hand geometry provides strong shape constraints for object reconstruction
3. **Advantages of end-to-end training**: Compared to the modular training of D-SCO, end-to-end optimization enables better synergy among components
4. **Suitable as initialization for optimization-based methods**: The feed-forward predictions can accelerate subsequent optimization-based refinement

## Limitations & Future Work
- Object rotation is not predicted (due to symmetry issues), leaving pose estimation incomplete for asymmetric objects
- Performance depends on the quality of hand pose estimation; failures in hand reconstruction degrade object reconstruction
- Point cloud representations cannot be directly used in applications requiring mesh topology (e.g., physics simulation)

## Related Work & Insights
- Hand-held object reconstruction: IHOI, AlignSDF, gSDF, D-SCO
- Implicit 3D representations: SDF, DDF, NeRF
- Point cloud generation: PoinTr, SnowflakeNet, Michelangelo
- Hand reconstruction: MANO, HaMeR, WiLoR

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The coarse-to-fine Transformer framework combined with hand geometry encoding is novel and practical
- **Technical Depth**: ⭐⭐⭐⭐ — Multi-coordinate-system hand encoding and joint decoding design are elegant
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, comprehensive ablations, and inference speed comparisons
- **Value**: ⭐⭐⭐⭐⭐ — 0.08-second inference enables real-time application potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)

</div>

<!-- RELATED:END -->
