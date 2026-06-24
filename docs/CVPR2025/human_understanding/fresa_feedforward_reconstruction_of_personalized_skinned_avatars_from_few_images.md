---
title: >-
  [Paper Note] FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images
description: >-
  [CVPR 2025][Human Understanding][Avatar Reconstruction] This paper proposes FRESA, which learns a general clothed human prior to jointly infer personalized canonical shape, skinning weights, and pose-dependent deformations in a feedforward manner (18 seconds) from a few images. This achieves high-quality animatable 3D human avatar reconstruction with zero-shot generalization to mobile phone photos.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Avatar Reconstruction"
  - "Feedforward Inference"
  - "Personalized Skinning"
  - "Linear Blend Skinning"
  - "Animatable"
date: 2026-05-08
content_hash: a03b3aae2216b0e2
---

# FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images

**Conference**: CVPR 2025  
**arXiv**: [2503.19207](https://arxiv.org/abs/2503.19207)  
**Code**: [https://github.com/rongakowang/FRESA](https://github.com/rongakowang/FRESA)  
**Area**: Human Understanding  
**Keywords**: Avatar Reconstruction, Feedforward Inference, Personalized Skinning, Linear Blend Skinning, Animatable

## TL;DR

This paper proposes FRESA, which learns a general clothed human prior to jointly infer personalized canonical shape, skinning weights, and pose-dependent deformations in a feedforward manner (18 seconds) from a few images. This achieves high-quality animatable 3D human avatar reconstruction with zero-shot generalization to mobile phone photos.

## Background & Motivation

**Background**: Significant progress has been made in 3D clothed human reconstruction (e.g., PIFu, ICON), but most focus on reconstructing single-frame static shapes. To obtain animatable avatars, it is necessary to reconstruct the geometry in the canonical space and animate it through Linear Blend Skinning (LBS) based on skinning weights.

**Limitations of Prior Work**: Current animatable avatar reconstruction methods suffer from two main limitations: (1) Feedforward methods like ARCH++, while fast, use nearest-neighbor skinning weights from a template body to bind the avatar, leading to distortion artifacts (e.g., over-stretched underarm triangles) under extreme poses and body shapes. (2) Some methods attempt to jointly optimize personalized skinning weights but lack a unified prior across body shapes and clothing types, relying on per-subject optimization that requires hours of inference time.

**Key Challenge**: Personalized skinning weights are critical to animation quality (different body shapes and clothing require different skinning strategies), but learning such weights requires large-scale diverse data to build a general prior. Meanwhile, there is a coupling ambiguity between the canonical shape and skinning weights; an incorrect canonical shape combined with incorrect skinning weights can accidentally produce a correct posed shape.

**Goal**: How to jointly infer personalized canonical geometry, skinning weights, and pose-dependent deformations in a feedforward manner from a few images, without per-subject optimization?

**Key Insight**: The authors collected a large-scale dome-captured dataset featuring over 1,100 subjects wearing diverse clothing types with up to 100 poses per person, allowing them to learn a general prior across body shapes and clothing categories. Explicit 3D canonicalization generates pixel-aligned initial conditions to simplify feature extraction, while multi-frame aggregation eliminates canonicalization artifacts and fuses intrinsic human information.

**Core Idea**: Learning general priors from thousand-scale data to achieve feedforward joint inference of personalized skinned avatars via 3D canonicalization, multi-frame aggregation, and multi-stage training.

## Method

### Overall Architecture

Inputs: N frames of clothed human images (front and back views) and estimated 3D poses. Outputs: avatar mesh $M$ in the canonical space, skinning weight matrix $W$, and pose-dependent displacements $\Delta V$ under any target poses. The pipeline consists of three steps: (1) 3D Canonicalization unposes the posed images into the canonical space to generate pixel-aligned initial conditions; (2) Multi-frame encoder aggregation and decoder jointly predict geometry, skinning, and deformation; (3) Multi-stage training decouples canonical and posed supervisions.

### Key Designs

1. **3D Canonicalization**:

    - Function: Eliminate pose differences in input images to generate pixel-aligned initial conditions in a unified space.
    - Mechanism: First, a foundation model is deployed to estimate normal maps and segmentation masks, which are reconstructed into 3D front/back surface meshes via normal integration. Next, inverse LBS transforms these meshes into the canonical space: $[v;1] = (\sum_{j=1}^J w_j T_j)^{-1}[\hat{u};1]$. At this stage, deterministic unposing is performed using template nearest-neighbor skinning weights (although artifacts occur, their consistent patterns are easily corrected by downstream networks). Finally, canonical normals and segmentation masks are rendered using a fixed orthographic camera to serve as network inputs.
    - Design Motivation: Sampling features directly from posed images leads to misalignment due to pose variations, resulting in overly smoothed reconstructions. After canonicalization, the same body parts consistently map to the same locations in feature space, significantly lowering the learning difficulty.

2. **Multi-frame Feature Aggregation**:

    - Function: Fuse cross-frame features to eliminate canonicalization artifacts and extract intrinsic human identity features.
    - Mechanism: For each frame, its canonical normal and segmentation maps are fed into a shallow CNN to extract high-resolution features $H_i^v$ and DeepLabV3 to extract low-resolution global features $L_i^v$. These multi-frame features are aggregated into a single triplane feature $B = (B^f \oplus B^b)$ via simple averaging, where $B^v = \frac{1}{N}\sum_{i=1}^N f_b(H_i^v \oplus L_i^v)$.
    - Design Motivation: Unposing artifacts vary across poses, but the user's intrinsic information (body shape, clothing style) remains consistent. Simple averaging naturally preserves common features and filters out frame-specific artifacts. Experiments show that 5 frames are sufficient to converge to high-quality results.

3. **Joint Decoding: Geometry + Skinning + Pose Deformation**:

    - Function: Jointly predict three coupled outputs from the aggregated features.
    - Mechanism:
        - Geometry Decoder: On the canonical tetrahedral grid, each vertex is projected to sample the bi-plane features, and an MLP predicts SDF values and displacements. The mesh is then extracted using Marching Tetrahedra.
        - Skinning Weight Decoder: An independent MLP predicts $J$-joint skinning weights for each canonical vertex (normalized via Softmax to ensure validity), regularized against the template's nearest-neighbor weights.
        - Pose Deformation Module: Conditioned on a rendered position map of the target pose, together with the rendered normals of the canonical mesh, a CNN + MLP predicts per-vertex displacements $\Delta v_t$. The final animation is computed as: $[\hat{v}_t;1] = \text{LBS}(v + \Delta v_t, w, \hat{T})$.
    - Design Motivation: Jointly optimizing the three outputs is more effective than separate optimizations (skinning weights affect geometric quality, while geometric shape determines skinning plausibility). Multiple training stages are employed to resolve the coupling ambiguity.

### Loss & Training

**Multi-stage training** resolves the coupling ambiguity between the canonical shape and skinning weights:

- **Canonical Stage**: Only the encoder and geometry decoder are trained, supervised by pseudo-GT canonical meshes (high-quality unposed results obtained through optimization): $\mathcal{L}_c = \|\mathcal{N} - \mathcal{N}_i^\star\|_1 + \|\mathcal{D} - \mathcal{D}_i^\star\|_1$.
- **Posed Stage**: All modules are jointly trained, supervised by GT scans in the posed space: $\mathcal{L} = \lambda_p \mathcal{L}_p + \lambda_s \mathcal{L}_s + \lambda_e \mathcal{L}_e$, where $\mathcal{L}_p$ includes normal L1 + depth L1 + perceptual losses, $\mathcal{L}_s$ regularizes skinning weights to stay close to the template, and $\mathcal{L}_e$ penalizes excessively stretched triangle edges.

## Key Experimental Results

### Main Results

| Method | Normal↓ | P2S(cm)↓ | CD(cm)↓ | Inference Time |
|------|---------|----------|---------|---------|
| ARCH++ (Feedforward) | 0.338 | 4.52 | 5.07 | 26s |
| PuzzleAvatar (Diffusion) | 0.104 | 1.47 | 1.63 | 3h |
| Vid2Avatar (Optimization) | 0.072 | 0.98 | 1.12 | 8h |
| **FRESA (LBS Only)** | 0.030 | 0.43 | 0.49 | 18s |
| **FRESA (Full)** | **0.026** | **0.37** | **0.43** | **18s** |

Zero-shot generalization on the RenderPeople dataset also leads by a wide margin (CD: 0.34 vs. 1.91) and generalizes directly to mobile phone photos.

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| w/o Canonicalization | Overly smoothed geometry | Features sampled directly from posed images are misaligned |
| Single frame vs. 5-frame aggregation | Multi-frame is more accurate | Artifacts are averaged out; skirts and hair are more plausible |
| Template skinning vs. Personalized skinning | Personalization reduces underarm artifacts | Skinning weights trained on multiple frames are more robust |
| w/o Pose deformation | Lacks dynamic wrinkles | Deformation module corrects LBS artifacts and generates realistic wrinkles |

### Key Findings

- Feedforward inference takes only 18 seconds, which is 600-1600x faster than optimization-based methods, yet yields superior quality thanks to general priors learned from large-scale data.
- Personalized skinning weights significantly improve animation quality under extreme poses, especially in regions like the armpits and elbow bends.
- The pose-dependent deformation module brings three benefits: correcting LBS artifacts, generating realistic clothing dynamics (e.g., sagging sleeves when arms are raised), and refining detailed geometric features.

## Highlights & Insights

- **The "unpose-then-correct" strategy is highly practical**: Although unposing artifacts are imperfect, they provide pixel-aligned starting points, freeing the network to learn "corrective residuals" rather than "understanding poses from scratch." This significantly eases learning difficulty.
- **The simplicity and efficacy of average multi-frame aggregation are impressive**: Without complex attention mechanisms or alignment operations, a simple average leverages cross-frame consistency to filter out artifacts. This demonstrates that simple methods thrive when initial conditions are robust.
- **The general prior driven by large-scale data represents the core advantage**: The dome dataset of 1100+ subjects serves as the "moat" of this work, enabling feedforward inference to outperform optimization-based approaches in both quality and speed.

## Limitations & Future Work

- Geometric accuracy is constrained by the resolution of tetrahedral grids, occasionally losing fine accessories (e.g., earrings, necklaces).
- The model only simulates pose-driven deformations, ignoring body-clothing contact dynamics and complex movements of loose clothing or long hair.
- The approach relies on front-and-back dual-view inputs; single-view scenarios require additional view completion strategies.
- The training dataset is private (Meta Reality Labs internal dome data), limiting system reproducibility.
- Generating canonical pseudo-GTs requires 20 minutes of optimization per frame, which limits the scale expansion of training data.

## Related Work & Insights

- **vs. ARCH++**: ARCH++ is also a feedforward method but uses fixed skinning weights and handcrafted spatial encodings, leading to poor animation quality and coarse geometry. FRESA achieves order-of-magnitude improvements using personalized skinning and explicit canonicalization.
- **vs. PuzzleAvatar**: PuzzleAvatar utilizes diffusion models with SDS losses for avatar generation. Although quality is acceptable, it requires 3 hours of computation and risks losing facial identity. FRESA preserves personalized details while being 600x faster.
- **vs. Vid2Avatar**: Based on multi-view video optimization, Vid2Avatar offers good quality but is extremely time-consuming. Powered by a general prior, FRESA outperforms optimization-based methods in quality via a feedforward pipeline.

## Rating

- Novelty: ⭐⭐⭐⭐ The feedforward framework for joint inference of geometry, skinning, and deformation is highly novel, and the decoupled multi-stage training is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across multiple datasets, with zero-shot generalization on mobile photos and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, complete mathematical formulations, and high-quality figures.
- Value: ⭐⭐⭐⭐⭐ Achieves simultaneous breakthroughs in both speed and quality, offering direct practical value to the virtual avatar industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Few-Shot Personalized Scanpath Prediction](few-shot_personalized_scanpath_prediction.md)
- [\[CVPR 2025\] 3D Face Reconstruction From Radar Images](3d_face_reconstruction_from_radar_images.md)
- [\[CVPR 2025\] PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [\[CVPR 2025\] ShowMak3r++: Compositional Entertainment Video Reconstruction](showmak3r_compositional_tv_show_reconstruction.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)

</div>

<!-- RELATED:END -->
