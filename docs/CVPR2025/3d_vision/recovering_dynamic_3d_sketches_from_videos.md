---
title: >-
  [Paper Note] Recovering Dynamic 3D Sketches from Videos
description: >-
  [CVPR 2025][3D Vision][Dynamic Sketch] Liv3Stroke proposes the first method to extract dynamic 3D sketches from videos. It abstracts object motion using a set of deformable 3D Bézier curves, achieving viewpoint-consistent motion sketch reconstruction by learning point cloud motion guidance and stroke-by-stroke deformation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dynamic Sketch"
  - "3D Motion Abstraction"
  - "Bézier Curve"
  - "Differentiable Rendering"
  - "Motion Representation"
date: 2026-05-08
content_hash: 5f59782c02fb6ed4
---

# Recovering Dynamic 3D Sketches from Videos

**Conference**: CVPR 2025  
**arXiv**: [2503.20321](https://arxiv.org/abs/2503.20321)  
**Code**: [https://jaeah.me/liv3stroke_web](https://jaeah.me/liv3stroke_web) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Dynamic Sketch, 3D Motion Abstraction, Bézier Curve, Differentiable Rendering, Motion Representation

## TL;DR

Liv3Stroke proposes the first method to extract dynamic 3D sketches from videos. It abstracts object motion using a set of deformable 3D Bézier curves, achieving viewpoint-consistent motion sketch reconstruction by learning point cloud motion guidance and stroke-by-stroke deformation.

## Background & Motivation

**Background**: Understanding 3D motion from videos is a core problem in computer vision. Existing methods either represent motion using unstructured dense motion vectors (e.g., optical flow fields) or rely on pre-defined articulated templates (e.g., SMPL). Although neural radiance fields for dynamic scenes (D-NeRF, 4DGS) can achieve photorealistic reconstruction, they solve for high-dimensional, complex variables and are prone to errors under appearance changes.

**Limitations of Prior Work**: Dense motion representations are computationally expensive and make it difficult to directly analyze the core structure of motion; template-based methods are only applicable to specific object categories; dynamic NeRF/3DGS methods pursue pixel-level accurate reconstruction but fail to provide an abstract understanding of motion.

**Key Challenge**: The need for a compact intermediate representation that can also express motion of arbitrary topologies—one that is neither as redundant as dense fields nor as restricted to specific objects as templates.

**Goal**: Abstract the core motion features of objects in videos using a small set of deformable 3D curves (strokes).

**Key Insight**: Inspired by sketching—humans can convey the core information of a scene with just a few simple lines. Recent works like CLIPasso have demonstrated that abstract sketches can be automatically generated from images. The authors extend this concept to "abstracting motion with deformable 3D strokes."

**Core Idea**: Define the sketch as a set of deformable 3D Bézier curves, representing motion through the translation/rotation of each curve and control point adjustments, and align the sketch with the ground-truth frames in a latent space using perceptual losses (LPIPS + CLIP).

## Method

### Overall Architecture

The pipeline of Liv3Stroke is divided into two stages: (1) Learning 3D motion guidance from video frames—obtaining a coarse 3D motion layout using point clouds and deformation MLP; (2) Initializing 3D curves and learning their deformation based on the motion guidance—fitting the position and shape of each stroke at each timestep through a rotation/translation MLP and a control point adjustment MLP. The final output is a set of 3D Bézier curves for each frame, which can be rendered into 2D sketches from any viewpoint.

### Key Designs

1. **3D Motion Guidance Learning**:

    - Function: Extract a coarse 3D motion field from video frames to provide guidance for subsequent stroke initialization and deformation.
    - Mechanism: Represent the scene with 10k 3D points and use a deformation MLP network (with positional encoding + temporal encoding as input) to predict the displacement of each point at each timestep. Crucially, the LPIPS perceptual loss (rather than pixel-level L1/L2 loss) is used to align the rendered point cloud maps with the ground-truth frames, as point clouds render grayscale images instead of natural images. Velocity continuity regularization and rigidity regularization are added to ensure smooth motion.
    - Design Motivation: LPIPS loss captures structural differences rather than pixel details. Experiments demonstrate that it preserves object boundaries and shape integrity better than L1/L2 loss. This stage does not aim for exact reconstruction but rather to obtain coarse information of "where the motion is and how it moves."

2. **Dynamic 3D Stroke Deformation**:

    - Function: Represent motion as a combination of rigid transformation and shape deformation for each Bézier curve.
    - Mechanism: The motion of each stroke is decomposed into two levels: (1) rotation $R_i^t$ and translation $T_i^t$ per stroke (predicted by two MLPs $\mathcal{M}_R$ and $\mathcal{M}_T$), and (2) local displacement of each control point $\Delta p_i^j$ (predicted by the MLP $\mathcal{M}_L$). A coarse-to-fine strategy is adopted: global rotation/translation is learned at low resolution first, followed by control point fine-tuning at high resolution. $\mathcal{M}_T$ is initialized with the pre-trained weights from the motion guidance stage.
    - Design Motivation: Separating rigid motion from non-rigid deformation allows better control over different levels of motion—global displacement is handled by the rotation/translation MLP, while local shape changes (such as the details of a bird flapping its wings) are handled by control point adjustments.

3. **Perceptual Loss-Driven Sketch Optimization**:

    - Function: Align the rendered sketch with the input video frames in a semantic feature space.
    - Mechanism: Integrate LPIPS loss (capturing structural similarity) and CLIP distance loss (capturing semantic similarity): $\mathcal{L}_{frame}^s = \lambda_s \rho(LPIPS(\mathcal{I}, \mathcal{S})) + dist(CLIP(\mathcal{I}), CLIP(\mathcal{S}))$. Add velocity continuity regularization $\mathcal{L}_{temp}^s$ and motion magnitude regularization $\mathcal{L}_{reg}$. A correction function $\xi(\mathbf{x})$ is also used to suppress minor movements during static periods.
    - Design Motivation: Sketches are abstract representations and cannot be aligned with natural images in pixel space; they must be aligned in semantic/perceptual space. CLIP loss ensures the sketch maintains semantic correctness, while LPIPS loss ensures structural alignment.

### Loss & Training

The total loss for the motion guidance stage is $\mathcal{L}_{guidance} = 0.1 \cdot \mathcal{L}_{frame}^g + 0.05 \cdot \mathcal{L}_{temp}^g + 10^{-4} \cdot \mathcal{L}_{rigid}$. The total loss for the sketch generation stage is $\mathcal{L}_{sketch} = \mathcal{L}_{frame}^s + \mathcal{L}_{temp}^s + \mathcal{L}_{reg}$. Using the Adam optimizer, all MLP networks are trained independently.

## Key Experimental Results

### Main Results

| Method | Structural Alignment (Novel View↑) | Structural Alignment (Fixed View↑) | Motion Prompt Similarity (Novel View↑) | Motion Prompt Similarity (Fixed View↑) |
|------|-----------------|-----------------|---------------------|---------------------|
| CLIPasso | 0.760±0.107 | 0.740±0.127 | 0.659±0.007 | 0.664±0.011 |
| Sketch Video Syn. | 0.663±0.115 | 0.657±0.135 | 0.654±0.011 | 0.658±0.011 |
| Suggestive Contours | 0.784±0.102 | 0.750±0.119 | 0.661±0.013 | 0.656±0.016 |
| **Liv3Stroke** | 0.693±0.096 | 0.683±0.108 | **0.656±0.006** | **0.656±0.008** |

3D motion guidance accuracy evaluation:

| Method | Chamfer Distance↓ | Motion Velocity Error↓ |
|------|-------------|-------------|
| 4DGS | 0.205±0.046 | 4.60±3.00 |
| DG-Mesh | 0.277±0.059 | 4.10±2.70 |
| **Liv3Stroke** | 0.252±0.049 | **4.16±2.34** |

### Ablation Study

| Configuration | Chamfer Distance↓ | Motion Velocity Error↓ |
|------|-------------|-------------|
| w/o $\mathcal{L}_{temp}^g$ | 0.258±0.043 | 6.81±5.41 |
| w/o $\mathcal{L}_{rigid}$ | 0.253±0.048 | 5.45±3.74 |
| Replace $\mathcal{L}_{rigid}$ with L2 | 0.253±0.045 | 4.61±2.94 |
| Full Model | 0.252±0.049 | 4.16±2.34 |

### Key Findings

- Liv3Stroke exhibits the smallest standard deviation across all indicators, indicating that this method is the most stable across different scenes and viewpoints.
- Directly optimizing strokes (without guidance) leads to unstructured, broken line segments; omitting guidance in the coarse stage results in the loss of global structure (e.g., a bird's wings).
- Velocity continuity regularization is crucial for motion quality—removing it causes the motion velocity error to surge from 4.16 to 6.81.
- Although Liv3Stroke does not aim for photorealistic reconstruction, its point cloud motion accuracy is comparable to DG-Mesh, which specializes in precise mesh reconstruction.

## Highlights & Insights

- **The philosophy of "representing complexity with simplicity"** is highly impressive—using only dozens of line segments to capture the essence of complex 3D motion. This is not only a technical innovation but also poses a deeper question: "What is the simplest form in which motion can be expressed?"
- **Using LPIPS loss for structural alignment of non-natural images** is a reusable trick: in any scenario where abstract representations need to be aligned with natural images, perceptual loss outperforms pixel loss.
- The separation strategy of rigid motion + control point deformation can be transferred to any scenario that requires representing the motion of deformable objects, such as tracking flexible objects in robotic manipulation.

## Limitations & Future Work

- It only considers viewpoint-independent strokes and cannot render viewpoint-dependent silhouettes (such as the silhouette of a sphere).
- The number of strokes must be manually set by the user, lacking an adaptive mechanism to determine the optimal number of strokes.
- It performs well for motion under pure translation, but highly complex non-rigid deformations (such as fluids) may exceed the expression capability of Bézier curves.
- Future work could introduce physical attributes of strokes (such as stiffness, mass) to achieve stroke-based physical simulation control.

## Related Work & Insights

- **vs CLIPasso**: CLIPasso only processes single-frame 2D sketches, resulting in poor temporal consistency. Liv3Stroke defines strokes in 3D space, naturally ensuring multi-view consistency.
- **vs Sketch Video Synthesis**: This method is based on a layered neural atlas, which cannot handle large movements and operates entirely in 2D space. Liv3Stroke scales up to 3D and handles large motions using deformation MLPs.
- **vs 3Doodle/EMAP**: These methods process 3D sketches of static scenes. Liv3Stroke is the first to extend this to dynamic scenes, with the key difference being the introduction of the deformation MLP framework.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first dynamic 3D sketch reconstruction method; the problem definition itself is very novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient quantitative and qualitative evaluations, but lacks user studies to evaluate the comprehensibility of the sketches.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of methodology.
- Value: ⭐⭐⭐⭐ Pioneers a new direction in motion abstraction, though practical application scenarios require further exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)
- [\[CVPR 2025\] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity](freegave_3d_physics_learning_from_dynamic_videos_by_gaussian_velocity.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[CVPR 2026\] Recovering Physically Plausible Human-Object Interactions from Monocular Videos](../../CVPR2026/3d_vision/recovering_physically_plausible_human-object_interactions_from_monocular_videos.md)

</div>

<!-- RELATED:END -->
