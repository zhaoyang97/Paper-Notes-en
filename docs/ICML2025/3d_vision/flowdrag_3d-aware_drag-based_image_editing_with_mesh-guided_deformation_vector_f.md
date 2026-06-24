---
title: >-
  [Paper Note] FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields
description: >-
  [ICML 2025 (Spotlight)][3D Vision][drag-based editing] Proposed FlowDrag, which constructs a 3D mesh from an image and generates continuous 2D vector flow fields through progressive SR-ARAP deformation. This injects global geometric priors into the motion supervision process of diffusion models, leading to comprehensive state-of-the-art performance on DragBench (MD=22.88) and the newly proposed VFD-Bench (PSNR=18.55, 1-LPIPS=0.82, MD=28.23).
tags:
  - "ICML 2025 (Spotlight)"
  - "3D Vision"
  - "drag-based editing"
  - "3D mesh deformation"
  - "SR-ARAP"
  - "vector flow field"
  - "geometric consistency"
date: 2026-05-08
content_hash: 756a272249273b86
---

# FlowDrag: 3D-aware Drag-based Image Editing with Mesh-guided Deformation Vector Flow Fields

**Conference**: ICML 2025 (Spotlight)  
**arXiv**: [2507.08285](https://arxiv.org/abs/2507.08285)  
**Code**: None  
**Area**: 3D Vision / Image Editing  
**Keywords**: drag-based editing, 3D mesh deformation, SR-ARAP, vector flow field, geometric consistency

## TL;DR

Proposed FlowDrag, which constructs a 3D mesh from an image and generates continuous 2D vector flow fields through progressive SR-ARAP deformation. This injects global geometric priors into the motion supervision process of diffusion models, leading to comprehensive state-of-the-art performance on DragBench (MD=22.88) and the newly proposed VFD-Bench (PSNR=18.55, 1-LPIPS=0.82, MD=28.23).

## Background & Motivation

**Drag-based image editing** precisely controls object transformations through user-defined dragging points (handle→target), and has achieved significant progress in the diffusion model era. However, **existing methods suffer from severe geometric inconsistency**: they only optimize handle point features to move toward target points, completely ignoring the global geometric structure of the object.

**Key Challenge**: When editing involves rigid transformations (rotation, translation, pose changes), various parts of the object should move coordinately. However, methods like DragDiffusion and GoodDrag only focus on local matching of sparse control points, leading to structural tearing and artifacts—a typical case is the deformation of the arm/torch when rotating the Statue of Liberty, or the hat and hand detaching when rotating a face.

**Core Idea**: Introduce geometric priors from 3D mesh deformation into 2D drag editing—by constructing a 3D mesh, performing rigid deformation, and projecting it onto 2D vector flow fields, the pixels across the entire editing region receive geometrically consistent displacement guidance, rather than relying solely on a few user-defined drag points. Concurrently, the first drag editing benchmark with ground truth, VFD-Bench, is proposed to resolve the fundamental issue that existing benchmarks like DragBench cannot evaluate editing quality.

## Method

### Overall Architecture

The pipeline of FlowDrag consists of three phases:
1. **3D Mesh Generation**: From the input image, a 3D mesh $M=(V,F)$ is generated via depth estimation (Marigold) or image-to-3D diffusion models (Hunyuan3D 2.0).
2. **Progressive SR-ARAP Deformation + 2D Vector Flow Field Generation**: Performs rigidity-preserving deformation on the mesh based on user dragging points, and projects the difference between the original and deformed meshes as a 2D displacement field $\Phi$.
3. **Vector Flow Field-guided Drag Editing**: Integrates $\Phi$ into the motion supervision and point tracking of the UNet denoising process, while injecting layout features of the deformed mesh.

### Key Designs

1. **Progressive SR-ARAP Mesh Deformation**:
    - **Function**: Moves handle vertices in the 3D mesh to target positions while preserving local rigidity.
    - **Mechanism**: Adds a rotation consistency term and an inter-step smoothness term to the classical ARAP energy function. The SR-ARAP energy is formulated as $E_{SR\text{-}ARAP}(M) = E_{ARAP}(M) + \alpha \sum_{i \in V} \sum_{j \in N(i)} \|R_i - R_j\|^2$. The vertices are moved progressively in $K$ steps: $v_h^{(k+1)} = v_h^{(k)} + \lambda(v_t - v_h^{(k)})$, with an Inter-Step Smoothness term $\beta \sum_{i} \|\hat{v}_i^{(k+1)} - \hat{v}_i^{(k)}\|^2$ to prevent abrupt changes.
    - **Design Motivation**: Direct large-scale displacement of vertices leads to local distortion and sub-optimal convergence; the rotation consistency term ensures smooth rotation transitions between adjacent vertices.

2. **2D Vector Flow Field Generation and Sampling**:
    - **Function**: Projects the 3D mesh deformation result to a 2D continuous displacement field, replacing the sparse discrete drag points used in prior work.
    - **Mechanism**: Calculates the difference between the 2D projections of the original and deformed meshes $\Phi = \{(\Delta x_i, \Delta y_i)\}$. Within the editing mask, $N \times N$ ($N=20$) grid candidate vectors are uniformly sampled. Then, 5-30 most representative vectors are chosen via magnitude-based sampling for motion supervision.
    - **Design Motivation**: A continuous displacement field covers the entire editing region, providing richer global geometric guidance than discrete drag points.

3. **Layout Feature Injection**:
    - **Function**: Injects the attention features, obtained from DDIM inversion of the deformed mesh's 2D projection $\pi(\hat{M})$, into the main editing branch during early denoising steps.
    - **Mechanism**: Leverages the property of early timesteps establishing structural outlines to inject layout information only before $t' = 30$, avoiding over-constraining the details.
    - **Design Motivation**: The vector flow field provides point-level displacement guidance, while layout injection provides global structural context, complementing each other.

### Loss & Training

Based on the Stable Diffusion 1.5 pre-trained model, fine-tuning is performed using LoRA (rank=16) for 200 steps. DDIM Inversion is done up to step 38 (75% of 50 steps), and layout feature injection occurs at $t'=30$. The motion supervision loss is expanded to constrain the sampled vectors plus a regularization term outside the mask. The vector flow field uses magnitude-based sampling, which outperforms uniform sub-sampling (MD 28.23 vs 30.21).

## Key Experimental Results

### Main Results — DragBench (205 images, 349 drag pairs)

| Method | 1-LPIPS (IF)↑ | MD↓ |
|------|-------------|-----|
| DiffEditor | 0.89 | 28.46 |
| DragDiffusion | 0.89 | 33.70 |
| DragNoise | 0.63 | 33.41 |
| FreeDrag | 0.70 | 35.00 |
| GoodDrag | 0.86 | 22.96 |
| **FlowDrag** | **0.82** | **22.88** |

### Main Results — VFD-Bench (250 images, with GT)

| Method | PSNR↑ | 1-LPIPS↑ | MD↓ |
|------|-------|---------|-----|
| DiffEditor | 16.23 | 0.67 | 43.35 |
| DragDiffusion | 17.55 | 0.76 | 38.42 |
| DragNoise | 16.58 | 0.71 | 40.52 |
| FreeDrag | 17.38 | 0.72 | 42.78 |
| GoodDrag | 18.14 | 0.79 | 35.31 |
| **FlowDrag** | **18.55** | **0.82** | **28.23** |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| β=0.2 / 0.4 / 0.6 / **0.8** / 1.0 | MELR: 0.87/0.88/0.92/**0.94**/0.93 | β=0.8 yields the best rigidity preservation |
| mARAPError | 17.24/14.91/12.56/**10.12**/11.60 | β=0.8 minimizes local twisting |
| Magnitude vs. Uniform Sampling | MD: **28.23** vs 30.21, 1-LPIPS: **0.82** vs 0.80 | Magnitude sampling is more effective |
| Number of vectors = 10 | Highest PSNR and 1-LPIPS | 10 vectors yield the optimal balance |

### Key Findings

- FlowDrag performs best across all three metrics (PSNR/1-LPIPS/MD) on VFD-Bench, validating the value of the geometric prior.
- On DragBench, DiffEditor achieves the highest IF but has worse MD—because its editing magnitude is small, resulting in high similarity to the original image.
- β=0.8 is optimal: it balances inter-step smoothness with deformation flexibility, yielding an MELR closest to 1.0 (indicating no distortion).
- Mesh deformation takes about 5 seconds per sample on average, which is acceptable for interactive editing.
- User study (25 participants $\times$ 50 images): FlowDrag received the highest ratings in both drag accuracy and image quality.

## Highlights & Insights

- **Systematic introduction of 3D geometric priors**: Moving from "discrete point matching" to "global continuous displacement field guidance" represents a paradigm shift in drag-based editing.
- **VFD-Bench fills the evaluation gap**: It constructs 250 evaluation pairs with ground truth derived from continuous video frames (Animal 140/Human 65/Object 45), establishing the first drag-based editing benchmark with ground truth.
- **Progressive Deformation + Inter-Step Smoothness**: Simple and effective solution to prevent mesh collapse under large-magnitude dragging.
- Robust to mesh simplification and diffusion sampling step counts (stable with DepthMesh reduction ratio 0.001-1.0, and stable with DiffMesh 10-40 steps).
- ICML 2025 Spotlight; reviewers recognized its contribution to geometrically consistent editing.

## Limitations & Future Work

- The feasible drag distance is bounded by the stability range of mesh deformation; extremely large edits may fail.
- Primarily supports rigid edits (rotation/translation/pose), and is not applicable to non-rigid editing (scaling/bending/content creation).
- Projecting 3D meshes to 2D discards depth information, and SD 1.5 lacks explicit 3D understanding, imposing an upper limit on geometry preservation.
- Depth estimation quality directly affects DepthMesh accuracy; scenes with heavy occlusion may fail.
- Future work can explore 3D-aware or video diffusion models to better capture object dynamics and 3D structures.

## Related Work & Insights

- **DragGAN** (Pan et al., 2023): Pioneer of point-based drag editing, though with limited GAN capability.
- **DragDiffusion** (Shi et al., 2024): Diffusion-based drag editing, optimizing specific timestep DDIM latents.
- **GoodDrag** (Zhang et al., 2024): Alternating drag and denoising (AlDD) to reduce accumulated errors.
- **FreeDrag** (Ling et al., 2023): Tracking-free point dragging.
- **ARAP/SR-ARAP** (Sorkine & Alexa, 2007; Levi & Gotsman, 2014): Classical rigidity-preserving mesh deformation.
- **Marigold** (Ke et al., 2024): Diffusion-based monocular depth estimation.
- Insight: The 3D geometric prior can be extended to other 2D editing tasks such as video editing and style transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ The approach of transitioning from 3D mesh deformation to 2D vector flow fields is novel. Each component is based on existing techniques, but the integrated pipeline is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Employs two benchmarks, a user study, multi-dimensional ablation, and sensitivity analyses. VFD-Bench successfully fills the evaluation gap.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with an intuitive pipeline diagram; step-by-step description makes the methodology easy to follow.
- Value: ⭐⭐⭐⭐ Spotlight paper. Bringing 3D geometric priors into drag editing is a highly promising direction, and VFD-Bench offers long-term utility to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reference-Based 3D-Aware Image Editing with Triplanes](../../CVPR2025/3d_vision/reference-based_3d-aware_image_editing_with_triplanes.md)
- [\[ICCV 2025\] Image-Guided Shape-from-Template Using Mesh Inextensibility Constraints](../../ICCV2025/3d_vision/image-guided_shape-from-template_using_mesh_inextensibility_constraints.md)
- [\[ICCV 2025\] 3D Mesh Editing using Masked LRMs](../../ICCV2025/3d_vision/3d_mesh_editing_using_masked_lrms.md)
- [\[CVPR 2025\] GenVDM: Generating Vector Displacement Maps From a Single Image](../../CVPR2025/3d_vision/genvdm_generating_vector_displacement_maps_from_a_single_image.md)
- [\[CVPR 2026\] ObjectMorpher: 3D-Aware Image Editing via Deformable 3DGS](../../CVPR2026/3d_vision/objectmorpher_3d-aware_image_editing_via_deformable_3dgs.md)

</div>

<!-- RELATED:END -->
