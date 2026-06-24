---
title: >-
  [Paper Note] 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency
description: >-
  [CVPR 2025][3D Vision][3D Inpainting] This paper proposes 3DGIC, a framework that achieves object removal and inpainting in 3D Gaussian Splatting scenes through **depth-guided cross-view consistent inpainting**. By leveraging rendered depth maps, it projects background pixels visible from other views onto the masked region to refine the inpainting mask. Then, 2D inpainting results from a reference view are projected onto 3D space to constrain cross-view consistency for other…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Inpainting"
  - "Object Removal"
  - "Cross-View Consistency"
  - "Depth-Guided Mask"
  - "3DGS"
date: 2026-05-08
content_hash: c149159e921c6990
---

# 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency

**Conference**: CVPR 2025  
**arXiv**: [2502.11801](https://arxiv.org/abs/2502.11801)  
**Code**: [https://peterjohnsonhuang.github.io/3dgic-pages](https://peterjohnsonhuang.github.io/3dgic-pages)  
**Area**: 3D Vision / Scene Editing / 3D Gaussian Splatting  
**Keywords**: 3D Inpainting, Object Removal, Cross-View Consistency, Depth-Guided Mask, 3DGS  

## TL;DR
This paper proposes 3DGIC, a framework that achieves object removal and inpainting in 3D Gaussian Splatting scenes through **depth-guided cross-view consistent inpainting**. By leveraging rendered depth maps, it projects background pixels visible from other views onto the masked region to refine the inpainting mask. Then, 2D inpainting results from a reference view are projected onto 3D space to constrain cross-view consistency for other views. The proposed method outperforms existing approaches in FID and LPIPS on the SPIn-NeRF dataset.

## Background & Motivation

### Background

**Background**: 3D scene inpainting (filling holes after object removal) is a core requirement for VR/AR editing. The main challenge lies in **cross-view consistency**—applying 2D inpainting to individual multi-view images independently often yields inconsistent results across views. Existing methods face two main limitations: (1) they directly use Segment Anything (SAM)-generated object masks as inpainting masks, which may include background pixels that are actually visible from other viewpoints. Passing these known regions to a 2D inpainter introduces inconsistent content. (2) They perform 2D inpainting on each view independently, lacking geometric constraints across viewpoints.

### Solution

**Goal**: How to achieve **high-fidelity, multi-view consistent** 3D inpainting after object removal in 3D Gaussian Splatting (3DGS) scenes? The key challenges are: (1) accurately determining the "true region that requires inpainting" in each view (excluding backgrounds that can be observed indirectly from other views); (2) ensuring that the inpainted content remains geometrically consistent across all viewpoints.

## Method

### Overall Architecture
The proposed framework consists of a two-stage pipeline:
1. **Depth-Guided Inpainting Mask Inference**: Background pixels from each view are projected into other views using rendered depth maps to progressively shrink the inpainting mask, ensuring it only contains regions that are occluded from all perspectives.
2. **Inpainting-Guided 3DGS Refinement**: The view with the largest refined inpainting mask is selected as the reference view. After performing 2D inpainting on this view, its results are projected into 3D space to provide cross-view consistent supervision for other views.

### Key Designs
1. **Depth-Guided Inpainting Mask Inference**: For a mask $M_1$ of view $\xi_1$, background pixels $I_2^B$ from another view $\xi_2$ are projected into 3D space using depth $D_2$, and then re-projected back to $\xi_1$. Pixels falling inside $M_1$ indicate that "this region can see the background from $\xi_2$", and are thus removed from the mask. After traversing all views, the final refined mask $M_1'$ only contains the true occluded regions that are invisible across all viewpoints. This is a **deterministic** process and does not require training. During projection, a z-buffer check is utilized to avoid misprojection of occluded points.

2. **Reference View Selection**: The view with the largest refined mask is chosen as the reference view, as its 2D inpainting can cover the largest 3D space, providing the maximum amount of consistency information for other views.

3. **Cross-View Consistency Loss**: The 2D inpainting result of the reference view is projected onto the 3D point cloud $P_1$, and then re-projected back to other views $\xi_k$ to serve as supervision for the inpainted regions. The LPIPS perceptual loss is used to measure the discrepancy between the rendered image and the projected result: $$\mathcal{L}_{cross} = \sum_k \mathcal{L}_{LPIPS}(I_k', I_k^P)$$

4. **Joint Inpainting of Color and Depth**: Instead of inpainting only RGB images, depth maps are inpainted concurrently. When using a Latent Diffusion Model (LDM), RGB and depth maps are concatenated into a single 1024×1024 image to be inpainted simultaneously, ensuring joint geometric and texture consistency.

5. **Backbone 3DGS**: Combines Gaussian Grouping (semantic segmentation to automatically generate object masks) and Relightable Gaussians (for more reliable depth estimation).

### Loss & Training
$$\mathcal{L}_{inpaint} = \mathcal{L}_{rgb} + \mathcal{L}_{depth} + \mathcal{L}_{cross}$$
- $\mathcal{L}_{rgb} = \|I_1' - I_1^{In}\|_1 + \mathcal{L}_{SSIM}$
- $\mathcal{L}_{depth} = \|D_1' - D_1^{In}\|_1$
- 2D inpainting is updated every 500 iterations (using a progressive DDIM step reduction strategy).
- Optimized on an RTX 3090, taking 5000 iterations per scene in PyTorch.

## Key Experimental Results

### SPIn-NeRF Dataset

| Method | Representation | 2D Inpainter | FID↓ | m-FID↓ | LPIPS↓ | m-LPIPS↓ |
|------|------|----------|------|--------|--------|---------|
| SPIn-NeRF | NeRF | LAMA | 49.6 | 153.4 | 0.31 | 0.053 |
| MVIP-NeRF | NeRF | LDM | 50.5 | 173.4 | 0.31 | 0.050 |
| Gaussian Grouping | 3DGS | LAMA | 44.7 | 132.5 | 0.30 | 0.037 |
| MALD-NeRF | NeRF | LDM | 44.9 | 113.5 | 0.26 | 0.031 |
| GScream | 3DGS | LDM | 38.6 | 101.6 | 0.28 | 0.033 |
| **3DGIC (LAMA)** | 3DGS | LAMA | 41.7 | 102.4 | 0.28 | 0.032 |
| **3DGIC (LDM)** | 3DGS | LDM | **36.4** | **96.3** | **0.26** | **0.028** |

The LDM-based version achieves the best performance across all four metrics. The LAMA-based version (non-diffusion) also outperforms MVIP-NeRF and MALD-NeRF, both of which utilize LDM.

### Ablation Study (Bear Scene)
- **Original Mask Only + No Cross-View Consistency**: Results are blurry and inconsistent.
- **Original Mask + Cross-View Consistency**: Improves fidelity but alters the visible background.
- **Depth-Guided Mask + No Cross-View Consistency**: Preserves the background well but results in blurry inpainting regions.
- **Full Method**: Preserves background details while achieving consistent inpainting.

### Qualitative Results
- Background details (e.g., wall sockets on a desk) are preserved across 10 scenes in the SPIn-NeRF dataset, demonstrating cross-view consistency in inpainted regions.
- Performs well on 360° scenes (Figurines, Counter, Kitchen, Bear), robustly handling wide viewpoint variations.
- The refined mask size is reduced by 30-60% on average, indicating that a substantial portion of the "to-be-inpainted" regions is actually visible from other viewpoints.

### Efficiency Analysis
- The mask refinement stage involves only geometric computations, resulting in negligible overhead (<1s per scene).
- The overall optimization takes approximately 15 minutes per scene for 5000 iterations on an RTX 3090, which is comparable to GScream.

## Highlights & Insights
- **Key Insight of Mask Refinement**: Object masks generated by SAM contain backgrounds that are visible from other views. By "reclaiming" these regions through depth projection, the inpainter focuses only on areas that genuinely need to be filled, directly improving consistency from the source.
- **Combining Deterministic and Learned Processes**: Mask refinement is a deterministic geometric operation (no training required), while 3DGS optimization is a learning process—the two complement each other.
- **Decoupled from 2D Inpainters**: The framework is compatible with various inpainters such as LAMA (non-diffusion) and LDM (diffusion), outperforming strong baselines even when using a weaker inpainter.
- **Joint Depth-Color Inpainting**: Inpainting RGB and depth maps concurrently guarantees consistent geometry.

## Limitations & Future Work
- Relies on the accuracy of rendered depth maps—unreliable depth under sparse views leads to failures in mask refinement.
- SAM classification/segmentation may be inaccurate for small objects.
- Reference view selection relies on a simple heuristic (the largest mask), without considering inpainting difficulty or texture complexity.
- Updating 2D inpainting results every 500 steps increases computational overhead (each DDIM sampling takes ~10s).
- Scenarios with concurrent multi-object removal are not discussed—depth projections may become ambiguous when masks overlap.
- Outdoor large-scale scenes exhibit noisier depth estimates, leaving the effectiveness of mask refinement to be verified.
- Lacks user studies for perceptual quality evaluation.

## Related Work & Insights
- **GScream**: Uses reference-view depth prediction and cross-view feature consistency. 3DGIC introduces mask refinement to avoid modifying visible backgrounds.
- **MALD-NeRF**: Leverages LoRA fine-tuning of a diffusion model to perform scene-specific inpainting. 3DGIC does not require fine-tuning the diffusion model, making it more lightweight.
- **Gaussian Grouping**: Highlights inpainted areas using GroundedSAM with a "black blurry hole" prompt, which often causes false positives. 3DGIC deterministically refines masks via depth projection.
- **SPIn-NeRF**: Demands manual mask annotations, whereas 3DGIC is fully automatic (using SAM + depth guidance).

## Related Work & Insights
- The concept of depth-guided mask refinement can be extended to other task formats requiring cross-view consistent editing (e.g., texture editing, relighting).
- The deterministic process of "recovering occluded regions from other views" is conceptually similar to traditional image-based rendering.
- Concurrently back-propagating and inpainting RGB and depth in a concatenated manner is simple yet effective, bypassing the need for a separate depth inpainting model.
- Mask refinement essentially serves as pre-processing for cross-view information fusion, which can be extended to video inpainting via inter-frame optical flow.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of depth-guided mask refinement is simple yet effective, and the cross-view projection constraint is logically sound.
- Experimental Thoroughness: ⭐⭐⭐⭐ SPIn-NeRF + multiple 360° scenes + ablation study + qualitative comparison.
- Writing Quality: ⭐⭐⭐⭐ Problem formulations are clear, and the method step diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Object removal in 3D scenes is a highly practical demand, and the mask refinement approach offers broad generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement](imfine_3d_inpainting_via_geometry-guided_multi-view_refinement.md)
- [\[CVPR 2025\] GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](geal_generalizable_3d_affordance_learning_with_cross-modal_consistency.md)
- [\[CVPR 2026\] No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency](../../CVPR2026/3d_vision/no_calibration_no_depth_no_problem_cross-sensor_view_synthesis_with_3d_consisten.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
