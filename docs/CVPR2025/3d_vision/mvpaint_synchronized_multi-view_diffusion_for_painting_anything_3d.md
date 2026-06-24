---
title: >-
  [Paper Note] MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D
description: >-
  [CVPR 2025][3D Vision][3D texture generation] MVPaint proposes a three-stage 3D texture generation framework consisting of Synchronized Multi-View Generation (SMG) + Spatial-Aware 3D Inpainting (S3I) + UV Refinement (UVR). By synchronizing multiple views in the image domain rather than the latent domain, performing inpainting in the 3D point cloud space rather than the UV space, and utilizing a spatial-aware seam smoothing algorithm, it comprehensively outperforms existing st…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D texture generation"
  - "multi-view consistency"
  - "UV refinement"
  - "3D inpainting"
  - "diffusion models"
date: 2026-05-08
content_hash: fe9edff32b95723b
---

# MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D

**Conference**: CVPR 2025  
**arXiv**: [2411.02336](https://arxiv.org/abs/2411.02336)  
**Code**: [http://mvpaint.github.io](http://mvpaint.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D texture generation, multi-view consistency, UV refinement, 3D inpainting, diffusion models

## TL;DR
MVPaint proposes a three-stage 3D texture generation framework consisting of Synchronized Multi-View Generation (SMG) + Spatial-Aware 3D Inpainting (S3I) + UV Refinement (UVR). By synchronizing multiple views in the image domain rather than the latent domain, performing inpainting in the 3D point cloud space rather than the UV space, and utilizing a spatial-aware seam smoothing algorithm, it comprehensively outperforms existing state-of-the-art (SOTA) methods on both the Objaverse and GSO T2T benchmarks.

## Background & Motivation

1. **Background**: 3D texture generation (Text-to-Texture, T2T) is a crucial step in 3D asset production. Existing methods are mainly divided into two categories: (a) iterative methods (TEXTure, Text2Tex) that sequentially render depth maps and generate textures using diffusion models; (b) synchronization methods (SyncMVD) that synchronize UV-space latents during the multi-view DDIM process.

2. **Limitations of Prior Work**: (a) **Multi-view inconsistency**: Independent multi-view generation causes local style/texture discontinuities; (b) **Janus problem**: The attention reuse in SyncMVD is limited to adjacent views, frequently generating multi-faced artifacts; (c) **Heavy reliance on UV layout**: Paint3D and Meta 3D TextureGen rely on continuous UV unwrapping, causing texture discontinuities when the UV atlas is randomly packed into UV images; (d) **Oversmoothed textures** lacking fine details.

3. **Key Challenge**: The latent space (e.g., 32×32) has a resolution too low to establish a precise mapping with the UV space, resulting in poor synchronization in the latent domain. Meanwhile, UV unwrapping maps adjacent 3D regions to non-adjacent 2D regions, meaning that direct inpainting in the UV space cannot guarantee 3D spatial consistency.

4. **Goal**: (a) How to generate seamless, consistent 3D textures without relying on the quality of UV unwrapping? (b) How to efficiently inpaint textures for unobserved regions? (c) How to repair UV seams while improving resolution?

5. **Key Insight**: The authors propose to synchronize multiple views in the decoded image domain (256×256) rather than the latent domain (32×32), perform inpainting in the 3D point cloud space instead of the UV space, and implement seam repair within the 3D local neighborhood structure.

6. **Core Idea**: To address the three major challenges of texture generation (consistency, completeness, and smoothness) by solving each of them in their most suitable representation spaces: image-domain synchronization, 3D-space inpainting, and 3D neighborhood smoothing.

## Method

### Overall Architecture
The framework consists of three stages: Stage 1 (SMG) uses a ControlNet-guided MVDream model to generate low-resolution multi-view images, synchronizing multiple views via the UV space in the image domain during the denoising process, and then refining them to 1K using an SDXL I2I model; Stage 2 (S3I) projects the multi-view images onto the UV space to obtain an incomplete UV map, reformulates the unpainted areas as a 3D point cloud inpainting problem, and completes it iteratively via spatial-aware color propagation; Stage 3 (UVR) super-resolves the UV map (1K→2K) and uses a spatial-aware seam smoothing algorithm to fix texture fractures caused by UV unwrapping.

### Key Designs

1. **Synchronized Multi-view Generation (SMG)**:

    - **Function**: Generates multi-view consistent initial textures, avoiding the Janus problem.
    - **Mechanism**: MVDream is used as the base T2MV model, with a ControlNet trained to provide guidance using depth or normal maps. The key innovation lies in the synchronization method—instead of synchronizing in the 32×32 latent space (where the resolution is too low to establish accurate UV mappings), intermediate latents are decoded into 256×256 images during the denoising process. These images are projected into the UV space via inverse UV mapping (weighted and fused based on the cosine angle between the view direction and surface normal: $\mathbf{T}'_{\text{sync}} = \sum_i^N \cos(\mathbf{v}_i, \mathbf{n}_{uv}) \mathbf{T}'_i$). The synchronized UV map is then rasterized back to each view and encoded into synchronized latents to continue denoising. A single synchronization step is sufficient; too many synchronization steps cause instability.
    - **Design Motivation**: Methods like SVD perform poorly in latent-domain synchronization due to low latent resolution. Although synchronizing in the image domain introduces an extra decode-and-encode step, it offers much more precise mapping. The inherent multi-view prior of MVDream effectively eliminates the Janus problem.

2. **Spatial-aware 3D Inpainting (S3I)**:

    - **Function**: Completes texture inpainting for regions unobserved during the SMG stage.
    - **Mechanism**: The problem is reformulated from the UV space to the 3D point cloud space. From the incomplete UV map, colored pixels are converted into 3D points with RGB values $\mathbf{P}_v$, while uncolored pixels are mapped to zero-colored 3D points $\mathbf{P}_u$. These are iteratively colored using a Spatial-aware Color Propagation (SCP) algorithm: in each iteration, for every uncolored point, its $k$-nearest colored neighbors are identified, and its color is calculated based on Euclidean distance $d_j$ and normal similarity $\mathbf{n}_j \cdot \mathbf{n}_i$. The normal similarity is mapped using a piecewise function $f(x)$—when the normal difference is large ($<0.5$), the weight is assigned an extremely low value ($10^{-8}$), whereas it increases to 10 when nearly parallel ($>0.9$). This ensures that colors only propagate across surfaces with consistent normals.
    - **Design Motivation**: The issue with direct inpainting in the UV space is that 3D adjacent regions may be mapped to non-adjacent positions in the UV layout, particularly when the UV atlas is highly fragmented. Performing inpainting in 3D space naturally avoids this problem and is completely independent of the UV unwrapping results.

3. **UV Refinement Module (UVR)**:

    - **Function**: Super-resolves the UV map to 2K and repairs UV unwrapping-induced seams.
    - **Mechanism**: A two-step process: (a) UV Super-Resolution: An Image-to-Image upscale diffusion model is utilized to upscale the 1K UV map to 2K, enhancing texture details. (b) Spatial-aware Seam Smoothing (SSA): The binary mask of the valid region in the UV map is extracted, and the seam mask $\mathbf{m}_{\text{seam}}$ is located via connectivity analysis and edge detection. The UV map is resampled as a colored 3D point cloud, a kd-tree of non-seam points is constructed, and the colors of the seam points are recalculated using weights based on normal cosine similarity and distance.
    - **Design Motivation**: Seams become more noticeable after UV super-resolution, as super-resolution models do not understand the discontinuities of UV unwrapping. Finding neighbors in 3-dimensional space to perform color smoothing fixes these texture fractures that are discontinuous in 2D but actually adjacent in 3D space.

### Loss & Training
- The ControlNet for SMG is trained on 102K Objaverse samples using a standard ControlNet training paradigm.
- 8 views are uniformly distributed in azimuth, with alternating ±30° elevation angles.
- S3I and SSA are learning-free algorithms and require no training.
- Text annotations are described by CogVLM-2 regarding the category, texture, and appearance of the 3D objects, followed by LLM-based keyword extraction.

## Key Experimental Results

### Main Results

Objaverse T2T Benchmark:

| Method | FID↓ | KID↓ | CLIP↑ | User Rating (Overall)↑ | Consistency↑ |
|------|------|------|-------|----------------|---------|
| TEXTure | 28.03 | 7.60 | **20.30** | 3.81 | 3.31 |
| Paint3D | 25.28 | 5.19 | 19.27 | 3.85 | 3.51 |
| SyncMVD | 26.99 | 5.72 | 20.19 | 3.96 | 3.59 |
| **MVPaint** | **20.89** | **3.45** | 19.87 | **4.19** | **3.98** |

GSO T2T Benchmark (Generalization Test):

| Method | FID↓ | KID↓ | Seamlessness↑ | Consistency↑ |
|------|------|------|---------|---------|
| TEXTure | 24.76 | 5.50 | 3.97 | 3.65 |
| Paint3D | 37.29 | 10.24 | 3.14 | 3.45 |
| SyncMVD | 26.96 | 5.37 | 4.12 | 3.71 |
| **MVPaint** | **20.02** | **3.12** | **4.51** | **4.21** |

### Ablation Study

| Configuration | FID↓ | KID↓ | CLIP↑ |
|------|------|------|-------|
| Full MVPaint | **20.89** | **3.45** | 19.87 |
| w/o MV Sync | 21.42 | 3.72 | 19.90 |
| w/o MV Diff (replacing MVDream with single-view diffusion) | 27.63 | 5.82 | 20.44 |
| w/o Geo. Refinement | 21.17 | 3.67 | 20.00 |
| w/o 3D Inpainting | 20.91 | 3.56 | 19.87 |
| w/o Seam Smoothing | 20.82 | 3.54 | 19.92 |

### Key Findings
- **Multi-view diffusion model (MVDream) contributes the most**: Removing MVDream and substituting it with a single-view diffusion model degrades the FID by 6.74 points. This represents the largest impact among all ablations, demonstrating that multi-view priors are essential to avoiding the Janus problem.
- **Image-domain synchronization is effective but not the most critical**: Removing synchronization only degrades the FID by 0.53, indicating that MVDream's native multi-view consistency is already robust, making synchronization an additional boost.
- **Paint3D degrades significantly on GSO** (FID rises from 25.28 to 37.29), owing to its submodules overfitting during training on Objaverse. MVPaint maintains strong cross-domain generalization.
- **Substantial gap in seamlessness scores**: MVPaint achieves a seamlessness score of 4.51 on GSO vs. 4.12 for SyncMVD, verifying the efficacy of the SSA algorithm.

## Highlights & Insights
- **"Solving the right problem in the right space"** is the most central design philosophy of this work—synchronization is performed in the image domain (high resolution), inpainting is done in 3D space (preserving topological continuity), and seam repair is executed in the 3D local neighborhood (crossing UV boundaries). This approach is highly inspiring for any task involving multi-space representations.
- **The learning-free inpainting algorithm in S3I** is highly impressive—relying purely on a simple $k$-nearest neighbors + normal-weighted color propagation, it completes textures for complex geometries without training any models. Such geometry-based inpainting is particularly valuable in data-constrained scenarios.
- **The meticulous design of the piecewise normal weighting function $f(x)$** is highly ingenious—when the normal angle is less than 0.5 (approximately 60°), the color practically does not propagate, whereas the propagation weight increases tenfold when it exceeds 0.9 (approx. 26°). This effectively prevents colors from leaking across distinct surfaces. This trick can be readily transferred to point cloud coloring, 3D semantic propagation, and similar tasks.

## Limitations & Future Work
- The requirement to train a ControlNet to guide MVDream introduces additional training overhead.
- The initial generation resolution for the 8 views is relatively low (256×256), which constrains the level of detail in the starting textures.
- The iterative color propagation of S3I operates as a progressive diffusion process, which can lead to oversmoothed patterns for large, unobserved regions.
- The CLIP score is not optimal (TEXTure achieves a higher score), because the Janus artifacts in TEXTure may conversely increase the CLIP matching score from certain angles.
- The potential of utilizing video diffusion models as multi-view priors was not explored.

## Related Work & Insights
- **vs. TEXTure/Text2Tex**: Iterative methods generate textures view-by-view, which are multi-view inconsistent and slow. MVPaint's SMG generates multi-view consistent textures in a single run.
- **vs. SyncMVD**: SyncMVD synchronizes in the latent domain but suffers from low resolution, still displaying the Janus problem. MVPaint synchronizes in the image domain more precisely and fundamentally avoids the Janus problem using MVDream.
- **vs. Paint3D**: Paint3D relies on continuous UV unwrapping and generalizes poorly. MVPaint’s S3I and SSA operate in 3D space, making them immune to the quality of the UV unwrapping.

## Rating
- Novelty: ⭐⭐⭐⭐ Three modules are each creative, and the overall framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducted two benchmarks, user studies, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Structured clearly, with intuitive illustrations and well-formulated problem statements.
- Value: ⭐⭐⭐⭐ Provides an industrially viable solution for 3D texture generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Material Anything: Generating Materials for Any 3D Object via Diffusion](material_anything_generating_materials_for_any_3d_object_via_diffusion.md)
- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement](imfine_3d_inpainting_via_geometry-guided_multi-view_refinement.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] 3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)

</div>

<!-- RELATED:END -->
