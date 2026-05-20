---
title: >-
  [Paper Note] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes GaussianGrow, which replaces the conventional paradigm of jointly predicting geometry and appearance from scratch by "growing" 3D Gaussians from readily a…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Point Clouds"
  - "Text Guidance"
  - "Multi-view Diffusion"
  - "Appearance Generation"
date: 2026-05-08
content_hash: cd4a6e5d2e3ee4f4
---

# GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance

**Conference**: CVPR 2026
**arXiv**: [2604.05721](https://arxiv.org/abs/2604.05721)  
**Code**: [https://weiqi-zhang.github.io/GaussianGrow](https://weiqi-zhang.github.io/GaussianGrow)  
**Area**: 3D Vision / 3D Generation
**Keywords**: 3D Gaussian Splatting, Point Clouds, Text Guidance, Multi-view Diffusion, Appearance Generation

## TL;DR
This paper proposes GaussianGrow, which replaces the conventional paradigm of jointly predicting geometry and appearance from scratch by "growing" 3D Gaussians from readily available 3D point clouds. It employs a geometry-aware multi-view diffusion model to generate consistent appearance supervision, and addresses view-fusion artifacts and invisible-region problems through an overlap-region detection mechanism coupled with an iterative inpainting strategy, achieving substantial improvements over state-of-the-art methods on both synthetic and real-scan point clouds.

## Background & Motivation
1. **Background**: 3D Gaussian Splatting (3DGS) has become the dominant representation for high-fidelity 3D modeling, yet generating high-quality 3D Gaussians remains challenging. Existing generative methods (GVGEN, DiffSplat, etc.) must simultaneously learn geometric structure and appearance; inaccurate geometry predictions severely degrade overall generation quality.
2. **Limitations of Prior Work**: Some methods attempt to infer Gaussian primitives by predicting point maps as geometric references, but the estimated geometry is unreliable, leading to poor generation quality. Another line of work generates appearance by texturing 3D meshes, but meshes require extensive manual modeling, and reliance on UV unwrapping introduces texture seams and distortions.
3. **Key Challenge**: Joint learning of geometry and appearance makes models highly sensitive to geometry prediction errors, while obtaining reliable geometric priors is costly (mesh modeling demands substantial manual effort).
4. **Goal**: How can readily accessible geometric priors (3D point clouds) be exploited to significantly improve 3D Gaussian generation quality?
5. **Key Insight**: With the proliferation of LiDAR sensors and depth cameras, acquiring clean point cloud data has become highly convenient. Point clouds can serve as reliable geometric priors, reducing the generation task from "joint geometry–appearance learning" to "growing appearance on given geometry."
6. **Core Idea**: Fix the centers of Gaussian primitives at point cloud positions and leverage a multi-view diffusion model to generate appearance supervision for "growing" the color and opacity attributes of the Gaussians.

## Method

### Overall Architecture
The pipeline consists of two stages. **Stage 1**: A depth-aware ControlNet generates a reference image for the primary view, after which a geometry-aware multi-view diffusion model (Hunyuan3D-Paint) produces 6 canonical views plus 4 additional views optimized for overlap regions—10 views in total—as appearance supervision for optimizing Gaussian attributes. **Stage 2**: Unseen regions are iteratively detected; camera poses are optimized to observe the largest unseen region; a 2D diffusion model inpaints the rendered views, which then serve as supervision to continue growing the Gaussians until full coverage is achieved. **Input**: 3D point cloud + text prompt. **Output**: A complete 3D Gaussian set.

### Key Designs

1. **Initialization and Geometry Extraction**

    - **Function**: Establish a reliable geometric foundation from the point cloud.
    - **Mechanism**: Each Gaussian center is initialized at its corresponding point cloud position. An unsigned distance field (UDF) is optimized from the point cloud using CAP-UDF, from which normals are computed as $n_i = \nabla f_u(p_i) / \|\nabla f_u(p_i)\|$. A 2D Gaussian Splatting representation (oriented discs rather than ellipsoids) is adopted, with rotation matrices set automatically from the normals. Depth maps (via ray marching), normal maps (via gradient inference), and position maps (pixel→XYZ coordinates) are extracted from the UDF as geometric conditioning for subsequent view generation.
    - **Design Motivation**: UDF is preferred over SDF because it can represent open topologies and complex structures without requiring watertight surfaces. Direct initialization from point clouds inherently guarantees geometric accuracy.

2. **Overlap Region Detection and Pose Optimization**

    - **Function**: Resolve appearance inconsistencies in overlapping regions between adjacent canonical views.
    - **Mechanism**: Ray tracing identifies the set of visible Gaussians for each viewpoint; the intersection of adjacent viewpoints yields the overlap region $R_{i,j}$. A new camera pose is optimized for each overlap region to maximize alignment between camera ray directions and the normals of Gaussians within that region:
    $$\mathcal{L}_{\text{align}} = \sum_{g \in R_{i,j}} \left(1 - \left|\frac{\mathbf{d}_{i,j} \cdot \mathbf{n}_g}{\|\mathbf{d}_{i,j}\| \|\mathbf{n}_g\|}\right|\right)$$
    This ensures that additional views observe overlap regions from the most frontal angle, reducing projection distortion and thereby generating more consistent appearance. Camera positions are constrained to lie on the unit sphere during optimization. A CUDA-parallelized detection algorithm reduces computation time from minutes to seconds.
    - **Design Motivation**: The standard 6 preset views inevitably produce large overlapping areas between adjacent views, and multi-view diffusion models frequently generate inconsistent results in these regions. Generating appearance for overlap regions from the optimal viewpoint is key to resolving this issue.

3. **Iterative Gaussian Inpainting**

    - **Function**: Cover point cloud regions that remain unseen after multi-view generation.
    - **Mechanism**: Visibility analysis automatically predicts the optimal camera pose for observing the largest unseen region. The core optimization objective minimizes the number of unoptimized Gaussians occluded by already-optimized ones:
    $$\mathcal{L}_{\text{occ}} = \sum_{i,j} \sigma\!\left((\tau(\rho_i+\rho_j)^2 - \|q_i-q_j\|^2)\right) \sigma(\tau(z_i-z_j))$$
    where $q$ denotes 2D projections, $\rho$ is the projected radius, and $z$ is depth. After finding the optimal viewpoint, the current view (containing occlusion holes) is rendered and a depth-aware inpainting diffusion model fills the holes; the inpainted result supervises optimization of the corresponding Gaussians. This process iterates until all Gaussians are covered (typically within 6 iterations). A final Spatial Inpainting post-processing step propagates attributes from optimized Gaussians to neighboring unoptimized ones.
    - **Design Motivation**: The geometric structures of different objects vary substantially; a fixed dense viewpoint set cannot cover all regions. Adaptively discovering and inpainting unseen regions is more efficient and complete than predefined viewpoint patterns.

### Loss & Training
Gaussian optimization follows a view-specific scheme—only front-facing Gaussians visible from the current viewpoint are optimized, preventing interference from back-facing Gaussians. The 6 canonical views are optimized first, followed by the 4 additional views targeting overlap regions. Hunyuan3D-Paint is used as the multi-view diffusion model; primary view generation employs Stable Diffusion with a depth-aware ControlNet.

## Key Experimental Results

### Main Results (Objaverse Dataset, Text-guided Appearance Generation)

| Method | FID ↓ | KID ↓ | CLIP ↑ | User Study (Overall) ↑ |
|--------|-------|-------|--------|----------------------|
| TexTure | 42.63 | 7.84 | 26.84 | 1.49 |
| Text2Tex | 41.62 | 6.45 | 26.73 | 2.37 |
| SyncMVD | 40.85 | 5.77 | 27.24 | 4.13 |
| GAP | 40.39 | 5.28 | 27.26 | 3.37 |
| **GaussianGrow** | **36.07** | **3.04** | **27.30** | **4.67** |

### Ablation Study

| Configuration | FID ↓ | KID ↓ | CLIP ↑ |
|--------------|-------|-------|--------|
| **Full Model** | **36.07** | **3.04** | **27.30** |
| W/o Overlap Processing | 40.48 | 4.81 | 26.73 |
| W/o Inpaint | 40.46 | 4.68 | 26.71 |

| Views K | FID ↓ | KID ↓ | CLIP ↑ |
|---------|-------|-------|--------|
| K=6 (canonical views only) | 40.48 | 4.81 | 26.73 |
| **K=10** | **36.07** | **3.04** | **27.30** |
| K=12 | 36.57 | 2.88 | 26.48 |

### Key Findings
- **Both overlap processing and inpainting are essential**: Removing either module raises FID from 36 to above 40, with contributions of roughly equal magnitude.
- **K=10 is the optimal view count**: Four additional views focused on the most critical overlap regions suffice; increasing to K=12 yields a marginal KID improvement but slightly worsens CLIP and FID.
- **Point clouds outperform reconstructed meshes**: Baseline methods exhibit a significant performance drop (FID increases by 15–25 points) when operating on reconstructed meshes (BPA/CAP-UDF), demonstrating that the point cloud→mesh→UV unwrapping pipeline introduces substantial geometric distortion. GaussianGrow bypasses these intermediate steps entirely.
- On the T3Bench text-to-3D benchmark, GaussianGrow combined with a Uni3D retrieval scheme surpasses DiffSplat, GVGEN, LGM, and other methods across all metrics.
- The method generalizes to real-scan point clouds (DeepFashion3D), demonstrating robustness to noise and density variation.
- A single point cloud paired with different text prompts can produce diverse appearance styles, demonstrating flexibility.

## Highlights & Insights
- **The "growing Gaussians from point clouds" perspective shift**: Reducing 3D generation from "jointly learning geometry and appearance" to "learning appearance on existing geometry" is a simple yet highly effective insight. Point clouds as geometric priors are more reliable than predicted point maps, and their acquisition cost (LiDAR scanning or cross-modal retrieval) continues to decrease.
- **Fine-grained overlap region handling**: Optimizing camera poses via normal–ray alignment to observe overlap regions is a highly engineered yet effective design choice; the CUDA-parallel implementation further reflects a commitment to practical efficiency.
- **Adaptive inpainting strategy**: Rather than using predefined viewpoints, the model autonomously identifies regions most in need of inpainting—this "on-demand generation" paradigm is more elegant than brute-force dense-view approaches.

## Limitations & Future Work
- The method depends on the quality of the external multi-view diffusion model (Hunyuan3D-Paint); poor generation quality for certain object categories cannot be remedied by GaussianGrow itself.
- Iterative inpainting requires multiple rendering passes and diffusion model inference steps, incurring greater computational overhead than end-to-end approaches.
- Primary view generation (ControlNet + Stable Diffusion) is a single-pass sample; a poor reference image will compromise the consistency of all subsequent views.
- Current evaluation is primarily at the object level; applicability to scene-level point clouds has not been validated.

## Related Work & Insights
- **vs. DiffSplat**: DiffSplat employs an image diffusion model to directly generate Gaussians, with geometry and appearance learned jointly. GaussianGrow mitigates the risk of geometry prediction failure by decoupling geometry (provided by point clouds) from appearance (generated by the diffusion model).
- **vs. DreamGaussian**: DreamGaussian optimizes appearance via Score Distillation Sampling (SDS), which is prone to over-saturation and unnatural results. GaussianGrow provides explicit supervision through multi-view diffusion, yielding more natural appearance.
- **vs. TriplaneGaussian**: Resolution constraints of the triplane representation limit fine-grained appearance recovery. GaussianGrow optimizes Gaussian primitives directly in 3D space, unconstrained by the resolution of any intermediate representation.
- **vs. mesh texturing methods (TexTure, Text2Tex, etc.)**: GaussianGrow circumvents the UV unwrapping bottleneck; the substantial performance degradation of baseline methods after point-cloud-to-mesh reconstruction further confirms this advantage.

## Rating
- Novelty: ⭐⭐⭐⭐ The "growing Gaussians from point clouds" framing is original; the engineering design of overlap region handling and iterative inpainting is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers Objaverse synthetic + DeepFashion3D real scans + T3Bench text-to-3D + multi-method comparison + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear, with good coordination between equations and figures.
- Value: ⭐⭐⭐⭐ Offers a new paradigm for 3D generation, though dependence on an external multi-view diffusion model limits self-contained applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PointINS: Instance-Aware Self-Supervised Learning for Point Clouds](pointins_instance-aware_self-supervised_learning_for_point_clouds.md)
- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](../../ICCV2025/3d_vision/egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds](3d_sans_3d_scans_scalable_pre-training_from_video-generated_point_clouds.md)

</div>

<!-- RELATED:END -->
