---
title: >-
  [Paper Note] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images
description: >-
  [CVPR 2025][3D Vision][Inverse Rendering] IRIS proposes an inverse rendering framework to jointly recover HDR lighting, physical materials, and camera response functions from multi-view LDR images. By explicitly modeling tone mapping, automatically detecting emitters, and employing an iterative optimization strategy, it achieves high-quality material estimation, relighting, and virtual object insertion on both real and synthetic indoor scenes.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Inverse Rendering"
  - "Indoor Scenes"
  - "Low Dynamic Range"
  - "HDR Reconstruction"
  - "Camera Response Function"
date: 2026-05-08
content_hash: 6d18530ba2490335
---

# IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images

**Conference**: CVPR 2025  
**arXiv**: [2401.12977](https://arxiv.org/abs/2401.12977)  
**Code**: [https://irisldr.github.io/](https://irisldr.github.io/)  
**Area**: 3D Vision  
**Keywords**: Inverse Rendering, Indoor Scenes, Low Dynamic Range, HDR Reconstruction, Camera Response Function

## TL;DR
IRIS proposes an inverse rendering framework to jointly recover HDR lighting, physical materials, and camera response functions from multi-view LDR images. By explicitly modeling tone mapping, automatically detecting emitters, and employing an iterative optimization strategy, it achieves high-quality material estimation, relighting, and virtual object insertion on both real and synthetic indoor scenes.

## Background & Motivation

1. **Background**: Physics-based inverse rendering aims to decompose geometry, materials, and lighting from images to support applications such as relighting, material editing, and object insertion. Current dominant methods (e.g., FIPT, NeILF++) typically rely on HDR images as input to capture complete light transport information.

2. **Limitations of Prior Work**: HDR acquisition requires specialized hardware or multi-exposure fusion, which is user-unfriendly. Existing methods attempting to use LDR inputs either assume infinitely distant light sources (unsuitable for indoor scenes), require additional emitter masks as inputs, or ignore multi-bounce light transport, making it difficult to handle the complex spatially-varying illumination in indoor scenes.

3. **Key Challenge**: After dynamic range clipping and non-linear CRF mapping, crucial high-dynamic-range lighting information in LDR images (such as the actual brightness of windows and lights) is irreversibly lost. Without reconstructing HDR lighting, material estimation is severely degraded.

4. **Goal**: To reconstruct HDR spatially-varying lighting, physical materials (BRDF), and the camera response function (CRF) simultaneously from casual LDR photos, making inverse rendering accessible to everyday users.

5. **Key Insight**: Utilize physics-based rendering equations to bridge LDR observations and HDR scene parameters. Through differentiable path tracing and CRF modeling, the HDR intensity of overexposed regions is naturally restored during the optimization process.

6. **Core Idea**: Explicitly model the LDR imaging pipeline (clipping + CRF) and decouple the light-material-CRF ambiguity in inverse rendering by alternately optimizing HDR emission, materials, and the CRF.

## Method

### Overall Architecture
IRIS takes multi-view posed LDR images and a reconstructed surface mesh as input, and operates via a two-stage pipeline: an initialization phase (BRDF initialization + surface light field extraction + emitter detection) and an iterative optimization phase (HDR radiance recovery $\rightarrow$ shading baking $\rightarrow$ joint BRDF & CRF optimization, looping until convergence). The outputs are spatially-varying HDR illumination, Cook-Torrance BRDF parameters, and the CRF curve.

### Key Designs

1. **CRF Modeling and HDR Emission Recovery**:

    - **Function**: Recovery of camera response functions and high-dynamic-range emitter radiance from LDR images.
    - **Mechanism**: Parameterize the CRF using the EMoR model: $\mathbf{g} = \bar{\mathbf{g}} + \Sigma_b w_b \mathbf{g}_b$, where $\bar{\mathbf{g}}$ is the mean curve of 201 real-world CRFs, and $\mathbf{g}_b$ represents the PCA bases. Emitters are detected via multi-view saturation check: a surface point is labeled as an emitter if its average LDR value across all visible views is $\ge 0.99$. HDR radiance is recovered through differentiable path tracing by minimizing the photometric loss: $\min_{L_e} \sum_i \|CRF(\min(L_o \cdot \Delta t_i, 1)) - I_i\|_2$.
    - **Design Motivation**: The CRF is crucial yet typically unknown for LDR-to-HDR conversion. The EMoR model is based on a database of real-world CRFs, maintaining a small parameter space and allowing regularization, which is more stable than MLP parameterization. The multi-view saturation check reliably distinguishes emitters from reflective surfaces.

2. **Light Transport Decomposition and Shading Baking**:

    - **Function**: Efficiently compute global illumination effects, avoiding expensive path tracing during material optimization.
    - **Mechanism**: Decompose light transport into diffuse shading $L_d$, rough specular shading $L_s^0$, and smooth specular shading $L_s^1$, which are precomputed and stored. During ray tracing, if a secondary ray hits an emitter, the learnable HDR radiance $L_e$ is used; otherwise, the precomputed surface light field $L_{SLF}$ is fetched to approximate global field illumination. This decomposition allows BRDF optimization to proceed efficiently.
    - **Design Motivation**: Doing multi-bounce path tracing directly during optimization is computationally expensive and unstable. The decomposition allows alternating updates between shading and BRDFs. The surface light field is used to approximate indirect illumination, avoiding recursive light transport.

3. **Alternating Optimization Strategy**:

    - **Function**: Resolve the ambiguity during the joint estimation of HDR lighting, materials, and the CRF.
    - **Mechanism**: Alternately execute three steps: (1) Fix the BRDF and CRF, and optimize the emitter HDR radiance $L_e$ via differentiable rendering; (2) Bake new shading maps using the updated HDR lighting; (3) Fix the shading and jointly optimize the BRDF parameters and CRF coefficients. The optimization goal is: $\min_{a,m,\sigma,g} \mathcal{L}_{photo} + \lambda_a\mathcal{L}_{albedo} + \lambda_c\mathcal{L}_{CRF} + \lambda_m\mathcal{L}_{mat}$.
    - **Design Motivation**: Illumination, materials, and the CRF are highly coupled; direct joint optimization is unstable. Alternating optimization handles only a subset of variables at each step, progressively improving the quality of each component.

### Loss & Training
- Photometric loss $\mathcal{L}_{photo}$: Rendered results mapped through the CRF are compared with LDR observations.
- Albedo regularization $\mathcal{L}_{albedo}$: Monocular albedo estimation is utilized to provide initialization and regularization using a scale-invariant loss.
- CRF regularization $\mathcal{L}_{CRF}$: $L_2$ regularization on PCA coefficients plus monotonicity constraints.
- Material regularization $\mathcal{L}_{mat}$: Consistency constraints on roughness and metallic values within the same semantic instance.
- Geometry reconstruction employs BakedSDF, with normals regularized by an off-the-shelf estimation method.

## Key Experimental Results

### Main Results

| Dataset | Metric | IRIS (LDR) | FIPT* (LDR) | NeILF (LDR) | FIPT (HDR) |
|--------|------|------------|-------------|-------------|------------|
| FIPT Synthetic | $k_d$ PSNR↑ | **22.33** | 15.49 | 16.85 | 29.95 |
| FIPT Synthetic | $a'$ PSNR↑ | **17.92** | 09.74 | 14.02 | 25.98 |
| FIPT Synthetic | $\sigma$ PSNR↑ | **21.38** | 04.99 | 16.96 | 26.37 |
| FIPT Synthetic | $L_e$ IoU↑ | **0.69** | 0.69 | 0.35 | 0.86 |
| FIPT Synthetic | $L_e$ L2↓ | **0.12** | 0.28 | 2.29 | 0.03 |

### Ablation Study
The qualitative evaluation of IRIS is mainly conducted through visual comparisons:
- NeILF, due to single-bounce path tracing, fails to eliminate shadows in the diffuse albedo, and the roughness estimation falsely identifies walls as smoother than mirrors.
- FIPT* (an LDR-adapted version of FIPT provided with emitter masks) suffers from a lack of HDR information, leading to severely inaccurate material estimation and a tendency to underestimate roughness, which results in unrealistic reflections.
- IRIS produces nearly shadow-free diffuse albedo fields, correctly identifies mirrors as low-roughness regions, and successfully recovers the HDR illumination of windows and ceiling lights.

### Key Findings
- LDR-to-HDR lighting recovery is the core advantage of IRIS: the recovered HDR illumination makes light transport more accurate, significantly improving material estimation.
- Even with LDR inputs, the quality of IRIS's material estimation significantly outperforms other LDR methods (with a gain of approximately 6dB in $k_d$ PSNR).
- The alternating optimization strategy effectively decouples the illumination-material-CRF ambiguity.
- CRF estimation enables IRIS to handle input images with different exposure levels.
- A performance gap still exists compared to FIPT with HDR inputs, indicating the inherent value of HDR information.

## Highlights & Insights
- **LDR as Input for Inverse Rendering**: Explicit modeling of the CRF and tone mapping allows casual photos from mobile phones to be used for inverse rendering. This significantly lowers the barrier to using inverse rendering, carrying great practical significance.
- **Automatic Emitter Detection**: Multi-view saturation consistency detection is simple yet effective, successfully distinguishing between emitters and reflective surfaces (saturation in one view $\neq$ emitter, whereas saturation in all views does). This strategy can be transferred to any scenario that requires distinguishing self-emission from reflection.
- **EMoR Parameterization of CRF**: PCA parameterization based on a real-world CRF database is physically more plausible and easier to regularize than MLPs or simple gamma curves.

## Limitations & Future Work
- It assumes that the primary light sources are visible within the input images and cannot handle completely hidden light sources.
- The quality of geometry depends on the reconstruction accuracy of BakedSDF.
- Currently, it assumes all input images share the same CRF (i.e., captured by the same camera) and does not support mixed inputs from multiple cameras.
- Quantitative evaluation is only conducted on synthetic scenes (due to the lack of ground truth for real scenes), while real scenes only have qualitative results.
- Future directions for improvement: introducing diffusion-model-based HDR priors, or integrating 3DGS for more efficient rendering.

## Related Work & Insights
- **vs FIPT**: FIPT requires HDR inputs and cannot handle the CRF issues of LDR. IRIS extends the capability of FIPT to the LDR domain through CRF modeling and HDR reconstruction.
- **vs NeILF/NeILF++**: NeILF represents illumination with a neural light field but only performs single-bounce path tracing, while NeILF++ uses a learnable gamma, which is overly simplistic. IRIS achieves more accurate HDR recovery through light transport decomposition and the EMoR CRF.
- **vs Li et al.**: Li et al. estimate the BRDF and illumination from a single image (data-driven), which is limited by the domain gap of the training data. IRIS is optimization-based and more robust on real-world scenes.

## Rating
- Novelty: ⭐⭐⭐⭐ It systematically addresses indoor inverse rendering with LDR inputs for the first time. The combined strategy of CRF + HDR reconstruction + alternating optimization is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation is performed on both synthetic and real scenes, with convincing demos of relighting and object insertion, though real scenes lack quantitative metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem definition is clear, the modeling logic of the imaging pipeline is rigorous, and the illustrations are of high quality.
- Value: ⭐⭐⭐⭐ High practical value, bringing inverse rendering from the laboratory to consumer-grade devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](pbr-nerf_inverse_rendering_with_physics-based_neural_fields.md)
- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)
- [\[CVPR 2026\] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering](../../CVPR2026/3d_vision/sgs-intrinsic_semantic-invariant_gaussian_splatting_for_sparse-view_indoor_invers.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_extended_dr_3d.md)
- [\[CVPR 2025\] MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)

</div>

<!-- RELATED:END -->
