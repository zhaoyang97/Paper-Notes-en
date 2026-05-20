---
title: >-
  [Paper Note] Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation
description: >-
  [NeurIPS 2025][3D Vision][Single-image 3D generation] Cue3D is the first model-agnostic framework for quantifying the importance of image cues in single-image 3D generation. By systematically perturbing six visual cues—i…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Single-image 3D generation"
  - "visual cues"
  - "interpretability"
  - "illumination"
  - "texture"
date: 2026-05-08
content_hash: 18a059cc6153a545
---

# Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation

**Conference**: NeurIPS 2025
**arXiv**: [2511.22121](https://arxiv.org/abs/2511.22121)  
**Code**: [Available](https://ryanxli.github.io/cue3d)  
**Area**: 3D Vision
**Keywords**: Single-image 3D generation, visual cues, interpretability, illumination, texture

## TL;DR

Cue3D is the first model-agnostic framework for quantifying the importance of image cues in single-image 3D generation. By systematically perturbing six visual cues—illumination, texture, silhouette, perspective, edges, and local continuity—across seven methods spanning three paradigms (regression-based, multi-view, and native 3D generation), it reveals key insights: shape meaningfulness rather than texture governs generalization ability, illumination matters more than texture, and models are overly dependent on input silhouettes.

## Background & Motivation

Single-image 3D generation has advanced remarkably in recent years, progressing from regression-based methods (LRM, SF3D) to multi-view methods (CRM, InstantMesh) and native 3D generative methods (Trellis, Hunyuan3D-2), enabling high-quality textured 3D mesh generation from a single image. However, a critical question has been overlooked: **what information do these models actually exploit from an image to infer 3D structure?**

Classical computer vision research has long formalized monocular 3D cues—shape from shading, texture gradients, contour outlines, perspective effects, and so on—building decades of theoretical understanding. Yet whether end-to-end trained deep learning models implicitly internalize these classical visual priors, or instead rely on unknown shortcuts or high-level semantic information, remains entirely unclear.

This opacity is both a scientific gap (preventing a connection between deep learning and visual science) and a practical liability—it impedes diagnosing model failure modes, predicting which inputs will cause collapse, and guiding targeted model improvements. Cue3D is designed to fill this gap by establishing a unified, model-agnostic framework that uses controlled perturbation experiments to quantify each image cue's contribution to 3D generation.

## Method

### Overall Architecture

The Cue3D pipeline consists of three stages: (1) establishing baselines by evaluating seven state-of-the-art methods uniformly on two standard datasets (GSO and Toys4K); (2) applying six targeted cue perturbations to input images, where each perturbation modifies a single cue while preserving others as much as possible; and (3) quantifying the importance of each cue via the magnitude of performance degradation before and after perturbation. Evaluation covers four dimensions: 2D appearance quality (PSNR/SSIM/LPIPS), 3D geometry quality (Chamfer Distance/F-score), visible-surface quality, and symmetry consistency.

### Key Designs

1. **Six Cue Perturbation Designs**:
    - Function: Each perturbation targets a specific visual cue; importance is quantified by comparing performance before and after perturbation.
    - **Style perturbation**: Images are transferred into six artistic styles (ink wash, line drawing, pointillism, flat design, oil painting, sculpture) using the CSGO style transfer method, preserving high-level semantics while disrupting geometric cues such as realistic illumination and texture.
    - **Illumination–texture disentanglement**: Independently controlled in a Blender rendering pipeline—five texture conditions (original / checkerboard / Perlin noise / random texture / flat gray) × two illumination conditions (with/without lighting), yielding ten combinations.
    - **Silhouette perturbation**: Dilation of the object alpha mask (without altering interior pixels) combined with three levels of occlusion simulation.
    - **Edge perturbation**: Retaining only Canny edges (testing whether edges alone suffice for shape inference) and Gaussian blurring of edge regions (testing whether precise edges are necessary).
    - **Perspective perturbation**: Switching the rendering camera from perspective projection to orthographic projection.
    - **Local continuity**: Dividing the foreground into an $n \times n$ grid and randomly shuffling pixels within each cell, preserving global structure while disrupting local detail.
    - Design Motivation: Cue selection is grounded in established monocular depth cue theory from visual psychology and classical computer vision, ensuring a solid theoretical foundation for the analysis.

2. **Unified Multi-Dimensional Evaluation Framework**:
    - Function: Fairly comparing all methods under standardized conditions, avoiding the incomparability introduced by each paper selecting metrics favorable to its own approach.
    - Mechanism: For each predicted mesh, four dimensions are assessed—overall 2D appearance (PSNR/SSIM/LPIPS rendered from 16 viewpoints), overall 3D geometry (Chamfer Distance and F-score), visible-surface quality (2D and 3D metrics at the input viewpoint), and symmetry (reflection symmetry F1 matching between prediction and ground truth).
    - Design Motivation: Prior papers each select metric sets favorable to their own methods, precluding a unified baseline. Multi-dimensional evaluation avoids the blind spots of any single metric.

3. **Shape Meaningfulness Experiments**:
    - Function: Testing whether the input image must correspond to a "meaningful" object shape.
    - Mechanism: Three strategies are applied progressively—the Zeroverse dataset (textured random primitive assemblies with entirely meaningless shapes); standard CutMix (replacing 1/8 of a GSO object's volume with the corresponding portion of another object); octant CutMix (replacing each of the eight octants with parts from different random objects); and half-half splicing (front–back, left–right, and top–bottom concatenation of two objects).
    - Design Motivation: Disentangling the effects of "distribution shift" and "shape meaningfulness"—half-half splicing introduces minimal distribution shift while impairing meaningfulness, thereby confirming that meaningfulness is the critical factor.

### Loss & Training

Cue3D is an analysis framework and does not train any models. All seven evaluated methods use their respective official implementations and pretrained weights. Experiments are conducted on 8 NVIDIA L40S GPUs. The GSO evaluation set comprises 412 objects and the Toys4K evaluation set comprises 500 randomly sampled objects; each object is rendered in Blender with random camera angles and Poly Haven HDRI lighting.

## Key Experimental Results

### Main Results

Unified evaluation (GSO dataset; CD×1000↓ denotes lower is better):

| Method | Paradigm | CD×1000↓ | F-score↑ | Symmetry F1↑ |
|--------|----------|-----------|----------|--------------|
| LGM | Multi-view | 83.01 | 0.034 | 0.188 |
| OpenLRM | Regression | 80.89 | 0.033 | 0.391 |
| CRM | Multi-view | 68.07 | 0.043 | 0.285 |
| SF3D | Regression | 61.58 | 0.059 | 0.488 |
| InstantMesh | Multi-view | 54.54 | 0.072 | 0.715 |
| **Hunyuan3D-2** | **Native 3D** | **41.82** | **0.087** | **0.894** |
| **Trellis** | **Native 3D** | **39.64** | **0.092** | 0.867 |

Degradation on Zeroverse meaningless shapes:

| Method | GSO CD×1000 | Zeroverse CD×1000 | Degradation |
|--------|-------------|-------------------|-------------|
| Hunyuan3D-2 | 41.82 | 78.09 | +87% |
| Trellis | 39.64 | 78.14 | +97% |
| InstantMesh | 54.54 | 89.47 | +64% |

### Ablation Study

Illumination–texture disentanglement experiment (change in CD×1000 on GSO; larger = cue more important):

| Configuration | Trellis Δ | Hunyuan3D-2 Δ | SF3D Δ | Note |
|---------------|-----------|----------------|--------|------|
| Replace texture + retain lighting | Negligible (~1–3) | Negligible (~1–3) | Negligible (~2–4) | Texture unimportant |
| Remove lighting + retain texture | Moderate (~5–10) | Moderate (~5–10) | Moderate (~5–8) | Lighting important |
| Remove lighting + replace texture | Large (~8–15) | Large (~8–15) | Large (~7–12) | Compounding effect of missing lighting and replaced texture |
| Dilate silhouette | Small | Large | Large | Trellis more robust to silhouette |

### Key Findings

- **Shape meaningfulness is the most critical cue**: Among all perturbation types, disrupting shape meaningfulness (Zeroverse, CutMix) causes the most severe performance degradation. Even standard CutMix, which replaces only 1/8 of the volume, degrades Hunyuan3D-2 by 20 CD points. Half-half splicing (minimal distribution shift) still causes degradation exceeding 10 points—confirming that the decisive factor is meaningfulness rather than distribution shift.
- **Illumination >> Texture**: Replacing original texture with gray/noise/random alternatives while retaining lighting has almost no effect on the best-performing methods; removing lighting while retaining texture causes significant degradation. An interaction effect is also present: with lighting, texture is irrelevant, but without lighting, retaining original texture is preferable to replacing it.
- **Over-reliance on silhouettes is a source of fragility**: Dilating the silhouette by a few pixels causes severe degradation in regression-based and multi-view methods, whereas Trellis is relatively robust—suggesting it has learned a degree of silhouette invariance.
- **The three paradigms exhibit distinct failure modes**: On meaningless shapes, regression-based methods generate smooth, averaged-out back faces (losing normal detail); multi-view methods collapse due to multi-view inconsistency (drop in DINOv2 similarity); native 3D methods tend toward hallucinated symmetric completion.
- **Low inter-cue correlation**: Spearman rank-correlation analysis shows that per-object effects of different cues are largely independent (correlation coefficients 0.19–0.66), indicating that individual cue effects are relatively isolated at the object level.

## Highlights & Insights

- **"Illumination matters more than texture" is the most counterintuitive finding**: Intuition suggests texture is highly informative, yet quantitative analysis of 3D generation models demonstrates that shading (illumination) is the core geometric cue. This aligns with classical shape-from-shading theory, indicating that deep learning models do internalize this classical visual prior.
- **The shape meaningfulness finding has far-reaching implications**: The models have not learned a general 3D reconstruction capability (which would be effective for any shape), but rather structured priors based on meaningful shapes in the training distribution. This suggests that current methods are fundamentally performing "shape memory and transformation" rather than "genuine 3D understanding."
- **Native 3D generation paradigm leads comprehensively**: Trellis and Hunyuan3D-2 substantially outperform other paradigms on nearly all metrics and exhibit greater robustness to various perturbations, establishing native 3D generative methods as the current dominant paradigm.
- **The CutMix experimental design is elegant**: By progressively controlling the degree of meaningfulness disruption (half-half → standard CutMix → octant CutMix), the influence of shape meaningfulness is isolated while controlling for distribution shift.
- **Cue correlation analysis enhances rigor**: Spearman correlation matrices demonstrate that different perturbations are largely independent at the object level, strengthening the credibility of conclusions drawn for each cue.

## Limitations & Future Work

- **Perturbations may introduce unintended effects**: Texture replacement may subtly alter other cues (e.g., edge patterns), and style transfer cannot perfectly preserve semantics. The paper mitigates this by cross-validating across multiple perturbation strategies.
- **Evaluation is limited to synthetic rendered images**: All evaluation images are rendered in Blender; real photographs (with noise, varying lighting conditions, and imperfect segmentation) may yield different conclusions.
- **Text-guided 3D generation is not covered**: The analysis is restricted to image-to-3D; text-to-3D may exhibit entirely different cue dependency patterns.
- **Future directions**: Leveraging findings to guide data augmentation (e.g., enriching training data with lighting variation), improving model design (e.g., adding illumination-aware modules and reducing silhouette dependence), and extending the framework to video-to-3D.

## Related Work & Insights

- The shape–texture conflict experiments of Geirhos et al. inspired Cue3D's perturbation design philosophy, though extending it from classification to 3D generation represents an important paradigm shift.
- Classical work on shape from shading, shape from texture, and related topics finds an "echo" here—deep models do internally learn analogous geometric inference capabilities.
- The analysis provides clear improvement directions for future 3D generation methods: enhanced utilization of illumination, reduced silhouette dependence, and improved handling of meaningless shapes.
- The analysis framework itself is transferable to other generative tasks (image super-resolution, video generation, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic framework for image cue analysis in single-image 3D generation, with genuinely insightful findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models × 2 datasets × 6 cue perturbations × multi-dimensional metrics; the scope and coverage are exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, concise summaries of findings, excellent visualizations, and precise expression of conclusions.
- Value: ⭐⭐⭐⭐ Significant practical value for the 3D generation community in understanding model behavior and guiding improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](../../ICCV2025/3d_vision/ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions](../../ICCV2025/3d_vision/wonderplay_dynamic_3d_scene_generation_from_a_single_image_and_actions.md)
- [\[NeurIPS 2025\] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] A Recipe for Generating 3D Worlds from a Single Image](../../ICCV2025/3d_vision/a_recipe_for_generating_3d_worlds_from_a_single_image.md)

</div>

<!-- RELATED:END -->
