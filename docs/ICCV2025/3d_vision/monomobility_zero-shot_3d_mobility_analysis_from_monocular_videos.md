---
title: >-
  [Paper Note] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos
description: >-
  [ICCV 2025][3D Vision][Articulated object analysis] MonoMobility presents the first framework for zero-shot analysis of moving parts and motion attributes (motion axis and motion type) of articulated objects from monocul…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Articulated object analysis"
  - "monocular video"
  - "zero-shot"
  - "motion attribute estimation"
  - "2D Gaussian splatting"
date: 2026-05-08
content_hash: 2cac39b9973da50e
---

# MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos

**Conference**: ICCV 2025
**arXiv**: [2505.11868](https://arxiv.org/abs/2505.11868)  
**Code**: [https://monomobility.github.io/MonoMobility](https://monomobility.github.io/MonoMobility) (project page)  
**Area**: 3D Vision / Articulated Object Analysis
**Keywords**: Articulated object analysis, monocular video, zero-shot, motion attribute estimation, 2D Gaussian splatting

## TL;DR

MonoMobility presents the first framework for zero-shot analysis of moving parts and motion attributes (motion axis and motion type) of articulated objects from monocular video. It combines off-the-shelf tools—depth estimation, optical flow, and segmentation—for coarse initialization, and then refines the results via self-supervised optimization of a dynamic scene represented with 2D Gaussian splatting together with a specially designed articulated-object dynamic scene optimization algorithm. The method requires no annotated data and handles rotational, translational, and compound motion.

## Background & Motivation

**Background**: Accurately parsing the moving parts and motion attributes of articulated objects in dynamic environments is a fundamental prerequisite for embodied intelligence and robotic manipulation. Everyday environments are populated by articulated objects (drawers, swivel chairs, staplers, pump bottles, etc.), and efficient robot interaction with these objects depends on precise understanding of moving parts and motion parameters.

**Limitations of Prior Work**: (1) Data-driven methods (Shape2Motion, OPDMulti) rely on large-scale annotated datasets, generalize poorly to unseen objects, and OPDMulti is further restricted to openable objects (doors/windows), making it unable to handle irregular articulated objects such as staplers. (2) Multi-view reconstruction methods (PARIS, Weng et al.) require dense RGB(D) multi-view images of the initial and final states together with accurate camera poses, which are difficult to obtain in practical limited-field-of-view settings; they are also confined to object-level manipulation and cannot scale to complex scenes. (3) 4D Gaussian splatting methods (4D-GS, Shape of Motion) treat each Gaussian independently, ignoring the part-level rigidity constraints inherent to articulated objects.

**Key Challenge**: Robots are typically equipped only with RGB cameras and operate under limited viewpoints, necessitating articulated object analysis from monocular video. Existing methods either require dense multi-view input (hard to acquire), annotated data (expensive and poorly generalizable), or have restricted capability (limited to specific object types).

**Goal**: Achieve scene-level articulated object moving-part identification and motion attribute analysis from a single monocular video, entirely without annotated data (zero-shot).

**Key Insight**: Two key insights—(1) dynamic videos of articulated objects inherently contain rich 3D motion information, from which geometric and motion priors can be extracted via off-the-shelf tools for initial analysis; (2) the accuracy of motion analysis improves as the estimated dynamic process becomes more consistent with the true motion pattern, so optimizing the dynamic scene can refine the results.

**Core Idea**: Coarse initial estimation + self-supervised dynamic scene optimization = zero-shot monocular articulated object motion analysis.

## Method

### Overall Architecture

A three-stage pipeline: initial analysis → scene representation → end-to-end dynamic optimization. Input: monocular video. Output: moving-part segmentation and motion attributes (motion axis, motion type: rotation / translation / compound).

### Key Designs

1. **Initial Analysis and Scene Initialization**:

    - **Function**: Leverages a series of off-the-shelf methods to extract geometric and motion information from the video, constructing an initial moving-part segmentation and motion axis estimate, and initializing the 2D Gaussian splatting scene representation.
    - **Mechanism**: Camera pose estimation (DUSt3R/DROID-SLAM) → depth estimation (DepthAnything) → optical flow analysis (RAFT/FlowFormer) → optical flow map segmentation (SAM) to obtain moving-part masks → generation of moving-part segmented point clouds → first/last frame point cloud registration (ICP, etc.) to initialize the motion axis. All moving parts are initially assumed to undergo compound motion (rotation + translation).
    - **Design Motivation**: Rather than training detectors from scratch, the approach maximizes reuse of existing foundation models, endowing the method with natural zero-shot generalization to novel articulated object categories. The initial estimates may contain mis-segmentations (static regions misidentified as moving parts) and inaccurate motion axes, but these are automatically corrected in the subsequent optimization stage.

2. **End-to-End Dynamic Scene Optimization**:

    - **Function**: The core innovation—a self-supervised dynamic scene optimization algorithm specifically designed for articulated objects, which iteratively samples frame pairs, transforms Gaussians, and jointly optimizes multiple objectives to refine motion parameters.
    - **Mechanism**: Iteratively sample random frame pairs $(I_a, I_b)$ → apply a unified rigid transformation (based on current motion axis and motion magnitude) to all Gaussians belonging to the moving part → render to the target frame → compare with the real frame → backpropagate a joint loss to update the motion axis parameters, motion magnitude, and Gaussian attributes. The key distinction is that all Gaussians within the same moving part share a single rigid transformation (rotation + translation), as opposed to the per-Gaussian independent motion in 4D-GS.
    - **Design Motivation**: The moving parts of articulated objects are rigid bodies; constraining all Gaussians within the same part to move consistently constitutes a strong prior—it both reduces the parameter space to ease optimization and prevents physically implausible motion patterns. The three losses provide supervision from complementary perspectives: the rendering loss ensures visual consistency, the normal loss ensures geometric accuracy, and the motion loss ensures consistency between the estimated dynamics and point cloud transformations.

3. **Motion Type Determination and Part Pruning**:

    - **Function**: After optimization converges, automatically determines the motion type (pure rotation / pure translation / compound) of each part based on the final motion magnitudes, and prunes falsely identified moving parts.
    - **Mechanism**: If the cumulative motion magnitude of a "moving part" after optimization is not significant (below a threshold), it is classified as a static region false detection and removed from the moving-part list. For genuine moving parts, the motion type is classified by the relative magnitudes of the rotation and translation components: only rotation significant → pure rotation; only translation significant → pure translation; both significant → compound motion.
    - **Design Motivation**: Initial analysis inevitably introduces noise (optical flow segmentation may over-segment), and the automatic pruning mechanism endows the system with self-correction capability. Motion type classification eliminates the need to specify the object type in advance, making the method applicable to arbitrary articulated objects.

### Loss & Training

- **Rendering Loss**: The transformed Gaussians are rendered to the target frame and an RGB reconstruction loss (L1 + SSIM) is computed against the real frame, ensuring visual consistency.
- **Normal Loss**: A consistency loss between Gaussian normals and estimated surface normals, improving geometric reconstruction accuracy.
- **Motion Loss**: A consistency loss between the estimated motion transformation and the inter-frame point cloud registration transformation, ensuring that the motion pattern is coherent under both the independent geometric estimation and the joint optimization perspectives.
- The three losses jointly optimize the motion axis (direction and position), motion magnitude (rotation angle and translation distance), and Gaussian attributes (position, color, scale, etc.).

## Key Experimental Results

### Datasets

The authors construct a comprehensive evaluation dataset:
- **Synthetic scenes**: Generated with a simulator, containing precise ground-truth motion axes and magnitudes.
- **Real scenes**: Multiple articulated objects (drawers, doors, staplers, pump bottles, etc.) with manual annotations.

### Main Results

| Motion Type | Typical Objects | Capability |
|-------------|----------------|------------|
| Pure rotation | Doors, hinges, flip covers | Accurate estimation of rotation axis and angle |
| Pure translation | Drawers, sliding rails | Accurate estimation of translation direction and distance |
| Compound motion | Staplers, pump bottles | Simultaneous estimation of rotation + translation parameters |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Initial analysis only (no optimization) | Large error | Noticeable deviation in motion axis direction |
| With dynamic optimization | **Significant improvement** | Both axis direction and motion magnitude are refined |
| Without normal loss | Reduced geometric accuracy | Part boundaries become blurred |
| Without motion loss | Inaccurate motion parameters | May converge to physically implausible solutions |

### Key Findings

- Dynamic scene optimization yields a decisive improvement over the initial analysis results, confirming the core insight of "refining analysis through dynamic scene optimization."
- The system works effectively on both synthetic and real scenes, handling rotational, translational, and compound motion with strong performance.
- Part-level rigidity constraints (vs. per-Gaussian independent motion) are critical for obtaining physically plausible motion.
- Zero-shot generalization enables the method to handle novel articulated object categories not seen during any training.
- Operating at the scene level rather than the object level better matches practical application requirements.

## Highlights & Insights

- The zero-shot paradigm design philosophy is instructive—rather than training specialized detectors or segmenters, the method cleverly orchestrates off-the-shelf tools followed by self-supervised optimization refinement, achieving generalization to arbitrary novel categories.
- Part-level rigidity constraints inject articulated-object priors into the 3DGS framework—compared to the per-point independent motion assumption in 4D-GS and Shape of Motion, this constitutes a stronger and more appropriate inductive bias.
- Support for compound motion (rotation + translation) broadens applicability—staplers, pump dispensers, and similar objects are ubiquitous in everyday and industrial settings, yet most prior methods handle only pure rotation or pure translation.
- The automatic part pruning and motion type determination mechanisms endow the system with self-correction and self-classification capability, requiring no manual intervention or pre-specified object types.

## Limitations & Future Work

- Cascaded dependencies—the quality of upstream tools (depth estimation, optical flow, segmentation) directly affects final results; failure at any stage (e.g., optical flow failure on reflective surfaces) propagates to subsequent steps.
- Depth ambiguity in monocular video affects motion axis accuracy—the lack of absolute scale in monocular depth may cause axis position offsets.
- The optimization process requires non-trivial computation time—each video clip requires thousands of iterative optimization steps, making the method unsuitable for real-time applications.
- The paper lacks detailed quantitative comparison with existing methods—most results are qualitative visualizations without numerical comparisons under standardized metrics.
- Only rigid articulated motion is handled—non-rigid deformation (e.g., elastic objects) is not supported.

## Related Work & Insights

- **Shape2Motion / OPDMulti**: Data-driven methods requiring annotated data with limited generalization → MonoMobility avoids these limitations through zero-shot self-supervision.
- **PARIS / Ditto**: Require dense multi-view images or two-frame point clouds → MonoMobility requires only monocular video, substantially lowering input requirements.
- **4D-GS / Shape of Motion**: Per-Gaussian independent motion → MonoMobility's part-level rigidity constraints are better suited to articulated objects.
- **A-SDF**: Neural implicit articulated reconstruction requiring 3D supervision → MonoMobility is fully self-supervised.

## Rating

- Novelty: ⭐⭐⭐⭐ Zero-shot monocular articulated analysis defines a new task; part-level rigidity constraints combined with 2DGS is a novel formulation.
- Experimental Thoroughness: ⭐⭐⭐ The dataset is self-constructed, but standardized quantitative comparisons with existing methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; the method pipeline is well-organized and logically structured.
- Value: ⭐⭐⭐⭐ Has direct application value for robotic manipulation and embodied intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] one look is enough seamless patchwise refinement for zero-shot monocular depth e](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICCV 2025\] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)
- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)

</div>

<!-- RELATED:END -->
