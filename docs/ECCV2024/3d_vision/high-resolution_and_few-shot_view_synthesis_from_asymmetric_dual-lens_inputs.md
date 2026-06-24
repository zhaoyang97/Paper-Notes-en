---
title: >-
  [Paper Note] High-Resolution and Few-shot View Synthesis from Asymmetric Dual-Lens Inputs
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This paper proposes DL-GS (Dual-Lens 3D-GS), which addresses two major issues of 3D-GS in few-shot training and super-resolution rendering by leveraging stereo geometric constraints and high-resolution guidance from asymmetric dual-lens systems (wide-angle + telephoto) commonly found on mobile devices. It achieves SOTA performance through a consistency-aware training strategy and a multi-reference guided refinement module.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Dual-Lens System"
  - "Few-Shot View Synthesis"
  - "High-Resolution Rendering"
  - "Super-Resolution"
date: 2026-05-08
content_hash: c19fd457d2f3796d
---

# High-Resolution and Few-shot View Synthesis from Asymmetric Dual-Lens Inputs

**Conference**: ECCV 2024  
**Code**: [https://github.com/XrKang/DL-GS](https://github.com/XrKang/DL-GS)  
**Area**: 3D Vision / Novel View Synthesis  
**Keywords**: 3D Gaussian Splatting, Dual-Lens System, Few-Shot View Synthesis, High-Resolution Rendering, Super-Resolution

## TL;DR

This paper proposes DL-GS (Dual-Lens 3D-GS), which addresses two major issues of 3D-GS in few-shot training and super-resolution rendering by leveraging stereo geometric constraints and high-resolution guidance from asymmetric dual-lens systems (wide-angle + telephoto) commonly found on mobile devices. It achieves SOTA performance through a consistency-aware training strategy and a multi-reference guided refinement module.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3D-GS), as an emerging scene representation method, enables high-quality real-time rendering. However, standard 3D-GS requires dense multi-view images (typically 30–100 images) as training inputs and can only render at the training resolution, making it fail to generate novel high-resolution views exceeding the input resolution.

**Limitations of Prior Work**: (1) **Few-shot degradation**: When training views are reduced to 3–5, the performance of 3D-GS drops sharply, as sparse views lack sufficient geometric constraints, causing Gaussian primitives to overfit to visible areas and ignore occluded regions. Existing few-shot methods (e.g., FSGS, DNGaussian) rely on depth priors or regularization but offer limited effectiveness. (2) **Resolution constraints**: 3D-GS performs well at training resolution but cannot render views at higher resolutions, because the density and details of Gaussian primitives are optimized for the training resolution. Existing approaches either require additional super-resolution post-processing networks or multi-scale training.

**Key Challenge**: Few-shot and high-resolution are two conflicting objectives—few-shot implies sparse information, whereas high-resolution rendering demands richer information. Traditional pipelines struggle to address both simultaneously since they are treated as independent sub-tasks.

**Goal**: How to exploit off-the-shelf hardware properties (dual-lens smartphones) to simultaneously provide geometric constraints (resolving the few-shot challenge) and high-frequency details (resolving the resolution challenge), killing two birds with one stone.

**Key Insight**: Modern smartphones are commonly equipped with asymmetric dual-lens systems—a wide-angle lens with a large field of view (FoV) but relatively low resolution, and a telephoto lens with a narrow FoV but higher resolution. The two lenses are spatially separated by a small baseline (a few millimeters to one centimeter), forming a natural stereo pair. The authors harness this: the wide-angle camera provides broad scene coverage and geometric cues, while the telephoto camera provides local high-resolution details. Combining the two directly addresses both the few-shot and high-resolution challenges.

**Core Idea**: Leveraging the asymmetric stereo configuration of mobile dual-lens systems as a natural source of geometric constraints and resolution enhancement to achieve few-shot, high-resolution 3D-GS reconstruction.

## Method

### Overall Architecture

The inputs to DL-GS are a few (3–5 poses) dual-lens image pairs, where at each pose, a wide-angle image and a telephoto image are captured simultaneously. First, SfM (COLMAP) is applied to estimate camera parameters for all images to initialize the 3D Gaussian primitives. Then, during the 3D-GS optimization: (a) a consistency-aware training strategy leverages the stereo relationship of the dual-lens system to apply geometric constraint regularization; (b) once optimization is complete, a multi-reference guided refinement module utilizes the high-frequency information of telephoto images to enhance the resolution of novel view rendering. Finally, high-resolution novel views are rendered.

### Key Designs

1. **Consistency-Aware Training Strategy**:

    - **Function**: Leverages stereo consistency between the dual lenses to provide additional geometric constraints for 3D-GS optimization, alleviating the overfitting problem in few-shot scenarios.
    - **Mechanism**: Since dual lenses capture the same scene from slightly different perspectives at the same instance, a rigid geometric consistency relation exists between them. This strategy introduces a dual-lens consistency loss $\mathcal{L}_{consist}$: the depth map rendered from the wide-angle view, when projected to the telephoto view using the known relative dual-lens camera pose, should be consistent with the depth map directly rendered from the telephoto view. Specifically, for each Gaussian primitive, rendering is performed from both the wide-angle and telephoto perspectives, and the difference in warped depth is calculated as a loss. This cross-view geometric consistency constraint acts as a "free" extra view supervision for each training pose, effectively increasing constraint density.
    - **Design Motivation**: The bottleneck of few-shot 3D-GS is geometric degradation caused by insufficient constraints. The stereo baseline provided by dual lenses is the most natural and reliable source of extra geometry—it does not rely on any pre-trained depth estimation models, thus avoiding the introduction of extra estimation errors.

2. **Multi-Reference-Guided Refinement Module**:

    - **Function**: Utilizes the telephoto and wide-angle images in the training set to provide high-frequency detail enhancement for newly synthesized views, achieving super-resolution rendering.
    - **Mechanism**: For each novel target view to be rendered, K nearest reference images (including both wide-angle and telephoto) are selected from the training set based on camera pose distances. Then, an attention-based fusion network transfers the high-frequency texture details from reference images to the rendered result. Specifically: (i) the low-resolution rendering from 3D-GS is upsampled to the target resolution to serve as a base; (ii) multi-scale features of the reference and base images are extracted via feature extractors; (iii) deformable attention is used to query high-frequency details from the reference features corresponding to the base image; (iv) the queried high-frequency features are fused back into the base image. This process resembles reference-based super-resolution, but leverages multiple reference images with different focal lengths and views.
    - **Design Motivation**: The novel view rendered directly by 3D-GS contains correct geometry and coarse color but lacks high-frequency details beyond the training resolution. The telephoto images perfectly capture high-frequency details of local areas. Through intelligent reference selection and feature fusion, rendering resolution can be enhanced without retraining the 3D-GS model.

3. **Adaptive Reference Selection Strategy**:

    - **Function**: Selects the most suitable set of reference images for each novel view.
    - **Mechanism**: A pose distance metric is defined as $d(v_{new}, v_{ref}) = w_t \|t_{new} - t_{ref}\| + w_r \angle(R_{new}, R_{ref})$, taking both translation distance and rotation angle into account. Top-K nearest neighbors are selected for telephoto and wide-angle images respectively. Preference is given to telephoto images that align well with the novel view direction (since telephoto lenses have narrow FoVs and are only effective when views are close), while angle constraints are relaxed for wide-angle images (since they have wide FoVs covering ample content). The weights $w_t$ and $w_r$ are adaptively adjusted according to the lens type.
    - **Design Motivation**: The quality of reference images directly impacts the refinement performance. Mismatched references can introduce flaws; thus, a meticulously designed selection strategy is essential.

### Loss & Training

The total training loss is formulated as $\mathcal{L} = \mathcal{L}_{rgb} + \lambda_1 \mathcal{L}_{ssim} + \lambda_2 \mathcal{L}_{consist}$, where $\mathcal{L}_{rgb}$ and $\mathcal{L}_{ssim}$ are standard color and structural similarity losses in 3D-GS, and $\mathcal{L}_{consist}$ is the dual-lens consistency loss. The refinement module is trained separately, supervised by an L1 + perceptual loss, with training data constructed by downsampling telephoto images.

## Key Experimental Results

### Main Results

**Synthetic Dataset (3 Training Views)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3D-GS (baseline) | 22.14 | 0.791 | 0.218 |
| FSGS | 24.31 | 0.834 | 0.176 |
| DNGaussian | 24.87 | 0.841 | 0.168 |
| DL-GS (Ours) | **27.52** | **0.893** | **0.112** |

**Real-world Dataset (5 Training Views, 2× Super-Resolution)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3D-GS + SR | 24.63 | 0.845 | 0.153 |
| Mip-NeRF 360 | 25.18 | 0.856 | 0.141 |
| DL-GS (Ours) | **28.37** | **0.911** | **0.089** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Description |
|------|-------|-------|------|
| Full DL-GS | 27.52 | 0.893 | Full model |
| w/o Consistency Loss | 25.41 | 0.852 | Insufficient geometric constraints, drops by 2.11dB |
| w/o Refinement Module | 25.89 | 0.861 | No super-resolution capability |
| w/o Adaptive Selection | 26.73 | 0.878 | Improper reference selection introduces artifacts |
| Wide-only training | 24.87 | 0.838 | No telephoto guidance |
| Telephoto-only training | 23.56 | 0.812 | Insufficient FoV coverage |

### Key Findings

- Consistency loss makes the greatest contribution (+2.11 dB), demonstrating that stereo geometric constraints from dual lenses are critical for few-shot scenarios.
- The refinement module significantly improves results in super-resolution scenarios (+1.63 dB), particularly in high-frequency texture regions.
- Using only wide-angle or only telephoto lenses yields results significantly inferior to joint usage, demonstrating that the complementarity of the two lens types is fundamental.
- The performance gains are even more pronounced on real-world datasets, illustrating the strong robustness of the method to real scenes.
- The contribution of the reference selection strategy is significant (+0.79 dB); incorrect reference selections can introduce noticeable artifacts.

## Highlights & Insights

- **Turning hardware features into algorithmic advantages** is the most ingenious idea of this paper. Dual-lens smartphones are ubiquitous, but they were not previously viewed as a natural advantage for 3D reconstruction. Compared to few-shot methods relying on pre-trained depth models, the physical geometric constraints in this method are grounded in reality and thus much more reliable.
- **An elegant decomposition of challenges**: the few-shot challenge is addressed by the stereo baseline, and the high-resolution challenge is solved by telephoto details. Two seemingly independent, difficult problems are resolved in a unified manner using a single hardware feature.
- **Reusable design patterns**: The design of the multi-reference-guided refinement module can be directly transferred to other 3D representation methods requiring super-resolution rendering (e.g., NeRF, NeuS).

## Limitations & Future Work

- It relies heavily on precise dual-lens calibration. If there are errors in intrinsic or extrinsic calibration, the consistency loss will introduce misleading constraints.
- The refinement module requires additional forward inference time, which could be a bottleneck for real-time rendering pipelines.
- The framework is currently only validated on static scenes. The synchronization issue of dual lenses in dynamic scenes (shutter timing discrepancy between the two lenses) is not considered.
- The capacity of super-resolution is bound by the resolution of the telephoto images. If the target resolution vastly exceeds the telephoto resolution, the refinement effect may be limited.
- The system's performance under changing illumination conditions has not been explored (e.g., auto-exposure might not be synchronized between the two cameras).

## Related Work & Insights

- **vs FSGS**: Solves few-shot 3D-GS via density initialization and semantic guidance but lacks strict physical geometric constraints, underperforming relative to DL-GS by 2.65 dB.
- **vs DNGaussian**: Introduces depth prior regularization, but the pre-trained depth estimation model can be inaccurate, and it does not address the high-resolution rendering bottleneck.
- **vs Mip-NeRF 360**: While it supports continuous multi-scale rendering, its inference speed is vastly slower than 3D-GS, and it struggles with degradation in few-shot settings.
- **Insight**: Utilizing the physical properties of consumer-grade hardware (such as LiDAR, ToF, multi-cameras) to enhance 3D reconstruction is an extremely promising direction for future research.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of leveraging dual lenses is simple yet effective, though the individual technical components (consistency constraints, reference-based super-resolution) are not entirely novel from scratch.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evaluation on both synthetic and real-world data, solid ablation and comparative experiments, and codebase released.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, smooth method description, and high-quality charts.
- Value: ⭐⭐⭐⭐ Highly practical, directly targeting 3D reconstruction on consumer mobile devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SemanticHuman-HD: High-Resolution Semantic Disentangled 3D Human Generation](semantichuman-hd_high-resolution_semantic_disentangled_3d_human_generation.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ECCV 2024\] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)
- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)

</div>

<!-- RELATED:END -->
