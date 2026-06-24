---
title: >-
  [Paper Note] ActiveGAMER: Active GAussian Mapping through Efficient Rendering
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] ActiveGAMER is proposed, representing the first attempt to utilize 3D Gaussian Splatting for active mapping. By efficiently selecting the optimal next-best-view via a rendering-based information gain module, integrated with a coarse-to-fine exploration strategy, post-refinement, and a global-local keyframe policy, ActiveGAMER significantly outperforms NeRF-based methods in both geometric accuracy and rendering fidelity on the Repl…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Active Mapping"
  - "Next-Best-View"
  - "Real-time Rendering"
  - "SLAM"
date: 2026-05-08
content_hash: 770e199c74777975
---

# ActiveGAMER: Active GAussian Mapping through Efficient Rendering

**Conference**: CVPR 2025  
**arXiv**: [2501.06897](https://arxiv.org/abs/2501.06897)  
**Code**: None  
**Area**: 3D Vision / Active Reconstruction  
**Keywords**: 3D Gaussian Splatting, Active Mapping, Next-Best-View, Real-time Rendering, SLAM

## TL;DR

ActiveGAMER is proposed, representing the first attempt to utilize 3D Gaussian Splatting for active mapping. By efficiently selecting the optimal next-best-view via a rendering-based information gain module, integrated with a coarse-to-fine exploration strategy, post-refinement, and a global-local keyframe policy, ActiveGAMER significantly outperforms NeRF-based methods in both geometric accuracy and rendering fidelity on the Replica and MP3D datasets.

## Background & Motivation

Active reconstruction is a core capability for autonomous robots: robots must autonomously determine observation positions to achieve reconstruction of 3D scenes that are as complete and accurate as possible. This is fundamentally a joint optimization problem of planning and mapping.

In recent years, Neural Radiance Fields (NeRFs) have been introduced to the field of active reconstruction, but they face two major bottlenecks:

**High computational overhead**: Volume rendering in NeRF requires dense MLP sampling along each ray, resulting in extremely slow rendering speeds. This makes it difficult to evaluate the information gain of a large number of candidate viewpoints in real time.

**Neglected photometric reconstruction**: Due to slow rendering, existing NeRF-based methods (such as NARUTO) primarily focus on geometric reconstruction, leaving the rendering quality (RGB fidelity) insufficiently optimized.

**Restricted motion**: Many methods are limited to 2D planar motion or discrete jumps, failing to explore freely in complex 3D spaces.

3D Gaussian Splatting (3DGS) provides an efficient alternative by explicitly representing the scene with sparse Gaussian ellipsoids, achieving rendering speeds orders of magnitude faster than NeRF. The core idea of this work is to **leverage the real-time rendering capability of 3DGS to drive active mapping**—generating a large number of candidate views, rapidly evaluating information gain, and autonomously deciding the next observation position, while establishing high-quality geometric and photometric reconstructions simultaneously.

## Method

### Overall Architecture

The pipeline of ActiveGAMER can be summarized as a loop:
1. **Input**: HabitatSim simulator provides posed RGB-D images ($680 \times 1200$).
2. **Gaussian Mapping**: Incrementally update the Gaussian map using a simplified 3DGS from SplaTAM.
3. **Rendering-Based Planning**: Render silhouette masks on candidate views to evaluate information gain $\rightarrow$ select the next-best-view.
4. **Path Planning**: Plan collision-free paths in free space using RRT.
5. **Execution**: The robot moves to the target pose, obtains new observations $\rightarrow$ returns to step 1.
6. **Post-Refinement**: Further optimize rendering quality using global keyframes after exploration is completed.

It supports unconstrained 6DoF motion, not limited to 2D planes.

### Key Designs

1. **Simplified Gaussian Representation and Real-time Rendering**:

    - Function: Represent the scene using isotropic Gaussians (color $c$, position $\boldsymbol{\mu}$, radius $r$, opacity $o$) to reduce the parameter count.
    - Mechanism: During rendering, 3D Gaussians are projected onto the image plane, sorted front-to-back, and alpha-blended:
    $C(\mathbf{p}) = \sum_{i=1}^{n} c_i f_i(\mathbf{p}) \prod_{j=1}^{i-1}(1 - f_j(\mathbf{p}))$
      Depth maps and silhouette masks are rendered similarly. The optimization loss is:
    $L = \sum_{\mathbf{p}} (S(\mathbf{p}) > 0.99)(L_1(D(\mathbf{p})) + 0.5 L_1(C(\mathbf{p})))$
    - Design Motivation: Rendering each frame in NeRF takes several seconds, whereas 3DGS achieves real-time speeds—making it feasible to evaluate hundreds of candidate viewpoints in a single step.

2. **Rendering-Based Information Gain Module**:

    - Function: Calculate an information gain score for each candidate viewpoint to select the optimal next-best-view.
    - Mechanism: Render the silhouette mask $S$ at the candidate pose, count the number of missing pixels $N_{S_i}$ (pixels with a value of 0), while also considering the movement cost:
    $\mathcal{I} = (1 - \sigma(l_i)) \cdot \sigma(\log(N_{S_i}))$
      where $l_i = \|T_{i,x} - T_{t,x}\|_2$ represents the distance, and $\sigma$ denotes softmax normalization.
    - Design Motivation: More missing pixels imply that the viewpoint can observe more unreconstructed areas. Multiplying by a distance decay factor ensures that nearby targets are chosen when the information gain is similar, thereby reducing total travel distance.

3. **Coarse-to-Fine Exploration Strategy**:

    - Function: Efficiently cover the entire scene in two stages.
    - Mechanism:
        - **Coarse Exploration**: Candidate viewpoints are sampled on a single height plane ($v_1=1$m interval, $v_2=5$ orientations) to quickly cover large areas.
        - **Fine Exploration**: Multi-height layers with denser sampling ($v_1=0.5$m, $v_2=15$ orientations) to refine missed regions.
    - Maintain an **exploration candidate pool**: Candidate viewpoints are incrementally sampled based on an occupancy grid, and candidates that have been sufficiently observed ($N_{S_i} < 0.5\%$ of total pixels) are removed from the pool. Full free space is re-sampled when switching from coarse to fine exploration.
    - Design Motivation: Excessive candidate sampling increases computational evaluation overhead; the coarse-to-fine strategy balances efficiency and completeness.

4. **Global-Local Keyframe Selection**:

    - Function: Improve SplaTAM's strategy of relying solely on local keyframes for optimization, mitigating local overfitting.
    - Mechanism: SplaTAM selects $k$ local keyframes with the highest overlap to optimize the map. This work instead uses half local keyframes and half global keyframes. The selection criteria for global keyframes are:
        - Completeness-based: new pixels in the silhouette mask $> 10\%$.
        - Quality-based: rendering quality falls below a predefined threshold.
    - Design Motivation: Relying purely on local keyframes causes Gaussians within the view frustum but behind the main surface to suffer from overly reduced opacity. Global keyframes provide remote supervision, preventing local overfitting.

5. **Post-Refinement**:

    - Function: After exploration is completed, further optimize the rendering quality of the Gaussian map using global keyframes.
    - Mechanism: Increase the number of optimization iterations (from 15 to 60) and perform densification using full-resolution images.
    - Design Motivation: Rendering quality is constrained during the exploration phase due to only 15 iterations per step and low resolution. Post-refinement sacrifices a small amount of geometric completeness (by pruning redundant Gaussians) to achieve significant photometric improvements.

### Loss & Training

- Gaussian map optimization: $L_1$ depth loss + $0.5 \times L_1$ color loss, calculated only on pixels where $S > 0.99$.
- Densification mask: Combines low-density regions ($S<0.5$) and regions with excessively high depth errors ($> 50 \times$ median depth error).
- No learning components—completely based on rules and rendering evaluations.

## Key Experimental Results

### Main Results: Geometric Reconstruction (MP3D)

| Method | Accuracy (cm) ↓ | Completion (cm) ↓ | Comp. Ratio (%) ↑ |
|------|-----------------|-------------------|-------------------|
| FBE | / | 9.78 | 71.18 |
| ANM | 7.80 | 9.11 | 73.15 |
| NARUTO | 6.31 | 3.00 | 90.18 |
| **ActiveGAMER** | **1.66** | **2.30** | **95.32** |

### Main Results: Rendering Quality (Replica 8-Scene Average)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | L1-Depth ↓ |
|------|--------|--------|---------|------------|
| SplaTAM (Passive) | 29.08 | 0.95 | 0.14 | 1.38 |
| NARUTO | 26.01 | 0.89 | 0.41 | 9.54 |
| **ActiveGAMER** | **32.02** | **0.97** | **0.11** | **1.12** |

### Ablation Study (Replica)

| Configuration | Comp. (cm) ↓ | Comp. Ratio ↑ | PSNR ↑ | L1-D ↓ |
|------|-------------|--------------|--------|--------|
| Coarse Exploration Only | 1.77 | 94.53 | 29.77 | 1.80 |
| Without Global Keyframes | 2.19 | 94.87 | 30.73 | 1.23 |
| Without Post-Refinement | 1.56 | **96.50** | 30.67 | 1.42 |
| **Full System** | **1.80** | **95.45** | **32.02** | **1.12** |

### Key Findings

- **3DGS completely outperforms NeRF in active mapping**: Geometric accuracy is improved by $\sim 4\times$ (MP3D Accuracy from $6.31\text{cm}$ to $1.66\text{cm}$), and PSNR is elevated by $6\text{dB}+$.
- The coarse-to-fine strategy enables the coarse exploration phase to achieve over $94\%$ completeness, while the fine exploration contributes an additional $\sim 2\%$ and improves rendering.
- Post-refinement significantly raises PSNR (from $30.67$ to $32.02$), though it slightly degrades geometric completeness (as redundant Gaussians are pruned).
- Global keyframes are essential to prevent overfitting: removing them leads to decreased completeness and increased depth errors.

## Highlights & Insights

- **First 3DGS-based active mapping system**: Exploits the real-time rendering advantage of 3DGS to make rendering-based information gain evaluation practically feasible.
- **Excellent system engineering design**: Coarse-to-fine exploration, candidate pool management, global/local keyframe selection, and post-refinement—each component has a clear design motivation and ablation validation.
- **Dual optimization of geometry and photometry**: Unlike prior methods that focus on only one aspect, this work simultaneously pursues complete geometry and high-fidelity rendering.
- **6DoF free motion**: No restriction to 2D planes or discrete jumps, staying closer to realistic robotic scenarios.

## Limitations & Future Work

- **Assumed ground-truth localization**: Real-world scenarios require integrating a SLAM localization module.
- **Ignorance of double-sided objects**: Rendering silhouette masks cannot reveal the back-side of objects, leading to unexplored backsides (Figure 9 illustrates a typical failure case).
- **Restricted candidate sampling near surfaces**: To prevent the renderer from ignoring nearby Gaussians, candidate sampling deliberately avoids areas close to surfaces, preventing coverage of certain regions.
- **Lack of kinematic constraints**: The kinematic constraints of physical robots (e.g., wheel-based robots cannot perform arbitrary 6DoF movements) are not considered.

## Related Work & Insights

- NARUTO pioneered 6DoF active NeRF but was limited by rendering speed; ActiveGAMER addresses this core bottleneck using 3DGS.
- AG-SLAM (concurrent work) utilizes Fisher Information + 3DGS, but focuses primarily on SLAM rather than pure reconstruction.
- Insights for embodied AI: Real-time, high-quality rendering makes the "map-while-plan" paradigm truly feasible, establishing an important cornerstone for active perception systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of 3DGS and active mapping is novel, though the methodological innovation of individual components is somewhat limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on dual datasets (Replica + MP3D) across dual metrics (geometry + rendering), featuring detailed ablations and runtime analysis.
- Writing Quality: ⭐⭐⭐⭐ The system is described clearly with a complete algorithmic pipeline, though some sections are somewhat verbose.
- Value: ⭐⭐⭐⭐ Significantly advances the active reconstruction field, validating the viability of 3DGS as a foundation for active vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](../../CVPR2026/3d_vision/magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [\[CVPR 2026\] Uncertainty-driven 3D Gaussian Splatting Active Mapping via Anisotropic Visibility Field](../../CVPR2026/3d_vision/uncertainty-driven_3d_gaussian_splatting_active_mapping_via_anisotropic_visibili.md)
- [\[CVPR 2025\] UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping](uvgs_reimagining_unstructured_3d_gaussian_splatting_using_uv_mapping.md)
- [\[CVPR 2025\] ERUPT: Efficient Rendering with Unposed Patch Transformer](erupt_efficient_rendering_with_unposed_patch_transformer.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)

</div>

<!-- RELATED:END -->
