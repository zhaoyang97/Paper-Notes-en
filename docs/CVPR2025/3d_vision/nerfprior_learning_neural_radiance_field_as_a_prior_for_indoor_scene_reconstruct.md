---
title: >-
  [Paper Note] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction
description: >-
  [CVPR 2025][3D Vision][NeRF prior] NeRFPrior utilizes a rapidly trained Grid-NeRF (TensoRF, 30 minutes) as a scene-specific prior to guide SDF learning through multi-view consistency constraints and a confidence-weighted depth consistency loss. The F1 score on ScanNet improves from 0.310 (MonoSDF) to 0.930 (+200%), with a total training time of only 4.7 hours (2.2x faster than MonoSDF).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "NeRF prior"
  - "SDF reconstruction"
  - "multi-view consistency"
  - "depth prior"
  - "indoor scenes"
date: 2026-05-08
content_hash: c61acc8e986752cd
---

# NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2503.18361](https://arxiv.org/abs/2503.18361)  
**Code**: [https://wen-yuan-zhang.github.io/NeRFPrior/](https://wen-yuan-zhang.github.io/NeRFPrior/)  
**Area**: 3D Vision  
**Keywords**: NeRF prior, SDF reconstruction, multi-view consistency, depth prior, indoor scenes

## TL;DR

NeRFPrior utilizes a rapidly trained Grid-NeRF (TensoRF, 30 minutes) as a scene-specific prior to guide SDF learning through multi-view consistency constraints and a confidence-weighted depth consistency loss. The F1 score on ScanNet improves from 0.310 (MonoSDF) to 0.930 (+200%), with a total training time of only 4.7 hours (2.2x faster than MonoSDF).

## Background & Motivation

1. **Background**: Neural surface reconstruction (such as NeuS, MonoSDF) reconstructs 3D surfaces from multi-view images by optimizing SDF through volume rendering. Indoor scenes are highly challenging due to large textureless areas (white walls, ceilings) and weak lighting.
2. **Limitations of Prior Work**: (1) Methods like NeuS degrade severely in textureless regions, where color supervision provides almost no information on white walls; (2) MonoSDF relies on pre-trained monocular depth estimation networks, which suffer from out-of-domain generalization issues and slow training (10.6h); (3) Geo-NeuS uses COLMAP as a prior, but COLMAP itself is time-consuming and yields unstable quality in indoor scenes.
3. **Key Challenge**: Textureless regions require additional prior constraints, but general priors (monocular depth networks) are inaccurate, whereas scene-specific priors (COLMAP) are too slow.
4. **Goal**: To use the target scene's own NeRF as a prior, which is scene-specific and fast to train.
5. **Key Insight**: Grid-based NeRFs (such as TensoRF) require only 30 minutes to train high-quality color and density fields. Although their geometric surfaces are less precise than those of SDFs, their density and color fields can provide valuable prior constraints.
6. **Core Idea**: The NeRF prior provides density supervision (where the surface is likely located) and color supervision (multi-view consistency checks), combined with a depth consistency loss specifically designed for textureless regions.

## Method

### Overall Architecture

Target scene multi-view images $\rightarrow$ TensoRF training for 30 minutes to obtain the NeRF prior (density field $F_\sigma$ + color field $F_c$) $\rightarrow$ NeuS SDF optimization (100K steps of base training + 50K steps of multi-view constraints + 50K steps of depth consistency) $\rightarrow$ Marching Cubes to extract the surface mesh.

### Key Designs

1. **Multi-View Consistency Constraint**

    - **Function**: Utilizing the NeRF prior to verify the consistency of SDF-predicted surface points across other views.
    - **Mechanism**: SDF root search finds the surface intersection point $p^*$ $\rightarrow$ cast rays back to source views $\rightarrow$ perform local volume rendering in the NeRF prior to estimate the source view color $c_s^{proj}$ $\rightarrow$ visibility check (if $|c_s^* - c_s^{proj}| < t_0$) $\rightarrow$ visible source views participate in the multi-view matching loss.
    - **Design Motivation**: Directly using photometric loss fails in textureless regions. The NeRF prior provides occlusion detection and color consistency judgment at the volume rendering level.

2. **Confidence-Weighted Depth Consistency Loss**

    - **Function**: Constraints specifically designed for textureless planar regions.
    - **Mechanism**: $\mathcal{L}_{depth} = \sum ||\hat D(r) - \bar D| \cos\langle n, r \rangle|| \cdot sgn_c \cdot sgn_\sigma$. It is only activated in regions satisfying two conditions: $sgn_c=1$ (color variance < $t_1$, determined as textureless) and $sgn_\sigma=1$ (density variance < $t_2$, determined as planar).
    - **Design Motivation**: Textureless planar regions (such as white walls) are highly prone to SDF degradation. The conditional restrictions ensure that depth constraints are only applied in "confident" regions, avoiding erroneous enforcement on architecturally complex areas.

3. **Density and Color Supervision from NeRF Prior**

    - **Function**: Provides a coarse but fast reference for geometry and appearance to guide the SDF.
    - **Mechanism**: $\mathcal{L}_\sigma = ||\sigma_{SDF} - \sigma_{prior}||$ and $\mathcal{L}_c = ||c_{SDF} - c_{prior}||$ with exponentially decaying weights (strong constraints in early stages, relaxed later to let the SDF refine autonomously).
    - **Design Motivation**: Although the geometry of the NeRF prior is less precise than the SDF (lacking surface definition), its density field provides a rough formulation of "where the surface roughly is".

### Loss & Training

$\mathcal{L} = \mathcal{L}_{rgb} + \lambda_1 \mathcal{L}_\sigma + \lambda_2 \mathcal{L}_c + \lambda_3 \mathcal{L}_{reg} + \lambda_4 \mathcal{L}_{depth}$. Here, $\lambda_1 = \lambda_2 = 0.1$ (exponentially decaying), $\lambda_3 = 0.05$, and $\lambda_4 = 0.5$. The multi-view constraint is enabled after 100K steps, and the depth constraint is enabled after 150K steps. NeRF prior training takes 30 minutes, and SDF training takes 4.2 hours, totaling 4.7 hours.

## Key Experimental Results

### Main Results

| Method | ScanNet F1↑ | Replica F1↑ | BlendSwap F1↑ | Training Time |
|------|-------------|-------------|---------------|---------|
| NeuS | 0.291 | 0.665 | 0.483 | 7.2h |
| MonoSDF | 0.310 | 0.632 | - | 10.6h |
| Geo-NeuS | 0.291 | - | - | 9.0h |
| NeuralAngelo | 0.292 | - | - | - |
| **NeRFPrior** | **0.732** | **0.813** | **0.621** | **4.7h** |
| NeRFPrior+depth | **0.930** | - | - | - |

### Ablation Study

| Configuration | Replica CD↓ | Replica NC↑ | Replica F1↑ |
|------|-------------|-------------|-------------|
| Base only | 0.083 | 0.832 | 0.619 |
| +NeRF prior | 0.051 | 0.893 | 0.781 |
| +Multi-view constraint | 0.049 | 0.763 | 0.673 |
| +Depth constraint | 0.050 | 0.887 | 0.744 |
| **Full** | **0.038** | **0.912** | **0.813** |

### Key Findings

- The NeRF prior alone contributes an F1 increase from 0.619 to 0.781 (+26%), which is the largest single improvement.
- The combination of multi-view and depth constraints performs better than using either alone, demonstrating their complementarity.
- Adding the depth prior on ScanNet (real indoor) boosts F1 from 0.732 to 0.930, indicating that depth constraints in textureless regions are highly critical.
- The training speed is 4.7h vs 10.6h for MonoSDF (2.2x speedup), with NeRF prior training taking only 30 minutes (comprising just 11% of the total time).

## Highlights & Insights

- **"Coarse-to-fine" two-stage reconstruction strategy**: The 30-minute NeRF provides a coarse prior $\rightarrow$ followed by 4.2 hours of SDF refinement. The concept of using a fast, coarse model to guide a slower, precise model is highly versatile.
- **Confidence-weighted conditional loss design**: Depth constraints are only applied in "confident" regions, avoiding the catastrophic consequences of blind enforcement.
- **No external pre-trained models required**: Unlike MonoSDF, which relies on monocular depth networks, NeRFPrior's prior is entirely derived from the target scene itself, entailing zero out-of-domain risk.

## Limitations & Future Work

- The quality of the NeRF prior itself degrades under extremely sparse views, leading to method failure.
- Reasonable camera coverage is required (no completely missing areas).
- Sensitivity to hyperparameters ($t_0, t_1, t_2, \lambda$) is relatively high.
- The depth consistency assumes piecewise planar textureless areas; non-planar textureless regions may not be handled well.

## Related Work & Insights

- **vs NeuS**: Pure color supervision degrades in indoor scenes. NeRFPrior thoroughly addresses the textureless issue through the NeRF prior and depth constraints.
- **vs MonoSDF**: Relies on a pre-trained monocular depth network, leading to generalization issues and a 10.6h training time. NeRFPrior's scene-specific prior is more accurate and faster.
- **vs Geo-NeuS**: Relies on COLMAP point clouds as a prior. NeRFPrior replaces COLMAP with NeRF, which is denser and faster (30 min vs 1.5h+ and unstable).

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using NeRF as an SDF prior is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets (real + synthetic) + detailed ablations + time comparisons.
- Writing Quality: ⭐⭐⭐⭐ Highly clear motivation and design logic.
- Value: ⭐⭐⭐⭐⭐ A practical breakthrough in indoor reconstruction—the F1 score leap from 0.31 to 0.93 is a qualitative breakthrough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Decompositional Neural Scene Reconstruction with Generative Diffusion Prior](decompositional_neural_scene_reconstruction_with_generative_diffusion_prior.md)
- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[CVPR 2025\] ProbeSDF: Light Field Probes for Neural Surface Reconstruction](probesdf_light_field_probes_for_neural_surface_reconstruction.md)
- [\[CVPR 2025\] LookCloser: Frequency-aware Radiance Field for Tiny-Detail Scene (FA-NeRF)](lookcloser_frequency-aware_radiance_field_for_tiny-detail_scene.md)
- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)

</div>

<!-- RELATED:END -->
