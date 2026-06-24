---
title: >-
  [Paper Note] PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects
description: >-
  [ECCV 2024][3D Vision][Polarimetric reconstruction] This paper proposes PISR, which leverages the geometric constraints of polarized light (the correspondence between the angle of polarization and the azimuth of surface normals) to directly regularize neural implicit surface shapes. Combined with hash grid acceleration and image-space normal smoothing, it achieves high-precision reconstruction on textureless and specular objects with a 0.5mm Chamfer distance and a 99.5% F-sco…
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Polarimetric reconstruction"
  - "Neural implicit surfaces"
  - "SDF"
  - "Textureless and specular objects"
  - "Multi-view reconstruction"
date: 2026-05-08
content_hash: 1793683c832d1340
---

# PISR: Polarimetric Neural Implicit Surface Reconstruction for Textureless and Specular Objects

**Conference**: ECCV 2024  
**arXiv**: [2409.14331](https://arxiv.org/abs/2409.14331)  
**Code**: Yes ([https://github.com/GCChen97/PISR](https://github.com/GCChen97/PISR))  
**Area**: Others  
**Keywords**: Polarimetric reconstruction, Neural implicit surfaces, SDF, Textureless and specular objects, Multi-view reconstruction

## TL;DR

This paper proposes PISR, which leverages the geometric constraints of polarized light (the correspondence between the angle of polarization and the azimuth of surface normals) to directly regularize neural implicit surface shapes. Combined with hash grid acceleration and image-space normal smoothing, it achieves high-precision reconstruction on textureless and specular objects with a 0.5mm Chamfer distance and a 99.5% F-score, while running 4 to 30 times faster than previous polarimetric methods.

## Background & Motivation

Multi-view neural implicit surface reconstruction (e.g., NeuS) has shown excellent performance on regular objects but still fails on **textureless and specular objects**. The root cause is the **shape-radiance ambiguity**:

- Textureless regions lack photometric consistency constraints, leaving shape estimation under-constrained.
- Specular reflections cause the optimizer to distort the shape to fit the reflected colors, resulting in severe reconstruction errors.
- Existing methods (e.g., NeRO, Ref-NeuS, UniSDF) improve radiance modeling, but optimization still relies on image reconstruction loss, tightly coupling shape and appearance.

**Polarized light provides geometric constraints independent of appearance**: The angle of polarization (AoP) of light equals the azimuth of the surface normal (the direction projected onto the image plane), subject to $\pi/2$ and $\pi$ ambiguities but independent of light intensity. This property allows surface normal constraints to be bound without involving color or material modeling.

Unlike recent polarimetric works (e.g., PANDORA, NeISF), which indirectly utilize polarimetric information through polarimetric color volume rendering and may still suffer from shape-radiance ambiguity, PISR directly regularizes the shape using polarimetric constraints, **independent of appearance**.

## Method

### Overall Architecture

The reconstruction workflow of PISR is divided into three stages:

1. **Coarse Shape Initialization**: Estimates a coarse shape using only the photometric loss $\mathcal{L}_{color}$.
2. **Shape Correction**: Incorporates the polarimetric loss $\mathcal{L}_{pol}^p$ and the normal smoothing loss $\mathcal{L}_{normal}$ to correct shape distortion.
3. **Refinement**: Retains the polarimetric loss while gradually reducing normal smoothing to preserve details.

| Component | Function | Description |
|------|------|------|
| Multi-resolution hash grid $\mathcal{G}$ | Spatial feature storage | 16 resolution levels, from 32 to 2700 |
| SDF MLP $\Phi_s$ | Decodes SDF values | 1 hidden layer (64-dimensional) |
| Color MLP $\Phi_c$ | Decodes color | 2 hidden layers (64-dimensional) |
| NeuS volume rendering | Differentiable image rendering | Unbiased weight function |
| Instant-NGP | Background rendering | Open-scene handling |

### Key Designs

**1. Perspective Polarimetric Constraint Loss**

Previous methods utilized orthographic polarimetric constraints: $[\sin(\varphi+\Delta), \cos(\varphi+\Delta), 0] \cdot \hat{\mathbf{n}} = 0$, which neglects the perspective effect of the lens. PISR adopts a more accurate perspective polarimetric constraint:

$$[\nu_z\sin\varphi', \nu_z\cos\varphi', -(\nu_y\cos\varphi' + \nu_x\sin\varphi')] \cdot \hat{\mathbf{n}} = 0$$

where $\mathbf{v}$ represents the camera ray direction. Experiments show that this improvement **reduces the Chamfer distance by 30%**.

**2. $\pi/2$ Ambiguity Resolution**

The AoP difference between diffuse and specular reflections is $\pi/2$, making disambiguation challenging. PISR proposes a piecewise loss based on the Degree of Polarization (DoP):

$$f^p(\varphi, \mathbf{v}, \mathbf{n}, \rho) = \begin{cases} h^p(\varphi,0,\mathbf{v},\mathbf{n}) \cdot h^p(\varphi,\pi/2,\mathbf{v},\mathbf{n}) & \rho < \theta \\ h^p(\varphi,\pi/2,\mathbf{v},\mathbf{n}) & \rho \geq \theta \end{cases}$$

When DoP < 0.3 (dominated by diffuse reflection), the product of the two disambiguations is taken (only requiring one to be zero). When DoP $\geq$ 0.3, it is regarded as specular-dominated, and $\Delta=\pi/2$ is directly applied.

**3. Criss-Cross Normal Smoothing**

The discrete nature of hash grids causes the SDF to be insufficiently smooth, leading to holes and cracks. Directly smoothing normals in 3D space results in unevenness (since sampled normals are only constrained by the smoothing loss). PISR performs normal smoothing in the **image space**, so that each sampled point is simultaneously constrained by photometric and polarimetric losses.

A criss-cross pattern is used to sample neighboring pixels. Under a fixed total pixel count $|S|$, the balance between the number of center pixels $|\mathcal{S}_c|$ and neighboring pixels $|\mathcal{N}_u|$ is maintained: more center pixels capture the global structure, while more neighboring pixels enhance smoothing.

### Loss & Training

Total loss function:

$$\mathcal{L}_{all} = \mathcal{L}_{color} + \lambda_p\mathcal{L}_{pol}^p + \lambda_n\mathcal{L}_{normal} + \lambda_e\mathcal{L}_{eikonal}$$

Progressive training strategy:

| Stage | Iterations | $\lambda_p$ | $\lambda_n$ | Description |
|------|--------|-------------|-------------|------|
| Initialization | 0-2.5k | 0 | 0 | Photometric and Eikonal loss only, establishing coarse shape |
| Correction | 2.5k-5k | 0→2.0 | 0→1.0 | Linearly increasing polarimetric and normal constraints |
| Refinement | 5k-7.5k | 2.0 | 1.0→0 | Decreasing smoothing to preserve details |
| Convergence | 7.5k-20k | 2.0 | 0 | Polarimetric constraints and photometric loss |

The hash grid starts with the fourth coarsest resolution, adding one level every 1000 steps. The Eikonal loss coefficient is consistently set to $\lambda_e=0.1$.

## Key Experimental Results

### Main Results

Reconstruction accuracy on self-collected polarimetric datasets (4 objects, 40-60 viewpoint polarimetric images):

| Method | Input | Black Dragon CD↓ | Red Dragon CD↓ | Rabbit S. CD↓ | Rabbit L. CD↓ | Mean CD↓ | Mean FS↑ |
|------|------|-----------------|----------------|--------------|--------------|---------|---------|
| Ref-NeuS | RGB | 2.1 | 2.1 | 1.2 | 1.0 | 1.6 | 76.6% |
| NeRO | RGB | 2.4 | 1.8 | 1.3 | 1.3 | 1.7 | 72.8% |
| NeuS | RGB | 1.9 | - | - | - | - | - |
| PANDORA | Pol.RGB | - | - | - | - | - | - |
| PMVIR | Pol.RGB | - | - | - | 0.6 | ~1.0 | ~90.5% |
| **PISR (Ours)** | **Pol.RGB** | **~0.5** | **~0.5** | **~0.4** | **0.6** | **0.5** | **99.5%** |

The mean Chamfer distance of PISR (0.5mm) is only half that of the second-best method PMVIR, while the F-score is approximately 9 percentage points higher.

### Ablation Study

Impact of polarimetric loss design choices:

| Configuration | Polarimetric Constraint | Normal Smoothing | Hash Grid | Mean CD↓ (mm) |
|------|---------|---------|---------|-------------|
| PISR-A (Photometric only) | None | None | Yes | Large |
| PISR-B | Orthographic constraint | Yes | Yes | ~0.7 |
| PISR-C | Perspective constraint | None | Yes | ~0.6 |
| **PISR (Full)** | **Perspective constraint** | **Yes** | **Yes** | **0.5** |

Key ablation findings:
- Perspective constraint vs. Orthographic constraint: **Chamfer distance is reduced by approximately 30%**.
- Role of normal smoothing: Eliminates artifacts caused by the discrete nature of the hash grid.
- Hash grid vs. pure MLP: Speeds up reconstruction by 4 to 30 times and corrects topological errors during optimization.

### Key Findings

1. **Decoupling appearance via polarimetric constraints is highly effective**: RGB-only methods (Ref-NeuS, NeRO) yield a Chamfer distance of 1.6-1.7mm on textureless/specular objects, which drops to 0.5mm with polarimetric constraints.
2. **Perspective effects are non-negligible**: Orthographic polarimetric constraints introduce systematic errors at image edges, whereas perspective constraints correct these, boosting accuracy by 30%.
3. **NeuS outperforms NeRO/Ref-NeuS**: Interestingly, methods designed specifically for specular objects perform worse than baseline NeuS, indicating that complex radiance modeling may introduce additional shape-radiance ambiguity.
4. **PANDORA exhibits poor performance**: Although using polarization, it relies on indirect polarimetric volume rendering, which still suffers from shape-radiance ambiguity.
5. **Significant speed advantage**: PISR's total optimization time is about 40 minutes (20k iterations), whereas PMVIR requires several hours or longer.

## Highlights & Insights

- **Core Insight**: "Shape constraints should be decoupled from appearance." When RGB losses tightly couple shape and appearance, polarization functions as an anchor to decouple shape.
- **Importance of perspective polarimetric constraints**: Allying physical accuracy with geometry, a seemingly minor mathematical refinement (accounting for ray direction) yields a 30% accuracy improvement.
- **Advantages of Hash Grid + Neural SDF**: Offers greater flexibility than mesh-based representations (like PMVIR); it can correct topological errors during optimization, whereas mesh representations cannot recover once topological errors occur.
- **Simplicity of Criss-Cross Sampling**: Efficiently balances global structure and local smoothness with a straightforward sampling strategy.

## Limitations & Future Work

1. **Requirement of a polarimetric camera**: Color polarimetric cameras are more expensive than standard RGB cameras, limiting the general applicability of the method.
2. **Small self-collected dataset**: Only 4 objects with ground truth and 2 without ground truth were evaluated, indicating a limited evaluation scale.
3. **Limited material range**: Mainly tested on ceramics and plastics, while highly specular materials like metals are not fully verified.
4. **Indoor natural illumination assumption**: While polarimetric constraints are relatively stable under natural illumination, strong directional light sources may introduce bias.
5. **Unmodeled refraction**: Ineffective for transparent/translucent objects.
6. **Manual DoP threshold setting**: The threshold $\theta=0.3$ is manually selected, and an adaptive threshold could be more robust.

## Related Work & Insights

- **Difference from PMVIR**: PMVIR relies on mesh representations (unable to correct topology), whereas PISR is based on neural SDFs (capable of correcting topology); PISR employs perspective constraints, while PMVIR leverages orthographic constraints.
- **Difference from PANDORA/NeISF**: The latter model appearance via polarimetric volume rendering, continuing to rely on image reconstruction loss, whereas PISR directly regularizes shape using polarimetric constraints.
- **Inspirations**: Polarimetric information can serve as a plug-and-play shape regularization technique that can theoretically be integrated into any SDF or NeRF-based reconstruction framework.
- **Potential Extensions**: Combining PISR with polarimetric inverse rendering (with PISR providing the initial shape) could realize a complete shape, material, and illumination joint estimation framework.

## Rating

| Dimension | Score (1-5) | Evaluation |
|------|-----------|------|
| Novelty | 4 | Cleverly designed perspective polarimetric constraints and appearance-independent shape regularization. |
| Technical Depth | 4.5 | Rigorous mathematical derivation of physical constraints and a well-designed multi-stage optimization strategy. |
| Experimental Thoroughness | 3.5 | Thorough ablations, though the self-collected dataset is small. |
| Writing Quality | 4 | Clear introduction of preliminaries and tight, logical presentation of the method. |
| Practical Value | 4 | High direct value for industrial-grade textureless and specular object reconstruction; code is open-sourced. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A Probability-guided Sampler for Neural Implicit Surface Rendering](a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)
- [\[CVPR 2026\] Opti-NeuS: Neural Reconstruction for Dual-Layered Transparent and Opaque Objects](../../CVPR2026/3d_vision/opti-neus_neural_reconstruction_for_dual-layered_transparent_and_opaque_objects.md)
- [\[ECCV 2024\] Ray-Distance Volume Rendering for Neural Scene Reconstruction](ray-distance_volume_rendering_for_neural_scene_reconstruction.md)
- [\[ECCV 2024\] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)
- [\[CVPR 2025\] ProbeSDF: Light Field Probes for Neural Surface Reconstruction](../../CVPR2025/3d_vision/probesdf_light_field_probes_for_neural_surface_reconstruction.md)

</div>

<!-- RELATED:END -->
