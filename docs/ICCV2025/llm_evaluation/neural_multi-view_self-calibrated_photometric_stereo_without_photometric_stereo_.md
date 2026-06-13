---
title: >-
  [Paper Note] Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues
description: >-
  [ICCV 2025][LLM Evaluation][multi-view photometric stereo] This paper proposes an end-to-end neural inverse rendering framework that jointly recovers geometry, spatially-varying reflectance…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "multi-view photometric stereo"
  - "neural inverse rendering"
  - "self-calibration"
  - "end-to-end optimization"
  - "neural BRDF"
  - "shadow-aware volume rendering"
date: 2026-05-08
content_hash: 78b055ab0f23efa3
---

# Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues

**Conference**: ICCV 2025
**arXiv**: [2507.23162](https://arxiv.org/abs/2507.23162)  
**Area**: 3D Reconstruction / Inverse Rendering / Photometric Stereo
**Keywords**: multi-view photometric stereo, neural inverse rendering, self-calibration, end-to-end optimization, neural BRDF, shadow-aware volume rendering

## TL;DR

This paper proposes an end-to-end neural inverse rendering framework that jointly recovers geometry, spatially-varying reflectance, and lighting parameters from multi-view images captured under varying illumination, requiring neither light source calibration nor intermediate photometric stereo cues (e.g., normal maps). The method outperforms existing multi-stage MVPS approaches.

## Background & Motivation

Recovering the intrinsic properties of a scene—geometry, reflectance, and illumination—from images is a longstanding core problem in computer vision. Photometric stereo (PS) acquires OLAT (one-light-at-a-time) images under a fixed viewpoint by sequentially activating light sources from different directions, and analyzes surface normals from illumination variation, constituting a classical paradigm for this class of problems. Multi-view photometric stereo (MVPS) extends this to capture stacks of OLAT images from multiple viewpoints, enabling complete 3D reconstruction.

However, existing MVPS methods suffer from fundamental architectural limitations:

**Error accumulation from multi-stage processing**: The typical pipeline—light calibration → per-view PS cue estimation → multi-view fusion—propagates errors from each stage downstream.

**Per-view PS ignores cross-view information**: Estimating normal maps independently for each viewpoint fails to exploit complementary cross-view observations and is prone to **cross-view inconsistency**.

**Dependency on light source calibration**: Calibration equipment such as chrome spheres or Lambertian white boards is required.

**Requirement for view-aligned OLAT**: Traditional methods require a complete OLAT image stack for each viewpoint, constraining acquisition flexibility.

**Core insight**: Given that multi-view OLAT images provide known, dense photometric observations, why not jointly optimize all scene parameters end-to-end directly from raw pixels? Multi-stage processing discards information rather than exploiting it.

## Method

### Overall Architecture

The framework takes multi-view OLAT images (with foreground masks and camera parameters) as input and jointly optimizes three scene properties:

1. **Geometry**: neural SDF (Signed Distance Function)
2. **Reflectance**: neural implicit BRDF
3. **Illumination**: per-light-source direction and RGB intensity

The core rendering pipeline consists of three MLPs:
- **Spatial MLP**: predicts the SDF value and BRDF latent code at each scene point
- **BRDF MLP**: predicts reflectance values from the latent code and angular encoding
- **Shadow MLP**: refines the SDF transmittance-based shadow factor

### Key Design 1: Image Formation Model and Shadow-Aware Volume Rendering

The radiance at scene point $\mathbf{x}$ under viewing direction $\mathbf{v}$, light direction $\boldsymbol{\ell}$, and intensity $e$ is:

$$r(\mathbf{x}) = e \cdot f(\mathbf{x}, \mathbf{n}, \mathbf{v}, \boldsymbol{\ell}) \cdot (\mathbf{n}^\top \boldsymbol{\ell})_+$$

The pixel observation is computed via shadow-aware volume rendering:

$$r(\mathbf{p}) = s'(\mathbf{x}', \boldsymbol{\ell}) \cdot e \cdot \sum_k T_k \alpha_k f(\mathbf{x}_k, \mathbf{n}_k, \mathbf{v}, \boldsymbol{\ell}) (\mathbf{n}_k^\top \boldsymbol{\ell})_+$$

where $s'$ is the shadow factor, $T_k$ is the accumulated transmittance, and $\alpha_k$ is the opacity.

**Key approximation**: All sampled points along a ray are assumed to share the same shadow factor as the surface intersection point, reducing shadow computation complexity from $O(N^2)$ to $O(N)$.

### Key Design 2: Geometry Representation (Neural SDF + Hash Encoding)

A Spatial MLP jointly predicts both SDF values and BRDF latent codes:

$$(g(\mathbf{x}), \mathbf{b}(\mathbf{x})) = \mathcal{G}(\mathcal{H}(\mathbf{x}; \phi); \theta)$$

- $\mathcal{H}$: multi-resolution hash encoding (instant-ngp style), facilitating high-frequency detail recovery and training efficiency
- Surface normals are computed analytically from the SDF gradient: $\mathbf{n} = \overline{\nabla g}$
- Opacity is converted from the SDF via a sigmoid function with a learnable sharpness parameter

### Key Design 3: Neural Latent Code–Driven BRDF

Rather than decomposing the BRDF into diffuse and specular terms fitted by analytic models, the method employs a single MLP to directly predict BRDF values:

$$f(\mathbf{x}, \mathbf{n}, \mathbf{v}, \boldsymbol{\ell}) = \mathcal{F}(\mathbf{b}(\mathbf{x}), \mathcal{A}(\mathbf{n}, \mathbf{v}, \boldsymbol{\ell}); \psi)$$

**Angular Encoding** is one of the key innovations:

$$\mathcal{A}(\mathbf{n}, \mathbf{v}, \boldsymbol{\ell}) = [\mathbf{n}^\top \mathbf{h}, \boldsymbol{\ell}^\top \mathbf{h}, \mathbf{n}^\top \boldsymbol{\ell}, \mathbf{n}^\top \mathbf{v}, (\mathbf{n}^\top \mathbf{h})^{10}]^\top$$

where $\mathbf{h} = \overline{\boldsymbol{\ell} + \mathbf{v}}$ is the half-angle vector.

Design rationale for angular encoding:
- Converts normal–view–light directions into **rotation-invariant** scalar features
- Points sharing the same BRDF but with different world-coordinate normals are more readily mapped to the same latent code
- The MLP is relieved from implicitly learning rotation invariance, reducing learning difficulty

Distinction from prior work: NeRFactor requires pre-training the latent BRDF on measured BRDF datasets, whereas this work demonstrates that a **neural latent code–driven BRDF can be optimized from scratch from multi-view OLAT images**.

### Key Design 4: Shadow Modeling

**Two-step shadow strategy**:

1. **SDF-based volume rendering shadow**: A shadow ray is cast from the estimated surface intersection point along the light direction, and the shadow factor is computed via transmittance accumulation:

$$s = 1 - \sum_k T_k^{(s)} \alpha_k^{(s)}$$

2. **Shadow MLP refinement**: Volume-rendered shadows are nearly binary, yet shadow regions in practice retain residual brightness due to inter-reflections. A Shadow MLP refines this:

$$s' = \mathcal{S}(\mathbf{b}(\mathbf{x}'), s, \mathbf{v}; \varphi)$$

### Key Design 5: Lighting Self-Calibration

Each light source is parameterized by a direction $\boldsymbol{\ell}_j \in \mathcal{S}^2$ and RGB intensity $\mathbf{e}_j \in \mathbb{R}_+^3$ in camera coordinates. During rendering, the camera rotation matrix $\mathbf{R}_i$ transforms light directions to world coordinates. Light source parameters are optimized jointly with scene parameters, requiring no prior calibration.

### Key Design 6: Support for View-Unaligned OLAT

Traditional MVPS requires view-aligned OLAT (a complete illumination stack per viewpoint). This method additionally supports **view-unaligned** acquisition: one light source is kept active while a set of multi-view images is captured, then the next light source is activated for another set. Viewpoints across different light sources need not correspond one-to-one, substantially increasing practical acquisition flexibility.

### Loss & Training

The cached notes are partially truncated, but based on the framework design the core losses can be inferred as:
- **Weighted L1 color loss**: more robust to outliers than L2 loss
- **SDF regularization loss**: Eikonal regularization to enforce valid SDF
- **Foreground mask loss**: constrains the rendered silhouette to be consistent with the input mask

## Key Experimental Results

### Main Results

The cached notes are truncated and the experimental section is incomplete. Based on claims in the abstract and introduction:

- **Outperforms SOTA normal-guided methods** (e.g., SuperNormal) in shape reconstruction accuracy
- Surpasses multi-stage methods in illumination estimation accuracy
- Remains robust under sparse or even zero illumination variation, where multi-stage methods degrade due to inaccurate PS estimation
- Qualitative results validated on view-unaligned OLAT images
- Successfully handles challenging reflectance materials including ceramic and bronze metal

### Key Findings

- End-to-end optimization effectively avoids geometric artifacts caused by cross-view normal inconsistency in multi-stage pipelines
- Angular encoding significantly improves BRDF learning quality and overall reconstruction accuracy
- Shadow MLP effectively compensates for residual brightness in shadow regions caused by inter-reflections
- Light source directions and relative intensities can be recovered without prior calibration

## Highlights & Insights

1. **Philosophical correctness of end-to-end design**: Multi-stage pipelines are artifacts of historical constraints; end-to-end joint optimization fully exploits all available observations. The paper provides an information-theoretic argument that per-view independent PS discards complementary cross-view information.
2. **Elegant angular encoding design**: The 5-dimensional scalar feature $[\mathbf{n}^\top\mathbf{h}, \boldsymbol{\ell}^\top\mathbf{h}, \mathbf{n}^\top\boldsymbol{\ell}, \mathbf{n}^\top\mathbf{v}, (\mathbf{n}^\top\mathbf{h})^{10}]$ precisely captures the physical invariants of BRDFs; the $(\mathbf{n}^\top\mathbf{h})^{10}$ term is particularly clever in modeling the concentrated nature of specular reflection.
3. **Two-step shadow strategy is pragmatically effective**: A physical model (transmittance accumulation) yields a coarse shadow, and an MLP handles non-ideal effects such as inter-reflections—balancing physical correctness and practical flexibility.
4. **Lighting self-calibration eliminates dependence on calibration equipment**: This substantially lowers the barrier to deploying MVPS in practice.
5. **Practical value of view-unaligned acquisition**: The traditional requirement of a complete OLAT stack per viewpoint is inconvenient in real-world settings; supporting unaligned acquisition greatly enhances the method's usability.

## Limitations & Future Work

1. Severe cache truncation; the absence of complete quantitative experiments and ablation studies precludes comprehensive evaluation.
2. Directional lighting is assumed; the method may not generalize to point light sources or complex illumination conditions.
3. Inter-reflections are neglected, which may introduce errors for concave geometry or highly reflective materials.
4. The computational cost of end-to-end optimization may be substantially higher than multi-stage methods due to online volume rendering and shadow ray casting.
5. Test scenes are primarily small-to-medium-scale objects; scalability to large scenes is unverified.
6. Directly predicting BRDF via a single MLP offers less interpretability than analytic models—physical parameters such as diffuse albedo and roughness are not directly accessible.

## Related Work & Insights

- **Multi-stage MVPS**: SuperNormal (normal-guided), PS-NeRF (semi-end-to-end), SVNL (requires shadow cues)
- **End-to-end MVPS**: DPIR (point-based volume rendering, requires known lighting)
- **Neural inverse rendering**: NeRFactor, NvDiffRec, InvRender
- **Neural BRDF**: analytic BRDF (Disney BRDF), basis-function BRDF (spherical Gaussians), latent code–driven BRDF
- **Neural SDF**: NeuS, VolSDF, instant-ngp

## Rating

- **Novelty**: ★★★★☆ — The end-to-end MVPS framework design (without PS cues or light calibration) is conceptually clear and convincing.
- **Technical Depth**: ★★★★★ — The image formation model is rigorous; the physical motivation behind angular encoding and shadow modeling is well-grounded.
- **Experimental Quality**: ★★★☆☆ — Cache truncation prevents full assessment; based on stated claims, the scope of comparative experiments appears reasonable.
- **Practicality**: ★★★★☆ — Eliminating light calibration and view-alignment constraints substantially improves acquisition flexibility, with promising applications in cultural heritage digitization and material scanning.
- **Clarity of Presentation**: ★★★★★ — Derivations are detailed and rigorous; scene parameterization and the rendering pipeline are described clearly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](../../AAAI2026/llm_evaluation/dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)
- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](../../ACL2026/llm_evaluation/diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)
- [\[NeurIPS 2025\] AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](../../NeurIPS2025/llm_evaluation/adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](../../ACL2026/llm_evaluation/when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)

</div>

<!-- RELATED:END -->
