---
title: >-
  [Paper Note] Thermal Polarimetric Multi-view Stereo
description: >-
  [ICCV2025][thermal imaging] This paper proposes a method for high-fidelity 3D shape reconstruction using thermal polarimetric (long-wave infrared polarimetric) cues. It theoretically demonstrates that LWIR polarimetric observations are unaffected by illumination conditions and material optical properties, enabling accurate 3D reconstruction of transparent, translucent, and heterogeneous objects—significantly outperforming visible-light polarimetric methods.
tags:
  - ICCV2025
  - thermal imaging
  - polarimetric imaging
  - LWIR
  - multi-view stereo
  - SDF
  - shape from polarization
date: 2026-05-08
content_hash: 5efecad01d9c25da
---

# Thermal Polarimetric Multi-view Stereo

**Conference**: ICCV2025
**arXiv**: [2510.20972](https://arxiv.org/abs/2510.20972)
**Code**: None
**Area**: 3D Reconstruction / Thermal Imaging / Polarimetric Imaging
**Keywords**: thermal imaging, polarimetric imaging, LWIR, multi-view stereo, SDF, shape from polarization

## TL;DR

This paper proposes a method for high-fidelity 3D shape reconstruction using thermal polarimetric (long-wave infrared polarimetric) cues. It theoretically demonstrates that LWIR polarimetric observations are unaffected by illumination conditions and material optical properties, enabling accurate 3D reconstruction of transparent, translucent, and heterogeneous objects—significantly outperforming visible-light polarimetric methods.

## Background & Motivation

### Root Cause

**Key Challenge**: 3D shape reconstruction is a fundamental problem in computer vision, yet existing methods rely on strong assumptions about illumination and material properties:

- **Multi-view stereo**: relies on surface texture for correspondence matching
- **Structured light / photometric stereo**: requires specific illumination conditions and assumes opaque surfaces
- **Visible-light polarization (SfP)**: depends on illumination conditions and is affected by ambiguities from mixed specular and diffuse polarization

For transparent objects (glass), translucent objects (plastic), and low-reflectance objects (black surfaces), these methods suffer from severe limitations.

### Advantages of Thermal Imaging

Thermal imaging offers an attractive alternative:
- **Illumination-independent**: any object with temperature emits long-wave infrared (LWIR) radiation
- **Material opacity**: the vast majority of materials are opaque in the LWIR spectrum (except materials specifically designed for thermal optics)
- **Self-emission**: objects themselves act as light sources, requiring no external illumination

### Limitations of Existing Thermal 3D Reconstruction

- **Thermal multi-view stereo**: still relies on texture matching
- **Thermal photometric stereo**: requires actively heating/cooling objects, which is time-consuming and impractical
- **Thermal NeRF**: yields lower geometric accuracy
- **Depth estimation via absorption**: limited accuracy

### Core Insight: LWIR Polarization Is Unambiguous

In the visible spectrum, polarimetric observations are a mixture of specular, diffuse, and transmitted components whose relative intensities vary in complex ways with material and illumination. In the LWIR spectrum, however, most objects are opaque (transmission is negligible), and at ambient temperatures the reflected component is far smaller than the emitted component. Consequently, LWIR polarimetric observations reduce to pure emission polarization, which can be analytically expressed using Kirchhoff's law and the Fresnel equations—without ambiguity.

## Method

### LWIR Polarization Theory

#### Physical Basis of Polarization

Polarization states are described using Stokes parameters and Mueller matrices. A complete observation comprises four components: specular reflection polarization, diffuse reflection polarization, transmitted polarization, and emitted polarization.

#### Visible-light vs. LWIR Polarization

**Visible light**: observation = diffuse reflection + specular reflection + transmission. The relative intensities of these three components vary in complex ways with material and illumination, making analysis difficult. In experiments, visible-light AoLP exhibits severe inconsistencies across materials (black stone: contamination from specular reflection; glass: contamination from transmission; translucent plastic: scattering interference).

**LWIR**: observation ≈ pure emission polarization. Emission polarization is determined by the Mueller matrix of blackbody radiation and the object temperature, and can be analytically expressed via Kirchhoff's law (emissivity = 1 − reflectivity) and the Fresnel equations—without ambiguity. In experiments, LWIR AoLP remains highly consistent across all materials.

#### From Polarization to Surface Normals

- **Degree of linear polarization (DoLP)**: correlated with the zenith angle, but depends on the material refractive index
- **Angle of linear polarization (AoLP)**: directly equals the azimuth angle of the surface normal, **entirely independent of material properties**

This is the core advantage of the proposed approach: AoLP provides robust, material-agnostic azimuthal constraints.

### 3D Reconstruction Method

#### Shape Representation

An implicit SDF is used (represented by an 8-layer MLP with softplus activation and positional encoding), where the zero level set defines the object surface. Optimization is performed within the IDR (Implicit Differentiable Renderer) framework.

#### Loss & Training

Total loss = Tangent Space Consistency (TSC) loss + silhouette loss + Eikonal regularization

**Tangent Space Consistency (TSC) loss** (from the MVAS method): AoLP defines a projected tangent vector to which the surface normal must be orthogonal. The set of tangent vectors from multiple views strongly constrains the normal direction. A key distinction: visible-light MVAS suffers from a half-cycle ambiguity in the normal direction and requires a modified TSC loss; the proposed LWIR method has no such ambiguity and can directly use the standard TSC loss.

**Silhouette loss**: constrains the visual hull using a cross-entropy loss.

**Eikonal regularization**: enforces the SDF gradient norm to remain close to 1.

### Experimental System

- **Thermal polarimetric camera**: FLIR Boson 320 + 15 mm lens + rotating wire-grid polarizer (acquisitions at 0/45/90/135 degrees)
- **Visible-light polarimetric camera**: FLIR Blackfly (for pose estimation and baseline comparison)
- **Calibration**: Aruco markers + dual-camera calibration using a heated white aluminum plate
- **Dataset**: 7 objects of varying material and geometry, 20–30 viewpoints, with structured-light scan ground truth

## Key Experimental Results

### Quantitative and Qualitative Results

Compared methods: visible-light MVAS, thermal IDR

The proposed method outperforms both baselines on all test objects in terms of Chamfer distance and mean angular error.

Key qualitative findings:

- **Ceramic owl (coated, reflective)**: visible-light MVAS and thermal IDR lose details such as eyes and beak; the proposed method successfully recovers them
- **Glass container (with internal utensils)**: relief texture is clearly visible with the proposed method, while other methods produce blurry results
- **Black cup with white lid**: visible-light AoLP on the lid is too noisy, causing wavy surfaces; LWIR polarization yields a reasonable reconstruction
- **Transparent vase and bottle**: concave details are accurately reconstructed; other methods produce blurring
- **Overall trend**: visible-light MVAS exhibits wavy artifacts (AoLP noise); thermal IDR produces inflated surfaces (no normal information)
- Thin components (bottle necks) show artifacts due to the low spatial resolution of thermal cameras

## Highlights & Insights

### Highlights

- Rigorous theoretical foundation: the unambiguous nature of LWIR polarization is derived from the Stokes–Mueller formalism
- Illumination-independent: completely decoupled from environmental lighting
- Material-agnostic: AoLP is independent of material refractive index, making the method applicable to transparent, translucent, and heterogeneous materials
- No heating/cooling required: steady-state measurements suffice
- Achieves best Chamfer distance and angular error across all tested objects

### Limitations & Future Work

- The assumption that the emitted component dominates breaks down when high-temperature objects are present in the scene or when the object temperature is below ambient
- Polarimetric signals on metallic or rough surfaces are unstable
- Low spatial resolution of thermal cameras makes reconstruction of thin components difficult
- Four acquisitions with a rotating polarizer are required rather than a single shot
- The custom acquisition system limits practical applicability

## Personal Reflections

1. The theoretical contribution is clear: rigorous physical derivation demonstrates that LWIR polarization is free from the ambiguities present in visible-light polarization, with AoLP directly yielding the azimuth angle.
2. Advances in thermal polarimetric cameras (future single-shot LWIR polarimetric cameras) would greatly enhance practicality.
3. Significant potential exists in complementing visible-light methods: visible light provides texture and high resolution, while LWIR provides material-agnostic normal constraints.
4. Direct application value exists in industrial inspection (transparent/reflective objects) and security surveillance (illumination-free scenarios).
5. It would be worthwhile to incorporate DoLP as an additional constraint (although it depends on the refractive index, joint estimation of the refractive index and shape could be explored).

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ICCV 2025\] Multi-view Gaze Target Estimation](multi-view_gaze_target_estimation.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)

<!-- RELATED:END -->
