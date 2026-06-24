---
title: >-
  [Paper Note] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics
description: >-
  [CVPR 2025][Polarized Inverse Rendering] NeISF++ extends polarized inverse rendering from supporting only dielectrics to supporting both conductors and dielectrics. By introducing a generalized pBRDF model with a binary control variable $m$, complex refractive index modeling, and DoLP geometric initialization, it reduces the normal angular error on synthetic conductor scenes to 1.789° (an 83% reduction compared to NeISF's 10.303°).
tags:
  - "CVPR 2025"
  - "Polarized Inverse Rendering"
  - "Conductors"
  - "Complex Refractive Index"
  - "Stokes Vector"
  - "pBRDF"
date: 2026-05-08
content_hash: e9b67e4b345bdc2c
---

# NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics

**Conference**: CVPR 2025  
**arXiv**: [2411.10189](https://arxiv.org/abs/2411.10189)  
**Code**: To be released  
**Area**: Others  
**Keywords**: Polarized Inverse Rendering, Conductors, Complex Refractive Index, Stokes Vector, pBRDF

## TL;DR

NeISF++ extends polarized inverse rendering from supporting only dielectrics to supporting both conductors and dielectrics. By introducing a generalized pBRDF model with a binary control variable $m$, complex refractive index modeling, and DoLP geometric initialization, it reduces the normal angular error on synthetic conductor scenes to 1.789° (an 83% reduction compared to NeISF's 10.303°).

## Background & Motivation

1. **Background**: Polarized-based inverse rendering (such as NeISF, PANDORA) utilizes Stokes vector information captured by polarization cameras to reconstruct geometry, materials, and illumination. However, existing methods only support dielectrics (e.g., plastic, glass) and cannot handle conducting materials such as metals.
2. **Limitations of Prior Work**: (1) Conductors lack subsurface scattering (the diffuse term is zero), causing existing pBRDF models to fail completely; (2) Fresnel reflection of conductors requires a complex refractive index $ior = \eta - ki$, while existing methods only support real-valued refractive indices; (3) The strong specular reflection of conductors causes geometry initialization based on Normalized Cross-Correlation (NCC) / normal consistency to fail.
3. **Key Challenge**: In real-world scenes, conductors and dielectrics often coexist (e.g., a metal base with a plastic body), but existing methods can only handle one of these classes, making a unified framework highly necessary.
4. **Goal**: To design a polarized inverse rendering method that simultaneously supports both conductors and dielectrics.
5. **Key Insight**: Introduce a binary marker $m$ ($m=0$ for conductors, $m=1$ for dielectrics) to control the toggle of the diffuse term in the pBRDF, and utilize DoLP (Degree of Linear Polarization) instead of appearance for geometry initialization, as DoLP is invariant to light intensity and robust for conductors.
6. **Core Idea**: Generalized pBRDF + Complex Refractive Index + DoLP Initialization.

## Method

### Overall Architecture

Multi-view polarized images → DoLP computation → VolSDF geometry initialization (substituting normal consistency with DoLP consistency) → Joint optimization of SDF geometry $\mathbb{S}$, pBRDF material field $\mathbb{B}$ (outputting roughness $r$, albedo $a$, real refractive index $\eta$, extinction coefficient $k$), and incident Stokes field $\mathbb{L}$ → Optimization by rendering polarized images and comparing with observations.

### Key Designs

1. **Generalized Polarized BRDF (pBRDF)**

    - **Function**: Unify the modeling of polarized reflection for both conductors and dielectrics.
    - **Mechanism**: Introduce a binary marker $m$ based on the Baek pBRDF: $\mathbf{s} = \int_\Omega m \cdot \mathbf{R}^{cam}_{dif} \cdot \mathbf{M}_{dif} \cdot \mathbf{s}^r_{dif} + \mathbf{R}^{cam}_{spec} \cdot \mathbf{M}_{spec} \cdot \mathbf{s}^r_{spec} \, d\omega_i$. When $m=0$ (conductor), the diffuse term vanishes, leaving only the specular term.
    - **Design Motivation**: The physical nature of conductors is the absence of subsurface scattering; electromagnetic waves are entirely reflected at the metal surface. The introduction of $m$ directly maps this physical fact.

2. **Complex Refractive Index Modeling**

    - **Function**: Accurately describe the Fresnel reflection characteristics of conductors.
    - **Mechanism**: The material field outputs the complex refractive index $ior = \eta - ki$, where $\eta \in \mathbb{R}^3$ (real part, determining reflectivity) and $k \in \mathbb{R}^3$ (imaginary part/extinction coefficient, determining absorption). This leverages PyTorch's support for complex automatic differentiation.
    - **Design Motivation**: The colors of metals originate from wavelength-dependent complex refractive indices—copper's reddish hue and gold's yellowish hue are manifestations of $k(\lambda)$. Failing to model $k$ makes it impossible to recover metallic colors.

3. **DoLP Geometric Initialization**

    - **Function**: Provide robust geometric priors for conductor scenes.
    - **Mechanism**: The DoLP, defined as $\rho = \sqrt{s[1]^2 + s[2]^2}/s[0]$, is invariant to light intensity (since both the numerator and denominator scale simultaneously). The consistency of several $\rho$ observations is utilized to replace the traditional NCC consistency for VolSDF initialization.
    - **Design Motivation**: The strong specular reflection of conductors causes traditional NCC to fail (the color of the same point is completely different from different viewing angles), whereas DoLP reflects the degree of polarization rather than brightness.

### Loss & Training

Initialization: $\mathcal{L}_{init} = \lambda_\rho L_1(\rho, \hat\rho) + \lambda_I L_1(I, \hat I) + \lambda_{Eik} L_{Eik}$. Joint optimization: $\mathcal{L}_{joint} = \lambda_{\rho_s} L_1(\rho_s, \hat\rho) + \lambda_s L_1(s, \hat s) + \lambda_{Eik} L_{Eik}$. A conductor-dielectric mask needs to be provided by the user.

## Key Experimental Results

### Main Results

| Method | Helmet Normal Error ↓ | Stanford Scan Normal Error ↓ |
|------|-----------------|----------------------|
| VolSDF | 8.829° | 11.754° |
| PANDORA | 13.212° | 21.740° |
| NeRO | 5.001° | 13.352° |
| NeISF | 10.303° | 14.022° |
| **NeISF++** | **1.789°** | **6.487°** |

### Ablation Study

| Configuration | Helmet Normal Error ↓ | Description |
|------|-----------------|------|
| w/o DoLP Initialization | 2.400° | DoLP initialization contributes 0.6° |
| w/ DoLP Initialization | **1.789°** | Full model |
| VolSDF (Traditional NCC) | 8.829° | NCC fails on conductors |
| VolSDF-DoLP | 4.715° | Significant improvement is achieved solely by using DoLP |

### Key Findings

- NeISF++ achieves a normal error of only 1.789° on the conducting scene (Helmet), which is 83% lower than NeISF's 10.303°—proving that conductor modeling is essential.
- DoLP initialization alone reduces the VolSDF error from 8.829° to 4.715°, serving as an effective standalone technique.
- Material decomposition accuracy: The MAE of roughness drops from 0.2075 in NeISF to 0.0161, an improvement of an order of magnitude.

## Highlights & Insights

- **Physics-driven Method Design**: The $m$ marker, complex refractive index, and DoLP directly correspond to the physical characteristics of conductors. Instead of blindly stacking modules, every design has a solid physical foundation.
- **The Robustness of DoLP is a Valuable Finding**: The illumination-invariant property of DoLP might also be useful in high-reflection scenes beyond conductors, such as wet ground surfaces.
- **Engineering Value of PyTorch's Complex Auto-Differentiation**: It demonstrates that modern frameworks naturally support backpropagation of complex numbers, lowering the engineering barrier for complex refractive index modeling.

## Limitations & Future Work

- Manual labeling of the conductor-dielectric mask is required, which is the most time-consuming step.
- Dielectric regions are still assumed to have a fixed refractive index of 1.5, without estimation.
- Currently, only linear polarization is supported, while circular polarization signals are ignored.
- There is currently no capability for automatic material segmentation.

## Related Work & Insights

- **vs NeISF**: Supports only dielectrics. NeISF++ extends to conductors through the $m$ marker and complex refractive indices.
- **vs NeRO**: An environment map-based inverse rendering method that does not utilize polarization information. NeISF++ utilizes polarization to provide additional constraints for more accurate reconstruction.
- **vs PANDORA**: Also uses polarization but assumes everything is dielectric, yielding an error of 13.2° on conducting scenes compared to 1.8° for NeISF++.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to extend polarized inverse rendering to mixed conductor-dielectric scenes.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic and real-world data, though the number of scenes is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear physical derivation.
- Value: ⭐⭐⭐⭐ An important extension to polarized inverse rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TensoFlow: Tensorial Flow-based Sampler for Inverse Rendering](tensoflow_tensorial_flow-based_sampler_for_inverse_rendering.md)
- [\[CVPR 2025\] Locally Orderless Images for Optimization in Differentiable Rendering](locally_orderless_images_for_optimization_in_differentiable_rendering.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[CVPR 2025\] UniPhy: Learning a Unified Constitutive Model for Inverse Physics Simulation](uniphy_learning_a_unified_constitutive_model_for_inverse_physics_simulation.md)
- [\[CVPR 2025\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)

</div>

<!-- RELATED:END -->
