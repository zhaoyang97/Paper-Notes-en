---
title: >-
  [Paper Note] ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects
description: >-
  [CVPR 2026][3D Vision][Polarized imaging] This paper presents ICTPolarReal, the first large-scale real-world polarized reflection and material dataset, capturing 218 everyday objects using an 8-camera…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Polarized imaging"
  - "material dataset"
  - "inverse rendering"
  - "reflection separation"
  - "Light Stage"
date: 2026-05-08
content_hash: 6cccd16484f37560
---

# ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects

**Conference**: CVPR 2026
**arXiv**: [2603.24912](https://arxiv.org/abs/2603.24912)
**Code**: [https://jingyangcarl.github.io/ICTPolarReal](https://jingyangcarl.github.io/ICTPolarReal) (project page)
**Area**: 3D Vision
**Keywords**: Polarized imaging, material dataset, inverse rendering, reflection separation, Light Stage

## TL;DR

This paper presents ICTPolarReal, the first large-scale real-world polarized reflection and material dataset, capturing 218 everyday objects using an 8-camera, 346-light Light Stage system under cross- and parallel-polarization configurations. The dataset comprises over 1.2 million high-resolution images with ground-truth diffuse–specular reflection separation, and demonstrably improves inverse rendering, forward relighting, and sparse-view 3D reconstruction.

## Background & Motivation

**Background**: Inverse rendering (intrinsic image decomposition) seeks to factorize images into albedo, illumination, and specular components. Recent diffusion-based methods (e.g., RGB2X, Diffusion Renderer) have achieved notable progress, but rely heavily on synthetic datasets (e.g., Objaverse, Hypersim) for training.

**Limitations of Prior Work**: Although visually plausible, synthetic data is constrained by simplified illumination models and limited material fidelity. Commonly used shading models employ analytic BRDFs or sparse-sample approximations of bidirectional reflectance, neglecting effects such as multiple scattering, polarization, and subsurface transport that are ubiquitous in real objects. Consequently, models trained solely on synthetic data generalize poorly to real illumination and real photographs.

**Key Challenge**: A lack of real-world reflectance measurement data. Existing real-world datasets either provide photographs under varying illumination without intrinsic decomposition annotations (Multi-Illumination), are restricted to planar samples and two viewpoints (OpenSVBRDF), or cover only a very limited number of objects and lighting patterns (Open Illumination), making them unsuitable for supervising deep material decomposition networks.

**Goal**: (1) Construct a large-scale real-world reflectance dataset spanning diverse material categories with ground-truth diffuse/specular separation; (2) Verify whether training on real measured data substantially improves inverse rendering and relighting models in real-world scenarios.

**Key Insight**: Leveraging polarization optics—cross-polarized and parallel-polarized filters physically separate diffuse and specular reflectance. Malus's law guarantees accurate extraction of both reflection components under specific polarization configurations.

**Core Idea**: Employ a polarized Light Stage system to perform large-scale measurements of real objects, yielding the first real-world material dataset capable of directly supervising deep inverse rendering models.

## Method

### Overall Architecture

The work comprises two main parts: design of the data acquisition system and computation of material data, and validation experiments training inverse rendering and forward rendering models on the resulting dataset. The input is a multi-view polarized image sequence of real objects captured in the Light Stage; the output is per-object material parameters including diffuse albedo, specular albedo, and surface normals.

### Key Designs

1. **Polarized Light Stage Capture System**:

    - **Function**: Multi-view, multi-illumination, polarization imaging of real objects under controlled conditions.
    - **Mechanism**: The system consists of 346 LED light sources mounted on a geodesic sphere and 8 synchronized RED Komodo 6K global-shutter cameras. Cross- or parallel-polarizing linear filters are placed in front of the LEDs; cameras are also equipped with rotatable polarizers. For each illumination direction, two polarized images are captured—$I_{\perp}$ (cross-polarized) and $I_{\parallel}$ (parallel-polarized)—using OLAT (one-light-at-a-time) acquisition triggered in a spiral order from the front to the rear hemisphere.
    - **Design Motivation**: Polarization separation is grounded in Malus's law: cross-polarized images retain only the diffuse component (specular reflections preserve polarization direction), while parallel-polarized images contain both components. Physical reflection separation is achieved via $I_d = 2I_{\perp}$, $I_s = 2I_{\parallel} - 2I_{\perp}$.

2. **Material Parameter Computation Pipeline**:

    - **Function**: Derive diffuse albedo, specular albedo, and surface normals from the polarized image sequences.
    - **Mechanism**: Polarization separation first yields a diffuse sequence $\Lambda_d$ and a specular sequence $\Lambda_s$. Applying Lambert's cosine law, diffuse albedo $\rho_d$ and surface normal $n$ are jointly estimated per pixel by minimizing $L = \{\rho_d |n \cdot \omega_k|\}_{k=0}^{N} - \Lambda_d$. Specular albedo $\rho_s$ is approximated by integrating the specular reflectance function over all illumination directions.
    - **Design Motivation**: The 346 OLAT lights with known directions provide an over-constrained system of equations, making the joint estimation of normals and albedo stable and reliable.

3. **Illumination Augmentation and Rendering Model Training**:

    - **Function**: Support arbitrary novel illumination synthesis and model training validation.
    - **Mechanism**: Arbitrary environment map textures are projected onto the unit sphere aligned with the calibrated light sources; per-light weights are computed and used to blend all OLAT images, synthesizing relighted results. Two training pipelines are designed: a PBR pipeline (predicting physical material components) and a polarization pipeline (predicting cross/parallel polarized images). LoRA fine-tuning of RGB2X is adopted to avoid catastrophic forgetting.
    - **Design Motivation**: The linear superposition property of OLAT acquisition enables accurate synthesis of images under arbitrary illumination while preserving ground-truth diffuse/specular separation.

### Loss & Training

Both inverse rendering and forward rendering networks are fine-tuned from RGB2X via LoRA. Inverse rendering uses a prompt-conditioned mechanism to control generation of different target components (e.g., "albedo" or "surface normal"). Forward rendering is supervised with an L2 loss, with irradiance maps provided as additional input. Training data is expanded through an illumination augmentation strategy comprising three types: OLAT, synthetic HDRI, and uniform white illumination.

## Key Experimental Results

### Main Results

**Inverse Rendering Decomposition (HDRI illumination, Light Stage data)**:

| Method | Albedo MSE↓ | Albedo PSNR↑ | Normal PSNR↑ | Specular PSNR↑ |
|--------|-------------|--------------|--------------|----------------|
| DR-IR (original) | 0.035 | 20.01 | 20.48 | 22.61 |
| RGB2X (original) | 0.040 | 18.08 | 18.58 | 17.21 |
| Ours (fine-tuned) | **0.005** | **33.51** | **28.09** | **31.02** |

**Forward Relighting (HDRI illumination, Light Stage data)**:

| Method | MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| DR-FR | 0.058 | 16.97 | 0.775 | 0.386 |
| RGB2X | 0.038 | 18.50 | 0.514 | 0.514 |
| Ours-PBR | **0.005** | **27.80** | **0.904** | **0.211** |
| Ours-Polarization | 0.007 | 26.13 | 0.909 | 0.200 |

### Ablation Study

**Sparse-View 3D Reconstruction (8-view input, 50 real objects)**:

| Input / Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------------|-------|-------|--------|
| Dust3r + original images | 14.51 | 0.226 | 0.604 |
| Dust3r + predicted diffuse | 17.78 | 0.411 | 0.556 |
| Dust3r + predicted albedo | **20.30** | **0.513** | **0.506** |
| Mast3r + original images | 12.72 | 0.193 | 0.613 |
| Mast3r + predicted albedo | **15.57** | **0.282** | **0.603** |

### Key Findings

- Fine-tuning with real polarized data improves albedo PSNR from 20 dB to 33.5 dB (a 13.5 dB leap), indicating a substantial domain gap between synthetic data and real reflectance.
- The polarization pipeline achieves the lowest LPIPS (0.200) on relighting, suggesting that polarization supervision leads to more accurate reflection modeling.
- Using specular-free diffuse images as input to 3D reconstruction improves Dust3r PSNR from 14.5 to 20.3, demonstrating that specular reflections are the primary source of interference in sparse-view reconstruction.

## Highlights & Insights

- **Elegant exploitation of physical polarization separation**: Malus's law enables fully physics-driven reflection separation without any learned components, providing accurate ground truth for the dataset. This "physics first, learning second" paradigm is broadly instructive.
- **"Virtual polarization" concept**: Training models to predict polarization-equivalent outputs from ordinary non-polarized inputs effectively endows standard cameras with polarization capability—an idea transferable to many tasks requiring specialized imaging.
- **Linear superposition for illumination augmentation**: The key advantage of OLAT acquisition lies in the linear superposition property of real reflectance, enabling the generation of accurately annotated training data under an unlimited range of illumination conditions.

## Limitations & Future Work

- Data acquisition is restricted to static objects in a controlled Light Stage environment, precluding highly transparent, dynamic, or strongly anisotropic materials.
- The dataset does not include subsurface scattering parameters.
- Although the 218 objects cover a broad range, the count remains limited, and certain material categories may be underrepresented.
- Future work may explore extending polarization measurement to more complex materials and in-the-wild capture scenarios.

## Related Work & Insights

- **vs. Objaverse/Hypersim (synthetic datasets)**: These datasets offer large-scale annotations but lack real material fidelity. The proposed dataset, while smaller in scale, provides the advantage of real physical measurements; the two are complementary.
- **vs. Multi-Illumination**: The latter provides multi-illumination real photographs but lacks intrinsic decomposition annotations, making direct supervision of inverse rendering infeasible. The proposed dataset addresses this critical gap through polarization.
- **vs. OpenSVBRDF**: The latter is limited to planar samples and two viewpoints; this work extends coverage to full 3D objects and 8 viewpoints.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale real-world polarized material dataset, filling an important gap, though the methodology is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three downstream tasks—inverse rendering, relighting, and 3D reconstruction—with comparisons across multiple illumination conditions.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of the underlying physics.
- Value: ⭐⭐⭐⭐⭐ The dataset is of exceptional value and has the potential to become foundational infrastructure for the inverse rendering community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AnthroTAP: Learning Point Tracking with Real-World Motion](anthrotap_learning_point_tracking_with_real-world_motion.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)
- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](../../ICCV2025/3d_vision/demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](../../ICCV2025/3d_vision/revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)

</div>

<!-- RELATED:END -->
