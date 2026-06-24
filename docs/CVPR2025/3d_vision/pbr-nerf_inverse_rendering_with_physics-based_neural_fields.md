---
title: >-
  [Paper Note] PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields
description: >-
  [CVPR 2025][3D Vision][Inverse Rendering] Based on NeILF++, PBR-NeRF introduces two physics-based prior losses (conservation of energy loss and NDF-weighted specular loss), effectively constraining the material-lighting decomposition ambiguity in inverse rendering. It achieves SOTA material estimation without sacrificing the quality of novel view synthesis.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Inverse Rendering"
  - "Material Estimation"
  - "Physics Priors"
  - "NeRF"
  - "BRDF"
date: 2026-05-08
content_hash: dc9b5aaa37d8a9d2
---

# PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields

**Conference**: CVPR 2025  
**arXiv**: [2412.09680](https://arxiv.org/abs/2412.09680)  
**Code**: [https://github.com/s3anwu/pbrnerf](https://github.com/s3anwu/pbrnerf)  
**Area**: 3D Vision  
**Keywords**: Inverse Rendering, Material Estimation, Physics Priors, NeRF, BRDF

## TL;DR

Based on NeILF++, PBR-NeRF introduces two physics-based prior losses (conservation of energy loss and NDF-weighted specular loss), effectively constraining the material-lighting decomposition ambiguity in inverse rendering. It achieves SOTA material estimation without sacrificing the quality of novel view synthesis.

## Background & Motivation

**Background**: NeRF and 3DGS have made significant progress in novel view synthesis, but they treat the scene as a "black box", modeling only view-dependent appearance without distinguishing between scene material and illumination. This means they cannot support downstream tasks such as relighting and material editing. Inverse rendering methods like NeILF++ attempt to jointly estimate geometry, materials, and lighting, but face severe ambiguity challenges.

**Limitations of Prior Work**: Existing inverse rendering methods lack effective physical priors when decomposing material and illumination. The Lambertian prior used in NeILF++ is overly strong, forcing the assumption that the material is completely rough and non-metallic, which suppresses the specular lobe of the BRDF and leads to specular information being wrongly "baked" into the diffuse albedo. Although this does not affect RGB rendering quality, it drastically reduces the accuracy of material estimation.

**Key Challenge**: The fundamental difficulty of inverse rendering lies in the material-lighting ambiguity—the same image can be explained by an infinite number of combinations of materials, lighting, and geometry. Existing methods either use overly strong priors that limit expressiveness, or lack sufficient constraints resulting in inaccurate material estimation.

**Goal**: Design more reasonable physical priors to constrain inverse rendering, ensuring the physical validity of the BRDF while promoting the correct separation of diffuse and specular components.

**Key Insight**: Starting from physically-based rendering (PBR) theory, the authors observe that the Disney BRDF model itself does not guarantee energy conservation, and the diffuse lobe tends to overcompensate in specular reflection directions. These two physical-level defects can be corrected using explicit loss functions.

**Core Idea**: Directly constrain the behavior of the Disney BRDF through two intuitive physical losses—conservation of energy loss and NDF-weighted specular loss—thereby significantly improving material estimation quality without requiring additional networks or data.

## Method

### Overall Architecture

The overall pipeline of PBR-NeRF inherits the implicit differentiable renderer (IDR) of NeILF++, which contains three neural networks: (1) a NeRF SDF network predicting geometry (density) and color; (2) a NeILF MLP predicting incoming radiance; (3) a BRDF MLP predicting Disney BRDF parameters (albedo $b$, roughness $r$, metallicness $m$). The rendering equation is discretely approximated using a fixed set of incoming directions $S_L$ (256 directions) sampled via Fibonacci sampling. The core innovation of PBR-NeRF lies in introducing two new physical losses during the material estimation stage, which act on the diffuse and specular components of the Disney BRDF.

### Key Designs

1. **Conservation of Energy Loss**:

    - **Function**: Ensures the BRDF does not violate the law of conservation of energy, i.e., the total reflected energy does not exceed the incident energy.
    - **Mechanism**: Imposes an upper-bound constraint on the weighted integral of the BRDF over all incident directions. The discretized form is $\mathcal{L}_{\text{cons}} = \max\{(\frac{2\pi}{|S_L|}\sum f_r(\omega_i \cdot \mathbf{n})) - 1, 0\}$, employing a ReLU-style penalty—only generating a loss when the sum of reflection weights exceeds 1.
    - **Design Motivation**: The Disney BRDF and other microfacet models do not inherently guarantee energy conservation. Without constraints, the material might "create" energy (reflecting more light than received), which causes the estimated lighting to be darkened as compensation, thereby affecting downstream tasks like relighting.

2. **NDF-weighted Specular Loss**:

    - **Function**: Promotes the correct separation of the diffuse lobe and specular lobe of the BRDF, eliminating the "baked illumination" phenomenon.
    - **Mechanism**: Uses the Normal Distribution Function (NDF) as a weight to penalize the magnitude of the diffuse component in the specular reflection direction. For each incident direction, the NDF value at the half vector is calculated, normalized via softmax, and then used to weight the diffuse lobe. The NDF term is detached to block gradient flow, ensuring that only the diffuse parameters are updated.
    - **Design Motivation**: Without constraints, the diffuse lobe tends to overcompensate at specular reflection angles. By suppressing the diffuse reflection in specular areas, it indirectly forces the specular lobe to expand to explain the specular effects.

3. **Three-stage Joint Optimization**:

    - **Function**: Stably and jointly optimizes geometry, material, and lighting.
    - **Mechanism**: Splits training into three stages—(1) Geometry stage: only trains the NeRF SDF to initialize geometry; (2) Material stage: freezes the SDF and trains the NeILF and BRDF networks; (3) Joint stage: trains all networks together.
    - **Design Motivation**: Staged training avoids optimization difficulties caused by the coupling of geometry, material, and lighting.

### Loss & Training

The total loss includes the geometry loss and the material loss. Key hyperparameters are $\lambda_{\text{cons}}=0.01$, and $\lambda_{\text{spec}}$ is 0.5 on the NeILF++ dataset and 0.01 on DTU. Training is conducted on a single A6000 GPU for approximately 3-7.5 hours.

## Key Experimental Results

### Main Results

Comparison of material estimation on the NeILF++ dataset (averaged over 6 lighting conditions):

| Metric | Method | PSNR | SSIM |
|--------|------|------|------|
| RGB | NeILF++ | 30.51 | 86.59 |
| RGB | PBR-NeRF | **31.27** | **87.05** |
| Albedo | NeILF++ | 17.29 | 75.87 |
| Albedo | PBR-NeRF | **20.08** | **86.82** |
| Roughness | NeILF++ | 21.83 | 91.56 |
| Roughness | PBR-NeRF | **22.47** | **92.31** |
| Metallicness | NeILF++ | 18.84 | 83.46 |
| Metallicness | PBR-NeRF | **21.62** | **74.08** |

### Ablation Study

| Configuration | Albedo PSNR | RGB PSNR | Description |
|------|------------|----------|------|
| Full PBR-NeRF | 20.08 | 31.27 | Full model |
| w/o Conservation Loss | ~18.5 | ~31.0 | Without conservation of energy loss |
| w/o Specular Loss | ~18.0 | ~30.8 | Without specular separation loss |
| NeILF++ baseline | 17.29 | 30.51 | Without both losses |

### Key Findings

- The improvement in Albedo estimation is the most significant (+2.79 dB PSNR), indicating that the two physical losses effectively solve the specular baking problem.
- RGB rendering quality increases instead of decreasing (+0.76 dB), demonstrating that correct material-lighting decomposition also benefits novel view synthesis.
- The improvement is particularly noticeable under mixed lighting (Mix) scenarios, as near-field light sources and area light sources exaggerate the material-lighting ambiguity.
- The method is relatively robust to the choice of hyperparameters, requiring only fine-tuning of the specular loss weight across the two datasets.

## Highlights & Insights

- **Simplicity and Universality of Physical Losses**: Both losses act directly on the BRDF and do not rely on specific rendering frameworks or network architectures. Theoretically, they can be plugged into any inverse rendering system that uses the Disney BRDF.
- **Clever Application of the detach Operation**: Detaching the NDF term in the specular loss ensures that gradients only flow to the diffuse parameters without altering the NDF (roughness), preventing instability of both lobes being adjusted simultaneously.
- **Material Improvement without Sacrificing NVS Quality**: Many inverse rendering works degrade RGB rendering quality while pursuing correct material decomposition; PBR-NeRF achieves both simultaneously.

## Limitations & Future Work

- The method still relies on the three-stage training of NeILF++, leading to long training times (3-7.5 hours), significantly slower than 3DGS-based methods.
- Quantitative material evaluation is only performed on synthetic data; real-world scenes lack ground-truth materials.
- The conservation of energy loss only prevents energy creation but does not handle energy destruction issues.
- Comparison with 3DGS-based inverse rendering methods is missing.

## Related Work & Insights

- **vs NeILF++**: Direct base model. Achieves maximum gains with minimal modifications by adding only two loss terms.
- **vs PhySG/SG-ENV**: Early inverse rendering methods, trailing behind by a large margin on various metrics.
- **vs NeRFactor/TensoIR**: Full material-lighting decomposition but with higher computational cost; PBR-NeRF is more lightweight.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of the two physical losses is simple and elegant, but the overall framework represents an incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Exhaustive quantitative evaluation on two datasets, yet lacks real-world scene material GT.
- Writing Quality: ⭐⭐⭐⭐⭐ Formulas are clearly derived and the physical motivation is thoroughly explained.
- Value: ⭐⭐⭐⭐ The plug-and-play physical prior design holds practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)
- [\[CVPR 2025\] IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)
- [\[CVPR 2025\] FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)
- [\[CVPR 2025\] Preconditioners for the Stochastic Training of Neural Fields](preconditioners_for_the_stochastic_training_of_neural_fields.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)

</div>

<!-- RELATED:END -->
