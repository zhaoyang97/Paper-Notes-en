---
title: >-
  [Paper Note] A Real-world Display Inverse Rendering Dataset
description: >-
  [ICCV 2025][LLM Evaluation][Inverse Rendering] This paper presents the first real-world inverse rendering dataset built upon an LCD display-camera system…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Inverse Rendering"
  - "Display-Camera System"
  - "OLAT Illumination"
  - "Polarization Imaging"
  - "Photometric Stereo"
date: 2026-05-08
content_hash: 6303c90313b44d66
---

# A Real-world Display Inverse Rendering Dataset

**Conference**: ICCV 2025
**arXiv**: [2508.14411](https://arxiv.org/abs/2508.14411)  
**Code**: [https://michaelcsj.github.io/DIR/](https://michaelcsj.github.io/DIR/)  
**Area**: Computer Vision / Inverse Rendering
**Keywords**: Inverse Rendering, Display-Camera System, OLAT Illumination, Polarization Imaging, Photometric Stereo

## TL;DR
This paper presents the first real-world inverse rendering dataset built upon an LCD display-camera system, comprising stereo polarization images of 16 objects with diverse materials captured under OLAT illumination patterns alongside high-precision geometric ground truth. A simple yet effective display inverse rendering baseline is proposed, outperforming existing inverse rendering methods.

## Background & Motivation
- **Background**: Inverse rendering aims to recover geometry and reflectance from images. Existing methods rely on different imaging systems — light stages offer high-quality multi-light samples but are costly and bulky; flash photography requires repeated camera movement; displays, by contrast, are programmable, high-resolution, and compact.
- **Unique Advantages of Display-Camera Systems**: Each pixel can serve as a programmable point light source; LCD panels emit polarized light, naturally enabling diffuse/specular separation.
- **Key Challenge**: Despite their clear advantages, no publicly available real-world dataset exists for display-based systems. All existing inverse rendering datasets are collected using light stages, robotic rigs, or natural illumination, making it impossible to evaluate challenges unique to display systems (near-field lighting, low SNR, limited light-view angle sampling, etc.).
- **Goal**: To fill this gap by constructing the system, collecting data, providing benchmarks, and validating methods.

## Method

### Overall Architecture
The work consists of four components: (1) constructing and calibrating an LCD display plus stereo polarization camera system; (2) capturing polarization images of 16 objects under 144 OLAT illumination conditions; (3) providing geometric ground truth from structured-light scanning; (4) proposing a baseline inverse rendering method and evaluating existing approaches.

### Key Designs

1. **Display-Camera Imaging System**:

    - **Function**: Constructs an imaging system consisting of a Samsung Odyssey Ark LCD display and two FLIR polarization RGB cameras.
    - **Core Parameters**: Maximum display brightness is 600 cd/m²; per-pixel output is only 0.06 mcd. Display pixels are grouped into 144 superpixels ($16 \times 9$), each comprising $240 \times 240$ display pixels.
    - **Calibration**: (a) Spatially varying backlight $B_i$ modeling; (b) nonlinear response $\gamma$ calibration; (c) camera intrinsic and extrinsic calibration; (d) superpixel relative position estimation.
    - **Radiance Model**: $L_i = s(P_i + B_i)^\gamma$, where $s$ is a global scaling factor.

2. **Data Acquisition and Processing**:

    - **Function**: Captures polarization images of 16 objects with diverse materials under OLAT illumination and obtains geometric ground truth.
    - **Material Coverage**: Resin, ceramic, metallic paint, wood, clay, plastic, bronze, plaster, etc.
    - **Polarization Processing**: Converts four-angle polarization images into Stokes vectors $s_0, s_1, s_2$, separating specular reflection $I_{\text{specular}} = \sqrt{s_1^2 + s_2^2}$ and diffuse reflection $I_{\text{diffuse}} = s_0 - I_{\text{specular}}$.
    - **Geometric Ground Truth**: Obtained using an EinScan SP V2 high-precision 3D scanner (accuracy 0.05 mm); scanned meshes are aligned with images via mutual information.

3. **Image Formation Model and Arbitrary Illumination Synthesis**:

    - **Function**: Exploits the linear superposition property of incoherent light transport to support image synthesis under arbitrary display patterns.
    - **Core Formula**: $I(\mathcal{P}) = \text{clip}(\sum_{i=1}^{N} I_i \cdot s(P_i + B_i)^\gamma + \epsilon)$
    - **Design Motivation**: Researchers can synthesize images under arbitrary illumination patterns and adjust noise levels without re-capturing data.

4. **Baseline Inverse Rendering Method**:

    - **Function**: Proposes a simple and effective baseline for display inverse rendering.
    - **Pipeline**: (a) Estimate normal maps using analytical RGB photometric stereo; (b) estimate depth maps using RAFT Stereo; (c) iteratively optimize normals and reflectance via differentiable rendering with a Cook-Torrance BRDF basis representation.
    - **Key Technique**: Models spatially varying BRDFs as weighted sums of basis BRDFs to handle limited light-view angle sampling.
    - **Runtime**: Optimization completes in only 150 seconds.

### Loss & Training
- **Baseline Method**: Minimizes RMSE between rendered and input images.
- **Display Calibration**: Optimizes global scalar $s$, spatially varying backlight $B_i$, and nonlinearity exponent $\gamma$ to match rendered OLAT images with captured images.

## Key Experimental Results

### Main Results (Inverse Rendering Evaluation)

| Method | Illumination | PSNR ↑ | SSIM ↑ | MAE (Normal) ↓ |
|--------|-------------|--------|--------|----------------|
| SRSH | OLAT | 41.28 | 0.9895 | 25.25° |
| DPIR | OLAT | 34.30 | 0.9790 | 41.09° |
| IIR | OLAT | 38.20 | 0.9850 | 38.38° |
| **Ours** | **OLAT** | **39.33** | **0.9821** | **20.94°** |
| **Ours** | **Multiplexed** | **37.27** | **0.9766** | **23.97°** |

### Ablation Study (Photometric Stereo Evaluation — Normal Reconstruction MAE)

| Method | Type | Elephant | Owl | Cat | Pig | Mean |
|--------|------|----------|-----|-----|-----|------|
| Woodham | Calibrated | 27.02 | 26.60 | 21.05 | 17.02 | ~23° |
| PS-FCN | Calibrated | 20.26 | 15.17 | 10.61 | 15.80 | ~15° |
| SDM-UniPS | Uncalibrated | 18.83 | 14.37 | 9.70 | 15.33 | ~15° |
| UniPS | Uncalibrated | 25.14 | 17.34 | 19.69 | 25.77 | ~22° |

### Key Findings
- SDM-UniPS achieves the best performance under OLAT illumination; 144 OLAT images provide sufficient information for normal reconstruction.
- The proposed baseline consistently outperforms existing methods for display inverse rendering, effectively handling near-field lighting and backlight challenges.
- Using only 2 multiplexed illumination patterns yields reasonable normal reconstruction, though inverse rendering accuracy remains inferior to that achieved with 144 OLAT images.
- Applying polarization-based diffuse/specular separation improves normal reconstruction accuracy, though the degree of improvement varies across methods.
- Modeling light attenuation is critical — omitting it causes PSNR to drop from 39.78 to 37.43.

## Highlights & Insights
- **Pioneering Contribution**: The first real-world inverse rendering dataset targeting display-camera systems, filling an important research gap.
- **Thorough System Calibration**: Comprehensive calibration of backlight, nonlinearity, and geometry makes the dataset highly usable.
- **Polarization Separation**: Leveraging LCD polarization properties to separate diffuse and specular reflectance provides a unique advantage for downstream research.
- **Light-View Sampling Analysis**: Analysis in Rusinkiewicz coordinates reveals that the display system provides sufficient sampling along $\theta_h$ but limited coverage along $\theta_d$.
- **Synthesis Capability**: Based on the linearity of light transport, the dataset supports image synthesis under arbitrary illumination patterns and noise levels.

## Limitations & Future Work
- Per-pixel display luminance is extremely low (0.06 mcd), necessitating superpixel grouping at the cost of reduced illumination resolution.
- When superpixels are smaller than $240 \times 240$ pixels, captured images are too dark to be usable.
- Light-view angle sampling coverage is limited, particularly along the $\theta_d$ direction.
- The current setup supports only single-viewpoint OLAT capture; simultaneous multi-view acquisition is not supported.
- The number of objects (16) and material diversity can be further expanded.

## Related Work & Insights
- **vs. Light Stage Datasets**: Light stages provide far-field uniform illumination and denser angular sampling but are expensive and bulky; display systems are compact and low-cost yet face near-field effects.
- **vs. DiLiGenT and Similar Photometric Stereo Datasets**: Existing datasets use point light sources and do not address display-specific challenges such as backlight and near-field illumination.
- **vs. DDPS (Differentiable Display Photometric Stereo)**: DDPS uses 3D-printed objects with limited material diversity; this work captures real objects.

## Rating
- Novelty: ⭐⭐⭐⭐ — First display inverse rendering dataset, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluates a broad range of photometric stereo and inverse rendering methods with multi-dimensional ablations.
- Writing Quality: ⭐⭐⭐⭐ — System construction and dataset description are thorough and detailed.
- Value: ⭐⭐⭐⭐ — Provides a standardized benchmark for display-based inverse rendering research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/llm_evaluation/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[NeurIPS 2025\] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](../../NeurIPS2025/llm_evaluation/words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)
- [\[NeurIPS 2025\] Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking](../../NeurIPS2025/llm_evaluation/time_travel_is_cheating_going_live_with_deepfund_for_real-time_fund_investment_b.md)

</div>

<!-- RELATED:END -->
