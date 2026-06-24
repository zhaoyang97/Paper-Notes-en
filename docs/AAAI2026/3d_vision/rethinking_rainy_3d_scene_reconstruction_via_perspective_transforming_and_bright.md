---
title: >-
  [Paper Note] Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning
description: >-
  [AAAI 2026 Oral][3D Vision][Rainy 3D reconstruction] Proposes the OmniRain3D dataset (the first rainy 3D scene dataset that simultaneously models perspective heterogeneity and brightness dynamicity) alongside the REVR-GSNet end-to-end framework (joint recursive brightness enhancement + Gaussian primitives optimization + GS-guided deraining) to reconstruct high-fidelity clean 3D scenes from rain-degraded images.
tags:
  - "AAAI 2026 Oral"
  - "3D Vision"
  - "Rainy 3D reconstruction"
  - "3D Gaussian Splatting"
  - "deraining"
  - "brightness enhancement"
  - "end-to-end framework"
date: 2026-05-08
content_hash: 19c4fce9ff9395c4
---

# Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.06734](https://arxiv.org/abs/2511.06734)  
**Code**: [https://github.com/ncfjd/REVR-GSNet](https://github.com/ncfjd/REVR-GSNet)  
**Area**: 3D Vision  
**Keywords**: Rainy 3D reconstruction, 3D Gaussian Splatting, deraining, brightness enhancement, end-to-end framework

## TL;DR

Proposes the OmniRain3D dataset (the first rainy 3D scene dataset that simultaneously models perspective heterogeneity and brightness dynamicity) alongside the REVR-GSNet end-to-end framework (joint recursive brightness enhancement + Gaussian primitives optimization + GS-guided deraining) to reconstruct high-fidelity clean 3D scenes from rain-degraded images.

## Background & Motivation

### Background

3D scene reconstruction (e.g., NeRF, 3DGS) has achieved excellent results in clean scenarios. However, in adverse weather conditions like rain, multi-view images are affected by raindrop/rain streak occlusions and reduced visibility, breaking multi-view consistency and severely degrading reconstruction quality. This poses significant challenges for systems demanding all-weather operation, such as autonomous driving and robot navigation.

### Limitations of Prior Work

The authors identify two crucial features overlooked by existing works:

**Perspective Heterogeneity**: Raindrops in 3D space project onto 2D images under different perspectives with varying appearances (spreading as $\Lambda$-shapes when looking up, parallel horizontally, and converging as v-shapes when looking down; rain streaks tilt when the camera deviates from the rainfall direction). Existing datasets (such as HydroViews) directly overlay 2D rain layers linearly, lacking 3D physical consistency.

**Brightness Dynamicity**: In real rainy scenarios, cloud cover leads to reduced ambient light, and the heavier the rain, the darker the brightness. Although existing datasets (such as RainyScape) simulate rain effects in 3D space, they ignore the impact of rainfall on brightness, presenting a significant domain gap compared to real-world scenes.

### Key Challenge & Key Insight

Existing methods (DerainNeRF, DerainGS) typically adopt a **two-stage pipeline**: first removing degradation using a pre-trained deraining network, and then performing 3D reconstruction. This separated paradigm suffers from two limitations: (1) pre-trained models may overfit to specific rain patterns, and (2) they lack a brightness adjustment mechanism. This work proposes an end-to-end unified framework to handle both deraining and brightness restoration simultaneously.

## Method

### Overall Architecture

REVR-GSNet adopts a **joint alternating optimization** strategy, which involves the synergistic operation of three modules:

- **Phase 1**: Joint optimization of RBE (Recursive Brightness Enhancement) + GPO (Gaussian Primitives Optimization) — gradually increasing brightness and embedding it into 3DGS.
- **Phase 2**: Joint optimization of GPO + GRE (GS-guided Rain Elimination) — leveraging 3DGS-rendered images to guide deraining, with deraining results fed back to optimize 3DGS.
- **Phase 3**: GPO only — generating the final clean radiance field $V^M$.

### Key Designs

#### 1. OmniRain3D Dataset Construction

**Function**: Constructs the first rainy 3D scene dataset that simultaneously incorporates perspective heterogeneity and brightness dynamicity.

**Mechanism**:
- **Perspective Extraction**: Extracted all camera extrinsic parameters from clean background images using COLMAP to obtain inclination angle $\theta$ and azimuth angle $\phi$.
- **Dynamic Rain Streak Rendering**: Established a 3D rain model in Blender with six-dimensional meteorological parameters $S = \{\omega_{den}, \omega_{dep}, \omega_{str}, \omega_{dir}, \omega_{qty}, \omega_{scl}\}$ (density, depth, wind strength, wind direction, rain quantity, scale) to synchronously render rain streaks for each camera pose.
- **Adaptive Brightness Adjustment**: Based on the exponential decay model of the Beer-Lambert law:

$$L = L_0 e^{-\gamma \omega_{den}}$$

Three levels of rain density (light, medium, heavy rain) are configured to compute the corresponding brightness. Finally, **brightness-adjusted background + rain streaks of corresponding density = synthetic rainy image**.

Overall imaging model: $O_t(\theta_i, \phi_j) = L \odot (B_t(\theta_i, \phi_j) + R_t(\theta_i, \phi_j))$

**Design Motivation**: To overcome the limitations of HydroViews (only 2D overlay) and RainyScape (neglecting brightness), providing training/evaluation data closer to real-world scenarios.

#### 2. Recursive Brightness Enhancement (RBE)

**Function**: Progressively corrects the brightness of low-light rainy images.

**Mechanism**: Employs a lightweight CNN (CPEN, 7 convolutional layers + symmetric skip connections) to estimate brightness adjustment parameters, and recursively applies a quadratic brightness enhancement curve:

$$\mathbf{BE}(I_t, A_1) = I_t + A_1 I_t (1 - I_t)$$

It recurses for $n=4$ steps, gradually increasing brightness at each step with different parameters $A_a$.

**Design Motivation**: Single-step enhancement struggles with severe darkening. The recursive manner progressively approaches the target brightness, and the parameterized curve ensures controllability of the enhancement.

#### 3. Gaussian Primitives Optimization (GPO)

**Function**: Constructs and optimizes the 3D Gaussian scene representation using the enhanced multi-view images.

**Mechanism**:
- Estimize camera poses using COLMAP on the enhanced images $\{E_t^i\}$.
- Construct the 3DGS representation $V = \{\mu_z, \Sigma_z, \sigma_z, h_z\}$ (position, covariance, opacity, spherical harmonics coefficients).
- Optimize 3DGS attributes via differentiable rasterization.

**Key Insight**: Although the enhanced images still contain rain streaks, the radiance field optimization process effectively suppresses these artifacts by exploiting **cross-view consistency** and spatial correlation.

#### 4. GS-guided Rain Elimination (GRE)

**Function**: Guides the deraining process using reference images rendered by the current 3DGS.

**Mechanism**: Under multi-view fusion by 3DGS, the rendered image $R_t$ exhibits fewer rain artifacts and clearer structures than the enhanced image $E_t$. A Recursive Rain Estimation Network (RREN) is utilized:
- Concatenates $R_t$ and $E_t$ as input.
- Employs a recurrent U-Net architecture with LSTM cells and embedded Residual Recurrent Blocks (RRB).
- Recurses for $l=6$ steps, estimating the rain streak map $M_o$ at each step, and obtains the derained image $D_t$ via residual subtraction.
- Feeds back the derained image to GPO to continually optimize 3DGS.

$$D_t = \text{Cat}(R_t, E_t) - E_\phi(\text{Cat}(R_t, E_t))$$

**Design Motivation**: The 3DGS rendered images have already "implicitly derained" part of the corruption. Using them as guide signals helps the deraining network better distinguish between rain streaks and scene content.

### Loss & Training

- The overall framework is trained end-to-end on PyTorch with an RTX 3090.
- RBE and GRE utilize the Adam optimizer with an initial learning rate of $10^{-3}$.
- Different learning rates are configured for each 3DGS attribute in GPO (means: $1.6 \times 10^{-4}$, scaling: $5 \times 10^{-4}$, SH: $2.5 \times 10^{-3}$).
- All methods are trained for 30,000 steps for a fair comparison.

## Key Experimental Results

### Main Results

**OmniRain3D rain streak scenes** (normal brightness, 4 scenes):

| Scene | Metric | REVR-GSNet | DerainGS | RainyScape | DerainNeRF |
|------|------|-----------|---------|-----------|-----------|
| Francis | PSNR↑ | **24.56** | 23.40 | 22.99 | 16.17 |
| Garden | PSNR↑ | **25.35** | 25.30 | 22.58 | 21.74 |
| Garden | LPIPS↓ | **0.184** | 0.200 | 0.241 | 0.320 |
| Caterpillar | PSNR↑ | **21.48** | 20.26 | 19.90 | 13.99 |

**OmniRain3D low-light rainy scenes** (4 scenes, other methods require brightness pre-processing):

| Scene | Metric | REVR-GSNet | DerainGS† | RainyScape† | DerainNeRF† |
|------|------|-----------|----------|------------|------------|
| Bicycle | PSNR↑ | **19.06** | 18.88 | 18.63 | 18.13 |
| Family | PSNR↑ | **17.83** | 17.78 | 16.92 | 17.05 |
| Family | LPIPS↓ | **0.440** | 0.461 | 0.497 | 0.595 |

**HydroViews raindrop scenes** (average of 3 scenes):

| Scene | REVR-GSNet | RainyScape | DRSformer* | NeRD-Rain* |
|------|-----------|-----------|-----------|-----------|
| Stump (PSNR) | **22.61** | 22.59 | 18.23 | 19.79 |
| Stump (LPIPS) | **0.258** | 0.284 | 0.303 | 0.336 |

### Ablation Study

Component ablation on the HydroViews dataset:

| Configuration | GPO | RBE | GRE | PSNR↑ | SSIM↑ |
|------|-----|-----|-----|-------|-------|
| GPO Only | ✓ | | | 19.03 | 0.514 |
| GPO + RBE | ✓ | ✓ | | 22.71 | 0.615 |
| GPO + GRE | ✓ | | ✓ | 21.64 | 0.535 |
| **Full Model** | **✓** | **✓** | **✓** | **23.88** | **0.687** |

RBE contributes the most (PSNR +3.68), indicating that brightness restoration is crucial for low-light rainy reconstruction. GRE also contributes significantly (+2.61), with the combination of all three achieving the best results.

### Key Findings

1. The advantages of REVR-GSNet are more pronounced in low-light rainy scenes — other methods require brightness pre-processing and still underperform compared to the end-to-end scheme.
2. All baseline methods (3DGS, NeRF, RainyScape) perform better when trained on the OmniRain3D dataset than on HydroViews, validating the higher realism of the proposed dataset.
3. Luminance histogram analysis shows that the brightness distribution of OmniRain3D is closer to real-world rainy images.
4. It also demonstrates strong generalization performance on real-world rainy scenes.

## Highlights & Insights

1. **Precise Problem Definition**: It explicitly defines two overlooked key characteristics in rainy 3D reconstruction — "perspective heterogeneity" and "brightness dynamicity" — and constructs dedicated datasets accordingly.
2. **Closed-loop Design**: The alternating optimization of RBE → GPO → GRE forms a closed loop, where deraining improves reconstruction, and reconstruction in turn guides deraining, boosting each other.
3. **Physical Modeling**: The construction of OmniRain3D is based on physical models like the Beer-Lambert law, rather than simple image processing.
4. **High Practicality**: The end-to-end framework avoids error accumulation characteristic of multi-stage methods, generalising better to real-world scenarios.

## Limitations & Future Work

1. Although more realistic than HydroViews, the dataset remains synthetic and still has domain gaps with the real-world rainy conditions (e.g., fog and puddle specular reflections are not modeled).
2. The gain of REVR-GSNet is limited in certain scenes (e.g., in the Ignatius scene, PSNR is only 0.46 higher than DerainGS). An exceptionally complex scene may require a more powerful deraining module.
3. The brightness attenuation model assumes a simplistic exponential relationship between rain density and brightness, whereas real scenarios are also influenced by factors like time and light source direction.
4. Computational efficiency is not discussed — the alternating optimization strategy may increase training time.
5. Comparison is conducted against a limited number of baselines (5), lacking comparisons with more recent deraining methods.

## Related Work & Insights

- **Application of 3DGS under adverse weather** is an emerging direction; this work provides a paradigm reference for rainy scenarios.
- Similar ideas can be extended to other weather conditions such as fog (brightness attenuation $\rightarrow$ fog density attenuation) and snow.
- The recursive curve enhancement logic of RBE is derived from Zero-DCE (CVPR 2020), and its application in 3D reconstruction is novel.
- The end-to-end "rendering-guided restoration" paradigm (GRE) can be generalized to other degradation scenarios (e.g., motion blur, dust, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel perspective of the problem (two overlooked features), with contributions from both the dataset and the methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across multiple datasets + ablation study + real-world scenes + dataset comparisons, though with relatively few baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and framework, though some details (such as the timing for switching training strategies) could be more detailed.
- Value: ⭐⭐⭐⭐ — Both the dataset and method offer practical value, providing a foundation for 3D reconstruction in adverse weather.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[AAAI 2026\] SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion](splatssc_decoupled_depth-guided_gaussian_splatting_for_semantic_scene_completion.md)
- [\[ICLR 2026\] SpatialHand: Generative Object Manipulation from 3D Perspective](../../ICLR2026/3d_vision/spatialhand_generative_object_manipulation_from_3d_prespective.md)

</div>

<!-- RELATED:END -->
