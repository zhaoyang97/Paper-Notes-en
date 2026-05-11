---
title: >-
  [Paper Note] Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning
description: >-
  [AAAI 2026][3D Vision][Rainy 3D reconstruction] This paper proposes OmniRain3D, the first dataset that jointly models perspective heterogeneity and brightness dynamicity in rainy 3D scenes, along with REVR-GSNet…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Rainy 3D reconstruction"
  - "3D Gaussian splatting"
  - "rain removal"
  - "brightness enhancement"
  - "end-to-end framework"
date: 2026-05-08
content_hash: 0749f57348e799da
---

# Rethinking Rainy 3D Scene Reconstruction via Perspective Transforming and Brightness Tuning

**Conference**: AAAI 2026
**arXiv**: [2511.06734](https://arxiv.org/abs/2511.06734)
**Code**: [https://github.com/ncfjd/REVR-GSNet](https://github.com/ncfjd/REVR-GSNet)
**Area**: 3D Vision
**Keywords**: Rainy 3D reconstruction, 3D Gaussian splatting, rain removal, brightness enhancement, end-to-end framework

## TL;DR

This paper proposes OmniRain3D, the first dataset that jointly models perspective heterogeneity and brightness dynamicity in rainy 3D scenes, along with REVR-GSNet, an end-to-end framework integrating recursive brightness enhancement, Gaussian primitive optimization, and GS-guided rain elimination to reconstruct high-fidelity clean 3D scenes from rain-degraded images.

## Background & Motivation

### State of the Field

3D scene reconstruction methods such as NeRF and 3DGS have achieved strong performance under clean conditions. However, in adverse weather such as rain, multi-view images suffer from raindrop/streak occlusion and reduced visibility, which disrupts cross-view consistency and severely degrades reconstruction quality. This poses significant challenges for systems requiring all-weather operation, such as autonomous driving and robot navigation.

### Limitations of Prior Work

The authors identify two critical characteristics overlooked by existing work:

**Perspective Heterogeneity**: Rain streaks in 3D space project differently onto 2D images from different viewpoints—appearing as Λ-shaped divergence in upward views, parallel lines in horizontal views, and v-shaped convergence in downward views; the streak orientation also varies when the camera deviates from the rainfall direction. Existing datasets such as HydroViews apply 2D rain layers via simple linear superposition, lacking 3D physical consistency.

**Brightness Dynamicity**: In real rainy scenes, cloud cover reduces ambient brightness, with heavier rain producing darker images. Existing datasets such as RainyScape simulate rain in 3D space but neglect its effect on brightness, creating a significant domain gap with real-world conditions.

### Root Cause & Starting Point

Existing methods (DerainNeRF, DerainGS) typically adopt a **two-stage pipeline**: a pre-trained deraining network first removes degradation, followed by 3D reconstruction. This decoupled paradigm has two drawbacks: (1) pre-trained models may overfit specific rain patterns; and (2) no brightness adjustment mechanism is incorporated. This paper proposes a unified end-to-end framework that jointly handles rain removal and brightness restoration.

## Method

### Overall Architecture

REVR-GSNet adopts a **joint alternating optimization** strategy comprising three cooperating modules:

- **Phase 1**: Joint optimization of RBE (Recursive Brightness Enhancement) and GPO (Gaussian Primitive Optimization)—progressively improving brightness while embedding it into 3DGS.
- **Phase 2**: Joint optimization of GPO and GRE (GS-guided Rain Elimination)—using 3DGS-rendered references to guide rain removal, with the deraining results fed back to refine 3DGS.
- **Phase 3**: GPO only—generating the final clean radiance field $V^M$.

### Key Designs

#### 1. OmniRain3D Dataset Construction

**Function**: Constructing the first rainy 3D scene dataset that simultaneously captures perspective heterogeneity and brightness dynamicity.

**Core Pipeline**:
- **Pose Extraction**: COLMAP is applied to clean background images to extract all camera extrinsics, obtaining elevation angle $\theta$ and azimuth angle $\phi$.
- **Dynamic Rain Streak Rendering**: A 3D rain model is built in Blender with six meteorological parameters $S = \{\omega_{den}, \omega_{dep}, \omega_{str}, \omega_{dir}, \omega_{qty}, \omega_{scl}\}$ (density, depth, wind strength, wind direction, rainfall amount, scale); rain streaks are rendered synchronously for each camera pose.
- **Adaptive Brightness Adjustment**: An exponential decay model based on the Beer–Lambert law:

$$L = L_0 e^{-\gamma \omega_{den}}$$

Three rainfall intensity levels (light, moderate, heavy) are defined with corresponding brightness values. The final synthetic rainy image is formed by combining the brightness-adjusted background with rain streaks at the corresponding density level.

The overall imaging model is: $O_t(\theta_i, \phi_j) = L \odot (B_t(\theta_i, \phi_j) + R_t(\theta_i, \phi_j))$

**Design Motivation**: To address the limitations of HydroViews (2D-only superposition) and RainyScape (brightness ignored), providing training and evaluation data closer to real-world conditions.

#### 2. Recursive Brightness Enhancement (RBE)

**Function**: Progressively correcting the brightness of low-illumination rainy images.

**Mechanism**: A lightweight CNN (CPEN, 7 convolutional layers with symmetric skip connections) estimates brightness adjustment parameters, which are then applied recursively via a quadratic brightness enhancement curve:

$$\mathbf{BE}(I_t, A_1) = I_t + A_1 I_t (1 - I_t)$$

The process recurses for $n=4$ steps, each applying a distinct parameter $A_a$ to incrementally increase brightness.

**Design Motivation**: Single-step enhancement is insufficient for severely darkened images; the recursive scheme progressively approaches the target brightness, while the parameterized curve ensures controllable enhancement.

#### 3. Gaussian Primitives Optimization (GPO)

**Function**: Building and optimizing a 3D Gaussian scene representation from the enhanced multi-view images.

**Core Pipeline**:
- Camera poses are estimated from enhanced images $\{E_t^i\}$ via COLMAP.
- A 3DGS representation $V = \{\mu_z, \Sigma_z, \sigma_z, h_z\}$ (position, covariance, opacity, spherical harmonics coefficients) is constructed.
- 3DGS attributes are optimized through differentiable rasterization.

**Key Insight**: Although enhanced images still contain rain streaks, the radiance field optimization effectively suppresses these artifacts by exploiting **cross-view consistency** and spatial correlations.

#### 4. GS-guided Rain Elimination (GRE)

**Function**: Using the current 3DGS-rendered reference images to guide the rain removal process.

**Mechanism**: The rendered image $R_t$ contains fewer rain artifacts and cleaner structure than the enhanced image $E_t$, as 3DGS already performs multi-view fusion. A Recursive Rain Estimation Network (RREN) is employed:
- $R_t$ and $E_t$ are concatenated as input.
- A recurrent U-Net with LSTM units and embedded Residual Recurrent Blocks (RRB) iterates for $l=6$ steps, estimating a rain streak map $M_o$ at each step and obtaining the deraining result $D_t$ via residual subtraction.
- The deraining result is fed back to GPO to further refine 3DGS.

$$D_t = \text{Cat}(R_t, E_t) - E_\phi(\text{Cat}(R_t, E_t))$$

**Design Motivation**: The 3DGS-rendered image has already undergone implicit partial deraining; using it as a guidance signal helps the deraining network better distinguish rain streaks from scene content.

### Loss & Training

- The full system is trained end-to-end using PyTorch on an RTX 3090.
- RBE and GRE use the Adam optimizer with an initial learning rate of $10^{-3}$.
- GPO applies separate learning rates for different 3DGS attributes (means: $1.6 \times 10^{-4}$, scaling: $5 \times 10^{-4}$, SH: $2.5 \times 10^{-3}$).
- All methods are trained for 30,000 steps for fair comparison.

## Key Experimental Results

### Main Results

**OmniRain3D Rain Streak Scenes** (normal brightness, 4 scenes):

| Scene | Metric | REVR-GSNet | DerainGS | RainyScape | DerainNeRF |
|-------|--------|-----------|---------|-----------|-----------|
| Francis | PSNR↑ | **24.56** | 23.40 | 22.99 | 16.17 |
| Garden | PSNR↑ | **25.35** | 25.30 | 22.58 | 21.74 |
| Garden | LPIPS↓ | **0.184** | 0.200 | 0.241 | 0.320 |
| Caterpillar | PSNR↑ | **21.48** | 20.26 | 19.90 | 13.99 |

**OmniRain3D Low-Light Rainy Scenes** (4 scenes; competing methods require brightness pre-processing):

| Scene | Metric | REVR-GSNet | DerainGS† | RainyScape† | DerainNeRF† |
|-------|--------|-----------|----------|------------|------------|
| Bicycle | PSNR↑ | **19.06** | 18.88 | 18.63 | 18.13 |
| Family | PSNR↑ | **17.83** | 17.78 | 16.92 | 17.05 |
| Family | LPIPS↓ | **0.440** | 0.461 | 0.497 | 0.595 |

**HydroViews Raindrop Scenes** (average over 3 scenes):

| Scene | REVR-GSNet | RainyScape | DRSformer* | NeRD-Rain* |
|-------|-----------|-----------|-----------|-----------|
| Stump (PSNR) | **22.61** | 22.59 | 18.23 | 19.79 |
| Stump (LPIPS) | **0.258** | 0.284 | 0.303 | 0.336 |

### Ablation Study

Component ablation on the HydroViews dataset:

| Configuration | GPO | RBE | GRE | PSNR↑ | SSIM↑ |
|--------------|-----|-----|-----|-------|-------|
| GPO only | ✓ | | | 19.03 | 0.514 |
| GPO + RBE | ✓ | ✓ | | 22.71 | 0.615 |
| GPO + GRE | ✓ | | ✓ | 21.64 | 0.535 |
| **Full model** | **✓** | **✓** | **✓** | **23.88** | **0.687** |

RBE contributes the most (PSNR +3.68), demonstrating that brightness restoration is the critical factor in low-light rainy scene reconstruction. GRE also yields a significant gain (+2.61), and the full combination performs best.

### Key Findings

1. REVR-GSNet shows a more pronounced advantage in low-light rainy scenes—competing methods require brightness pre-processing yet still fall short of the end-to-end approach.
2. All baselines (3DGS, NeRF, RainyScape) perform better on OmniRain3D than on HydroViews, validating the higher photorealism of the proposed dataset.
3. Brightness histogram analysis confirms that OmniRain3D's brightness distribution is closer to that of real rainy images.
4. The model also demonstrates strong generalization to real-world rainy scenes.

## Highlights & Insights

1. **Precise Problem Formulation**: The paper is the first to explicitly identify perspective heterogeneity and brightness dynamicity as two overlooked critical characteristics in rainy 3D reconstruction, constructing a dedicated dataset accordingly.
2. **Closed-Loop Design**: The alternating optimization of RBE → GPO → GRE forms a closed loop—rain removal improves reconstruction, and reconstruction in turn guides rain removal, yielding mutual reinforcement.
3. **Physics-Based Modeling**: OmniRain3D is constructed using physical models such as the Beer–Lambert law, rather than simple image processing.
4. **Practical Value**: The end-to-end framework avoids error accumulation across pipeline stages, achieving better generalization to real-world scenes.

## Limitations & Future Work

1. Although OmniRain3D is more realistic than HydroViews, it remains synthetic; a domain gap with real-world rainy conditions persists due to unmodeled factors such as mist and water-surface reflections.
2. Performance gains of REVR-GSNet over baselines are marginal in some scenes (e.g., only +0.46 PSNR over DerainGS on the Ignatius scene), suggesting that more powerful deraining modules may be needed for highly complex scenarios.
3. The brightness attenuation model assumes a simple exponential relationship between rain density and brightness, whereas real-world conditions are also influenced by time of day, light source direction, and other factors.
4. Computational efficiency is not discussed; the alternating optimization strategy likely increases training time.
5. The evaluation is limited to a relatively small number of baselines (five), with insufficient comparison against more recent deraining methods.

## Related Work & Insights

- **Applying 3DGS to adverse weather conditions** is an emerging direction; this paper provides a paradigm reference for rainy scene reconstruction.
- The proposed approach can be extended analogously to foggy scenes (brightness attenuation → fog density attenuation), snowy scenes, and other weather conditions.
- The recursive curve enhancement idea in RBE originates from Zero-DCE (CVPR 2020); its application to 3D reconstruction is novel.
- The "rendering-guided restoration" paradigm embodied in GRE is generalizable to other degradation types, such as motion blur and dust.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel problem perspective (two overlooked characteristics); contributions in both dataset and method.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset evaluation, ablation study, real-scene testing, and dataset comparison; baseline coverage is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ — Motivation and framework are clearly presented; certain details (e.g., timing of phase transitions in the training strategy) could be elaborated further.
- Value: ⭐⭐⭐⭐ — Both the dataset and the method offer practical utility, establishing a foundation for adverse-weather 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[AAAI 2026\] Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos](dynamic_gaussian_scene_reconstruction_from_unsynchronized_videos.md)

</div>

<!-- RELATED:END -->
