---
title: >-
  [Paper Note] Estimating 2D Camera Motion with Hybrid Motion Basis
description: >-
  [ICCV 2025][3D Vision][Camera motion estimation] This paper proposes CamFlow, which represents complex 2D camera motion via a hybrid motion basis (12 physical bases + random noise bases), reveals the nonlinear nature of superimposed homography flow fields, and incorporates a Laplace distribution-based probabilistic loss function. CamFlow substantially outperforms existing homography and meshflow methods under both standard and cross-dataset zero-shot evaluation settings.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Camera motion estimation"
  - "homography"
  - "motion basis"
  - "optical flow"
  - "video stabilization"
date: 2026-05-08
content_hash: 9c37feca6bf8abf2
---

# Estimating 2D Camera Motion with Hybrid Motion Basis

**Conference**: ICCV 2025
**arXiv**: [2507.22480](https://arxiv.org/abs/2507.22480)  
**Code**: [lhaippp.github.io/CamFlow](https://lhaippp.github.io/CamFlow/)  
**Area**: 3D Vision
**Keywords**: Camera motion estimation, homography, motion basis, optical flow, video stabilization

## TL;DR

This paper proposes CamFlow, which represents complex 2D camera motion via a hybrid motion basis (12 physical bases + random noise bases), reveals the nonlinear nature of superimposed homography flow fields, and incorporates a Laplace distribution-based probabilistic loss function. CamFlow substantially outperforms existing homography and meshflow methods under both standard and cross-dataset zero-shot evaluation settings.

## Background & Motivation

### Problem Definition

2D camera motion estimation is a fundamental vision task that recovers the projection of 3D camera motion (rotation $\mathbf{R}$ + translation $\mathbf{t}$) onto the 2D image plane:
$$\mathbf{M} = \mathbf{K}(\mathbf{R} + \mathbf{t}\frac{\mathbf{n}^T}{d})\mathbf{K}^{-1}$$

Because scenes contain multiple depths and planes, different regions undergo different transformations, making the motion inherently nonlinear.

### Limitations of Prior Work

**Homography-based methods** (BasesHomo, HomoGAN): can only align a single plane and fail to handle parallax and multi-plane scenes.

**Meshflow methods**: partition the image into $N \times N$ grids and estimate a local homography per cell, but increasing the grid count introduces optimization difficulties.

**Flawed core assumption**: BasesHomo assumes that homographies can be linearly represented by 8 bases, yet **the superposition of multiple homography flow fields is nonlinear**—the sum of two homography flows cannot be represented by any single homography (rigorously demonstrated through experiments in the paper).

### Core Idea

A Taylor expansion is used to decompose homographic motion into 12 physical bases ($\{1, x, y, xy, x^2, y^2\}$ in both the $x$ and $y$ directions). Randomly sampled homography matrices are then processed via SVD to extract orthogonal components as additional random bases. Together, these form a high-dimensional hybrid motion basis space capable of expressing complex nonlinear camera motion.

## Method

### Overall Architecture

The CamFlow pipeline: image pair input → multi-scale feature pyramid → Motion Estimation Transformer (MET) → prediction of physical and random basis weights → linear combination of motion bases to obtain bidirectional dense motion fields → confidence mask network to filter dynamic objects → probabilistic loss optimization.

### Key Designs

#### 1. Physical Motion Bases (12 bases)

- **Function**: Derive fundamental motion patterns from the Taylor expansion of homographic transformations.
- **Core derivation**: A second-order Taylor expansion of the homography-induced displacement $\Delta x$ at $(0,0)$:
  $$\Delta x \approx w_1 \cdot 1 + w_2 \cdot x + w_3 \cdot y + w_4 \cdot xy + w_5 \cdot x^2 + w_6 \cdot y^2$$
  An analogous expansion for $\Delta y$ yields 12 basis functions in total: $\mathbf{F} = \{(b_i, 0)\} \cup \{(0, b_i)\}$, where $b = [1, x, y, xy, x^2, y^2]$.
- **Design Motivation**: These 12 bases cover fundamental geometric transformations including translation, rotation, scaling, and perspective.

#### 2. Random Motion Bases

- **Function**: Capture higher-order motion patterns through random sampling.
- **Core method**: Generate $K$ random $3 \times 3$ matrices (entries $\sim \mathcal{N}(0,1)$, $h_9=1$), convert them to flow fields, and apply SVD to extract $N-12$ orthogonal components.
- **Design Motivation**: The complete camera motion space is infinite-dimensional (higher-order Taylor terms); the random bases exploit the near-orthogonality of random vectors in high-dimensional spaces to approximate broad coverage.

#### 3. Hybrid Probabilistic Loss

- **Function**: Model uncertainty in motion estimation via a Laplace distribution.
- **Core formulation**: Horizontal and vertical components are each modeled as Laplace distributions, with the confidence mask $\mathbf{d}$ controlling variance.
- **Dual losses**:
    - Motion supervision loss $\ell_{NLL_m}$: negative log-likelihood using pseudo-labels.
    - Photometric loss $\ell_{NLL_p}$: negative log-likelihood of warped feature consistency.
    - Adaptive balancing: $\ell_{overall} = \ell_{NLL_p} + \mathbf{w} \times \frac{|\ell_{NLL_p}|}{|\ell_{NLL_m}|} \cdot \ell_{NLL_m}$
- **Design Motivation**: The photometric loss provides fine-grained constraints while the motion loss supplies coarse-grained guidance; the Laplace distribution is more robust than the Gaussian.

### Loss & Training

- Trained on the CAHomo dataset (460K training pairs).
- Zero-shot testing is conducted on the GHOF benchmark and the newly constructed GHOF-Cam benchmark.
- GHOF-Cam uses SAM to detect and mask dynamic objects, isolating pure camera motion for evaluation.

## Key Experimental Results

### Main Results

**CAHomo test set PME (Point Matching Error) ↓**

| Method | Type | AVG | RE | LT | LL | SF | LF |
|--------|------|-----|----|----|----|----|-----|
| SIFT+RANSAC | Traditional | 1.41 | 0.30 | 1.34 | 4.03 | 0.81 | 0.57 |
| SPSG+MAGSAC | Traditional | 0.63 | 0.36 | 0.79 | 0.70 | 0.71 | 0.70 |
| DMHomo | Supervised | 0.31 | 0.19 | 0.33 | 0.40 | 0.38 | 0.28 |
| HomoGAN | Unsupervised | 0.39 | 0.22 | 0.41 | 0.57 | 0.44 | 0.31 |
| **CamFlow** | Unsupervised | **0.32** | **0.19** | **0.32** | **0.39** | **0.39** | 0.31 |

**GHOF-Cam zero-shot EPE ↓**

| Method | AVG | RE | FOG | LL | RAIN | SNOW |
|--------|-----|----|----|-----|------|------|
| BasesHomo | 1.74 | 1.39 | 0.97 | 4.12 | 0.66 | 1.58 |
| MeshFlow | 2.15 | 1.09 | 2.21 | 5.57 | 0.44 | 1.69 |
| **CamFlow** | **1.10** | **1.08** | **0.74** | **2.15** | **0.46** | **1.05** |

**GHOF zero-shot PME ↓**

| Method | AVG | RE | FOG | LL | RAIN | SNOW |
|--------|-----|----|----|-----|------|------|
| RealSH | 1.72 | 1.60 | 0.88 | 4.42 | 0.43 | 1.28 |
| HomoGAN | 1.95 | 1.73 | 0.60 | 3.95 | 0.47 | 3.02 |
| **CamFlow** | **1.23** | **1.15** | **0.96** | **2.69** | **0.40** | **0.93** |

### Ablation Study

**Ablation on number of motion bases**

| # Bases | CAHomo PME | GHOF PME | GHOF-Cam EPE | Inference Time |
|---------|-----------|---------|-------------|----------------|
| 8 (physical only) | 0.37 | 1.68 | 1.45 | 76.42ms |
| 12 (physical) | 0.36 | 1.54 | 1.23 | 75.38ms |
| 24 (hybrid) | **0.33** | **1.23** | **1.10** | 79.63ms |
| 200 | 0.33 | 1.27 | 1.07 | 99.28ms |

**Ablation on hybrid loss**

| Motion Loss | Photometric Loss | CAHomo | GHOF | GHOF-Cam |
|------------|-----------------|--------|------|----------|
| ✓ | | 0.41 | 2.21 | 2.13 |
| | ✓ | 0.36 | 1.58 | 1.42 |
| ✓ | ✓ | **0.33** | **1.23** | **1.10** |

### Key Findings

- Zero-shot generalization is the most prominent advantage: PME on GHOF is reduced by 28.5% relative to the best supervised method RealSH and by 36.9% relative to the best unsupervised method HomoGAN.
- 24 hybrid bases represent the optimal trade-off: increasing to 200 bases yields only marginal improvement while inference time increases by 24.7%.
- Expanding the physical bases from 8 to 12 (adding second-order terms) improves performance across all datasets, validating the necessity of second-order Taylor terms.
- The confidence mask effectively identifies dynamic object regions (e.g., pedestrians, vehicles), improving the robustness of camera motion estimation.
- On perceptual quality metrics (PSNR/SSIM/LPIPS), CamFlow approaches the level of ground-truth homographies.

## Highlights & Insights

1. **Key observation on nonlinear superposition**: The paper rigorously demonstrates that the superposition of multiple homography flow fields no longer constitutes a homography—this invalidates the 8-dimensional linear basis assumption of BasesHomo and provides a theoretical foundation for higher-dimensional motion bases.
2. **Elegant combination of physical and random bases**: Physical bases capture known geometric transformations, while random bases, orthogonalized via SVD, cover unknown higher-order patterns; the two are complementary.
3. **GHOF-Cam benchmark**: By automatically masking dynamic objects with SAM, this benchmark provides the first evaluation dataset for pure camera motion, offering long-term value to the community.
4. **Simplicity of the probabilistic loss**: The Laplace distribution provides a unified treatment of motion supervision and photometric consistency, avoiding complex multi-loss weight tuning.

## Limitations & Future Work

1. **Training data dependency on CAHomo**: Despite strong generalization, the diversity of the training set may still be a bottleneck.
2. **Expressive capacity of 24 bases**: May be insufficient for extreme parallax or extreme rotation scenarios.
3. **Pseudo-label quality**: The motion supervision loss relies on pseudo-labels generated by other methods, which may introduce noise.
4. **End-to-end video stabilization not validated**: Although the motivation derives from video stabilization, only motion estimation accuracy is evaluated.
5. **Large-displacement scenes**: The accuracy of the Taylor expansion degrades at locations far from the origin.

## Related Work & Insights

- BasesHomo pioneered the direction of learning motion bases (8-dimensional linear basis); CamFlow extends this to a nonlinear high-dimensional space.
- MeshFlow and MeshHomoGAN are representative methods for multi-plane motion but are limited by grid resolution.
- HomoGAN employs GAN losses and a Transformer for coarse-to-fine refinement; CamFlow achieves superior results with a simpler probabilistic framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of the nonlinear superposition observation and the hybrid motion basis is a creative contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three benchmarks, comparisons against diverse methods, dual evaluation via dense/sparse motion metrics, perceptual quality assessment, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; theoretical derivations are concise.
- **Value**: ⭐⭐⭐⭐ — Directly benefits the video stabilization and homography estimation communities; the GHOF-Cam benchmark has long-term significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image](image_as_an_imu_estimating_camera_motion_from_a_single_motion-blurred_image.md)
- [\[ICCV 2025\] Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [\[ICCV 2025\] Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion](global_motion_corresponder_for_3d_point-based_scene_interpolation_under_large_mo.md)
- [\[CVPR 2025\] Estimating Body and Hand Motion in an Ego-sensed World](../../CVPR2025/3d_vision/estimating_body_and_hand_motion_in_an_ego-sensed_world.md)

</div>

<!-- RELATED:END -->
