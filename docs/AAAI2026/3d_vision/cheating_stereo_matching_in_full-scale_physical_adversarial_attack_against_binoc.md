---
title: >-
  [Paper Note] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation
description: >-
  [AAAI 2026][3D Vision][adversarial attack] This paper proposes the first full-surface 3D texture physical adversarial attack against stereo matching models. Through a stereo-aligned rendering module and a region-aware me…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "adversarial attack"
  - "stereo matching"
  - "binocular depth estimation"
  - "3D texture camouflage"
  - "physical adversarial examples"
date: 2026-05-08
content_hash: 998248761903d96a
---

# Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation

**Conference**: AAAI 2026
**arXiv**: [2511.14386](https://arxiv.org/abs/2511.14386)
**Code**: None
**Area**: 3D Vision
**Keywords**: adversarial attack, stereo matching, binocular depth estimation, 3D texture camouflage, physical adversarial examples

## TL;DR

This paper proposes the first full-surface 3D texture physical adversarial attack against stereo matching models. Through a stereo-aligned rendering module and a region-aware merging attack strategy, adversarial vehicles seamlessly blend into the background in the predicted depth map, causing severe failures in autonomous driving perception systems.

## Background & Motivation

Existing physical adversarial examples (PAEs) primarily target monocular depth estimation (MDE) and mostly adopt 2D patch forms, which face fundamental limitations in stereo matching-based binocular depth estimation (SM-BDE) scenarios:

**Limitations of 2D patches**: They affect only local regions of the target object, exhibit unstable effects across different viewpoints, and disrupt the physical disparity consistency required by stereo cameras.

**Insufficiency of existing SM attacks**: PASM and Adv-DM naively assume a direct displacement relationship between left and right views, ignoring the disparity geometric constraints of real stereo cameras, leading to failure during physical deployment.

**Incomplete hiding attacks**: Existing hiding attacks only push target depth to infinity, leaving object contours clearly visible and exposing attack intent.

**Growing adoption of SM-BDE in autonomous driving** (Baidu Apollo, Waymo, Mobileye) makes security evaluation increasingly urgent.

## Method

### Overall Architecture

The system comprises three core modules:
1. **Stereo-Aligned 3D Rendering Module**: Uses 3D object detection to obtain vehicle pose, parameterizes left and right camera viewpoints in spherical coordinates, and independently renders the 3D vehicle with adversarial texture.
2. **Merging Attack Texture Generation Module**: Achieves depth blending via boundary depth extraction → region segmentation → region-wise optimization.
3. **End-to-End Texture Optimization**: A differentiable rendering pipeline backpropagates losses to iteratively optimize the full-surface texture $\theta$.

Given a stereo background image pair $b = (b^l, b^r)$, the texture is mapped onto the 3D mesh $O$ and composited into the scene:

$$x_\theta = S(R(O, \theta, k), b, m)$$

The optimization objective is to make the SM model $F$ predict depth values close to the background target depth $d_t$:

$$\theta = \arg\min_\theta \mathcal{L}(F(x_\theta^l, x_\theta^r), d_t)$$

### Key Designs

**(1) Stereo-Aligned 3D Rendering (SAR)**

Unlike monocular rendering, SM attacks must satisfy three constraints:
- The adversarial object maintains geometrically consistent appearance in both left and right views.
- Background context remains coherent across both views.
- Camera viewpoints conform to a physically accurate stereo baseline.

The vehicle bounding box $\text{bbox} = \{t_x, t_y, t_z, t_l, t_w, t_h, t_r, t_c\}$ is obtained via 3D detection. Rendering viewpoints are parameterized in spherical coordinates as $k = \{\text{dist}, \text{elev}, \text{azim}\}$, and rendered separately for the left and right cameras $(k_l, k_r)$ to ensure disparity consistency.

**(2) Merging Attack Texture Generation**

A three-step pipeline achieves depth blending:

**Step 1: Boundary Depth Extraction** — The target mask $m$ is expanded via max pooling to obtain the boundary mask $m_{bg}$, from which surrounding background depth is extracted:

$$m_{bg} = \text{Maxpool}(m) - m, \quad d_{bg} = d \cdot m_{bg}$$

**Step 2: Region Segmentation** — The average background depth $d_{bg}^{avg}$ is computed, and the reference points closest to it are found on the left and right boundaries of the object. The vehicle depth map is then horizontally divided into upper and lower regions. The lower region is closer to the ground background, while the upper region exhibits a larger depth discrepancy with the background.

**Step 3: Region-wise Optimization** — The upper and lower regions are independently aligned to their respective local background depths:

$$\mathcal{L}_{\text{merge}}(\theta) = \text{MSE}(d_{obj}^{up}, d_{bg}^{up}) + \text{MSE}(d_{obj}^{bt}, d_{bg}^{bt})$$

**(3) Appearing Attack**

A complementary strategy that minimizes the perceived depth of the target object to make it appear close to the camera, forcing surrounding vehicles to brake abruptly:

$$\mathcal{L}_{\text{appear}}(\theta) = \text{MSE}(d_{obj}, D_{max})$$

### Loss & Training

The total loss consists of three terms:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{merge}}(\theta) + \alpha \mathcal{L}_{\text{nps}}(\theta) + \beta \mathcal{L}_{\text{tv}}(\theta)$$

- $\mathcal{L}_{\text{nps}}$: Non-printability score loss, constraining colors to the reproducible printing range ($\alpha=5$).
- $\mathcal{L}_{\text{tv}}$: Total variation smoothness loss, suppressing high-frequency noise ($\beta=0.1$).
- Expectation over Transformation (EoT) is applied for robustness: random light source perturbations in $[-3,3]$ m, ambient light intensity in $[0.3, 0.9]$, and Gaussian noise to simulate rain and fog.
- Adam optimizer, 100 epochs, lr=0.01 with cosine decay to $1e^{-4}$.

## Key Experimental Results

### Main Results

**Table 1: Merging attack performance comparison across five SM models**

| Method | PSMNet | GA-Net | RAFT-Stereo | CREStereo | AnyStereo |
|--------|--------|--------|-------------|-----------|-----------|
| | $\mathcal{E}_{blend}↓$ / $\mathcal{E}_{cover}↑$ / $\mathcal{E}_{sys}↑$ | Same | Same | Same | Same |
| Benign | 0.631/0.013/0 | 0.641/0.012/0 | 0.786/0.012/0 | 0.677/0.017/0 | 0.572/0.093/0 |
| PASM | 0.475/0.154/0.13 | 0.411/0.088/0.12 | 0.502/0.148/0.07 | 0.431/0.094/0.15 | 0.471/0.124/0.15 |
| Adv-DM | 0.510/0.176/0.04 | 0.449/0.075/0.12 | 0.614/0.143/0.05 | 0.444/0.077/0.17 | 0.480/0.119/0.09 |
| **Ours** | **0.058/0.553/0.74** | **0.069/0.588/0.69** | **0.082/0.571/0.62** | **0.071/0.598/0.70** | **0.056/0.576/0.76** |

The proposed method outperforms all baselines across five models: $\mathcal{E}_{blend}$ is reduced by approximately 10×, $\mathcal{E}_{cover}$ is improved by approximately 4×, and the collision rate in the Apollo system increases from <0.15 to 0.62–0.76.

**Table 2: Evaluation in real physical environments (1:30 scale 3D-printed model + iPhone binocular camera)**

| Condition | $\mathcal{E}_{blend}↓$ (Benign→Adv) | $\mathcal{E}_{cover}↑$ (Benign→Adv) |
|-----------|------|------|
| Noon | 0.481→0.087 | 0.036→0.519 |
| Sunset | 0.536→0.067 | 0.042→0.577 |
| Side viewpoint | 0.557→0.071 | 0.030→0.581 |
| 12m distance | 0.517→0.074 | 0.035→0.504 |

### Ablation Study

**Table 4: Module ablation (PSMNet, with $\mathcal{L}_{nps}$ + $\mathcal{L}_{tv}$)**

| Configuration | $\mathcal{E}_{blend}↓$ | $\mathcal{E}_{cover}↑$ |
|---------------|------|------|
| None | 0.631 | 0.015 |
| SAR only | 0.403 | 0.541 |
| Merge only | 0.611 | 0.024 |
| **Full** | **0.051** | **0.587** |

SAR is the foundation for effective SM attacks (without SAR, $\mathcal{E}_{cover}$ is extremely low), while the merging attack is key to achieving stealthy blending (without merging, $\mathcal{E}_{blend}$ remains high). Both components must work in concert to simultaneously achieve high coverage and low visibility.

### Key Findings

1. **Physical deployability verified**: A 1:30 scale model vehicle combined with iPhone binocular camera validates robustness across varying illumination, viewpoints, and distances in real environments.
2. **Full-angle robustness**: $\mathcal{E}_{blend} < 0.09$ across heading angles from 0° to 330°, significantly outperforming patch-based methods that fail at side viewpoints.
3. **System-level threat**: When integrated into the full Apollo perception and planning stack, the collision rate reaches up to 0.76.

## Highlights & Insights

1. **Pioneer contribution**: This is the first full-surface 3D texture adversarial attack targeting SM, addressing the fundamental failure of MDE-based attacks under BDE settings.
2. **Merging attack concept**: An upgrade from "hiding" to "merging" — region-wise depth alignment eliminates object contours, offering greater concealment than simply pushing depth to infinity.
3. **Comprehensive evaluation**: Three-level assessment covering digital simulation (CARLA), physical real-world validation (3D printing), and system-level evaluation (Apollo).

## Limitations & Future Work

1. Physical experiments use a 1:30 scale model; printing precision and weathering durability at full vehicle scale require further validation.
2. Evaluation is primarily conducted in static scenes; temporal consistency under dynamic driving conditions is not sufficiently verified.
3. Adversarial robustness against defense methods (e.g., stereo consistency checking) is not discussed.
4. The merging attack relies on accurate background depth estimation; generalizability in complex background scenes remains questionable.

## Related Work & Insights

- **PAE evolution**: 2D patch (Eykholt 2018) → local 3D patch (Liu 2024, Cheng 2021) → full-surface 3D texture (this work).
- **Comprehensive SM model coverage**: PSMNet → GA-Net → RAFT-Stereo → CREStereo → AnyStereo, demonstrating cross-architecture generalization.
- **Implications for autonomous driving security evaluation**: Reveals the security vulnerabilities of SM-BDE systems and provides an important reference for defense research.

## Rating

| Dimension | Score | Comments |
|-----------|-------|----------|
| Novelty | ★★★★☆ | First 3D texture physical attack against SM; merging attack concept is innovative |
| Technical Depth | ★★★★☆ | Stereo-aligned rendering and region-wise optimization are elegantly designed |
| Experimental Thoroughness | ★★★★★ | 5 models + CARLA + physical validation + Apollo system-level evaluation |
| Writing Quality | ★★★★☆ | Clear structure with rich illustrations |
| Value | ★★★★☆ | Reveals binocular perception security risks; valuable reference for defense research |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[CVPR 2026\] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](../../CVPR2026/3d_vision/spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[CVPR 2026\] PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](../../CVPR2026/3d_vision/pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)

</div>

<!-- RELATED:END -->
