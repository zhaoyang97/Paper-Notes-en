---
title: >-
  [Paper Note] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation
description: >-
  [AAAI 2026][3D Vision][Adversarial Attack] Proposes the first 3D full-surface texture physical adversarial attack targeting stereo matching models. Through a stereo-aligned rendering module and region-aware merging attacks, the adversarial vehicle is seamlessly blended into the background within depth maps, leading to severe failures in autonomous driving perception systems.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Adversarial Attack"
  - "Stereo Matching"
  - "Binocular Depth Estimation"
  - "3D Texture Camouflage"
  - "Physical Adversarial Examples"
date: 2026-05-08
content_hash: ef7adf58e9ba1b9a
---

# Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation

**Conference**: AAAI 2026  
**arXiv**: [2511.14386](https://arxiv.org/abs/2511.14386)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Adversarial Attack, Stereo Matching, Binocular Depth Estimation, 3D Texture Camouflage, Physical Adversarial Examples

## TL;DR

Proposes the first 3D full-surface texture physical adversarial attack targeting stereo matching models. Through a stereo-aligned rendering module and region-aware merging attacks, the adversarial vehicle is seamlessly blended into the background within depth maps, leading to severe failures in autonomous driving perception systems.

## Background & Motivation

Existing physical adversarial attacks (PAEs) primarily target monocular depth estimation (MDE) and mostly adopt 2D patches, facing fundamental limitations in stereo matching driven binocular depth estimation (SM-BDE) scenarios:

**Limitations of 2D Patches**: They only affect local regions of the target object, lack robustness across different viewpoints, and disrupt the physical disparity consistency required by stereo cameras.

**Limitations of Prior SM Attacks**: PASM and Adv-DM simply assume a direct shift relationship between left and right views, ignoring the disparity geometric constraints of real stereo cameras, which causes them to fail in physical deployment.

**Incomplete Hiding Attacks**: Existing hiding attacks only push the target depth to infinity, leaving the object's silhouette clearly visible and exposing the attack intent.

**Growing Adaptation of SM-BDE in Autonomous Driving**: With its increasing adoption in systems like Baidu Apollo, Waymo, and Mobileye, its security evaluation is urgently required.

## Method

### Overall Architecture

The system consists of three core modules:
1. **Stereo-Aligned 3D Rendering Module**: Utilizes 3D object detection to obtain the vehicle pose, parameterizes the left and right camera viewpoints in a spherical coordinate system, and independently renders the 3D vehicle with adversarial textures.
2. **Merging Attack Texture Generation Module**: Achieves depth merging through boundary depth extraction $\rightarrow$ region segmentation $\rightarrow$ region-wise optimization.
3. **End-to-End Texture Optimization**: Backpropagates losses through a differentiable rendering pipeline to iteratively optimize the full-surface texture $\theta$.

Given a stereo background image pair $b = (b^l, b^r)$, the texture is mapped to the 3D mesh $O$ and synthesized into the scene:

$$x_\theta = S(R(O, \theta, k), b, m)$$

The optimization objective is to drive the depth prediction of the SM model $F$ to approximate the target background depth $d_t$:

$$\theta = \arg\min_\theta \mathcal{L}(F(x_\theta^l, x_\theta^r), d_t)$$

### Key Designs

**（1）Stereo-Aligned 3D Rendering**

Unlike monocular rendering, SM attacks must satisfy three constraints:
- The adversarial object maintains a geometrically consistent appearance across left and right views.
- Background contexts remain coherent across both views.
- The camera viewpoints follow a physically accurate stereo baseline.

Utilizing the 3D-detected vehicle bounding box $\text{bbox} = \{t_x, t_y, t_z, t_l, t_w, t_h, t_r, t_c\}$, the rendering viewpoints $k = \{\text{dist}, \text{elev}, \text{azim}\}$ are parameterized under a spherical coordinate system. Rendering is performed separately for the left and right cameras $(k_l, k_r)$ to ensure disparity consistency.

**（2）Merging Attack Texture Generation**

A three-step workflow achieves depth merging:

**Step 1: Boundary Depth Extraction** — Performs max pooling expansion on the target mask $m$ to obtain the boundary mask $m_{bg}$, then extracts the surrounding background depth:

$$m_{bg} = \text{Maxpool}(m) - m, \quad d_{bg} = d \cdot m_{bg}$$

**Step 2: Region Segmentation** — Calculates the average background depth $d_{bg}^{avg}$ and identifies the closest reference points along the left and right boundaries of the object, horizontally dividing the vehicle depth map into upper and lower regions. The lower region is closer to the ground-level background, while the upper region exhibits a larger depth difference relative to the background.

**Step 3: Region-wise Optimization** — Aligns the upper and lower regions to their respective local background depths:

$$\mathcal{L}_{\text{merge}}(\theta) = \text{MSE}(d_{obj}^{up}, d_{bg}^{up}) + \text{MSE}(d_{obj}^{bt}, d_{bg}^{bt})$$

**（3）Appearing Attack**

A complementary strategy — Minimizes the perceived depth of the target to make it "appear" closer to the camera, forcing surrounding vehicles to perform emergency braking:

$$\mathcal{L}_{\text{appear}}(\theta) = \text{MSE}(d_{obj}, D_{max})$$

### Loss & Training

The total loss consists of three terms:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{merge}}(\theta) + \alpha \mathcal{L}_{\text{nps}}(\theta) + \beta \mathcal{L}_{\text{tv}}(\theta)$$

- $\mathcal{L}_{\text{nps}}$: Non-Printable Score (NPS) loss, constraining the colors within printable reproduction ranges ($\alpha=5$).
- $\mathcal{L}_{\text{tv}}$: Total Variation (TV) loss, suppressing high-frequency noise ($\beta=0.1$).
- Employs Expectation of Transformation (EoT) to enhance robustness: randomly perturbs physical parameters such as light positions of $[-3,3]$m, ambient light intensity of $[0.3, 0.9]$, and injects Gaussian noise to simulate rain and fog.
- Uses Adam optimizer, 100 epochs, lr=0.01 with cosine decay to $1e^{-4}$.

## Key Experimental Results

### Main Results

**Table 1: Comparison of merging attack effectiveness on five SM models**

| Method | PSMNet | GA-Net | RAFT-Stereo | CREStereo | AnyStereo |
|------|--------|--------|-------------|-----------|-----------|
| | $\mathcal{E}_{blend}↓$ / $\mathcal{E}_{cover}↑$ / $\mathcal{E}_{sys}↑$ | Same as left | Same as left | Same as left | Same as left |
| Benign | 0.631/0.013/0 | 0.641/0.012/0 | 0.786/0.012/0 | 0.677/0.017/0 | 0.572/0.093/0 |
| PASM | 0.475/0.154/0.13 | 0.411/0.088/0.12 | 0.502/0.148/0.07 | 0.431/0.094/0.15 | 0.471/0.124/0.15 |
| Adv-DM | 0.510/0.176/0.04 | 0.449/0.075/0.12 | 0.614/0.143/0.05 | 0.444/0.077/0.17 | 0.480/0.119/0.09 |
| **Ours** | **0.058/0.553/0.74** | **0.069/0.588/0.69** | **0.082/0.571/0.62** | **0.071/0.598/0.70** | **0.056/0.576/0.76** |

The proposed method consistently leads across all five models: $\mathcal{E}_{blend}$ decreases by approximately 10 times, $\mathcal{E}_{cover}$ increases of about 4 times, and the collision rate of the Apollo system rises from <0.15 to 0.62-0.76.

**Table 2: Real physical environmental evaluation (3D-printed 1:30 scale model + iPhone stereo camera)**

| Condition | $\mathcal{E}_{blend}↓$ (Benign→Adv) | $\mathcal{E}_{cover}↑$ (Benign→Adv) |
|------|------|------|
| Noon | 0.481→0.087 | 0.036→0.519 |
| Sunset | 0.536→0.067 | 0.042→0.577 |
| Side View | 0.557→0.071 | 0.030→0.581 |
| 12m Distance | 0.517→0.074 | 0.035→0.504 |

### Ablation Study

**Table 4: Ablation study of modules (PSMNet, including $\mathcal{L}_{nps}$ + $\mathcal{L}_{tv}$)**

| Configuration | $\mathcal{E}_{blend}↓$ | $\mathcal{E}_{cover}↑$ |
|------|------|------|
| None | 0.631 | 0.015 |
| SAR only | 0.403 | 0.541 |
| Merge only | 0.611 | 0.024 |
| **Full** | **0.051** | **0.587** |

SAR is foundational for effectively attacking SM (without SAR, $\mathcal{E}_{cover}$ is extremely low), and Merge is critical for achieving stealthy blending (without Merge, $\mathcal{E}_{blend}$ remains high). Their combination is necessary to achieve both high coverage and low visibility.

### Key Findings

1. **Physical Deployability Validation**: Evaluated under real-world environments using a 1:30 scale model car and an iPhone stereo camera, demonstrating robustness across different lighting conditions, viewpoints, and distances.
2. **All-Angle Robustness**: $\mathcal{E}_{blend} < 0.09$ is maintained under heading angles from 0°-330°, significantly outperforming patch-based methods that fail under side viewpoints.
3. **System-Level Threat**: When integrated into Apollo's full-stack perception and planning, the collision rate reaches up to 0.76.

## Highlights & Insights

1. **Pioneering Work**: Represents the first 3D full-surface texture adversarial attack against SM, successfully addressing the fundamental issue where MDE attacks fail under BDE.
2. **Merging Attack Concept**: Evolution from "hiding" to "merging." By using split-region depth alignment to eliminate object contours, it achieves higher stealthiness than simply pushing depths far away.
3. **Experimental Thoroughness**: Comprises a three-tiered evaluation including digital simulation (CARLA), physical reality (3D printing), and system-level testing (Apollo).

## Limitations & Future Work

1. The physical experiments utilize 1:30 scaled models; the texture printing precision and weatherability on full-scale vehicles require further validation.
2. The current evaluation mainly focuses on static scenes, and temporal consistency during dynamic driving has not been fully verified.
3. The adversarial robustness of defense methods (such as stereo-consistency checking) is not discussed.
4. The merging attack relies on accurate estimations of background depth, making its generalization across complex backgrounds potentially uncertain.

## Related Work & Insights

- **Development of PAEs**: 2D patches (Eykholt 2018) $\rightarrow$ local 3D patches (Liu 2024, Cheng 2021) $\rightarrow$ Ours full-surface 3D textures.
- **Comprehensive Coverage of SM Models**: PSMNet $\rightarrow$ GA-Net $\rightarrow$ RAFT-Stereo $\rightarrow$ CREStereo $\rightarrow$ AnyStereo, demonstrating cross-architecture generalization.
- **Insight for Autonomous Driving Safety Assessment**: Uncovers the security vulnerabilities of SM-BDE systems, providing a significant benchmark for future defense research.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ★★★★☆ | First 3D texture physical attack on SM, innovative merging attack concept |
| Technical Depth | ★★★★☆ | Solid design with stereo-aligned rendering and split-region optimization |
| Experimental Thoroughness | ★★★★★ | Five models + CARLA + physical validation + Apollo system-level evaluation |
| Writing Quality | ★★★★☆ | Clear structure, abundant illustrations |
| Value | ★★★★☆ | Uncovers safety hazards of stereo perception, highly valuable for defense studies |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](../../CVPR2025/3d_vision/defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting](../../CVPR2025/3d_vision/spectral_defense_against_resource-targeting_attack_in_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)

</div>

<!-- RELATED:END -->
