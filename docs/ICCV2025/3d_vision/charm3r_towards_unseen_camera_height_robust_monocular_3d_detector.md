---
title: >-
  [Paper Note] CHARM3R: Towards Unseen Camera Height Robust Monocular 3D Detector
description: >-
  [ICCV 2025][3D Vision][Monocular 3D detection] Through mathematical proof that regression-based depth and ground-plane depth exhibit opposite extrapolation trends under camera height variation, CHARM3R proposes a simple in-model average of the two depth estimates to cancel out these trends, achieving robust generalization of Mono3D detectors to unseen camera heights with AP3D improvements exceeding 45%.
tags:
  - ICCV 2025
  - 3D Vision
  - Monocular 3D detection
  - camera height robustness
  - depth estimation extrapolation
  - ground-plane depth
  - autonomous driving
date: 2026-05-08
content_hash: 967fa01ed3741880
---

# CHARM3R: Towards Unseen Camera Height Robust Monocular 3D Detector

**Conference**: ICCV 2025  
**arXiv**: [2508.11185](https://arxiv.org/abs/2508.11185)  
**Code**: [Available](https://github.com/abhi1kumar/CHARM3R)  
**Area**: 3D Vision  
**Keywords**: Monocular 3D detection, camera height robustness, depth estimation extrapolation, ground-plane depth, autonomous driving

## TL;DR

Through mathematical proof that regression-based depth and ground-plane depth exhibit opposite extrapolation trends under camera height variation, CHARM3R proposes a simple in-model average of the two depth estimates to cancel out these trends, achieving robust generalization of Mono3D detectors to unseen camera heights with AP3D improvements exceeding 45%.

## Background & Motivation

### Limitations of Prior Work

**Background**: Monocular 3D detection is critical for autonomous driving, yet existing models suffer severe performance degradation when training and inference are conducted at different camera heights:

**Practical urgency**: Camera heights vary substantially across autonomous driving platforms (small robots, sedans, trucks), but nearly all training data is collected at a specific height (e.g., sedan level), making re-collection and annotation for each height infeasible.

**Severe performance degradation**: SoTA detectors exhibit a drop of more than 35 absolute AP3D_70 points under a 0.76 m height shift.

**Insufficiency of existing solutions**: Strategies such as Plücker embeddings, image transformation (assuming constant depth), and data augmentation show limited effectiveness under larger height variations.

The authors first systematically analyze the impact of height variation, finding that **depth estimation is the primary factor** driving performance degradation, and identify that regression-based depth and ground-plane depth exhibit diametrically opposite extrapolation behaviors.

## Method

### Overall Architecture

CHARM3R simultaneously maintains two depth estimates (regression-based depth + ground-plane depth) within an existing Mono3D detector and takes a simple average as the final depth prediction. The core contribution lies in the complementary nature of the two depth types under camera height variation.

### Key Designs

**1. Ground-based Depth**

For a front-facing camera, the depth of a pixel on the ground plane can be precisely computed from camera parameters: $z = \frac{H - b_2}{\frac{v - v_0}{f}}$. CHARM3R queries the ground-plane depth using the predicted projected 3D bottom center: $v_b = v_c + \frac{1}{2}h_{2D} + \alpha(v_c - v_{c,2D})$, where $\alpha$ is a learnable correction coefficient.

**Theorem 1**: The mean error of ground-plane depth has a **positive slope** with respect to $\Delta H$—depth is overestimated when height increases and underestimated when height decreases.

**2. Regression-based Depth**

Standard Mono3D regresses depth from the pixel location of the projected 3D center.

**Theorem 2**: The mean error of regression-based depth has a **negative slope** with respect to $\Delta H$—depth is underestimated when height increases and overestimated when height decreases.

**3. Depth Fusion**

A simple average is taken: $\hat{z} = \frac{1}{2}(\hat{z}^r + \hat{z}^g)$. Since the two extrapolation trends are exactly opposite, averaging effectively cancels out the errors.

**4. ReLU Activation**: ReLU is applied to the denominator $(v_b - v_0)$ to ensure non-negativity and improve training stability.

### Loss & Training

- A ground-plane depth branch is added on top of an existing Mono3D detector and trained end-to-end.
- Training is conducted solely on sedan-height data, with direct generalization to unseen heights.
- Simple averaging outperforms learned weighting, which overfits to the training distribution.

## Key Experimental Results

### Main Results

**CARLA Val (GUP Net backbone, trained at $\Delta H=0$ m, tested across heights):**

| Method | AP3D_70 ($\Delta H=-0.7$) | AP3D_70 ($\Delta H=0$) | AP3D_70 ($\Delta H=+0.76$) | MDE ($-0.7$) | MDE ($+0.76$) |
|--------|--------------------------|------------------------|----------------------------|--------------|----------------|
| Source | 9.46 | 53.82 | 7.23 | +0.53 | −0.63 |
| Plucker | 8.43 | 55.56 | 10.13 | +0.55 | −0.63 |
| UniDrive++ | 10.83 | 53.82 | 12.27 | +0.39 | −0.48 |
| **CHARM3R** | **19.45** | **55.68** | **27.33** | **+0.07** | **−0.02** |
| Oracle | 70.96 | 53.82 | 62.25 | +0.03 | +0.03 |

The method is also effective with the DEVIANT backbone: AP3D_70 increases from 6.25 to 26.24 at $\Delta H=+0.76$ m.

### Ablation Study

**Design choice ablation (GUP Net + CHARM3R):**

| Design Variant | AP3D_70 (−0.7 m) | AP3D_70 (0 m) | AP3D_70 (+0.76 m) |
|----------------|------------------|---------------|-------------------|
| Regression depth only | 9.46 | 53.82 | 7.23 |
| Ground-plane depth only | 0.98 | 26.61 | 5.39 |
| Offline fusion | 12.86 | 47.66 | 18.36 |
| Learned weighted average | 8.25 | **56.49** | 9.53 |
| Without ReLU | 0.60 | 52.94 | 0.07 |
| **CHARM3R** | **19.45** | 55.68 | **27.33** |

### Key Findings

- MDE trend validation: the regression model's MDE follows a negative trend with height ($+0.53 \to -0.63$), the ground-plane model a positive trend ($-0.80 \to +0.55$), and CHARM3R remains near zero ($+0.07 \to -0.02$).
- Simple averaging outperforms learned weighting in OOD scenarios, as learned weights overfit to the training distribution.
- ReLU is necessary: removing it causes training instability or collapse.
- The method is also effective with a ResNet-18 backbone.

## Highlights & Insights

1. **Theory-driven design**: Two formal theorems provide a clear theoretical foundation for the method, rather than relying on empirical trial and error.
2. **The power of minimalism**: A simple average of two depth estimates yields 45%+ OOD improvement, elegantly embodying the principle of "cancellation through symmetry."
3. **Problem-oriented research paradigm**: The work first conducts a thorough analysis of the problem's root cause (depth error decomposition) before designing the solution accordingly.

## Limitations & Future Work

1. Validation is conducted solely on the simulated CARLA dataset; real-world multi-height experiments are absent.
2. The theorems rely on simplifying assumptions (linear regression model, $\Delta H \ll z$).
3. Only vertical height variation is considered; pitch angle variation is not addressed.
4. The ground-plane assumption limits applicability to non-flat road scenarios.
5. Extension to additional detector architectures (e.g., transformer-based) remains to be explored.

## Related Work & Insights

- Unlike multi-height training approaches such as BEVHeight, the proposed method requires only single-height training data.
- The complementarity idea may generalize to other extrapolation problems (e.g., varying focal lengths or tilt angles).
- The observation that simple averaging outperforms learned weighting echoes findings in the domain generalization literature.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Theoretically grounded with a concise and practical solution; highly elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Thorough ablations, but limited to simulated data)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logical chain; rigorous theorem proofs)
- Value: ⭐⭐⭐⭐ (Reveals an important overlooked problem; solution can be directly integrated into existing detectors)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](boost_3d_reconstruction_using_diffusionbased_monocular_camer.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[ICCV 2025\] Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)
- [\[ICCV 2025\] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction](timeformer_capturing_temporal_relationships_of_deformable_3d_gaussians_for_robus.md)
- [\[ICCV 2025\] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions](stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)

</div>

<!-- RELATED:END -->
