---
title: >-
  [Paper Note] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation
description: >-
  [ICCV 2025][Autonomous Driving][physical adversarial attack] This paper proposes PGA, the first physical adversarial attack framework based on 3DGS, which generates cross-view robust physical adversarial camouflage through fast and accurate target reconstruction, resolution of Gaussian mutual/self-occlusion issues, and a min-max background adversarial optimization strategy. PGA surpasses state-of-the-art methods in both digital and physical domains.
tags:
  - ICCV 2025
  - Autonomous Driving
  - physical adversarial attack
  - adversarial camouflage
  - 3D Gaussian Splatting
  - multi-view robustness
  - object detection
date: 2026-05-08
content_hash: 6c82c653c6ba550d
---

# 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation

**Conference**: ICCV 2025
**arXiv**: [2507.01367](https://arxiv.org/abs/2507.01367)
**Code**: [https://github.com/TRLou/PGA](https://github.com/TRLou/PGA)
**Institution**: Sun Yat-Sen University, NTU, NUS, Peng Cheng Lab
**Area**: Autonomous Driving / Adversarial Attack / Physical Adversarial Examples
**Keywords**: physical adversarial attack, adversarial camouflage, 3D Gaussian Splatting, multi-view robustness, autonomous driving, object detection

## TL;DR
This paper proposes PGA, the first physical adversarial attack framework based on 3DGS, which generates cross-view robust physical adversarial camouflage through fast and accurate target reconstruction, resolution of Gaussian mutual/self-occlusion issues, and a min-max background adversarial optimization strategy. PGA surpasses state-of-the-art methods in both digital and physical domains.

## Background & Motivation

### Limitations of Prior Work

**State of the Field**: Physical adversarial attacks have exposed the vulnerability of DNNs in safety-critical scenarios such as autonomous driving. Adversarial camouflage is more promising than adversarial patches, as it covers the entire object surface and exhibits stronger adversarial efficacy in complex physical environments. However, existing methods suffer from two major limitations: (1) they rely on mesh priors of target objects and simulators such as CARLA to construct virtual environments, which is time-consuming and introduces an inevitable gap with the real world; (2) the limited variety of background images during training makes it difficult for the optimized camouflage to generalize across viewpoints, causing optimization to converge to suboptimal solutions.

### Root Cause

**Paper Goals**: How can adversarial camouflage that is both effective and robust across multiple viewpoints and physical environments be generated? Three key challenges arise: (1) How can arbitrary target objects be modeled rapidly and accurately without mesh priors? (2) How can cross-view consistency of camouflage patterns be ensured? (3) How can the adversarial efficacy of camouflage be maintained under varying backgrounds, weather conditions, and distances?

## Method

### Overall Architecture (PGA)
PGA consists of three modules: **Reconstruction Module** → **Rendering Module** → **Attack Module**.
- Reconstruction Module: reconstructs the target scene via 3DGS from a small number of multi-view images.
- Rendering Module: renders images from specified camera viewpoints, extracts the target mask using SAM, and synthesizes images for detection.
- Attack Module: iteratively optimizes the spherical harmonic coefficients of the Gaussians to generate adversarial camouflage.

### Key Designs

1. **3DGS-Based Reconstruction and Rendering**:

    - 3DGS is employed to rapidly and accurately reconstruct the target vehicle and scene from a small number of images, eliminating the need for manual mesh construction.
    - 3DGS provides differentiable, photo-realistic multi-view rendering capabilities.
    - Only the spherical harmonic coefficients $k_g$ (controlling surface color) are optimized, while geometry parameters remain unchanged, ensuring physical deployability.
    - Iterative attack update: $k^{t+1} = k^t + \eta\nabla_k L_{det}$

2. **Resolving Cross-View Imaging Inconsistency (Core Contribution 1)**:

    - **Mutual Occlusion Problem**: In vanilla 3DGS, some Gaussians reside inside the object; as the viewpoint changes, occlusion relationships shift and cause camouflage inconsistency. Solution: SuGaR regularization is introduced to align Gaussians to the object surface and reduce their opacity, preventing internal Gaussians from occluding the surface.
    - **Self-Occlusion Problem**: High-order spherical harmonics cause a single Gaussian to exhibit drastically different colors across viewpoints. Solution: During the attack iteration, only zeroth-order spherical harmonic coefficients $\langle k\rangle_0$ are optimized, ensuring uniform surface color for each Gaussian.
    - These two improvements guarantee that the same camouflage pattern is optimized consistently across viewpoints during iterative optimization.

3. **Min-Max Robust Camouflage Optimization (Core Contribution 2)**:

    - The camouflage optimization is formulated as a Universal Adversarial Perturbation (UAP) problem.
    - Optimization proceeds viewpoint by viewpoint with an iteration budget; once an attack succeeds, the process advances to the next viewpoint.
    - **Background Adversarial Perturbation**: Prior to each camouflage optimization step, I-FGSM is applied to add noise $\sigma$ to the background region to maximize the detection loss; the camouflage is then optimized to minimize the detection loss.
    - Mathematical formulation: $G' = \arg\min_G \max_\sigma L_{det}(I_{det}(\theta_c, G) + \sigma \cdot (1-M))$, s.t. $\|\sigma\|_\infty \leq \varepsilon$
    - This effectively filters out non-robust adversarial features that depend on specific backgrounds.

4. **Additional Augmentation Techniques**:

    - EoT (Expectation over Transformations): simulates random transformations in the physical world.
    - NPS (Non-Printability Score): ensures color printability.
    - Dominant color regularization: improves the visual naturalness of the camouflage.

### Attack Objective
Detection loss $L_{det}$: minimizes the confidence score of the predicted bounding box with the highest IoU with the ground truth, causing the detector to miss or misclassify the target.

## Key Experimental Results

### Digital Domain Attack Performance (AP@0.5 %, lower is better = more effective attack)

| Distance | Method | Faster R-CNN | YOLO-V5* | Mask R-CNN* | D-DETR* | Average |
|----------|--------|-------------|----------|-------------|---------|---------|
| 5m | Clean | 71.86 | 70.57 | 73.18 | 79.76 | 73.72 |
| 5m | RAUCA | 21.71 | 46.94 | 31.90 | 36.54 | 37.16 |
| 5m | **PGA** | **4.52** | **39.10** | **10.62** | **28.31** | **23.46** |
| 10m | RAUCA | 18.88 | 56.70 | 31.00 | 44.85 | 39.25 |
| 10m | **PGA** | **1.40** | **45.53** | **8.44** | **30.89** | **21.78** |

- PGA reduces the AP of the white-box Faster R-CNN to 1–5%, far surpassing all state-of-the-art methods.
- Black-box transfer to Mask R-CNN is particularly strong (8–11% vs. RAUCA's 31%).

### Physical Domain Experiments
- Adversarial camouflage deployed on real vehicles maintains high attack success rates across various distances, pitch angles, and weather conditions.
- Physical domain performance also significantly outperforms all competing methods.

### Ablation Study
- Mutual occlusion regularization, self-occlusion resolution, and min-max optimization each contribute substantially.
- The combination of all components achieves optimal performance.

## Highlights & Insights
- **First 3DGS-based physical attack framework**: leverages 3DGS's fast reconstruction and photo-realistic rendering capabilities to completely eliminate reliance on mesh priors and simulators.
- **Insightful analysis and resolution of mutual/self-occlusion**: precisely identifies two critical failure modes of vanilla 3DGS when applied to camouflage generation.
- **Elegant min-max background adversarial strategy**: automatically filters non-robust features through adversarial game formulation.
- **Comprehensive experimental validation**: digital domain across multiple distances and weather conditions, physical domain real-world deployment, cross-model black-box transfer, and infrared detection extension.

## Limitations & Future Work
- Optimizing only zeroth-order spherical harmonics sacrifices view-dependent color information.
- Physical deployment requires converting Gaussians into mesh textures for printing, which may introduce precision loss during the conversion process.
- The background noise budget $\varepsilon$ in min-max optimization requires manual tuning.
- Validation is limited to the vehicle detection task.
- Robustness evaluation against defense methods is insufficient.

## Related Work & Insights
- **vs. DAS/FCA/ACTIVE/TAS**: conventional methods based on mesh and differentiable rendering require target mesh priors and simulators.
- **vs. RAUCA**: although weather factors are considered, it still relies on mesh and a neural renderer.
- **vs. NeRF-based attacks**: NeRF suffers from slow rendering, lower quality, and high memory consumption; PGA comprehensively outperforms NeRF-based approaches via 3DGS.
- **PGA advantages**: no mesh prior required, fast and accurate reconstruction, photo-realistic rendering, and guaranteed cross-view consistency.

## Related Work & Insights
- The application of 3DGS to adversarial attacks highlights the importance of high-fidelity rendering for the effectiveness of physical attacks.
- The analysis of mutual/self-occlusion offers reference value for 3DGS applications in any task requiring cross-view consistency.
- The min-max framework is transferable to other optimization problems that demand environmental robustness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce 3DGS into physical attacks; mutual/self-occlusion analysis and min-max strategy are both original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Digital domain + physical domain + ablation + infrared extension; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear, though notation is dense.
- Value: ⭐⭐⭐⭐ Advances the physical attack state of the art and provides reference for defense research.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)

<!-- RELATED:END -->
