---
title: >-
  [Paper Note] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation
description: >-
  [ICCV 2025][Autonomous Driving][3D Gaussian Splatting] This paper proposes PGA, the first physical adversarial attack framework based on 3D Gaussian Splatting (3DGS). By addressing mutual occlusion and self-occlusion amo…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D Gaussian Splatting"
  - "Physical Adversarial Attack"
  - "Adversarial Camouflage"
  - "Multi-View Robustness"
  - "Autonomous Driving Security"
date: 2026-05-08
content_hash: 8fbe7ff4743df0b3
---

# 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation

**Conference**: ICCV 2025
**arXiv**: [2507.01367](https://arxiv.org/abs/2507.01367)
**Code**: [https://github.com/TRLou/PGA](https://github.com/TRLou/PGA)
**Area**: Autonomous Driving
**Keywords**: 3D Gaussian Splatting, Physical Adversarial Attack, Adversarial Camouflage, Multi-View Robustness, Autonomous Driving Security

## TL;DR

This paper proposes PGA, the first physical adversarial attack framework based on 3D Gaussian Splatting (3DGS). By addressing mutual occlusion and self-occlusion among Gaussians to ensure cross-viewpoint consistency, and by designing a min-max optimization strategy to filter non-robust adversarial features, PGA substantially outperforms state-of-the-art methods in both the digital and physical domains.

## Background & Motivation

Physical adversarial attacks are a critical tool for exposing the vulnerabilities of deep neural networks, posing significant threats in safety-critical scenarios such as autonomous driving. Compared to adversarial patches, adversarial camouflage covers the entire surface of a target object, enabling stronger and more persistent attack effects in complex physical environments.

However, existing adversarial camouflage methods suffer from two core limitations:
1. **Reliance on mesh priors and simulation environments**: Methods such as DAS, FCA, and DTA require mesh models of the target object and depend on simulators like CARLA to construct virtual environments, incurring high acquisition costs and an unavoidable domain gap with the real world.
2. **Insufficient multi-view robustness**: Limited diversity of training backgrounds causes optimization to fall into local optima, resulting in camouflage patterns with poor generalizability across different viewpoints, distances, and weather conditions.

NeRF has recently been introduced for 3D modeling in adversarial attack research, but its inherent drawbacks—slow rendering speed, low quality, and high memory demands—limit its effectiveness. As a novel 3D representation, 3DGS offers fast and accurate reconstruction together with differentiable real-time rendering, providing a stronger foundation for physical adversarial attacks.

## Core Problem

How can the fast reconstruction and differentiable rendering capabilities of 3DGS be leveraged—starting from only a small number of real images without requiring mesh models or simulators—to generate adversarial camouflage that maintains high attack effectiveness and robustness under complex physical conditions including multiple viewpoints, distances, and weather conditions?

Two specific technical challenges must be addressed simultaneously:
- **Cross-viewpoint consistency**: In vanilla 3DGS, mutual occlusion between Gaussians and self-occlusion induced by spherical harmonics cause rendered camouflage patterns to be inconsistent across viewpoints, undermining adversarial effectiveness.
- **Multi-view robustness**: Limited training backgrounds cause optimization to converge to suboptimal solutions, making it difficult to generate adversarial features that are universally effective across diverse real-world environments.

## Method

### Overall Architecture

The PGA framework consists of three modules:

1. **Reconstruction Module**: Given sparse multi-view images as input, the 3DGS training pipeline reconstructs a 3D Gaussian representation $\mathcal{G} = \{g_1, g_2, \ldots, g_N\}$ of both the target object and the background scene.
2. **Rendering Module**: Multiple camera viewpoints covering varying distances, elevation angles, and azimuth angles are selected. Differentiable rendering is performed via the 3DGS rasterizer, and the SAM model is used to extract the target object mask, compositing a clean background with the adversarial camouflage region to form detection images.
3. **Attack Module**: The composited images are fed into the target detector; detection loss is computed and used to iteratively optimize the spherical harmonic coefficients of the Gaussians. The final adversarial camouflage texture is extracted and deployed in the physical world.

The entire pipeline realizes an end-to-end flow from real images → 3DGS reconstruction → multi-view rendering → iterative attack → physical deployment.

### Key Designs

1. **Cross-Viewpoint Consistency Improvement**

    - **Resolving mutual occlusion**: The SuGaR regularization term is introduced during the reconstruction stage to align Gaussians to the object surface and encourage lower opacity, preventing Gaussians from being optimized into the object interior and thereby avoiding color occlusion from other surface Gaussians as the viewpoint changes.
    - **Resolving self-occlusion**: High-order spherical harmonics endow individual Gaussians with strong color expressiveness, causing different regions of a single Gaussian to exhibit entirely different colors as the viewpoint changes. The proposed solution is to **optimize only the zeroth-order spherical harmonic coefficient** $\langle k \rangle_0$ during iterative attack, ensuring uniform and consistent color variation across each Gaussian's surface.

2. **Min-Max Multi-View Robust Optimization**

    - **Sequential per-viewpoint optimization**: Since multi-view optimization is inherently a universal adversarial perturbation (UAP) problem with varying attack difficulty per viewpoint, PGA iterates over each viewpoint sequentially with a per-viewpoint iteration cap—once a viewpoint is successfully attacked, the remaining iterations for that viewpoint are skipped to avoid over-optimization on easy viewpoints.
    - **Background "counter-attack"**: Before each camouflage optimization iteration, point-level noise $\sigma$ is added to the background, and I-FGSM is used to optimize the detector back to correct detection (i.e., the max step). The camouflage is then optimized against this "hardened" background (i.e., the min step). This min-max strategy forces the algorithm to discard non-robust adversarial features that rely on specific backgrounds, retaining only genuinely multi-view robust adversarial features. Formally:
    $$\mathcal{G}' = \arg\min_{\mathcal{G}} \max_{\sigma} \mathcal{L}_{\text{det}}(\mathcal{I}_{\text{det}}(\theta_c, \mathcal{G}) + \sigma \cdot (1 - \mathcal{M}))$$

3. **SAM-Assisted Segmentation**: The Segment Anything Model is used to extract the target object mask, ensuring that adversarial perturbations are applied only to the target object region while the background remains clean.

### Loss & Training

The overall loss function is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}}(T(\mathcal{I}_{\text{det}}(\theta_c, \mathcal{G}) + \sigma \cdot (1 - \mathcal{M}))) + \lambda(\text{NPS} + \mathcal{L}_{\text{clr}} + \|\langle k \rangle_0 - \langle k \rangle_0^{\text{ori}}\|_2)$$

Component descriptions:
- **$\mathcal{L}_{\text{det}}$**: Detection loss that minimizes the confidence of the correct class in the maximum-IoU bounding box.
- **EoT transform $T$**: Applies random scaling, contrast, brightness, and noise augmentations to the image to enhance physical robustness.
- **NPS (Non-Printability Score)**: Constrains camouflage colors to remain within the printable color gamut to reduce manufacturing error.
- **$\mathcal{L}_{\text{clr}}$ (dominant color regularization)**: Extracts the top-$k$ dominant colors from the background via K-means and constrains camouflage colors to stay close to the ambient color tone, enhancing visual concealment.
- **$\|\langle k \rangle_0 - \langle k \rangle_0^{\text{ori}}\|_2$**: Constrains the magnitude of change in spherical harmonic coefficients before and after the attack.

Only the zeroth-order spherical harmonic coefficients are updated: $\langle k_{t+1} \rangle_0 = \langle k_t \rangle_0 + \eta \nabla_{\langle k \rangle_0} \mathcal{L}_{\text{total}}$

## Key Experimental Results

### Digital Domain Experiments (CARLA, AP@0.5 (%) ↓ lower is stronger)

| Distance | Method | Faster R-CNN | YOLO-v5* | Mask R-CNN* | Def-DETR* | Avg. |
|----------|--------|-------------|----------|-------------|-----------|------|
| 5m | Clean | ~73 | ~73 | ~76 | ~73 | 73.72 |
| 5m | RAUCA | 21.71 | 46.94 | 31.90 | 36.54 | 37.16 |
| 5m | **PGA** | **4.52** | **39.10** | **10.62** | **28.31** | **23.46** |
| 10m | Clean | ~89 | ~88 | ~95 | ~91 | 88.56 |
| 10m | RAUCA | 18.88 | 56.70 | 31.00 | 44.85 | 39.25 |
| 10m | **PGA** | **1.40** | **45.53** | **8.44** | **30.89** | **21.78** |
| 20m | Clean | ~87 | ~87 | ~99 | ~92 | 90.20 |
| 20m | RAUCA | 37.29 | 59.34 | 59.07 | 48.60 | 49.50 |
| 20m | **PGA** | **1.85** | **43.95** | **14.60** | **23.14** | **20.73** |

*Note:* \* denotes black-box transfer attack (white-box training on Faster R-CNN).

### Elevation Angle Experiments (Faster R-CNN, mean AP@0.5)

| Method | 20° | 30° | 40° | 50° | 60° | Avg. |
|--------|-----|-----|-----|-----|-----|------|
| Clean | 91.30 | 87.00 | 88.04 | 78.70 | 65.46 | 82.10 |
| RAUCA | 46.36 | 43.69 | 46.72 | 23.47 | 9.63 | 33.97 |
| **PGA** | **21.01** | **4.62** | **4.11** | **3.90** | **0.00** | **6.73** |

### Physical Domain Experiments (1:24 scale model car, AP@0.5)

| Distance | Method | Faster R-CNN | YOLO-v5* | Mask R-CNN* | Def-DETR* | Avg. |
|----------|--------|-------------|----------|-------------|-----------|------|
| 50cm | Clean | 86.12 | 90.71 | 85.36 | 89.25 | 87.86 |
| 50cm | RAUCA | 28.86 | 50.67 | 32.09 | 35.14 | 36.69 |
| 50cm | **PGA** | **20.94** | **50.25** | **22.35** | **21.25** | **28.69** |
| 100cm | Clean | 90.19 | 92.95 | 89.32 | 93.02 | 91.37 |
| 100cm | RAUCA | 34.61 | 44.14 | 35.55 | 34.70 | 37.25 |
| 100cm | **PGA** | **21.77** | **41.82** | **23.92** | **25.54** | **28.26** |

### 1:1 Full-Scale Vehicle Physical Experiments

- Faster R-CNN AP@0.5: 88.48% → **25.67%** (camouflage deployed via stickers; multi-view capture by UAV)

### Ablation Study

| Consistency Improvement | Min-Max Optimization | Faster R-CNN | YOLO-v5* | Mask R-CNN* | Def-DETR* | Avg. |
|------------------------|---------------------|-------------|----------|-------------|-----------|------|
| ✗ | ✓ | 8.05 | 50.38 | 16.33 | 34.50 | 27.32 |
| ✓ | ✗ | 10.23 | 54.40 | 20.56 | 36.82 | 30.50 |
| **✓** | **✓** | **3.57** | **47.24** | **11.89** | **28.78** | **22.87** |

- **Complementary contributions**: Both the consistency improvement and min-max optimization are individually effective, and their combination achieves the best overall performance.
- The min-max optimization yields a particularly notable improvement in the white-box setting (Faster R-CNN: 8.05 → 3.57).
- The consistency improvement also positively affects cross-model transfer performance.

## Highlights & Insights

- **First 3DGS-based physical attack framework**: Completely eliminates dependence on mesh models and simulators; only a small number of real photographs are needed to quickly reconstruct and attack arbitrary objects, substantially lowering the barrier to attack.
- **Precise analysis of mutual and self-occlusion**: The paper provides an in-depth analysis of two fundamental issues in vanilla 3DGS under adversarial optimization and proposes targeted solutions—SuGaR regularization combined with zeroth-order-only spherical harmonic optimization—that are both simple and effective.
- **Clever min-max adversarial game**: The two-stage strategy of first "counter-attacking" the background and then optimizing the camouflage automatically filters out spurious adversarial features that rely on specific backgrounds; this idea is transferable to other robust optimization scenarios.
- **Practical dominant color regularization**: Extracting dominant color tones from the background to constrain camouflage colors simultaneously improves visual concealment and environmental adaptability.
- **Complete digital-to-physical validation chain**: The three-tier validation of CARLA simulation → 1:24 scale model car → 1:1 full-scale vehicle is highly convincing.

## Limitations & Future Work

- **Limited black-box transferability**: As shown in the experimental results, AP can be reduced to 1–4% in the white-box setting on Faster R-CNN, but transfer to YOLO-v5 still yields 40–55% AP, indicating a significant gap.
- **Sensitivity of background noise budget $\epsilon$**: The paper does not sufficiently discuss the sensitivity of this hyperparameter; too large a value may hinder camouflage optimization, while too small a value may fail to effectively filter non-robust features.
- **Limited evaluation on more advanced detectors**: Detectors such as YOLO-v8/v9 and RT-DETR have not been evaluated.
- **Physical deployment constraints**: Current deployment relies on stickers, which may not conform well to objects with complex curved surfaces.
- **Only color is optimized, not geometry**: For ease of physical deployment, only spherical harmonic coefficients are optimized; combining geometric perturbations for stronger attacks remains unexplored.

## Related Work & Insights

| Dimension | DAS/FCA/DTA/ACTIVE/TAS | RAUCA | PGA (Ours) |
|-----------|----------------------|-------|-----------|
| 3D Modeling | Mesh + simulator required | Mesh + enhanced neural renderer required | **3DGS, only sparse photos needed** |
| Rendering Pipeline | Differentiable Neural Renderer | Enhanced Neural Renderer | **3DGS rasterizer (faster and more realistic)** |
| Cross-View Consistency | No special handling | Weather augmentation | **Mutual/self-occlusion correction** |
| Robustness Optimization | Simple data augmentation | Weather + lighting augmentation | **Min-max adversarial training** |
| White-box Attack (FR-CNN) | 32–78% AP | 18–37% AP | **1–5% AP** |
| Physical Deployment Cost | High (mesh modeling required) | High | **Low (photos → stickers)** |

Compared with NeRF-based attack methods (Adv3D, Huang et al.), PGA inherits the comprehensive advantages of 3DGS in rendering speed, quality, and memory efficiency.

2. **Generality of min-max optimization**: This strategy is not limited to adversarial camouflage; it can be transferred to adversarial training, robust feature learning, and similar settings by adopting the paradigm of "first perturbing background/context, then optimizing the target" to enhance optimization robustness.
3. **3DGS as a general attack tool**: The paper notes that PGA can be extended to infrared object detection attacks, suggesting that 3DGS-based attack frameworks possess modality-agnostic generality.
4. **Finding on zeroth-order SH optimization**: This implies that high-order spherical harmonics in 3DGS may be a double-edged sword in security contexts—providing expressive power while simultaneously introducing security risks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First 3DGS-based physical attack framework with insightful analysis of mutual/self-occlusion; however, the min-max paradigm has precedents in adversarial training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-tier validation across digital and physical domains, comparison against 6 baselines, 4 detectors, and comprehensive coverage of multiple distances, viewpoints, and weather conditions.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical structure with well-matched problem–solution correspondence and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Significant reference value for autonomous driving security with strong attack performance; discussion of defenses is lacking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting](ccl-lgs_contrastive_codebook_learning_for_3d_language_gaussian_splatting.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](../../CVPR2026/autonomous_driving/f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)

</div>

<!-- RELATED:END -->
