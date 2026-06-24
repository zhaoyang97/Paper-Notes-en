---
title: >-
  [Paper Note] WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments
description: >-
  [CVPR 2025][3D Vision][SLAM] This paper presents WildGS-SLAM, a monocular RGB SLAM system based on 3D Gaussian Splatting, which guides dynamic object removal in both tracking and mapping via DINOv2 feature-driven uncertainty prediction. It significantly outperforms existing methods in tracking accuracy (ATE RMSE of 0.46 cm) in dynamic environments and high-quality novel view synthesis without artifacts.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "SLAM"
  - "3D Gaussian Splatting"
  - "Dynamic Environments"
  - "Uncertainty Estimation"
  - "Monocular Vision"
date: 2026-05-08
content_hash: 52831be97f78e0a3
---

# WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments

**Conference**: CVPR 2025  
**arXiv**: [2504.03886](https://arxiv.org/abs/2504.03886)  
**Code**: [https://wildgs-slam.github.io](https://wildgs-slam.github.io)  
**Area**: 3D Vision  
**Keywords**: SLAM, 3D Gaussian Splatting, Dynamic Environments, Uncertainty Estimation, Monocular Vision

## TL;DR

This paper presents WildGS-SLAM, a monocular RGB SLAM system based on 3D Gaussian Splatting, which guides dynamic object removal in both tracking and mapping via DINOv2 feature-driven uncertainty prediction. It significantly outperforms existing methods in tracking accuracy (ATE RMSE of 0.46 cm) in dynamic environments and high-quality novel view synthesis without artifacts.

## Background & Motivation

**Background**: Traditional SLAM systems (ORB-SLAM, DROID-SLAM) rely on the static scene assumption. Dynamic environments disrupt feature matching and photometric consistency due to moving objects, leading to tracking drift. Recently, 3DGS-based SLAM (MonoGS, Splat-SLAM) has achieved excellent results in static scenes, but its performance degrades sharply in dynamic environments.

**Limitations of Prior Work**: (1) Existing dynamic SLAM methods (DG-SLAM, DDN-SLAM) rely on semantic segmentation or object detection to identify dynamic regions, requiring pre-defined movable object categories, which fails to generalize to unknown dynamic objects; (2) Although NeRF On-the-go and WildGaussians demonstrate the ability of purely geometric methods to remove dynamic distractions, they only handle sparse view reconstruction and require known camera poses as input; (3) Existing monocular SLAM methods do not support handling dynamic environments without relying on prior category information.

**Key Challenge**: In the sequential input setting of monocular SLAM, how to simultaneously achieve robust removal of dynamic objects and high-precision camera tracking in a purely geometric manner without relying on semantic priors.

**Goal**: To design a monocular RGB SLAM system with a purely geometric approach that can simultaneously achieve precise camera tracking, high-fidelity static scene reconstruction, and artifact-free novel view synthesis in highly dynamic environments.

**Key Insight**: Inspired by NeRF On-the-go, pre-trained 3D-aware DINOv2 features are utilized as image representations to train a shallow MLP online for predicting pixel-wise uncertainty maps. This uncertainty is then injected into both the tracking (DBA) and mapping (rendering loss) pipelines.

**Core Idea**: An online-learned uncertainty MLP is used as a unified interface to translate the semantic understanding capability of DINOv2 into dynamic object awareness in tracking and mapping.

## Method

### Overall Architecture

WildGS-SLAM takes RGB image streams as input and primarily consists of three modules: (1) Uncertainty Prediction Module: extracts features using DINOv2 and decodes them into pixel-wise uncertainty maps via a shallow MLP; (2) Tracking Module: optical flow tracking based on DROID-SLAM, integrating uncertainty into the DBA layer and fusing monocular metric depth; (3) Mapping Module: maintains and optimizes the 3DGS map, with rendering losses weighted by uncertainty. The uncertainty MLP and the 3DGS map are optimized independently with isolated gradient flows.

### Key Designs

1. **Uncertainty Prediction Module**:

    - **Function**: To predict a pixel-wise uncertainty map for each input frame, identifying dynamic regions.
    - **Mechanism**: Uses 3D-aware fine-tuned DINOv2 to extract image features $F_i = \mathcal{F}(I_i)$, which are decoded into an uncertainty map $\beta_i = \mathcal{P}(F_i)$ by a shallow MLP $\mathcal{P}$. The MLP is trained online using a composition loss $\mathcal{L}_{\text{uncer}} = (\mathcal{L}_{\text{SSIM}}' + \lambda_1 \mathcal{L}_{\text{depth}})/\beta_i^2 + \lambda_2 \mathcal{L}_{\text{reg\_V}} + \lambda_3 \mathcal{L}_{\text{reg\_U}}$, where a new depth uncertainty loss term $\mathcal{L}_{\text{depth}} = |\hat{D}_i - \tilde{D}_i|_1$ (L1 difference between rendered depth and Metric3D v2 predicted depth) is introduced.
    - **Design Motivation**: Purely geometric uncertainty estimation requires no pre-defined dynamic object categories, and the 3D-aware capability of DINOv2 provides a strong prior. The newly introduced depth loss term effectively improves the ability to distinguish dynamic distractors.

2. **Uncertainty-Aware Tracking**:

    - **Function**: To reduce the influence weight of dynamic region pixels in DBA optimization.
    - **Mechanism**: Integrates the uncertainty $\beta_i$ into the Mahalanobis distance weight of the DBA objective function as $\|\tilde{p}_{ij} - \Pi_c(\omega_j^{-1}\omega_i \Pi_c^{-1}(p_i, d_i))\|_{\Sigma_{ij}/\beta_i^2}^2$, making dynamic region pixels contribute minimally to pose optimization. Meanwhile, a metric depth regularization term $\lambda_4 \sum_i \|M_i(d_i - 1/\tilde{D}_i)\|^2$ is introduced to stabilize pose estimation in the early stages when the uncertainty MLP has not yet converged.
    - **Design Motivation**: Because the uncertainty MLP is unreliable in the early stages of online learning, the metric depth regularization acts as a stable "safety net," particularly in extreme scenes dominated by dynamic objects.

3. **Uncertainty-Aware Mapping**:

    - **Function**: To suppress the contribution of dynamic regions to the rendering loss during 3DGS optimization.
    - **Mechanism**: The rendering loss is weighted by uncertainty as $\mathcal{L}_{\text{render}} = (\lambda_5 \mathcal{L}_{\text{color}} + \lambda_6 \mathcal{L}_{\text{depth}})/\beta^2 + \lambda_7 \mathcal{L}_{\text{iso}}$, where high uncertainty in dynamic regions automatically downweights them in map optimization. A key design is the complete separation of gradient flows between $\mathcal{P}$ and $\mathcal{G}$—the uncertainty loss does not affect the Gaussian map, and the rendering loss does not affect the uncertainty MLP.
    - **Design Motivation**: Without decoupled optimization, the uncertainty MLP might "cheat" the loss function by lowering the uncertainty of all regions, preventing dynamic objects from being correctly identified.

### Loss & Training

The uncertainty MLP is jointly trained using modified SSIM loss, depth uncertainty loss, and two regularization terms (minimization of feature similarity variance $\mathcal{L}_{\text{reg\_V}}$ and a $\log \beta$ regularization to prevent $\beta$ from becoming infinitely large). The 3DGS map is trained using L1 + SSIM color loss, L1 depth loss, and isotropic regularization loss. The two are optimized in parallel at each iteration with isolated gradients. Tracking utilizes the ConvGRU within the DROID-SLAM framework for iterative optical flow updates, integrating loop closure detection and online global BA.

## Key Experimental Results

### Main Results (Tracking Performance on Wild-SLAM MoCap Dataset, ATE RMSE↓ [cm])

| Method | Input | ANYmal1 | Ball | Crowd | Person | Table2 | Average |
|------|------|---------|------|-------|--------|--------|------|
| DROID-SLAM | Mono | 0.6 | 1.2 | 2.3 | 0.6 | 95.6 | 16.17 |
| Splat-SLAM | Mono | 0.4 | 0.3 | 0.7 | 0.8 | 73.6 | 8.71 |
| MegaSaM | Mono | 0.6 | 0.6 | 1.0 | 3.2 | 9.4 | 2.40 |
| DynaSLAM (N+G) | RGB-D | 1.6 | 0.5 | 1.7 | 0.5 | 34.8 | 7.84 |
| **WildGS-SLAM** | **Mono** | **0.2** | **0.2** | **0.3** | **0.8** | **1.3** | **0.46** |

### Ablation Study (Novel View Synthesis Quality)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Splat-SLAM | 17.23 | 0.699 | 0.346 |
| **WildGS-SLAM** | **20.59** | **0.783** | **0.209** |

### Key Findings

- The tracking accuracy of WildGS-SLAM (0.46 cm) achieves a 5.2x improvement over the strongest monocular baseline MegaSaM (2.40 cm), and significantly outperforms DynaSLAM with RGB-D input (7.84 cm).
- In the Table2 scene (desktop manipulation with large-scale occlusion), all baseline tracking errors are massive (Splat-SLAM 73.6 cm), whereas WildGS-SLAM maintains 1.3 cm, demonstrating robustness in extreme dynamic scenes.
- For novel view synthesis, PSNR increases by 3.36 dB and LPIPS decreases by 0.137, achieving artifact-free rendering across all scenes.
- Compared to DynaSLAM (RGB), which requires a semantic segmentation prior, WildGS-SLAM achieves a lower error using a purely geometric approach (0.46 vs 5.19 cm).

## Highlights & Insights

- **The decoupled optimization of the uncertainty MLP and 3DGS map** is a key design insight. If optimized jointly, the system will find a shortcut to "cheat". This adversarial-like training concept (resembling the transient branch in NeRF-W) is a valuable reference for many joint optimization scenarios.
- **The design of metric depth regularization as a "start-up safety net"** is highly practical. It provides stability in the critical initial stage before the uncertainty MLP converges, after which the uncertainty gradually takes over. This cold-start strategy is beneficial in various online learning systems.
- The biggest advantage of a purely geometric scheme is **zero category dependency**. It does not need to know whether the dynamic object is a person, a vehicle, or an animal, which is extremely important in open-world scenarios.

## Limitations & Future Work

- The authors collected a new Wild-SLAM dataset for evaluation but did not validate the method on larger public benchmarks (such as TUM-RGBD dynamic sequences).
- It relies on Metric3D v2 to provide metric depth and DINOv2 to provide features. The inference overhead of these two pre-trained models might affect real-time performance.
- Online training of the shallow MLP might not respond fast enough during rapid motion or sudden large bursts of new dynamic objects.
- The work has not explored how to leverage estimated uncertainty to recover trajectories of dynamic objects or perform dynamic scene understanding.

## Related Work & Insights

- **vs NeRF On-the-go / WildGaussians**: These methods also use uncertainty to remove dynamic distractions, but only handle sparse views (requiring known poses). WildGS-SLAM extends this idea to sequential SLAM configurations, which presents a more challenging scenario.
- **vs DG-SLAM / DDN-SLAM**: These dynamic SLAM frameworks rely on semantic segmentation or object detection to define dynamic regions, limiting their generalization capability to pre-defined categories. The purely geometric scheme of WildGS-SLAM is effective for any type of dynamic objects.
- **vs Splat-SLAM**: Current state-of-the-art monocular GS-SLAM, but assumes a static scene; WildGS-SLAM achieves an 18.9x tracking accuracy improvement in dynamic scenes.

## Rating

- Novelty: ⭐⭐⭐⭐ Incorporating uncertainty estimation into the tracking and mapping of GS-SLAM is a major contribution, though the core idea stems from NeRF On-the-go.
- Experimental Thoroughness: ⭐⭐⭐⭐ The self-collected dataset is well-designed (indoor/outdoor, various dynamic objects), and the quantitative and qualitative results are robust, though comparison on public benchmarks is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Method descriptions are highly clear, the system overview diagram is informative, and the mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐⭐ The first GS-SLAM to achieve purely geometric dynamic processing under monocular RGB, boosting tracking accuracy by an order of magnitude.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [\[CVPR 2026\] ODGS-SLAM: Omnidirectional Gaussian Splatting SLAM](../../CVPR2026/3d_vision/odgs-slam_omnidirectional_gaussian_splatting_slam.md)
- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](../../ECCV2024/3d_vision/sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[CVPR 2026\] AERGS-SLAM: Auto-Exposure-Robust Stereo 3D Gaussian Splatting SLAM](../../CVPR2026/3d_vision/aergs-slam_auto-exposure-robust_stereo_3d_gaussian_splatting_slam.md)

</div>

<!-- RELATED:END -->
