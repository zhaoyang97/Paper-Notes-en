---
title: >-
  [Paper Note] EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes
description: >-
  [NeurIPS 2025][3D Vision][Event camera] EAG3R integrates asynchronous event streams from event cameras into the MonST3R point map reconstruction framework. Through a Retinex enhancement module, an SNR-aware fusion mechanism, and an event photometric consistency loss, it achieves robust depth estimation, pose tracking, and 4D reconstruction in extreme low-light dynamic scenes, significantly outperforming RGB-only methods via zero-shot transfer to nighttime scenarios.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Event camera
  - 3D geometry estimation
  - low-light
  - point map reconstruction
  - dynamic scene reconstruction
date: 2026-05-08
content_hash: db7b8ae584675015
---

# EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes

**Conference**: NeurIPS 2025
**arXiv**: [2512.00771](https://arxiv.org/abs/2512.00771)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: Event camera, 3D geometry estimation, low-light, point map reconstruction, dynamic scene reconstruction

## TL;DR

EAG3R integrates asynchronous event streams from event cameras into the MonST3R point map reconstruction framework. Through a Retinex enhancement module, an SNR-aware fusion mechanism, and an event photometric consistency loss, it achieves robust depth estimation, pose tracking, and 4D reconstruction in extreme low-light dynamic scenes, significantly outperforming RGB-only methods via zero-shot transfer to nighttime scenarios.

## Background & Motivation

**Background**: DUSt3R/MonST3R leverages Transformers to directly regress dense point maps for pose-free 3D reconstruction, sparking a wave of research addressing challenging scenarios such as long sequences and dynamic scenes.

**Limitations of Prior Work**: In real-world scenarios such as autonomous driving, rapid motion and drastic illumination changes cause RGB image degradation including blur, overexposure, and underexposure. Existing RGB-only methods suffer severe performance degradation under these conditions.

**Key Challenge**: RGB cameras rely on long-exposure imaging, making them inherently ill-suited for extreme lighting and fast-motion scenes. Event cameras offer high temporal resolution and high dynamic range, yet have not been integrated into modern learning-based geometry estimation pipelines.

**Goal**: To effectively incorporate event camera data into a point map-based reconstruction framework, maintaining robustness in extreme low-light dynamic scenes.

**Key Insight**: Lightweight event adapters and SNR-guided adaptive fusion are added on top of the MonST3R backbone, while event streams are used to construct photometric consistency constraints in global optimization.

**Core Idea**: SNR-aware fusion trusts RGB features in high-SNR regions and event features in low-SNR regions, while brightness changes from events serve as additional supervision signals in global optimization.

## Method

### Overall Architecture

EAG3R enhances MonST3R in two aspects: (1) feature extraction stage — Retinex enhancement + lightweight event adapter + SNR-aware fusion; (2) global optimization stage — event photometric consistency loss. The inputs are low-light video and corresponding event streams; the outputs include per-frame depth maps, camera poses, and a global dynamic point cloud.

### Key Designs

1. **Retinex Image Enhancement Module**:

    - **Function**: Restores visibility of low-light images and generates an SNR confidence map.
    - **Design Motivation**: Directly using degraded low-light images causes feature extraction failure; a pixel-level "reliability" metric is needed to guide subsequent fusion.
    - **Mechanism**: A shallow network estimates the illumination map $L_{\text{illum}}^t$, producing the enhanced image $I_{\text{lu}}^t = I^t \odot L_{\text{illum}}^t$. The SNR map is then computed as:
    $\mathcal{M}_{\text{snr}}^t = \frac{\widetilde{I}_g^t}{|I_g^t - \widetilde{I}_g^t| + \epsilon}$
   where $\widetilde{I}_g^t$ is the mean-filtered result of the grayscale image.
    - **Novelty**: Unlike standalone low-light enhancement preprocessing (e.g., RetinexFormer), this module is jointly trained with downstream tasks and additionally outputs an SNR map for fusion.

2. **Lightweight Event Adapter**:

    - **Function**: Extracts high-fidelity features from sparse event streams.
    - **Design Motivation**: Event data can still capture structural information in low-light scenes, but requires a dedicated encoder.
    - **Mechanism**: A pretrained Swin Transformer serves as the event encoder. Events are voxelized and hierarchical features $\{F_{\text{evt},l}^t\}_{l=1}^4$ are extracted. Cross-attention is applied at each level to interact with image features:
    $F'_{\text{evt},l} = \text{CrossAttn}(Q=F_{\text{evt},l}^t, K=F_{\text{img},l}^t, V=F_{\text{img},l}^t)$
    - **Novelty**: The image encoder is frozen; only the event adapter is trained, preserving pretrained image features.

3. **SNR-Aware Feature Fusion**:

    - **Function**: Adaptively combines image and event features.
    - **Design Motivation**: Image quality varies significantly across regions — RGB is reliable in bright areas while events are more reliable in dark areas.
    - **Mechanism**: Weighted fusion using the normalized SNR map:
    $F_{\text{cat}}^t = (F_{\text{img-final}}^t \odot \hat{\mathcal{M}}_{\text{snr}}^t) \| (F_{\text{evt-final}}'^t \odot (1 - \hat{\mathcal{M}}_{\text{snr}}^t))$
   High-SNR regions favor image features; low-SNR regions favor event features.
    - **Novelty**: Compared to simple uniform fusion or attention-based fusion, SNR guidance provides a physically meaningful prior.

4. **Event Photometric Consistency Loss**:

    - **Function**: Introduces event-based spatiotemporal consistency constraints in global optimization.
    - **Design Motivation**: MonST3R's original optical flow constraints are unreliable under low light; event streams provide a more stable motion signal.
    - **Mechanism**: Salient patches are defined at Harris corner locations. The brightness increment observed from events, $\Delta L_{\mathcal{P}_m}(u)$, is compared against the predicted brightness increment derived from image gradients and motion fields, $\Delta \hat{L}_{\mathcal{P}_m}(u; X_{\text{global}})$. After normalization, an L2 residual is computed:
    $\mathcal{L}_{\text{event}} = \sum_{\mathcal{P}_m} \sum_{u \in \mathcal{P}_m} \left\| \frac{\Delta L_{\mathcal{P}_m}(u)}{\|\Delta L_{\mathcal{P}_m}\|} - \frac{\Delta \hat{L}_{\mathcal{P}_m}(u; X_{\text{global}})}{\|\Delta \hat{L}_{\mathcal{P}_m}\|} \right\|^2$
    - **Novelty**: Normalization eliminates the unknown contrast sensitivity threshold $C$, making the loss invariant to different event sensors.

### Loss & Training

The joint optimization objective is:
$$X_{\text{global}}^* = \arg\min \left(\mathcal{L}_{\text{align}} + w_{\text{smooth}}\mathcal{L}_{\text{smooth}} + w_{\text{flow}}\mathcal{L}_{\text{flow}} + w_{\text{event}}\mathcal{L}_{\text{event}}\right)$$

- Training uses only the MVSEC outdoor_day2 sequence (daytime); zero-shot evaluation is performed on outdoor_night1–3 (nighttime).
- Fine-tuned components include MonST3R's ViT-Base decoder, DPT head, enhancement network, and event adapter.
- Trained for 25 epochs on 4× RTX 3090 GPUs for approximately 24 hours.

## Key Experimental Results

### Main Results — Monocular Depth Estimation (MVSEC Night1–3)

| Method | Night1 Abs Rel↓ | Night1 δ<1.25↑ | Night2 Abs Rel↓ | Night2 δ<1.25↑ | Night3 Abs Rel↓ | Night3 δ<1.25↑ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| DUSt3R | 0.407 | 0.393 | 0.415 | 0.384 | 0.463 | 0.335 |
| MonST3R | 0.370 | 0.373 | 0.309 | 0.469 | 0.317 | 0.453 |
| DUSt3R (LightUp) | 0.425 | 0.351 | 0.462 | 0.347 | 0.525 | 0.293 |
| MonST3R (Finetune) | 0.376 | 0.426 | 0.328 | 0.472 | 0.302 | 0.509 |
| **EAG3R** | **0.353** | **0.491** | **0.307** | **0.518** | **0.288** | **0.533** |

### Main Results — Camera Pose Estimation (MVSEC Night1–3)

| Method | Night1 ATE↓ | Night2 ATE↓ | Night3 ATE↓ |
|------|:---:|:---:|:---:|
| DUSt3R | 1.474 | 3.921 | 4.109 |
| MonST3R | 0.559 | 0.626 | 0.733 |
| MonST3R (Finetune) | 0.580 | 0.467 | 0.402 |
| Easi3R_monst3r (Finetune) | 0.540 | 0.448 | 0.394 |
| **EAG3R** | **0.482** | **0.428** | 0.409 |

### Ablation Study (Night3 Depth Estimation)

| Method | Abs Rel↓ | δ<1.25↑ | RMSE log↓ |
|------|:---:|:---:|:---:|
| MonST3R (Baseline) | 0.317 | 0.453 | 0.418 |
| MonST3R (Finetune) | 0.302 | 0.509 | 0.401 |
| + Event | 0.297 | 0.518 | 0.396 |
| + Event + LightUp | 0.291 | 0.523 | 0.388 |
| + Event + LightUp + SNR Fusion (Full) | **0.288** | **0.533** | **0.371** |

### Key Findings

- **Event streams contribute most**: Adding event input alone improves Abs Rel from 0.302 to 0.297, validating the central role of event signals in low-light scenes.
- **LightUp enhancement is effective but limited**: Applying RetinexFormer as standalone preprocessing can actually worsen results (DUSt3R LightUp performs worse than the vanilla baseline), demonstrating that enhancement must be jointly optimized with downstream tasks.
- **SNR fusion is critical**: The final SNR fusion step reduces RMSE log from 0.388 to 0.371, indicating that adaptive weight allocation is more effective than simple concatenation.
- **Zero-shot nighttime generalization**: Training exclusively on daytime data achieves substantial improvements over all baselines at night, demonstrating the cross-scene generalization capability of event data.

## Highlights & Insights

- **First event-augmented point map-based reconstruction framework**: Integrating event cameras into the DUSt3R/MonST3R paradigm opens a new direction at the intersection of event cameras and geometric foundation models.
- **SNR-guided fusion**: Using signal-to-noise ratio as a physical prior to guide multimodal fusion is more interpretable and efficient than purely learned attention mechanisms.
- **Event photometric consistency loss**: The brightness change model from events is cleverly used as a constraint in global optimization; normalization renders the loss insensitive to sensor-specific parameters.
- **Zero-shot nighttime generalization**: Training solely on daytime data yet significantly outperforming all methods at night directly demonstrates the high dynamic range advantage of event cameras.

## Limitations & Future Work

- Validation is conducted only on the MVSEC dataset, which is relatively small and limited in scene diversity (outdoor driving only).
- Event data in training comes from real event cameras; V2E-synthesized events cause gradient explosion, limiting scalability to larger datasets.
- The event adapter uses Swin Transformer, but parameter count and computational overhead are not reported in detail.
- The impact of event streams under normal lighting conditions — whether they introduce noise — is not evaluated.
- Dynamic reconstruction is assessed only qualitatively, without quantitative metrics.

## Related Work & Insights

- **DUSt3R/MonST3R**: The backbone architecture of this work and the pioneering contributions to point map-based pose-free reconstruction.
- **RetinexFormer**: The inspiration for the Retinex enhancement module; this work demonstrates that standalone preprocessing is inferior to end-to-end joint training.
- **EvLight**: A pioneering work on adaptive event-image feature fusion.
- **Insights**: The SNR-aware fusion concept is generalizable to other multimodal tasks (e.g., RGB-LiDAR, RGB-Thermal fusion), where physically-informed fusion weights are more robust than purely data-driven approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ First integration of event cameras into the point map reconstruction framework; SNR-aware fusion is innovative.
- Experimental Thoroughness: ⭐⭐⭐ Three tasks with complete ablation studies, but evaluation on a single dataset is a limitation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and highly informative figures.
- Value: ⭐⭐⭐⭐ Provides a practical solution for 3D reconstruction under extreme conditions; the event + foundation model direction holds considerable promise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](../../ICCV2025/3d_vision/event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[NeurIPS 2025\] RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes](rgb-only_supervised_camera_parameter_optimization_in_dynamic_scenes.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->
