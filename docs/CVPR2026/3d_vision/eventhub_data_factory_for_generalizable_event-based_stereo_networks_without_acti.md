---
title: >-
  [Paper Note] EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors
description: >-
  [CVPR 2026][3D Vision][Event cameras] This paper proposes EventHub, a training data factory for event-based stereo matching that requires no annotation from active sensors such as LiDAR. It generates proxy event-depth pa…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Event cameras"
  - "stereo matching"
  - "data factory"
  - "novel view synthesis"
  - "cross-modal distillation"
date: 2026-05-08
content_hash: b7facf384f4f3947
---

# EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors

**Conference**: CVPR 2026
**arXiv**: [2604.02331](https://arxiv.org/abs/2604.02331)
**Code**: [https://bartn8.github.io/eventhub](https://bartn8.github.io/eventhub)
**Area**: 3D Vision / Stereo Matching / Event Cameras
**Keywords**: Event cameras, stereo matching, data factory, novel view synthesis, cross-modal distillation

## TL;DR
This paper proposes EventHub, a training data factory for event-based stereo matching that requires no annotation from active sensors such as LiDAR. It generates proxy event-depth pairs via novel view synthesis and transfers knowledge from RGB stereo models through cross-modal distillation. The resulting event stereo models surpass LiDAR-supervised counterparts in cross-domain generalization, reducing error by up to 50% on M3ED and MVSEC.

## Background & Motivation

1. **Background**: Deep learning-driven stereo matching has achieved remarkable progress in the RGB domain, with highly generalizable foundation models such as FoundationStereo emerging. Event cameras, with their microsecond temporal resolution, absence of motion blur, and high dynamic range, offer unique advantages in autonomous driving and robotic navigation.

2. **Limitations of Prior Work**: Annotated data for event-based stereo matching is extremely scarce — datasets such as DSEC, MVSEC, and M3ED are far smaller than their RGB counterparts. LiDAR-based annotation also suffers from inherent limitations: sparsity (requiring 7×7 dilation before use), accumulated errors in dynamic scenes, reprojection errors, and failure on non-Lambertian surfaces.

3. **Key Challenge**: Acquiring and annotating event camera data requires complex active sensor setups (e.g., LiDAR), which are costly and yield limited annotation quality, whereas RGB image data is abundant and supported by mature depth estimation tools. The central question is how to leverage inexpensive RGB data to train event camera models.

4. **Goal**: (1) How can event camera training data be generated from RGB images? (2) How can knowledge from RGB stereo models be transferred to the event domain? (3) How can event stereo models achieve unprecedented generalization ability?

5. **Key Insight**: RGB images are cheap and abundant, and RGB stereo foundation models already possess strong depth estimation capability. These resources serve as inputs to an automatic data factory for event-based stereo matching training data.

6. **Core Idea**: Combine novel view synthesis (to generate proxy events and depth labels) with cross-modal distillation (to transfer knowledge from RGB stereo models), constructing an event-based stereo matching data factory that requires no active sensors.

## Method

### Overall Architecture
EventHub generates training data through two complementary pathways: (i) **Event Data Factory** — using SVRaster to synthesize event stereo pairs and depth labels from sparse RGB image sequences via novel view synthesis; and (ii) **Stereo Cross-Modal Distillation** — in scenes where RGB-Event paired data already exist, an RGB stereo foundation model (e.g., FoundationStereo) generates proxy depth labels, which are aligned to the event domain via multi-view geometry. Data from both pathways are merged into the EventHub training set for training or adapting event stereo networks.

### Key Designs

1. **SVRaster-Based Synthetic Event Generation Pipeline**:

    - **Function**: Generates event camera stereo training data (synthetic event streams and proxy depth labels) from multi-view RGB images of static scenes.
    - **Mechanism**: (a) Camera poses and intrinsics are recovered from RGB images using COLMAP; (b) an SVRaster radiance field is trained with normal consistency regularization $\mathcal{L}_{N\text{-mean/med}}$ and monocular depth prior $\mathcal{L}_{\text{DAv2}}$ to improve depth map quality; (c) two virtual motion trajectories are designed — a local trajectory $\Gamma(\tau)$ with single-axis translation (suitable for object-centric scenes) and a global trajectory $\Omega(\tau)$ fitted with cubic splines (suitable for indoor scenes); (d) trinocular image pairs are rendered along the trajectory and fed into ESIM to generate synthetic event streams; (e) inter-frame intervals are dynamically adjusted based on optical flow-estimated pixel motion, with the number of intermediate renders set adaptively as $2^n$ where $n = \max(\lceil\log_2(|\mathbf{F}|_{\max})\rceil, 0)$, avoiding event simulation artifacts.
    - **Design Motivation**: SVRaster's fast rendering speed matches the high temporal resolution demands of event cameras. The virtual trajectory design induces brightness variation in static scenes to trigger events. Dynamic frame interval adjustment balances simulation quality and computational efficiency.

2. **Stereo Cross-Modal Distillation**:

    - **Function**: Transfers knowledge from an RGB stereo foundation model to the event domain in scenes where RGB-Event paired data are available.
    - **Mechanism**: An RGB stereo model $\Phi_c$ (e.g., FoundationStereo) processes RGB stereo pairs to obtain a disparity map $\mathbf{D}_c$, which is converted to depth as $\mathbf{Z}_c = (b_c \cdot f_c) / \mathbf{D}_c$. Using the known relative pose $\mathbf{T}_{c \to e}$ between the RGB and event cameras, proxy labels are back-projected, transformed, and re-projected to align from the RGB domain to the event domain.
    - **Design Motivation**: When RGB and event cameras are already calibrated and paired in the same environment (e.g., DSEC dataset), synthetic event generation is unnecessary; label transfer alone suffices to exploit the strong depth estimation capability of RGB foundation models.

3. **Repurposing RGB Stereo Models for the Event Domain**:

    - **Function**: Directly transfers the architecture and weights of pretrained RGB stereo models to serve as event stereo models.
    - **Mechanism**: Event streams are encoded into a 3-channel Tencode representation (positive polarity channel, timestamp channel, negative polarity channel), keeping the input channel count consistent with RGB inputs and enabling direct initialization of the event model $\Phi_e$ from RGB model weights. Fine-tuning is performed with a low learning rate ($5 \times 10^{-5}$) while the DAv2 prior is frozen.
    - **Design Motivation**: This fully leverages the strong priors learned by RGB stereo foundation models from large-scale data, avoiding training event stereo models from scratch.

### Loss & Training
- NVS-generated data are supervised with a NeRF-supervised loss incorporating trinocular photometric consistency and confidence weighting.
- Distilled data and non-EventHub data use the original losses of their respective models.
- SVRaster training: total loss $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}} + \mathcal{L}_{\text{reg}}$, where regularization includes normal consistency, sparsity, and monocular prior terms.
- A custom voxel-size confidence $\mathbf{C}_{\text{Vsize}}$ is introduced to improve depth label quality estimation.

## Key Experimental Results

### Main Results

**DSEC In-Domain Results (E-FoundationStereo)**:

| Training Scheme | 1PE ↓ | MAE ↓ |
|---|---|---|
| Photometric | 93.85 | 3.65 |
| EV-SceneFlow | 61.80 | 3.10 |
| MIX 3 (NVS+ScanNet++) | 20.99 | 0.89 |
| MIX 4 (NVS+ScanNet+++DSEC Distill.) | 20.42 | 0.87 |
| LiDAR (GT) | 12.53 | 0.60 |

**M3ED Cross-Domain Generalization (E-FoundationStereo)**:

| Training Scheme | Day MAE ↓ | Night MAE ↓ | Indoor MAE ↓ |
|---|---|---|---|
| MIX 4 (Ours) | **0.98** | **1.54** | 2.45 |
| LiDAR (GT) | 2.89 | 1.99 | 2.87 |

### Ablation Study

| Data Combination | Composition | E-FoundationStereo DSEC MAE ↓ | Avg Rank |
|---|---|---|---|
| MIX 1 | NeRF-Stereo only | 1.39 | 5.00 |
| MIX 2 | NeRF-Stereo + DSEC | 1.04 | 4.00 |
| MIX 3 | NeRF-Stereo + ScanNet++ | 0.89 | 2.75 |
| MIX 4 | NeRF-Stereo + ScanNet++ + DSEC | 0.87 | 2.25 |

### Key Findings
- **EventHub-trained models comprehensively outperform LiDAR-supervised models in cross-domain generalization**: On M3ED and MVSEC, models trained with MIX 4 achieve MAE more than 50% lower than LiDAR-supervised counterparts, indicating that label diversity matters more than LiDAR precision.
- **Data diversity is the key factor**: Performance improves consistently from MIX 1 to MIX 4 as each additional data source is incorporated.
- **Bidirectional knowledge transfer**: Event models can in turn improve RGB foundation model performance in nighttime scenarios, realizing a RGB→Event→RGB knowledge cycle.
- **EMatch degrades severely under domain shift**: LiDAR-trained EMatch reaches MAE=12.22 on M3ED Day, whereas MIX 4 achieves only 2.23.

## Highlights & Insights
- **Data factory paradigm**: The problem of "acquiring expensive annotated data" is reframed as "generating proxy annotations from cheap data." This paradigm is transferable to data-scarce scenarios in other sensor modalities (e.g., thermal infrared, radar).
- **Dynamic adaptive frame interval**: Automatically adjusting the number of rendered intermediate frames based on optical flow avoids simulation artifacts from excessively large intervals while eliminating redundant computation from excessively small ones — a design that is both practical and elegant.
- **Counterintuitive finding that proxy labels outperform LiDAR annotation**: Although proxy labels are less precise than LiDAR, their diversity (across scenes, resolutions, and baselines) yields models with stronger generalization, offering broader implications for data strategy in the stereo matching field.

## Limitations & Future Work
- **The NVS pipeline is limited to static scenes**: Dynamic objects cannot be handled by the current virtual trajectory scheme.
- **Domain gap between ESIM-simulated and real events persists**: Synthetic events lack the noise characteristics and non-ideal threshold behavior of real sensors.
- **Computational cost**: Each scene requires independent SVRaster training; the rendering overhead across 270 NeRF-Stereo scenes × 3 trajectories plus 403 ScanNet++ scenes is substantial.
- **Future direction**: Incorporating dynamic scene NVS (e.g., 4D Gaussian Splatting) to extend the data factory's capability.

## Related Work & Insights
- **vs. NeRF-Stereo [Tosi 2023]**: NeRF-Stereo pioneered the use of NeRF for synthesizing RGB stereo training data; EventHub extends this idea to event cameras by adding an event simulator and virtual trajectory design.
- **vs. EMatch**: EMatch is the current state-of-the-art in event-based stereo matching but relies heavily on in-domain LiDAR annotation and generalizes poorly. E-FoundationStereo trained with EventHub substantially outperforms EMatch out-of-domain.
- **vs. GS2E**: The concurrent work GS2E uses 3DGS to generate multi-view event data, but only for NVS and deblurring tasks, without exploring training data generation for stereo matching.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First to apply an NVS data factory combined with cross-modal distillation to event-based stereo matching; innovative in combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four models × four data combinations × three test sets, with comprehensive evaluation covering in-domain, out-of-domain, and nighttime conditions.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, detailed pipeline descriptions, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the core bottleneck in the event camera field — data scarcity — and the finding that proxy labels surpass LiDAR annotation is highly impactful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](../../AAAI2026/3d_vision/domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[CVPR 2026\] SASNet: Spatially-Adaptive Sinusoidal Networks for INRs](sasnet_spatially_adaptive_sinusoidal_networks_for_inrs.md)

</div>

<!-- RELATED:END -->
