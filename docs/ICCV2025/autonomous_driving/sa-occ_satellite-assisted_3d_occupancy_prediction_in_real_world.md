---
title: >-
  [Paper Note] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World
description: >-
  [ICCV 2025][Autonomous Driving][3D occupancy prediction] SA-Occ is proposed as the first method to leverage satellite imagery to assist onboard cameras in 3D occupancy prediction. Three modules—Dynamic-Decoupling Fusion, 3D Projection Guidance, and Uniform Sampling Alignment—address cross-view perception challenges, achieving 39.05% mIoU (+6.97%) on Occ3D-nuScenes with only 6.93 ms additional latency.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D occupancy prediction"
  - "satellite imagery"
  - "cross-view fusion"
  - "multi-sensor fusion"
date: 2026-05-08
content_hash: 27632b421920b29c
---

# SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World

**Conference**: ICCV 2025
**arXiv**: [2503.16399](https://arxiv.org/abs/2503.16399)  
**Code**: [https://github.com/chenchen235/SA-Occ](https://github.com/chenchen235/SA-Occ)  
**Area**: Autonomous Driving
**Keywords**: 3D occupancy prediction, satellite imagery, cross-view fusion, autonomous driving, multi-sensor fusion

## TL;DR

SA-Occ is proposed as the first method to leverage satellite imagery to assist onboard cameras in 3D occupancy prediction. Three modules—Dynamic-Decoupling Fusion, 3D Projection Guidance, and Uniform Sampling Alignment—address cross-view perception challenges, achieving 39.05% mIoU (+6.97%) on Occ3D-nuScenes with only 6.93 ms additional latency.

## Background & Motivation

**Background**: 3D occupancy prediction is a core perception task in autonomous driving, aiming to represent the surrounding environment as a dense voxel grid with per-voxel semantic labels. Dominant methods (e.g., BEVDet-Occ, FlashOCC, FB-OCC) rely entirely on onboard surround-view cameras, constructing BEV/3D features via 2D-to-3D view transformation.

**Limitations of Prior Work**: Purely onboard-view approaches suffer from two inherent limitations: (1) **Occlusion**—large vehicles obstruct objects behind them, and buildings occlude street corners; (2) **Far-range degradation**—distant regions have low resolution, and 3D projection accuracy degrades significantly. These are physical constraints of the onboard viewpoint that cannot be fully resolved algorithmically.

**Key Challenge**: Autonomous driving demands accurate perception of the entire surrounding environment, including occluded and far-range regions, yet onboard cameras are physically unable to cover these areas.

**Goal**: Introduce a complementary viewpoint (satellite top-down view) to compensate for the inherent limitations of the onboard perspective.

**Key Insight**: Satellite imagery provides a natural bird's-eye view—unaffected by ground-level occlusion and with uniform resolution across distances. Although satellite images are captured historically (non-real-time), GPS and IMU enable precise retrieval of the satellite patch corresponding to the vehicle's current location, and the static scene structure (buildings, roads, vegetation) is highly consistent between satellite imagery and real-time perception.

**Core Idea**: GPS/IMU aligns historical satellite imagery with real-time onboard images. Three carefully designed cross-view fusion modules inject the "God's-eye-view" information from satellites into 3D occupancy prediction, supplementing occlusion and far-range information missing from the onboard view.

## Method

### Overall Architecture

The input consists of two components: (1) six onboard surround-view camera images (real-time); (2) a satellite image patch cropped according to the vehicle's current GPS coordinates (historical). Onboard images are processed through a baseline method (FlashOCC/BEVDet-Occ) to extract BEV features; satellite images are processed through a dedicated encoder to extract top-down features. Three core modules—Dynamic-Decoupling Fusion, 3D Projection Guidance, and Uniform Sampling Alignment—effectively fuse the two-view features. The resulting fused BEV features are fed into an occupancy prediction head to produce voxel-level semantic labels. The overall framework is plug-and-play and can enhance any BEV-based occupancy prediction method.

### Key Designs

1. **Dynamic-Decoupling Fusion (DDF)**:

    - **Function**: Resolves spatiotemporal inconsistency between satellite and onboard imagery in dynamic object regions.
    - **Mechanism**: Satellite images are captured historically, so the positions of dynamic objects (vehicles, pedestrians) differ from the current real-time scene. Naive fusion would introduce errors in dynamic regions. DDF addresses this via a dynamic–static separation attention mechanism: motion cues (or semantic predictions) in the onboard BEV features generate a dynamic region mask, splitting features into dynamic and static components. In dynamic regions, only onboard features are used (satellite information is unreliable); in static regions, cross-attention fuses onboard and satellite features (satellite provides complementary information). The fusion formula is $F_{fused} = M_{static} \cdot \text{CrossAttn}(F_{street}, F_{sat}) + M_{dynamic} \cdot F_{street}$.
    - **Design Motivation**: Temporal asynchrony is the core challenge in cross-view fusion—without decoupling, dynamic regions suffer severe hallucinations (e.g., predicting vehicles that no longer exist). Dynamic–static decoupling allows the model to fully exploit satellite information in reliable regions while falling back to onboard perception in unreliable ones.

2. **3D Projection Guidance (3D-Proj Guidance)**:

    - **Function**: Extracts 3D information from 2D satellite imagery.
    - **Mechanism**: Satellite images are 2D top-down views that inherently lack height information, yet 3D occupancy prediction requires per-voxel height. The 3D-Proj module uses depth estimates from onboard images to generate a 3D point cloud, which is projected into the satellite coordinate system as anchor points guiding 3D structure extraction from satellite features. Concretely, the 3D point cloud from onboard depth estimation is transformed into satellite coordinates; for each BEV grid cell, the height distribution of 3D points within that cell is computed as a prior, and this height prior is used for weighted lifting of satellite features (analogous to Lift-Splat-Shoot but with weights derived from onboard depth).
    - **Design Motivation**: The absence of height information in satellite imagery is a physical limitation—precise 3D structure cannot be recovered from satellite images alone. However, onboard images provide depth information; using this as a guidance signal to "inject" into satellite features is an elegant complementary strategy—satellites provide complete XY-plane coverage while onboard cameras supply Z-axis information.

3. **Uniform Sampling Alignment (USA)**:

    - **Function**: Aligns the sampling density of onboard and satellite features on the BEV grid.
    - **Mechanism**: BEV features from onboard cameras are dense in near regions and sparse in far regions (due to perspective projection), whereas satellite features are spatially uniform across all distances. Direct fusion causes weight imbalance in far regions. USA employs deformable attention to adaptively adjust sampling offsets at each BEV grid position, equalizing the effective sampling density of onboard and satellite features across all distances. In far regions where onboard sampling is sparse, the module automatically increases the weight of satellite sampling points to compensate.
    - **Design Motivation**: Density inconsistency is a latent obstacle in cross-view fusion—without alignment, fused features primarily reflect satellite information at far distances (where onboard information is too sparse) and onboard information at close range, causing discontinuities in the transition zone.

### Loss & Training

The training loss comprises: (1) voxel-level semantic cross-entropy loss and lovász-softmax loss for the main task (handling class imbalance); (2) auxiliary BEV semantic segmentation loss (guiding BEV feature learning); (3) depth estimation auxiliary loss (supervising onboard depth prediction for 3D-Proj Guidance). During training, the satellite encoder and backbone are first frozen for joint training with the baseline, followed by end-to-end fine-tuning with all parameters unfrozen.

## Key Experimental Results

### Main Results

Comparison on Occ3D-nuScenes (single-frame methods):

| Method | Backbone | Frames | mIoU (%) | Extra Latency |
|--------|----------|--------|----------|---------------|
| BEVDetOCC | R50 | 1 | 31.60 | — |
| FlashOCC (M1) | R50 | 1 | 32.08 | — |
| **SA-OCC (V1)** | R50+R18(sat) | 1 | **39.05** | 6.93 ms |
| FlashOCC-4D-Stereo (M2) | R50 | 2 | 37.84 | — |
| **SA-OCC (V2)** | R50+R18(sat) | 2 | **40.65** | 6.93 ms |
| FlashOCC-4D-Stereo (M3) | Swin-B | 2 | 43.52 | — |
| **SA-OCC (V4)** | Swin-B+R18(sat) | 2 | **43.90** | 6.93 ms |
| **SA-OCC (V5)** | Swin-B+R50(sat) | 2 | **44.29** | 6.93 ms |

### Ablation Study

| Configuration | mIoU (%) | Note |
|---------------|----------|------|
| Baseline (FlashOCC) | 32.08 | No satellite assistance |
| + Direct satellite feature concatenation | 34.52 | Simple fusion already yields gains |
| + Dynamic-Decoupling Fusion | 36.83 | Dynamic decoupling +2.31% |
| + 3D-Proj Guidance | 38.17 | Height-guided information +1.34% |
| + Uniform Sampling Alignment | 39.05 | Density alignment +0.88% |
| w/o DDF (replaced by global fusion) | 35.91 | Without dynamic decoupling −3.14% |
| w/o 3D-Proj (uniform lifting) | 37.42 | Without depth guidance −1.63% |

### Key Findings

- **Satellite assistance is significant and efficient**: A gain of 6.97% mIoU is achieved with only 6.93 ms additional latency, yielding an exceptionally favorable cost-benefit ratio.
- **DDF contributes the most**: Dynamic-Decoupling Fusion is the largest contributor among the three modules (+2.31% / +3.14%), confirming the importance of handling temporal asynchrony.
- **Greater advantage in nighttime scenarios**: Satellite imagery is unaffected by illumination changes (historical images are typically captured in daylight), providing stable complementary information when onboard cameras degrade severely at night.
- **Largest gains at far range**: Prediction accuracy improves by approximately 10% in the 40–50 m range, with modest improvement (~2%) at close range (0–10 m), fully consistent with the design intent.
- SA-OCC continues to improve in multi-frame settings (V2: 40.65%), demonstrating that satellite information and temporal information are complementary.

## Highlights & Insights

- **The cross-view complementarity idea is natural and practical**: Satellite imagery serves as a "free" God's-eye-view information source (historical imagery, publicly available) that forms a natural complement to onboard perception. This idea generalizes to any autonomous driving task requiring perception beyond the onboard camera's field of view.
- **Plug-and-play design**: The three SA-Occ modules can be inserted as plugins to enhance any BEV-based method without modifying the baseline architecture, making the approach highly engineering-friendly.
- **Accompanying dataset Occ3D-NuScenes-SatExt**: The nuScenes benchmark is extended with satellite imagery data, enabling the community to directly leverage this resource and lowering the barrier for follow-up work.

## Limitations & Future Work

- The temporal currency of satellite imagery is a fundamental limitation—if the environment undergoes large-scale changes (new construction, road works), historical satellite images may introduce incorrect information.
- Validation is currently limited to nuScenes, which covers a restricted geographic area (Boston and Singapore); satellite image coverage and quality may vary across other regions.
- GPS/IMU localization accuracy directly affects satellite image patch alignment—poor GPS signal in urban canyons may cause severe misregistration.
- Multi-temporal satellite image fusion could be explored, leveraging images captured at different times for mutual complementation.
- Integration with LiDAR point clouds—using satellite imagery as static prior and LiDAR for real-time 3D information—may outperform purely vision-based solutions.

## Related Work & Insights

- **vs. FlashOCC**: SA-Occ directly builds on FlashOCC as the baseline, adding cross-view fusion modules after its BEV features. The 6.97% improvement stems from an entirely different information source (satellite viewpoint), representing an orthogonal contribution.
- **vs. DualBEV**: DualBEV fuses BEV features from front and rear stereo cameras, sharing a similar spirit but remaining within the onboard viewpoint range. SA-Occ introduces a fundamentally different satellite viewpoint with stronger complementary information.
- **vs. BEVFormer / SurroundOcc**: These methods improve prediction quality through stronger spatiotemporal Transformers but remain constrained by the onboard viewpoint. Satellite assistance in SA-Occ is also orthogonal to these methods and could theoretically be combined with them.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to introduce satellite imagery into 3D occupancy prediction, opening an entirely new dimension of information sources.
- Experimental Thoroughness: ⭐⭐⭐⭐ Main experiments and ablations are comprehensive, though validation is limited to nuScenes.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and technical diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ High practical value—satellite imagery is publicly available, latency overhead is minimal, and the plug-and-play design is deployment-friendly for real-world autonomous driving perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](../../CVPR2026/autonomous_driving/test-time_3d_occupancy_prediction.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)

</div>

<!-- RELATED:END -->
