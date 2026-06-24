---
title: >-
  [Paper Note] ZeroVO: Visual Odometry with Minimal Assumptions
description: >-
  [CVPR 2025][Autonomous Driving][Visual Odometry] This paper proposes ZeroVO, a Transformer-based monocular visual odometry method. Through a calibration-free geometry-aware network architecture, language prior integration, and a semi-supervised training paradigm, it achieves over 30% improvement in zero-shot generalization performance across KITTI, nuScenes, Argoverse 2, and a self-built GTA dataset.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Visual Odometry"
  - "Zero-Shot Generalization"
  - "Language Prior"
  - "Semi-Supervised Learning"
  - "Calibration-Free"
date: 2026-05-08
content_hash: 15850a41f3c094c8
---

# ZeroVO: Visual Odometry with Minimal Assumptions

**Conference**: CVPR 2025  
**arXiv**: [2506.08005](https://arxiv.org/abs/2506.08005)  
**Code**: [https://zvocvpr.github.io/](https://zvocvpr.github.io/)  
**Area**: Autonomous Driving / Visual Odometry  
**Keywords**: Visual Odometry, Zero-Shot Generalization, Language Prior, Semi-Supervised Learning, Calibration-Free

## TL;DR
This paper proposes ZeroVO, a Transformer-based monocular visual odometry method. Through a calibration-free geometry-aware network architecture, language prior integration, and a semi-supervised training paradigm, it achieves over 30% improvement in zero-shot generalization performance across KITTI, nuScenes, Argoverse 2, and a self-built GTA dataset.

## Background & Motivation

**Background**: Monocular visual odometry (VO) aims to estimate the relative camera pose (rotation + translation) from consecutive image frames. Learned VO methods replace traditional geometric approaches by training neural networks on specific datasets. However, they typically perform well on training sets while generalizability to new scenes drops dramatically.

**Limitations of Prior Work**: Existing methods suffer from multiple limitations: (1) dependence on pre-calibrated or known camera intrinsic parameters; (2) evaluation on the same dataset as training, which fails to reflect true generalization capability; (3) rapid failure under adverse conditions (nighttime, rain, intense glare, dirty lenses, etc.) due to broken feature tracking and unstable optimization. A VO method capable of zero-shot generalization across camera configurations and scenes is still lacking.

**Key Challenge**: Monocular VO faces scale ambiguity—determining absolute scale is impossible when recovering 3D translation from 2D images. Traditional methods alleviate this using known camera intrinsics and depth constraints, but this requires precise camera parameters. Once the camera configuration shifts (different vehicles, different devices), the methods require recalibration.

**Goal**: Design a "minimal assumptions" VO framework that requires no camera calibration, no fine-tuning on target datasets, and no ground-truth depth/optical flow labels, while achieving zero-shot generalization in autonomous driving.

**Key Insight**: Instead of relying on a single cue, the proposed method integrates multiple complementary image-level priors (estimated optical flow, metric depth, camera intrinsics, language descriptions) and utilizes a cross-attention mechanism to allow the network to adaptively handle noise within each cue.

**Core Idea**: Utilize language priors to provide high-level semantic completion for geometric estimation, leverage semi-supervised training to learn generalization capability from large-scale unlabeled videos, and employ a multimodal pseudo-label selection mechanism to filter noise.

## Method

### Overall Architecture
Given two consecutive image frames, ZeroVO extracts multimodal features (optical flow, metric depth, estimated intrinsics, language descriptions) and generates a unified representation through a language- and geometry-guided Transformer fusion module. Finally, a dual-branch MLP decoder predicts the translation vector and rotation matrix, respectively. An optional semi-supervised training phase leverages YouTube driving videos for self-training.

### Key Designs

1. **Calibration-Free Geometry-Aware Encoding**:

    - **Function**: Extract effective geometric features without requiring known camera intrinsics.
    - **Mechanism**: The WildCamera model is used to estimate camera intrinsics $\hat{\mathbf{K}}$ from a single image and encode them into an intrinsic map $\mathbf{I}_{\hat{K}}(u,v) = \frac{|u-c_U|}{f_U} + \frac{|v-c_V|}{f_V}$ of the same size as the image. Metric3Dv2 is utilized along with the estimated intrinsics to generate a metric depth map $\hat{\mathbf{D}}$, which is then back-projected to obtain a 3D point cloud $\hat{\mathbf{D}}_{3D}$. Simultaneously, the 2D optical flow is back-projected to generate a 3D scene flow $\hat{\mathbf{F}}_{3D}$. Although these 3D estimations are noisy, they are processed by the Transformer as structured priors for robust fusion.
    - **Design Motivation**: Traditional methods treat intrinsics as known constants, resulting in failures when they are inaccurate. ZeroVO inputs intrinsics as another noisy feature, training the network to make reasonable inferences even under noisy conditions.

2. **Language-Guided Depth and Optical Flow Refinement**:

    - **Function**: Utilize high-level semantic information to improve low-level geometric estimation.
    - **Mechanism**: LLaVA-NeXT is employed to generate a scene description for each frame (containing road type, weather, dynamic objects, etc.), which is encoded into a $15\times768$ language feature matrix $\mathbf{Z}_l$ by a Sentence Transformer. Then, a cross-attention mechanism injects the language features into depth and optical flow features: $\mathbf{Z} = \text{CA}(\text{PE}([\hat{\mathbf{D}}, \mathbf{I}_{\hat{K}}]), \mathbf{Z}_l)$ (language $\rightarrow$ depth). The refined features are then fused with 3D geometric information: $\mathbf{Z}_D = \text{CA}(\text{PE}(\hat{\mathbf{D}}_{3D}), \mathbf{Z})$. The optical flow branch is processed similarly.
    - **Design Motivation**: The reliability of depth and optical flow estimation varies significantly under different conditions, such as bright vs. dark or rainy vs. sunny. Language descriptions provide contextual information of the conditions ("rainy night"), helping the network dynamically adjust its level of trust in geometric features. In experiments, the language module brings an ATE reduction of 1.3 PQ.

3. **Multimodal Pseudo-Label Semi-Supervised Training**:

    - **Function**: Utilize large-scale unlabeled videos to improve generalization capability.
    - **Mechanism**: This runs in two phases. In the first phase, a teacher model is trained under supervision on labeled data (nuScenes-OneNorth). In the second phase, the teacher model generates pseudo-labels for YouTube driving videos, which are noisy. Two filtering mechanisms are designed: (1) Geometric consistency filtering: View synthesis is performed using estimated poses and depth, and the similarity between the synthesized and real images is evaluated via SSIM; samples with a normalized SSIM < 0.5 are discarded. (2) Language-guided filtering: The similarity of language features between the first and last frames within a temporal window of $H=10$ is compared. High similarity indicates minimal scene change (redundant information), and such samples are filtered out.
    - **Design Motivation**: Direct training on all pseudo-labels actually degrades performance because redundant and noisy samples pollute the learning process. The dual-filtering mechanism ensures that only high-quality, high-information pseudo-labels are preserved.

### Loss & Training
The supervised loss is defined as $\mathcal{L} = \|\mathbf{t} - \hat{\mathbf{t}}\|_2^2 - \log(p(\mathbf{R}|\hat{\Psi}))$, where translation is regressed using MSE and rotation uses the negative log-likelihood of the Matrix Fisher distribution. The network is trained for 100 epochs using the SGD optimizer with a batch size of 16 and an initial learning rate of 0.001. Data augmentation techniques include Random Crop and Resize (RCR, to simulate different intrinsics) and horizontal flipping.

## Key Experimental Results

### Main Results

| Method | KITTI ATE↓ | nuScenes ATE↓ | Argoverse ATE↓ | GTA ATE↓ |
|------|-----------|-------------|---------------|---------|
| XVO | 168.43 | 8.30 | 5.70 | 28.02 |
| M+DS (Metric3D+DROID) | 154.77 | 10.46 | 8.51 | 12.96 |
| ZeroVO | 105.07 | 6.79 | 4.10 | 8.55 |
| **ZeroVO+** | **104.69** | **6.03** | **3.05** | **8.24** |
| TartanVO* | 103.07 | 6.26 | 7.03 | 3.82 |
| DPVO* | 78.53 | 2.66 | 1.59 | 4.33 |

\*Note: TartanVO/DPVO use ground-truth scale alignment (privileged information) rather than metric-scale prediction.

### Ablation Study

| Flow | Depth | Lang | SSL | Filter | KITTI ATE | nuScenes ATE |
|------|-------|------|-----|--------|-----------|-------------|
| ✓ | | | | | 174.24 | 12.54 |
| ✓ | ✓ | | | | 123.42 | 8.40 |
| ✓ | ✓ | ✓ | | | 105.07 | 6.79 |
| ✓ | ✓ | ✓ | ✓ | | 117.49 | 7.53 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **104.69** | **6.03** |

### Key Findings
- **Metric depth is critical for scale recovery**: Introducing the depth module reduces the KITTI ATE from 174 to 123 (-29%), significantly decreasing scale error.
- **Language priors provide significant help**: They consistently reduce the ATE by 1-2 points across all datasets, showing more pronounced effects under adverse conditions (night/rain).
- **Unfiltered semi-supervised training is detrimental**: ATE rises from 105 to 117, indicating that pseudo-label noise pollutes the learning process. Incorporating the dual-filtering mechanism reduces the ATE below 105.
- Condition analysis indicates that nighttime and intense glare are the most challenging scenarios (ATE 10-13 vs 3.6 during the day), yet ZeroVO+ still substantially outperforms baseline methods under these conditions.
- ZeroVO+ is the best performing among all metric-scale methods, and approaches DPVO (which uses ground-truth scale alignment) on several metrics.

## Highlights & Insights
- **Language as a semantic prior for VO**: This is the first work to introduce VLM language descriptions into the visual odometry task. Language provides high-level understanding of "what the scene is" (e.g., night urban road vs. sunny highway), allowing the network to remain robust when low-level geometric features are unreliable. This strategy can be extended to other depth or motion estimation tasks (e.g., depth estimation, scene flow).
- **Intrinsics as a learnable feature rather than fixed parameters**: Treating camera intrinsics—usually assumed to be known constants—as noisy input features allows the network to learn to tolerate intrinsic errors. This significantly expands the applicability of the method.
- **Value of the GTA dataset**: The newly introduced high-fidelity synthetic data contains extreme scenarios rare in real-world datasets (heavy snow, water droplets on the lens, nighttime deserts), serving as an ideal supplement for robustness evaluation.

## Limitations & Future Work
- Slow inference speed: ZeroVO+ runs at only 0.6 FPS (primarily limited by the 0.7 FPS of LLaVA-NeXT), which is far from real-time deployment. LiteZeroVO+ (without the language module) achieves 5 FPS but at the cost of some performance degradation.
- Depth estimation failures propagate to VO: In scenarios with sky reflections, glass surfaces, or dirty lenses, the depth predictions of Metric3Dv2 are severely distorted.
- Two-frame methods inherently suffer from cumulative drift without global optimization (such as loop closure). Although ZeroVO can be used as a front-end for a larger SLAM system, the paper does not demonstrate this integration.
- Language descriptions are pre-computed; future work could explore lighter semantic encoding schemes to replace the heavy pipeline of LLaVA + Sentence Transformer.

## Related Work & Insights
- **vs TartanVO**: Uses random crop and resize to simulate different intrinsics for better generalization, but only predicts relative scale. ZeroVO utilizes metric depth to recover absolute scale.
- **vs XVO**: Also employs a multimodal architecture and self-training, but ZeroVO adds language priors and superior pseudo-label filtering, consistently outperforming it across all datasets.
- **vs DROID-SLAM (M+DS)**: M+DS leverages multi-frame optimization from Metric3Dv2 + DROID-SLAM, which is theoretically stronger but less stable than ZeroVO+ in adverse real-world conditions; multi-frame optimization fails completely when tracking is lost.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing language priors to VO, calibration-free design, and multimodal pseudo-label filtering are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four datasets (including the newly created GTA dataset), detailed ablations, condition-based analysis, and noise-robustness tests.
- Writing Quality: ⭐⭐⭐⭐ Detailed and clear methodological descriptions, with substantial supplementary materials.
- Value: ⭐⭐⭐⭐ Provides a robust, calibration-free cross-domain VO solution for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamVLO: Streaming Visual-LiDAR Odometry with Cumulative Drift Compensation](../../CVPR2026/autonomous_driving/streamvlo_streaming_visual-lidar_odometry_with_cumulative_drift_compensation.md)
- [\[CVPR 2025\] MITracker: Multi-View Integration for Visual Object Tracking](mitracker_multi-view_integration_for_visual_object_tracking.md)
- [\[ECCV 2024\] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment](../../ECCV2024/autonomous_driving/dvlo_deep_visual-lidar_odometry_with_local-to-global_feature_fusion_and_bi-direc.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](../../ICCV2025/autonomous_driving/splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[CVPR 2025\] Segment Anything, Even Occluded](segment_anything_even_occluded.md)

</div>

<!-- RELATED:END -->
