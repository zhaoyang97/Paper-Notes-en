---
title: >-
  [Paper Note] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation
description: >-
  [CVPR 2026][Human Understanding][cattle pose estimation] FSMC-Pose presents a lightweight cattle mounting pose estimation framework tailored for dense farm environments. By combining the frequency-spatial fusion backbone CattleMountNet with the multiscale self-calibrating prediction head SC2Head, the method achieves 89% AP with only 2.698M parameters and 4.4G FLOPs.
tags:
  - CVPR 2026
  - Human Understanding
  - cattle pose estimation
  - frequency-spatial fusion
  - multiscale self-calibration
  - mounting detection
  - lightweight
date: 2026-05-08
content_hash: 4aa33510d2901229
---

# FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation

**Conference**: CVPR 2026
**arXiv**: [2603.16596](https://arxiv.org/abs/2603.16596)
**Code**: [https://github.com/FSMC-Pose](https://github.com/FSMC-Pose)
**Area**: Human/Animal Pose Estimation
**Keywords**: cattle pose estimation, frequency-spatial fusion, multiscale self-calibration, mounting detection, lightweight

## TL;DR

FSMC-Pose presents a lightweight cattle mounting pose estimation framework tailored for dense farm environments. By combining the frequency-spatial fusion backbone CattleMountNet with the multiscale self-calibrating prediction head SC2Head, the method achieves 89% AP with only 2.698M parameters and 4.4G FLOPs.

## Background & Motivation

1. **State of the Field**: Animal pose estimation has largely adopted methods developed for human pose estimation (bottom-up/top-down), yet the complexity of agricultural environments makes direct deployment of these approaches challenging.
2. **Limitations of Prior Work**: Mounting behavior is a key visual indicator of estrus in cows; however, farm scenes present significant challenges including cluttered backgrounds, severe inter-animal occlusion, and confusion caused by similar coat colors. Public datasets for cattle mounting pose are also lacking.
3. **Root Cause**: During estrus, cattle tend to congregate, making mounting scenes denser than typical farm settings. Intertwined limbs lead to identity confusion, while real-time monitoring requirements demand low computational cost.
4. **Paper Goals**: To achieve accurate mounting pose estimation in dense, cluttered real-world farm environments while maintaining lightweight computation.
5. **Starting Point**: The paper addresses background interference, scale variation, and occlusion through three complementary perspectives: frequency-domain enhancement for foreground-background separation, multiscale receptive field aggregation, and spatial-channel self-calibration.
6. **Core Idea**: Wavelet-based frequency-domain processing enhances the separability of cattle from backgrounds; multi-receptive-field aggregation handles keypoint scale variation; a self-calibration branch corrects structural misalignment caused by occlusion.

## Method

### Overall Architecture

A top-down framework is adopted: bounding boxes of individual cattle are detected first, followed by keypoint localization for each animal. The backbone CattleMountNet is built on an inverted residual structure (depthwise separable convolution) and integrates two modules: SFEBlock and RABlock. The prediction head SC2Head extends RTMPose with spatial-channel attention and a self-calibration branch.

### Key Designs

1. **Spatial Frequency Enhancement Block (SFEBlock)**:
    - Function: Enhances the separability of cattle bodies from cluttered backgrounds.
    - Mechanism: Wavelet transform convolution (WTConv) decomposes features into low- and high-frequency subbands, which are convolved separately to enable multiscale frequency-domain modeling and receptive field enlargement. A fixed $5 \times 5$ Gaussian kernel then smooths responses and suppresses background noise. The fused features are compressed via $1 \times 1$ convolution, refined through element-wise multiplication, and combined with a residual connection to preserve original information.
    - Design Motivation: Mud and shadows in farm settings make cattle texture similar to the background, necessitating frequency-domain discrimination.

2. **Receptive Field Aggregation Block (RABlock)**:
    - Function: Captures multiscale contextual information to handle keypoint scale variation from small hooves to large torsos.
    - Mechanism: Three parallel $3 \times 3$ depthwise separable convolutions with dilation rates of 1, 3, and 5 are appended to the inverted residual unit to capture local, mid-range, and long-range context, respectively. The three feature branches are summed, normalized with LayerNorm, and stabilized via a residual connection.
    - Design Motivation: Single-scale features cannot simultaneously localize small joints and large body regions with high accuracy.

3. **Spatial-Channel Self-Calibration Head (SC2Head)**:
    - Function: Maintains structural consistency under occlusion and identity confusion.
    - Mechanism: Three branches are employed — a spatial attention branch (SAB) generates spatial weights via average/max pooling and convolution; a channel attention branch (CAB) generates channel weights via global pooling and dual-branch interaction; a self-calibration branch (SCB) establishes long-range dependencies through up- and down-sampling with convolution. The three branch outputs are fused as: $C_o = f_{1\times1}([SA, CA]) \odot SC + X$
    - Design Motivation: Backbone improvements primarily affect early feature extraction stages; the prediction head must additionally resolve structural confusion arising from overlapping cattle bodies.

### Loss & Training

- Keypoint prediction follows the coordinate regression strategy of RTMPose.
- The MOUNT-Cattle dataset (1,176 mounting instances) is constructed and merged with the publicly available NWAFU-Cattle dataset.
- Sixteen keypoints are annotated in COCO format, enabling plug-and-play compatibility with existing training pipelines.

## Key Experimental Results

### Main Results

| Method | Backbone | AP/% | AP75/% | AR/% | FLOPs/G | Params/M |
|--------|----------|------|--------|------|---------|----------|
| RTMPose | CSPNext | 88.6 | 90.6 | 89.0 | 1.926 | 13.550 |
| FSMC-Pose | CattleMountNet | **89.0** | **92.5** | **89.9** | 4.411 | **2.698** |
| SimCC | ResNet50 | 87.4 | 91.0 | 89.9 | 5.493 | 36.753 |
| DEKR | HRNet | 87.2 | 90.3 | 89.0 | 44.416 | 29.548 |

### Ablation Study

| Configuration | AP/% | Note |
|---------------|------|------|
| Baseline (RTMPose) | 88.6 | Baseline |
| + SFEBlock | Gain | Frequency enhancement improves foreground-background separation |
| + RABlock | Gain | Multiscale receptive fields improve handling of scale variation |
| + SC2Head | Further gain | Self-calibration improves structural consistency under occlusion |
| Full FSMC-Pose | 89.0 | AP +1.4%, parameters reduced by 80% |

### Key Findings

- FSMC-Pose improves AP over RTMPose by 1.4% while reducing parameter count by 80% (2.698M vs. 13.550M).
- The self-calibration branch of SC2Head yields the largest improvement in occluded scenarios.
- Frequency-domain enhancement (SFEBlock) is particularly effective in cluttered background conditions.

## Highlights & Insights

- **Introducing frequency-domain processing (wavelet transform + Gaussian smoothing) into animal pose estimation** elegantly addresses foreground-background separation under low contrast.
- **The MOUNT-Cattle dataset fills a critical data gap for cattle mounting pose estimation**, and its COCO-format annotations allow direct reuse of existing methods.
- An 80% reduction in parameters with simultaneously improved accuracy makes the method well suited for edge deployment in practical settings.

## Limitations & Future Work

- The dataset scale is relatively small (only 1,176 mounting instances), which may limit generalization capability.
- Validation is restricted to cattle scenarios; generalization to other large animals has not been tested.
- Temporal information is not incorporated, despite mounting behavior being inherently dynamic.

## Related Work & Insights

- **vs. RTMPose**: FSMC-Pose introduces frequency-domain enhancement and self-calibration on top of RTMPose, achieving higher accuracy with substantially fewer parameters.
- **vs. DeepLabCut**: Bottom-up methods struggle to distinguish individuals in crowded scenes; the top-down combined with self-calibration approach proposed here is better suited for dense scenarios.

## Rating

- Novelty: ⭐⭐⭐ Module designs are well-motivated but lack breakthrough innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-baseline comparisons and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method description is detailed.
- Value: ⭐⭐⭐⭐ Fills a research gap in cattle mounting pose estimation with practical application value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)

<!-- RELATED:END -->
