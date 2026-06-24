---
title: >-
  [Paper Note] FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video
description: >-
  [CVPR 2025 (Highlight)][Video Understanding][Egocentric Motion Capture] FRAME proposes an egocentric motion capture method based on a floor-aligned coordinate system. By establishing a lightweight VR data acquisition system, it collects a large-scale real-world dataset. A geometry-aware multimodal fusion architecture is designed to effectively combine device 6D poses with camera images, achieving state-of-the-art whole-body pose prediction at 300 FPS.
tags:
  - "CVPR 2025 (Highlight)"
  - "Video Understanding"
  - "Egocentric Motion Capture"
  - "Whole-Body Pose Estimation"
  - "VR/AR"
  - "Multimodal Fusion"
  - "Floor-Aligned Representation"
date: 2026-05-08
content_hash: 608269737154f282
---

# FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2503.23094](https://arxiv.org/abs/2503.23094)  
**Code**: [https://github.com/abcamiletto/frame](https://github.com/abcamiletto/frame)  
**Area**: Video Understanding / Human Motion Capture  
**Keywords**: Egocentric Motion Capture, Whole-Body Pose Estimation, VR/AR, Multimodal Fusion, Floor-Aligned Representation

## TL;DR
FRAME proposes an egocentric motion capture method based on a floor-aligned coordinate system. By establishing a lightweight VR data acquisition system, it collects a large-scale real-world dataset. A geometry-aware multimodal fusion architecture is designed to effectively combine device 6D poses with camera images, achieving state-of-the-art whole-body pose prediction at 300 FPS.

## Background & Motivation

**Background**: Egocentric motion capture is critical for VR/AR applications. Mainstream solutions use stereo cameras mounted on head-mounted displays (HMDs) pointing towards the body to estimate whole-body poses. Existing methods such as EgoEgo and EgoPoseFormer are typically pre-trained on synthetic data and then adapted to real-world scenarios.

**Limitations of Prior Work**: (1) Severe self-occlusion: HMD cameras can only observe parts of the body, especially the lower limbs which are frequently occluded; (2) Extreme scarcity of real-world annotated data: existing datasets are small in scale with limited motion types, leading to poor model generalization; (3) Synthetic-to-real domain gap: models pre-trained on synthetic data struggle to generate smooth and accurate predictions in real scenarios.

**Key Challenge**: Egocentric whole-body pose estimation requires inferring complete body movements from highly restricted visual information (close proximity, severe occlusion, and fisheye distortion). Relying solely on camera images is insufficient for solving global localization and predicting severely occluded body parts, particularly the lower limbs.

**Goal**: (1) Build a large-scale real-world data collection pipeline; (2) Effectively fuse heterogeneous modalities of device poses and camera images; (3) Achieve high-precision real-time whole-body motion capture.

**Key Insight**: The authors observe that VR HMDs inherently possess real-time 6D pose tracking capabilities, which is crucial for global motion localization but ignored or underutilized by prior work. Meanwhile, transforming the pose prediction problem into a floor-aligned coordinate system can explicitly encode gravity direction and ground-plane constraints.

**Core Idea**: Utilize 6D pose tracking from VR devices as an additional input, combine it with body-facing stereo cameras, and design a geometry-aware multimodal fusion architecture (FRAME) to predict whole-body poses in a floor-aligned coordinate system.

## Method

### Overall Architecture
The input of FRAME consists of two modalities: (1) HMD 6D pose sequences (position + rotation); (2) stereo camera image pairs mounted on the device pointing towards the body. The output is the complete human skeletal pose (joint positions and rotations) in the floor-aligned coordinate system. The model utilizes a Transformer architecture, extracting features from both modalities separately before combining them via a geometry-aware fusion module.

### Key Designs

1. **Floor-aligned Representation**:

    - **Function**: Represent all motion data uniformly in a floor-aligned global coordinate system.
    - **Mechanism**: Utilize the gravity direction information provided by VR devices to transform both device poses and body poses into a ground-referenced coordinate system. Specifically, the gravity direction is obtained from the device's IMU to define the ground normal as the $y$-axis, and the projection of the device on the ground as the $z$-axis. Thus, poses from all frames are represented under a unified, gravity-aligned reference frame.
    - **Design Motivation**: Traditional methods perform predictions in the local coordinate system of the device, but the device undergoes rapid rotation with head movements, leading to unstable predictions. The floor-aligned coordinate system eliminates the impact of head rotation, allowing the model to more easily learn ground-related motion patterns like standing and walking.

2. **Geometrically Sound Multimodal Integration**:

    - **Function**: Effectively integrate two highly heterogeneous input modalities: device poses (structured numerical signals) and camera images (unstructured visual signals).
    - **Mechanism**: Device poses are encoded into pose embeddings via an MLP, while camera images are processed through a lightweight CNN backbone to extract visual features. The critical aspect lies in the fusion method: instead of simple concatenation, the geometric framework provided by the device poses is used to "calibrate" the visual features. Specifically, using the known 6D poses of the device, visual features are projected onto the floor-aligned coordinate system, then the tokens of both modalities interact via cross-attention layers in a Transformer. This guarantees that fusion is performed in a geometrically consistent space.
    - **Design Motivation**: Poses and images have distinct data characteristics: poses are precise low-dimensional values, whereas images are high-dimensional but noisy visual data. Direct concatenation can bias the network toward the modality that is easier to optimize. Geometry-guided fusion ensures that both modalities complement each other within a unified spatial reference frame.

3. **Large-Scale Real-World Data Collection and Augmented Training Strategy**:

    - **Function**: Provide sufficient real-world training data and enhance the generalization capability of the model.
    - **Mechanism**: (1) Data Collection: Design a lightweight VR data acquisition system by installing body-facing stereo cameras and retroreflective markers on Quest headsets, utilizing an external motion capture system to obtain ground-truth whole-body joint annotations. It collects the largest egocentric body-facing camera dataset to date, containing a rich variety of motion types; (2) Training Strategy: Propose geometry-based data augmentation methods, including trajectory rotation augmentation in the floor-aligned coordinate system, adding noise to device poses to enhance robustness, and mixed training across different environments.
    - **Design Motivation**: Data is a key bottleneck in egocentric motion capture. Previous methods relied on pre-training on synthetic data and fine-tuning on limited real data, leading to poor generalization. Large-scale real-world data combined with geometric augmentation fundamentally improves generalization.

### Loss & Training
- Joint position L2 loss: $L_{\text{pos}} = \|J_{\text{pred}} - J_{\text{gt}}\|_2$
- Joint rotation loss: geodesic distance of rotation matrices
- Velocity smoothness loss: constrains consistency of joint velocities between adjacent frames
- Training employs a two-stage strategy: first pre-training on mixed (synthetic + real) data, then fine-tuning on real data.

## Key Experimental Results

### Main Results
On the real-world test set compared with existing SOTA methods (MPJPE in mm):

| Method | Full Body MPJPE↓ | Upper Body↓ | Lower Body↓ | Frame Rate (FPS) |
|------|------------|---------|---------|-----------|
| EgoEgo | ~95 | ~58 | ~132 | ~30 |
| EgoPoseFormer | ~82 | ~52 | ~118 | ~60 |
| AvatarPoser | ~78 | ~48 | ~112 | ~90 |
| **FRAME (Ours)** | **~55** | **~35** | **~78** | **~300** |

### Ablation Study

| Configuration | Full Body MPJPE↓ | Description |
|------|------------|------|
| Full model | ~55 | Full model |
| w/o Device pose input | ~82 | Camera images only, severe degradation of lower limbs |
| w/o Floor-aligned representation | ~68 | Prediction in device coordinate system, degraded global localization |
| w/o Geometric fusion (simple concatenation) | ~65 | Fusion method degrades to feature concatenation |
| w/o Real-world data (synthetic only) | ~90 | Domain gap leads to poor generalization |
| w/o Data augmentation | ~62 | Geometric augmentation strategy effectively improves generalization |

### Key Findings
- **Device pose input is the largest contributing factor**: Removing device pose increases the error by 49% (55 $\rightarrow$ 82), indicating that 6D pose information is critical for global localization and lower-limb prediction.
- **Floor-aligned coordinate system contributes significantly**: Error increases by 24% under the standard coordinate system, validating the importance of gravity-aligned representation for motion modeling.
- **Real-world data is more important than synthetic data**: Error almost doubles when using only synthetic data, indicating severe domain gap issues.
- **Inference speed reaches 300 FPS**: 3-5x faster than prior fastest methods, satisfying the real-time requirements of VR.

## Highlights & Insights
- **Ingenious data acquisition system design**—Uses the inherent pose tracking capabilities of VR HMDs without needing additional expensive sensors, establishing a large-scale real-world data collection pipeline. This scheme can be extended to other tasks requiring real-world annotations.
- **Floor-aligned representation is the correct inductive bias**—Most human movements are ground-related (walking, running, squatting). Modeling in a gravity-aligned coordinate system can explicitly exploit this prior.
- **The 300 FPS running speed** demonstrates the key value of lightweight architecture design in real-time scenarios like VR/AR, making it unnecessary to chase the largest models.

## Limitations & Future Work
- The current dataset is biased towards standing/walking motions, offering limited generalization to atypical movements like rolling or handstands.
- It relies on external motion capture systems for annotation, meaning data collection costs remain relatively high.
- It only supports single-person scenarios; handling conflicting device poses under multi-person interaction has not yet been explored.
- Fisheye camera distortion might affect the quality of visual features under extreme poses.

## Related Work & Insights
- **vs EgoEgo**: EgoEgo uses only camera images for two-stage prediction (first head motion, then full body). FRAME vastly improves accuracy and speed by introducing device poses and geometric fusion.
- **vs AvatarPoser**: AvatarPoser uses head and hand controller poses for pose estimation. FRAME adds camera visual information to compensate for the lack of lower-limb information.
- **vs QuestSim**: QuestSim utilizes physical simulation to ensure physical plausibility, whereas FRAME is purely data-driven, yet faster and more flexible.

## Rating
- Novelty: ⭐⭐⭐⭐ The ideas of floor-aligned representation and using VR device poses as auxiliary modalities are excellent; it well deserves CVPR Highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dataset contribution + detailed ablation + multi-baseline comparison + visualization, highly solid.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, easy-to-understand method description.
- Value: ⭐⭐⭐⭐⭐ Dataset + code + CAD designs are fully open-sourced, bringing real value to the VR/AR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [\[CVPR 2025\] BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions](bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)
- [\[ECCV 2024\] IAM-VFI: Interpolate Any Motion for Video Frame Interpolation with Motion Complexity Map](../../ECCV2024/video_understanding/iam-vfi_interpolate_any_motion_for_video_frame_interpolation_with_motion_complex.md)
- [\[CVPR 2025\] Progress-Aware Video Frame Captioning](progress-aware_video_frame_captioning.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)

</div>

<!-- RELATED:END -->
