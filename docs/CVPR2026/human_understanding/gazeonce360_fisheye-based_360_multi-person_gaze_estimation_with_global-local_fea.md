---
title: >-
  [Paper Note] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion
description: >-
  [CVPR 2026][Human Understanding][gaze estimation] This paper proposes GazeOnce360, an end-to-end dual-resolution CNN model for 360° multi-person gaze direction estimation using a single upward-facing tabletop fisheye cam…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "gaze estimation"
  - "fisheye camera"
  - "multi-person scene"
  - "dual-resolution fusion"
  - "synthetic data"
date: 2026-05-08
content_hash: 758670a80607146e
---

# GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion

**Conference**: CVPR 2026
**arXiv**: [2603.17161](https://arxiv.org/abs/2603.17161)  
**Code**: [https://caizhuojiang.github.io/GazeOnce360/](https://caizhuojiang.github.io/GazeOnce360/) (Project Page)  
**Area**: Human Understanding
**Keywords**: gaze estimation, fisheye camera, multi-person scene, dual-resolution fusion, synthetic data

## TL;DR
This paper proposes GazeOnce360, an end-to-end dual-resolution CNN model for 360° multi-person gaze direction estimation using a single upward-facing tabletop fisheye camera. The authors also construct MPSGaze360, the first large-scale synthetic dataset for this setting, achieving substantial improvements over the existing multi-stage method GAM360 in both accuracy and speed.

## Background & Motivation
Gaze estimation has broad applications in human-computer interaction, collaborative analysis, and VR. Single-person gaze estimation methods are relatively mature, driven by datasets such as MPIIGaze and ETH-XGaze, yet real-world scenarios typically involve multiple people.

Limitations of prior work on multi-person gaze estimation: (1) **conventional forward-facing cameras have a limited field of view**, requiring multiple synchronized devices to cover all directions; (2) existing attempts (e.g., GAM360) use fisheye cameras but rely on **multi-stage pipelines** (face detection → perspective projection → individual estimation), which are computationally expensive, prone to error accumulation, and may miss faces split at panoramic stitching boundaries.

The paper's starting point is that an upward-facing fisheye camera naturally provides 360° coverage, enabling a single device to capture people in all directions. However, fisheye images exhibit **severe geometric distortion** and **perspective variation**, and no public multi-person upward-facing fisheye gaze dataset exists. The authors address these challenges simultaneously from two angles: data (a synthetic dataset) and model design (an end-to-end dual-resolution architecture).

## Method

### Overall Architecture
GazeOnce360 is an anchor-based detection-and-regression network. The input fisheye image is processed by a global low-resolution branch to extract contextual features and detect face bounding boxes. For each detected face region, a high-resolution local branch extracts fine-grained eye features. The two feature streams are fused via cross-attention, and a multi-task head predicts face bounding boxes, head pose, gaze direction, and face/eye landmarks. Ground-truth boxes are used for cropping during training; predicted boxes are used during inference.

### Key Designs

1. **Rotational Convolution**:

    - Function: Replaces standard convolution to handle rotational distortion in fisheye images.
    - Mechanism: The convolutional kernel is replicated into four orthogonally rotated versions, and the feature responses from each orientation are combined via weighted averaging to enhance rotational invariance.
    - Design Motivation: An upward-facing fisheye camera captures faces in highly varying orientations. Standard CNNs possess only translational invariance and cannot handle such rotation. Rotational convolution is integrated at the top level of the FPN to endow high-level features with rotational adaptability. Experiments show it outperforms deformable convolution (DCN) for fisheye distortion (10.39° vs. 11.05°).

2. **Dual-Resolution Feature Fusion**:

    - Function: Processes a low-resolution global image and high-resolution face crops in parallel, fusing them via cross-attention.
    - Mechanism: The global branch extracts spatial layout information at 512×512 resolution; the local branch extracts fine-grained eye features from cropped face regions at 1024×1024. Fusion is performed via cross-attention: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{QK}^T}{\sqrt{d_k}})\mathbf{V}$, where global features serve as Query and local face features serve as Key/Value. A spatial mask restricts attention to the corresponding face region.
    - Design Motivation: Gaze estimation requires highly precise eye features (iris and pupil positions), yet processing the full fisheye image at high resolution is inefficient due to the large proportion of background pixels. The dual-resolution scheme maintains accuracy while improving speed by 22%.

3. **Multi-Task Supervision and Synthetic Dataset MPSGaze360**:

    - Function: Generates large-scale synthetic training data using Unreal Engine 5 and MetaHuman, with face and eye landmarks as auxiliary supervision signals.
    - Mechanism: The dataset contains 23,496 fisheye images with 1–7 persons per frame and 69 distinct character models. Five orthogonal perspective views are rendered and projected into equidistant fisheye images. Annotations include 3D gaze vectors, 2D face/eye landmarks, face bounding boxes, and 3D head pose.
    - Design Motivation: Obtaining accurate multi-person gaze annotations (especially eye landmarks such as pupil centers) from real upward-facing fisheye footage is practically infeasible. Synthetic data enables pixel-level precise annotation, and prior work (GazeGene) has demonstrated the feasibility of synthetic-to-real generalization.

### Loss & Training
A multi-task joint loss is used: $\mathcal{L} = \lambda_1\mathcal{L}_c + \lambda_2\mathcal{L}_b + \lambda_3\mathcal{L}_d + \lambda_4\mathcal{L}_h + \lambda_5\mathcal{L}_g + \lambda_6\mathcal{L}_{fl} + \lambda_7\mathcal{L}_{el}$, where $\mathcal{L}_c$ is a balanced cross-entropy classification loss and all remaining terms are Smooth L1 losses. The model is trained for 150 epochs using the Adam optimizer with an initial learning rate of $10^{-3}$, decayed at epochs 30 and 100.

## Key Experimental Results

### Main Results

| Method | Gaze Error (°) ↓ | Adjusted Gaze Error (°) ↓ | FPS ↑ |
|--------|-----------------|--------------------------|-------|
| GAM360 (multi-stage) | 18.96 | 18.76 | 4.23 |
| **GazeOnce360** | **10.39** | **9.99** | **16.23** |
| Gain | −8.57 | −8.77 | +12.00 |

### Ablation Study

| Configuration | Precision ↑ | Recall ↑ | Gaze Error (°) ↓ | FPS ↑ | Note |
|---------------|-------------|----------|-----------------|-------|------|
| Baseline (no RotConv, no landmarks) | 0.984 | 0.993 | 12.14 | — | Baseline |
| +RotConv | 0.992 | 0.993 | 11.14 | — | Rotational invariance reduces error by ~1° |
| +RotConv+Eye Landmarks | 0.994 | 0.994 | **8.89** | — | Eye supervision contributes most |
| Single-resolution (512) | 0.996 | 0.992 | 16.50 | 20.49 | Low resolution, poor accuracy |
| Single-resolution (1024) | 0.998 | 0.993 | 8.945 | 13.30 | High accuracy but slow |
| Dual-resolution (512+1024) | 0.999 | 0.993 | 8.968 | **16.23** | Accuracy ≈ high-res, speed +22% |
| RotConv vs. DCN | — | — | 10.39 vs. 11.05 | — | RotConv outperforms DCN |

### Key Findings
- Eye landmark supervision is the single largest contributor to gaze accuracy improvement (12.14° → 8.89°, a 26.8% reduction).
- Rotational convolution clearly outperforms deformable convolution, indicating that the core challenge of fisheye distortion is rotation rather than general spatial deformation.
- Under a cross-scene and cross-identity evaluation setting, gaze error increases only marginally from 8.945° to 10.39°, demonstrating good generalization.
- Models trained purely on synthetic data produce plausible gaze predictions on real fisheye images.

## Highlights & Insights
- The problem formulation is valuable: 360° multi-person gaze estimation from a single tabletop fisheye camera has clear application prospects in smart meeting rooms and service robotics.
- The end-to-end approach versus the multi-stage pipeline yields nearly 2× accuracy improvement and nearly 4× speed improvement.
- The synthetic data generation pipeline is well-designed — from UE5 MetaHuman to the fisheye projection model — and is reproducible and extensible.

## Limitations & Future Work
- Training and evaluation are currently conducted solely on synthetic data; quantitative results on real-world scenes are absent (only qualitative visualizations are provided).
- MPSGaze360 is relatively small in scale (23K images, 69 character models), and its diversity may be insufficient to cover the complexity of real-world scenes.
- The fisheye projection model assumes only the equidistant projection; distortion model variation across different lenses could pose challenges in practical deployment.
- Eye region resolution for distant subjects remains low, potentially limiting the effectiveness of high-resolution cropping.

## Related Work & Insights
- GazeOnce360 is a fisheye extension of GazeOnce (multi-person gaze estimation with forward-facing cameras), inheriting its anchor-based multi-task design.
- The effectiveness of rotational convolution in fisheye perception is transferable to other fisheye tasks such as fisheye object detection and segmentation.
- The viability of a sim-to-real strategy is validated; future work could further improve generalization through domain adaptation or domain randomization.

## Rating
- Novelty: ⭐⭐⭐⭐ The problem formulation is novel, though individual technical components (rotational convolution, dual-resolution fusion, synthetic data) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation is limited to synthetic data; quantitative comparisons on real data and additional baselines are lacking.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with rich illustrations and a thorough description of the dataset generation pipeline.
- Value: ⭐⭐⭐⭐ Presents the first end-to-end solution for fisheye gaze estimation with clear practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-view Gaze Target Estimation](../../ICCV2025/human_understanding/multi-view_gaze_target_estimation.md)
- [\[CVPR 2026\] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation](fsmc-pose_frequency_and_spatial_fusion_with_multiscale_selfcalibration_for_cattle.md)
- [\[ICCV 2025\] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation](../../ICCV2025/human_understanding/high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)
- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](mmgait_multi_modal_gait_recognition.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](../../NeurIPS2025/human_understanding/omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
