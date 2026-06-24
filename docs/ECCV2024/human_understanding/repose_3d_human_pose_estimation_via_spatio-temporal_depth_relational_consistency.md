---
title: >-
  [Paper Note] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency
description: >-
  [ECCV 2024][Human Understanding][3D Human Pose Estimation] RePOSE proposes replacing traditional absolute depth supervision signals with spatio-temporal relative depth consistency constraints. This shifts 3D human pose estimation in occluded scenarios from "learning absolute depth values" to "learning the relative depth order of keypoints." With an extremely simple implementation (requiring only a few lines of code), it significantly improves the robustness and accuracy of po…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "3D Human Pose Estimation"
  - "Occlusion Handling"
  - "Relative Depth Consistency"
  - "Spatio-Temporal Relations"
  - "Video Pose Estimation"
date: 2026-05-08
content_hash: b9fd0dfb238f67a7
---

# RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Human Understanding / 3D Vision  
**Keywords**: 3D Human Pose Estimation, Occlusion Handling, Relative Depth Consistency, Spatio-Temporal Relations, Video Pose Estimation

## TL;DR

RePOSE proposes replacing traditional absolute depth supervision signals with spatio-temporal relative depth consistency constraints. This shifts 3D human pose estimation in occluded scenarios from "learning absolute depth values" to "learning the relative depth order of keypoints." With an extremely simple implementation (requiring only a few lines of code), it significantly improves the robustness and accuracy of pose estimation under occlusions.

## Background & Motivation

**Background**: Estimating 3D human poses from video (3D HPE) is one of the core tasks in computer vision, widely applied in action recognition, animation driving, sports analysis, and other scenarios. Currently, mainstream methods adopt a lifting strategy—first extracting keypoint heatmaps using a 2D detector, and then lifting the 2D keypoint sequence into 3D space using a temporal model (such as TCN, Transformer). During training, absolute depth values (i.e., the z-coordinate of keypoints) are usually used as supervision signals.

**Limitations of Prior Work**: When keypoints are occluded, the outputs of 2D detectors become unreliable, resulting in noisy inputs for the lifting model. More crucially, the absolute depth annotation of occluded keypoints itself is ambiguous—the same occluded pose can correspond to multiple plausible absolute depth values. Using absolute depth as supervision transmits ambiguous and inconsistent learning signals to the network, leading to highly unstable predictions in occluded regions.

**Key Challenge**: Absolute depth supervision works well on visible keypoints but is unreliable on occluded keypoints. Under occlusions, determining accurate absolute depth values is difficult, but the relative depth orders (who is in front, who is behind) between keypoints are usually deterministic and temporally consistent. Existing methods neglect this structural prior.

**Goal**: (1) How to provide more reliable supervision signals for occluded keypoints; (2) How to leverage spatio-temporal context to enhance prediction consistency under occluded scenarios; (3) How to obtain significant performance gains with minimal implementation cost.

**Key Insight**: The authors observe that while the absolute depth values of occluded keypoints are uncertain, their relative depth relations with neighboring keypoints (spatial domain) and the same keypoints across adjacent frames (temporal domain) are typically stable. For instance, even if a knee is occluded, its depth order between the hip and ankle is deterministic and remains consistent across consecutive frames.

**Core Idea**: Replacing absolute depth supervision with relative depth consistency losses in both spatial and temporal dimensions allows the network to learn the correct ordering of keypoints instead of their precise depth values, thereby enabling more robust estimation under occlusions.

## Method

### Overall Architecture

The input of RePOSE is a sequence of 2D keypoints from consecutive frames in a video, and the output is the 3D keypoint coordinates for each frame. The overall architecture adopts the standard 2D-to-3D lifting paradigm and can be paired with any temporal backbone (such as VideoPose3D, MixSTE, etc.). All core innovations reside in the loss functions—introducing additional spatial and temporal relative depth consistency losses on top of the standard 3D coordinate regression loss, without modifying the network architecture.

### Key Designs

1. **Spatial Relational Depth Consistency Loss**:

    - **Function**: Constrains the depth ordering relationship between different keypoints within the same frame.
    - **Mechanism**: For any two keypoints $i$ and $j$ in the same frame, this loss measures whether the signs of their difference in predicted depth and ground-truth depth are consistent. Specifically, defining spatial relation matrices as $R_s^{gt}(i,j) = \text{sign}(z_i^{gt} - z_j^{gt})$ and $R_s^{pred}(i,j) = \text{sign}(z_i^{pred} - z_j^{pred})$, the loss function penalizes inconsistencies between the two. Optimization is made differentiable using soft ranking or a margin-based loss. This ensures that even if absolute depth values are biased, the front-to-back order of keypoints remains correct.
    - **Design Motivation**: While the absolute depth of occluded keypoints might have multiple plausible values, its relative relation with neighboring keypoints is usually uniquely determined. Enforcing correct ordering relations provides more reliable gradient signals for the network.

2. **Temporal Relational Depth Consistency Loss**:

    - **Function**: Constrains the consistency of depth changes for the same keypoint across adjacent frames.
    - **Mechanism**: For the same keypoint $i$ in consecutive frames $t$ and $t+1$, the predicted direction of depth change is constrained to align with the ground truth: $\text{sign}(z_i^{t+1,pred} - z_i^{t,pred})$ should equal $\text{sign}(z_i^{t+1,gt} - z_i^{t,gt})$. This leverages the continuity of motion—keypoints do not abruptly flip their depth directions between adjacent frames. For occluded frames, temporal consistency constraints can stabilize predictions through information propagation from visible frames.
    - **Design Motivation**: Occlusions are usually short-lived, and the motion trend of a keypoint before and after occlusion is continuous. The temporal consistency loss utilizes this continuity constraint to prevent unreasonable jumps in depth predictions for occluded frames.

3. **Joint Training Strategy**:

    - **Function**: Balances absolute depth learning and relative depth learning.
    - **Mechanism**: The final loss is a weighted sum of the standard 3D coordinate regression loss (e.g., MPJPE loss) and the spatial and temporal relative depth consistency losses: $\mathcal{L} = \mathcal{L}_{abs} + \lambda_s \mathcal{L}_{spatial} + \lambda_t \mathcal{L}_{temporal}$. For visible keypoints, absolute depth supervision provides precise guidance; for occluded keypoints, relative depth consistency provides robust constraints. The two types of losses complement rather than replace each other.
    - **Design Motivation**: Completely abandoning absolute depth supervision would degrade the accuracy of visible keypoints. By combining them via weighted summation, the model maintains high accuracy in visible regions while gaining better robustness in occluded regions.

### Loss & Training

Total loss = MPJPE regression loss + $\lambda_s$ × Spatial Relational Depth Consistency loss + $\lambda_t$ × Temporal Relational Depth Consistency loss. The entire method can be implemented in just a few lines of code (calculating the sign of depth differences and adding a ranking loss), requiring no modifications to the network architecture. It can be applied as a plug-and-play loss module to any lifting method.

## Key Experimental Results

### Main Results

| Method | Backbone | Human3.6M MPJPE↓ | Occlusion MPJPE↓ | Gain |
|------|----------|-------------------|----------------|------|
| VideoPose3D | TCN | 46.8 | - | baseline |
| + RePOSE | TCN | ~44.5 | Significant | ~2.3mm |
| MixSTE | Transformer | 40.9 | - | baseline |
| + RePOSE | Transformer | ~39.2 | Significant | ~1.7mm |
| PoseFormerV2 | Transformer | 45.2 | - | baseline |
| + RePOSE | Transformer | ~43.5 | Significant | ~1.7mm |

### Ablation Study

| Config | MPJPE↓ | Description |
|------|--------|------|
| Baseline (abs only) | 46.8 | Absolute depth supervision only |
| + Spatial RDC | ~45.3 | Added spatial consistency, gain ~1.5mm |
| + Temporal RDC | ~45.8 | Added temporal consistency, gain ~1.0mm |
| + Both (Full) | ~44.5 | Spatial + temporal joint, gain ~2.3mm |

### Key Findings

- The contribution of spatial relative depth consistency is slightly larger than that of temporal consistency, as spatial relationships directly constrain the topological structure between joints.
- The improvement is more pronounced in severe occlusion scenarios, indicating that the method indeed resolves the ambiguity of supervision signals caused by occlusion.
- The method consistently improves different backbones, validating its plug-and-play nature.
- The implementation is simple (requiring only a few lines of code) with no additional inference overhead.

## Highlights & Insights

- **Minimalist yet Effective Design**: The core idea is extremely simple—shifting from teaching the network "the depth of this keypoint is 3.5" to teaching the network "this keypoint is in front of that keypoint". Redefining the supervision signal requires only a few lines of code, yet yields significant improvements under occluded scenarios, manifesting the research philosophy that "great ideas do not have to be complex".
- **Innovation at the Supervision Level**: While most 3D HPE methods focus on designing complex network architectures, this paper approaches the problem from the perspective of the loss function, complementing rather than replacing original losses. This line of thought can be transferred to other tasks handling uncertain supervision signals, such as depth estimation and 6DoF object pose estimation.
- **Spatio-Temporal Joint Structured Priors**: Simultaneously encoding the spatial topological structure of the human skeleton and the temporal continuity of motion into the loss function highlights the value of physical world priors in regularizing ill-posed problems.

## Limitations & Future Work

- The current method assumes that occlusions are temporary. If a keypoint is occluded for a long period, the temporal consistency constraint may fail.
- The spatial relationship only considers the ordering in the depth dimension, without utilizing the structured information in the $x/y$ dimensions.
- The construction of relative depth relationships is based on all keypoint pairs, leading to a complexity of $O(K^2)$ (where $K$ is the number of keypoints). Constructing relationships only between skeleton-connected neighboring keypoints could be considered to reduce computation.
- Occlusion relations between different individuals in multi-person scenarios are not considered.
- It can be extended to self-supervised or semi-supervised settings, utilizing relative depth consistency as self-supervised signals for unlabeled data.

## Related Work & Insights

- **vs VideoPose3D**: VideoPose3D uses TCN to model temporal information but relies solely on absolute depth supervision. RePOSE obtains significant improvements by simply adding the relative depth consistency loss on top of it.
- **vs OccFormer/P-STMO**: These methods handle occlusions by modifying network architectures (e.g., occlusion mask prediction, mask tokens), whereas RePOSE modifies only the loss function without altering the architecture. This is a lighter yet complementary approach.
- **vs Ordinal Ranking Loss**: Previous ordinal ranking losses only consider spatial relationships. RePOSE extends this to the joint spatio-temporal domain, providing a more comprehensive relative depth constraint.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The approach of addressing occlusion from the perspective of supervision signals is novel, and the design of joint spatio-temporal relative depth consistency is unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Highly comprehensive, with multi-backbone validation, occluded scenario analysis, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The motivation is clearly articulated, the core idea is simple and elegant, and emphasizing "only a few lines of code" is highly convincing.
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play solution for handling occlusions, offering practical value to the pose estimation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [\[ECCV 2024\] UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)
- [\[ICLR 2026\] From Sparse to Dense: Spatio-Temporal Fusion for Multi-View 3D Human Pose Estimation with DenseWarper](../../ICLR2026/human_understanding/from_sparse_to_dense_spatio-temporal_fusion_for_multi-view_3d_human_pose_estimat.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
