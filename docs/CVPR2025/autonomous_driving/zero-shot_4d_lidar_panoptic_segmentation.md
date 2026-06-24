---
title: >-
  [Paper Note] Zero-Shot 4D Lidar Panoptic Segmentation
description: >-
  [CVPR 2025][Autonomous Driving][Zero-Shot Segmentation] This paper proposes SAL-4D (Segment Anything in Lidar-4D), which utilizes a multimodal sensor setup as a bridge to distill Video Object Segmentation (VOS) models and CLIP vision-language features into the LiDAR space. This achieves zero-shot 4D LiDAR panoptic segmentation, outperforming prior methods on 3D zero-shot LPS by over $5+$ PQ.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Zero-Shot Segmentation"
  - "4D LiDAR"
  - "Panoptic Segmentation"
  - "Video Object Segmentation"
  - "Vision-Language Model Distillation"
date: 2026-05-08
content_hash: c6e8388e2c6bd5d6
---

# Zero-Shot 4D Lidar Panoptic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2504.00848](https://arxiv.org/abs/2504.00848)  
**Code**: None  
**Area**: Autonomous Driving / Point Cloud Segmentation  
**Keywords**: Zero-Shot Segmentation, 4D LiDAR, Panoptic Segmentation, Video Object Segmentation, Vision-Language Model Distillation

## TL;DR
This paper proposes SAL-4D (Segment Anything in Lidar-4D), which utilizes a multimodal sensor setup as a bridge to distill Video Object Segmentation (VOS) models and CLIP vision-language features into the LiDAR space. This achieves zero-shot 4D LiDAR panoptic segmentation, outperforming prior methods on 3D zero-shot LPS by over $5+$ PQ.

## Background & Motivation

**Background**: 4D (3D space + time) scene understanding is crucial for embodied navigation and autonomous driving, with applications ranging from streaming perception to semantic mapping and localization. LiDAR Panoptic Segmentation (LPS) requires predicting the semantic class and instance ID for each point; however, existing methods heavily rely on a large volume of manually annotated LiDAR data.

**Limitations of Prior Work**: LiDAR annotation is extremely expensive (annotating a single point cloud frame takes about 1 hour), and existing annotated datasets have limited category diversity (e.g., nuScenes has only 16 foreground classes). While zero-shot methods can recognize objects of arbitrary classes, progress in the LiDAR domain has been slow—3D zero-shot LPS is in its infancy, and the 4D dimension remains virtually unexplored. The core challenge stems from the lack of sufficiently diverse and large-scale annotated data.

**Key Challenge**: The 2D vision domain possesses a wealth of foundation models (SAM, CLIP, VOS models), whereas the LiDAR domain lacks similar general-purpose models. Directly applying 2D models to 3D point clouds encounters a massive modality gap.

**Goal**: To design a method that, without using any manual LiDAR annotations, achieves zero-shot 4D panoptic segmentation by transferring knowledge from 2D vision foundation models to 3D LiDAR.

**Key Insight**: Leveraging the calibration alignment between cameras and LiDAR on autonomous driving platforms as a natural cross-modal bridge. This approach tracks objects in video using VOS models to obtain temporally consistent tracklets, assigns semantics to each tracklet using CLIP, and lifts them into the 4D LiDAR space via calibrated sensor projection relationships.

**Core Idea**: Generate pseudo-label tracklets in the 2D video domain using VOS + CLIP, project them into 4D LiDAR to generate training data, and distill them to train the SAL-4D model.

## Method

### Overall Architecture
The training pipeline of SAL-4D is as follows: (1) Use an off-the-shelf VOS model (such as SAM 2) to track all visible objects in short video clips, obtaining temporally consistent 2D mask tracklets; (2) Compute sequence-level CLIP tokens for each tracklet as semantic descriptions; (3) Lift 2D tracklets to the 4D LiDAR point cloud space via camera-LiDAR calibration matrices to generate pseudo-labels; (4) Train the SAL-4D model on these pseudo-labels. During inference, SAL-4D directly takes LiDAR point clouds as input, without requiring camera data.

### Key Designs

1. **VOS-Driven 2D Tracklet Generation**:

    - **Function**: Obtain temporally consistent object segmentation in videos
    - **Mechanism**: Leverage the latest VOS models (such as SAM 2) to automatically segment and track short video clips without requiring any manual prompts or annotations. By discovering all objects in the first frame and tracking them in subsequent frames, the VOS model ensures that the same object obtains a consistent ID across different time steps. The key lies in the "class-agnostic" nature of VOS—it tracks any visible object rather than being restricted to predefined categories, which serves as the foundation for achieving zero-shot capacity.
    - **Design Motivation**: VOS models already achieve high-quality zero-shot tracking on 2D images, but this capability cannot be directly applied to LiDAR. Utilizing calibrated multimodal sensor systems as a bridge allows "transmitting" this 2D tracking capability to 3D.

2. **Sequence-Level CLIP Semantic Labeling**:

    - **Function**: Assign open-vocabulary semantic features to each tracked object
    - **Mechanism**: For each tracklet, crop images of the object across different frames along the temporal dimension, extract features via the CLIP image encoder for each, and average them to obtain a stable, sequence-level CLIP token. Instead of a fixed category label, this token is a continuous semantic vector, preserving CLIP's open-vocabulary capability. Averaging across multiple frames mitigates feature noise caused by single-frame occlusions or perspective changes.
    - **Design Motivation**: Traditional methods compute semantics for each object per frame individually, which can cause inconsistent labels for the same object across different frames. Sequence-level aggregation ensures temporally consistent semantic representations.

3. **2D-to-4D Pseudo-Label Lifting and SAL-4D Model Distillation**:

    - **Function**: Transfer segmentation knowledge from the 2D video domain to the 4D LiDAR domain
    - **Mechanism**: Map pixels corresponding to each 2D mask to 3D LiDAR points using the camera-to-LiDAR calibration matrix. A LiDAR point may be covered by masks from multiple frames or cameras; the final label is determined through voting or confidence weighting. Temporally, projections of the same tracklet ID into LiDAR over multiple frames constitute a 4D tracklet. The SAL-4D model takes LiDAR point cloud sequences as input, predicting each point's instance ID and CLIP semantic vector. The loss incorporates both instance segmentation loss and CLIP token regression loss.
    - **Design Motivation**: Although pseudo-labels contain noise (projection errors, occlusions, etc.), the statistical advantage of large-scale pseudo-labeling compensates for individual sample quality deficiencies. Furthermore, SAL-4D can learn 3D geometric reasoning capabilities during distillation that are absent in 2D pseudo-labels.

### Loss & Training
The training loss of SAL-4D consists of three components: (1) Instance segmentation loss—using Hungarian matching to pair predicted instances with pseudo-label instances, followed by mask and classification loss computation; (2) CLIP feature regression loss—the $L_2$ distance between predicted point-level CLIP features and pseudo-label CLIP tokens; (3) Temporal consistency loss—encouraging predicted features of the same instance in adjacent frames to remain consistent.

## Key Experimental Results

### Main Results

| Method | Dataset | 3D LPS PQ | Zero-Shot | Temporal Consistency |
|------|--------|-----------|--------|---------|
| OpenScene | nuScenes | 18.3 | ✓ | ✗ |
| LidarCLIP | nuScenes | 15.7 | ✓ | ✗ |
| Prev. 3D SOTA | nuScenes | ~20.0 | ✓ | ✗ |
| **SAL-4D (3D)** | **nuScenes** | **25.2** | **✓** | **✗** |
| **SAL-4D (4D)** | **nuScenes** | **27.8** | **✓** | **✓** |

### Ablation Study

| Configuration | PQ | Description |
|------|------|------|
| Full SAL-4D | 27.8 | Full model (4D) |
| w/o Temporal Consistency | 25.2 | Degrades to frame-by-frame 3D, -2.6 |
| w/o Sequence-Level CLIP | 23.4 | Unstable single-frame CLIP, -4.4 |
| w/o VOS Tracking | 19.8 | Degrades to frame-by-frame segmentation without tracking, -8.0 |
| Single Camera → Multi-Camera | 22.1→27.8 | Multi-camera offers better coverage, +5.7 |

### Key Findings
- **Temporal consistency is the key advantage of 4D**: 4D SAL-4D outperforms the 3D version by 2.6 PQ, as temporal information helps cope with occlusions and sparsity in individual frames.
- **VOS tracking contributes the most**: Removing VOS leads to a performance drop of 8.0 PQ, indicating that high-quality tracklets are fundamental to the entire method.
- The multi-camera setup significantly impacts pseudo-label quality—expanding the coverage from the limited perspective of a single camera to nearly 360°.
- Under the zero-shot setting, categories unseen in the training set (e.g., animals, construction tools) can still be recognized, demonstrating the generalization capability of CLIP features.

## Highlights & Insights
- **Modal Bridge Approach**: Utilizing calibrated multimodal sensor systems as a natural channel for 2D-to-3D knowledge transfer avoids complex cross-modal learning. This "detour" strategy is generally applicable in robotic systems with multi-sensor setups.
- **Information Gain from the Temporal Dimension**: 4D is not merely a stack of 3D snapshots—temporal consistency constraints can correct prediction errors in individual frames, similar to temporal smoothing in videos. This is particularly helpful in addressing the sparsity of LiDAR point clouds.
- **Preservation of Open Vocabulary**: By distilling CLIP features rather than fixed category labels, SAL-4D retains the ability to recognize arbitrary categories in a zero-shot manner.

## Limitations & Future Work
- Pseudo-label quality is limited by the tracking accuracy of the VOS model—it can fail under fast motion, severe occlusion, or for distant objects.
- The 2D-to-3D projection exhibits alignment errors near the edges of LiDAR point clouds, impacting the accuracy of instance boundaries.
- The current method relies on a calibrated camera-LiDAR setup and cannot be applied to scenarios equipped only with LiDAR.
- Inference speed is not fully discussed in the paper; the real-time processing of point cloud sequences by the 4D model is a critical issue for practical deployment.
- Future work could explore introducing SAM 2's prompting mechanism into the LiDAR domain to achieve interactive 4D segmentation.

## Related Work & Insights
- **vs OpenScene/LidarCLIP**: These methods directly distill CLIP features onto 3D points without performing instance segmentation, whereas SAL-4D additionally utilizes VOS to provide instance-level information and extends this to 4D.
- **vs SAL (3D Version)**: SAL-4D adds the temporal dimension on top of SAL, proving that 4D instance tracking can significantly improve segmentation quality.
- **vs Supervised 4D-LPS Methods**: Supervised methods perform better on closed category sets but cannot recognize novel categories. The PQ of SAL-4D under the zero-shot setting is already close to some earlier supervised methods.

## Rating
- Novelty: ⭐⭐⭐⭐ The first to achieve zero-shot 4D LiDAR panoptic segmentation; the combination of VOS + CLIP + LiDAR is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Relative comprehensive ablations, though comparisons with more recent methods could be more thorough.
- Writing Quality: ⭐⭐⭐⭐ The methodology is described clearly, and the pipeline diagram is intuitive.
- Value: ⭐⭐⭐⭐ It offers practical advancements toward zero-annotation autonomous driving scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads](../../ICCV2025/autonomous_driving/4dsegstreamer_streaming_4d_panoptic_segmentation_via_dual_threads.md)
- [\[CVPR 2025\] 3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [\[CVPR 2025\] A Dataset for Semantic Segmentation in the Presence of Unknowns](a_dataset_for_semantic_segmentation_in_the_presence_of_unknowns.md)

</div>

<!-- RELATED:END -->
