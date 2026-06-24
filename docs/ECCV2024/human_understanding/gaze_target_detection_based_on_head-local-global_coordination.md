---
title: >-
  [Paper Note] Gaze Target Detection Based on Head-Local-Global Coordination
description: >-
  [ECCV 2024][Human Understanding][Gaze Target Detection] A gaze target detection method based on head-local-global tri-view coordination is proposed. By introducing a field of view (FOV)-based local view and designing global-local position and representation consistency mechanisms, the accuracy of gaze target prediction is significantly improved.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Gaze Target Detection"
  - "Head-Local-Global Coordination"
  - "Field of View"
  - "Multi-view Fusion"
  - "Gaze Prediction"
date: 2026-05-08
content_hash: d2e1381a4cd706d6
---

# Gaze Target Detection Based on Head-Local-Global Coordination

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Gaze Target Detection, Head-Local-Global Coordination, Field of View, Multi-view Fusion, Gaze Prediction

## TL;DR

A gaze target detection method based on head-local-global tri-view coordination is proposed. By introducing a field of view (FOV)-based local view and designing global-local position and representation consistency mechanisms, the accuracy of gaze target prediction is significantly improved.

## Background & Motivation

1. **Background**: Gaze target detection is an important task in computer vision, aiming to predict the location or object that a person in an image is looking at. This task is widely used in human-computer interaction, social behavior analysis, and pedestrian intention understanding in autonomous driving. Existing methods typically adopt a two-stage strategy: (a) estimating the gaze direction from the head region, and (b) identifying salient targets in the global image, then combining both to determine the gaze target.

2. **Limitations of Prior Work**: 
    - **Limitations of Relying on Global Views**: Traditional methods mainly rely on head crops and global views (full images). Global views contain a large amount of gaze-irrelevant background information, which easily introduces noise, especially when the scene is complex with multiple potential targets.
    - **Lack of Local Context**: There is an excessively large scale gap between the global view and the head view—the head region is too small while the global image is too large, lacking a mid-scale context to help the model understand objects in the gaze direction.
    - **Insufficient Information Integration Across Views**: Simple concatenation or attention-based fusion of head and global features fails to effectively utilize spatial positional relationships.

3. **Key Challenge**: Information in the global view is overly redundant and lacks focus, whereas the head view is too limited—an improved perspective is required to bridge the information gap between the head and global views.

4. **Goal**: To design a more effective multi-view fusion scheme by introducing a local view to supplement the missing mid-scale information between head and global views, and designing corresponding view coordination mechanisms to integrate features from all three views.

5. **Key Insight**: Introducing a local view based on the human field of view (FOV) as the third view. The FOV local view focuses on the image region within a certain range in front of the gaze direction, which is more targeted than the global view and provides richer context than the head view.

6. **Core Idea**: By constructing a head-local-global tri-view framework and designing position and representation consistency mechanisms for multi-view coordination, more accurate gaze target prediction is achieved.

## Method

### Overall Architecture

The method consists of four core modules:
1. **View Construction**: Extracting the head view, FOV local view, and global view from the input image.
2. **Feature Extraction**: Extracting visual features from each of the three views independently.
3. **View Coordination**: Integrating tri-view features through global-local position and representation consistency mechanisms.
4. **Target Prediction**: Predicting the gaze target position (in the form of a heatmap) based on the fused features.

### Key Designs

1. **FOV-based Local View Construction**: 
    - **Function**: Cropping a local region in front of the gaze direction from the global image to serve as the local view.
    - **Mechanism**: First, a rough gaze direction is estimated from the head region. Then, centered on the head, a specific-sized region is cropped from the global image along the gaze direction. The size and orientation of this region are determined by the estimated gaze direction and the human field of view (FOV).
    - **Design Motivation**: Human gaze targets are highly likely to be located within a limited area in front of the gaze direction. The FOV local view effectively narrows down the target search space while providing a mid-scale context between the head and global scales. This design mimics the human gaze mechanism—first determining the rough direction, and then precisely locating the target within the local area.

2. **Global-Local Position Consistency**: 
    - **Function**: Establishing spatial position correspondence between the global view and the local view.
    - **Mechanism**: Since the local view is cropped from the global view, a clear spatial correspondence exists between them. This module maps the target position detected in the local view back to the global coordinate system via positional encoding and coordinate transformation, ensuring spatial consistency of the prediction results.
    - **Design Motivation**: Avoiding contradictory position predictions between the local and global views, ensuring that information from both scales can complement each other.

3. **Global-Local Representation Consistency**: 
    - **Function**: Fusing the representation of the three views at the feature level to ensure semantic consistency.
    - **Mechanism**: Designing a cross-view attention mechanism to interact features among the head view, local view, and global view. Specifically, the head feature is utilized as a query to attend to features of the local and global views respectively, obtaining information related to the gaze direction. Meanwhile, a consistency constraint is applied to the local and global features to ensure their representations of the same region are consistent.
    - **Design Motivation**: The three views observe the same scene from different scales and perspectives. It is necessary to unify their representations in the feature space to effectively leverage complementary information from each view.

### Loss & Training

- **Gaze Heatmap Loss**: Binary cross-entropy loss is used to supervise the predicted gaze target heatmap.
- **Gaze Direction Regression Loss**: L2 loss is used to supervise the estimated gaze direction angle.
- **Consistency Loss**: Contrastive learning or MSE constraints are applied to regularize the consistency of global-local representations.
- **In-frame / Out-of-frame Classification Loss**: Binary classification loss is used to determine whether the gaze target is within the image frame.
- **Multi-task Joint Training**: Simultaneously optimizing position prediction, direction estimation, and in-frame/out-of-frame judgment.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|---------|------|
| GazeFollow | AUC | SOTA | Previous best method | Significant improvement |
| GazeFollow | Avg. Dist. | SOTA (Lower) | Previous best method | Significant reduction |
| GazeFollow | Min. Dist. | SOTA (Lower) | Previous best method | Significant reduction |
| VideoAttentionTarget | AUC | SOTA | Previous best method | Significant improvement |
| VideoAttentionTarget | Avg. Dist. | SOTA | Previous best method | Significant reduction |
| ChildPlay | AUC | SOTA | Previous best method | Improvement |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Global + Head only (No local view) | Lower AUC | Baseline of the traditional two-view scheme |
| With local view (No coordination mechanism) | AUC improvement | Local view itself contributes to performance |
| With position consistency | Further AUC improvement | Spatial correspondence helps precise localization |
| With representation consistency | Continuous AUC improvement | Feature fusion at the semantic level is critical |
| Full model (All components) | Highest AUC | Three components cooperate complementarily |
| Different FOV sizes | Optimal value exists | Too large degrades to global, too small fails to cover target |

### Key Findings

- The introduction of the local view brings systematic and consistent performance improvements to gaze target detection.
- The design of the FOV-based local view is more effective than random cropping or fixed-size cropping.
- The proposed method has good scalability and can act as a plug-and-play module to enhance existing gaze target detection methods.
- State-of-the-art performance is achieved across multiple benchmark datasets.
- Both components of the view coordination mechanism (position consistency and representation consistency) make independent contributions.

## Highlights & Insights

- **Elegant Design Intuition**: The introduction of the FOV local view mimics the human gaze mechanism—first estimating a rough direction from the head, then searching for targets within the field of view, which aligns well with cognitive intuition.
- **Scalability**: The paper demonstrates the capability of applying this framework to enhance existing gaze detection methods, proving its generalizability.
- **Complementary Multi-scale Information**: The head view provides face orientation information, the local view provides target candidate regions, and the global view provides scene context—forming effective information complementarity.
- **Dual Consistency of Position and Representation**: Consistency is pursued not only in spatial position but also in feature representation, presenting a comprehensive fusion strategy.

## Limitations & Future Work

- The construction of the FOV local view depends on initial gaze direction estimation; if the initial estimation has a large deviation, the local view may fail to cover the actual gaze target.
- When multiple people are present in the scene, local views of different characters may overlap, increasing processing complexity.
- For cases where the gaze target is at the image edge or outside the image, the effectiveness of the local view may be reduced.
- Exploring temporal information (such as gaze trajectories across consecutive frames in videos) to further improve prediction accuracy is a promising direction.
- Combining depth information could help resolve depth ambiguities of gaze targets in 3D space.

## Related Work & Insights

- **GazeFollow / VideoAttentionTarget**: Classic datasets and baseline methods for gaze target detection.
- **Detecting Attended Visual Targets in Video**: Pioneering work in video gaze target detection.
- **Gaze360**: Gaze estimation method in 360-degree environments.
- **Where are they looking / Who is looking at who**: Gaze analysis methods in social scenarios.
- **Insight**: Combining multi-scale view designs with human visual cognitive mechanisms is an effective methodology.

## Rating

- Novelty: ⭐⭐⭐ companionship Introducing the FOV local view and tri-view coordination mechanism represents an entirely new framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, comprehensive ablation studies, and scalability validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and natural design logic.
- Value: ⭐⭐⭐⭐ Proposes an extensible framework that practically advances the field of gaze target detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Toward Gaze Target Detection in Young Autistic Children](../../AAAI2026/human_understanding/toward_gaze_target_detection_of_young_autistic_children.md)
- [\[CVPR 2026\] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](../../CVPR2026/human_understanding/gazeonce360_fisheye-based_360deg_multi-person_gaze_estimation_with_global-local_.md)
- [\[ECCV 2024\] De-confounded Gaze Estimation](de-confounded_gaze_estimation.md)
- [\[CVPR 2026\] Gaze Target Estimation Anywhere with Concepts](../../CVPR2026/human_understanding/gaze_target_estimation_anywhere_with_concepts.md)
- [\[ICCV 2025\] Multi-view Gaze Target Estimation](../../ICCV2025/human_understanding/multi-view_gaze_target_estimation.md)

</div>

<!-- RELATED:END -->
