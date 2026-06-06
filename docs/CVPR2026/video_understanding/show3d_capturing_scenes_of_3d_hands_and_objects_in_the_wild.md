---
title: >-
  [Paper Note] SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild
description: >-
  [CVPR 2026][Video Understanding][hand-object interaction dataset] This paper introduces SHOW3D, the first hand-object interaction dataset with accurate 3D annotations captured in truly in-the-wild environments. Through a…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "hand-object interaction dataset"
  - "in-the-wild 3D annotation"
  - "multi-camera capture"
  - "egocentric vision"
  - "hand pose estimation"
date: 2026-05-08
content_hash: 06f17953698eb3f7
---

# SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild

**Conference**: CVPR 2026
**arXiv**: [2603.28760](https://arxiv.org/abs/2603.28760)  
**Code**: [https://show3d-dataset.github.io/](https://show3d-dataset.github.io/)  
**Area**: Video Understanding
**Keywords**: hand-object interaction dataset, in-the-wild 3D annotation, multi-camera capture, egocentric vision, hand pose estimation

## TL;DR

This paper introduces SHOW3D, the first hand-object interaction dataset with accurate 3D annotations captured in truly in-the-wild environments. Through a lightweight wearable multi-camera backpack system and an ego-exo fusion annotation pipeline, the dataset comprises 4.3 million frames of multi-view data, achieving sub-centimeter annotation accuracy for both hands and objects. Cross-dataset experiments validate the generalization advantage of models trained on SHOW3D.

## Background & Motivation

1. **Background**: 3D understanding of hand-object interaction is critical for AR/VR and robotics. Existing datasets (GigaHands, HOT3D, ARCTIC, etc.) are primarily collected in indoor studios using motion capture systems or fixed multi-camera arrays.
2. **Limitations of Prior Work**: Studio environments constrain scene diversity and ecological validity — fixed equipment restricts freedom of movement, and markers alter the visual appearance of hands and objects. At the other extreme, datasets like Ego-Exo4D offer diverse environments but lack precise 3D annotations.
3. **Key Challenge**: A fundamental trade-off exists between environmental realism and 3D annotation accuracy: either annotations are precise but environments are constrained, or environments are diverse but annotations are absent.
4. **Goal**: Break this trade-off by obtaining accurate 3D annotations of hands and objects in truly in-the-wild environments.
5. **Key Insight**: Design an approximately 8 kg backpack-mounted multi-camera system that requires no markers, and achieve marker-free automatic 3D annotation via state-of-the-art 2D detection combined with multi-view triangulation.
6. **Core Idea**: Leverage a wearable multi-camera system with an ego-exo automatic annotation pipeline to obtain studio-comparable 3D hand-object annotation accuracy in the wild.

## Method

### Overall Architecture

The system consists of three components: (1) a backpack-mounted multi-camera capture system (8 outward-facing cameras plus 2 head-mounted egocentric cameras, totaling 10 synchronized fisheye cameras at 60 Hz); (2) an ego-exo 3D hand pose annotation pipeline; and (3) a CAD-based 3D object pose annotation pipeline. The inputs are synchronized multi-view grayscale images; the outputs include 3D hand keypoints/meshes, 6-DoF object poses, segmentation masks, contact regions, and text descriptions.

### Key Designs

1. **Wearable Multi-Camera Capture System**:

    - Function: Capture synchronized multi-angle footage without restricting the user's freedom of movement.
    - Mechanism: Eight grayscale fisheye cameras (1024×1280, 152°×116° FoV) are mounted hemispherically on a backpack frame, supplemented by two egocentric cameras from a Meta Quest 3. Five MoCap cameras track optical markers on a helmet to recover the relative pose between the helmet and the backpack. All cameras are hardware-synchronized, and the reference coordinate frame moves with the user.
    - Design Motivation: The approximately 8 kg weight does not significantly restrict natural movement; fisheye lenses maximize visual coverage; the helmet is not rigidly attached to the backpack, allowing natural head motion.

2. **Ego-Exo Hand 3D Annotation**:

    - Function: Automatically obtain sub-centimeter-accurate 3D hand keypoints and meshes from multi-view images.
    - Mechanism: The Sapiens model first detects 21 hand keypoints on full frames; InterNet then refines detections on cropped perspective patches. RANSAC-based robust triangulation fuses the two sets of 2D keypoints into 3D keypoints. A personalized linear blend skinning model then fits a detailed hand mesh via inverse kinematics. Low-quality annotations are automatically filtered through Bayesian confidence estimation combining keypoint reprojection error and IK residuals.
    - Design Motivation: Full-frame detection by Sapiens provides broad coverage but insufficient hand resolution, while InterNet's cropped detection is precise but requires coarse localization first — the two are complementary. Egocentric views contribute unique perspectives that supplement blind spots in the exocentric cameras.

3. **CAD-Based Object 6-DoF Annotation**:

    - Function: Automatically obtain accurate per-frame 6-DoF object poses.
    - Mechanism: A three-stage pipeline — CNOS for 2D object detection, FoundPose for coarse pose estimation, and GoTrack for 6-DoF pose refinement — is extended to multi-view inputs throughout, replacing standard PnP with multi-view gPnP. When per-frame confidence is sufficiently high, only the refinement stage is executed (initialized from the previous frame), improving efficiency and robustness to occlusion. All stages rely on DINOv2 features without object-specific training.
    - Design Motivation: Multi-view inputs fundamentally improve pose accuracy and confidence reliability; eliminating object-specific training allows the pipeline to be rapidly applied to any object with a CAD model.

### Loss & Training

The annotation pipeline does not involve end-to-end training; instead, it combines 2D detection with geometric triangulation and optimization. For hands, confidence is estimated via a Bayesian formulation over keypoint detection/triangulation error and IK residuals. For objects, the multi-view confidence score from the GoTrack refiner serves as the filtering threshold.

## Key Experimental Results

### Main Results

Cross-dataset generalization for 3D hand pose estimation (MKPE mm↓):

| Train Set | Test Set | MKPE (mm) |
|-----------|----------|-----------|
| UmeTrack | SHOW3D | 22.2 (+55%) |
| HOT3D | SHOW3D | 19.6 (+37%) |
| UmeTrack+HOT3D | SHOW3D | 16.4 (+15%) |
| SHOW3D | SHOW3D | 15.5 (+8%) |
| All three | SHOW3D | **14.3** |
| HOT3D | HOT3D | 14.0 (+14%) |
| All three | HOT3D | **12.3** |

### Ablation Study

Cross-dataset generalization for interaction field estimation (ADE mm↓):

| Train Set | Test Set | ADE (mm) | ACC (m/s²) |
|-----------|----------|----------|------------|
| SHOW3D | HOT3D | 14.70 | 4.05 |
| HOT3D | HOT3D | 11.29 | 3.21 |
| HOT3D+SHOW3D | HOT3D | **8.80** | **2.16** |
| HOT3D | SHOW3D | 22.57 | 5.61 |
| SHOW3D | SHOW3D | 13.82 | 3.79 |

Text-driven 6-DoF object trajectory prediction (mean translation error mm↓):

| Prediction Horizon | w/o Text | w/ Text | Gain |
|--------------------|----------|---------|------|
| 30 frames | 42.7 | 30.4 | −29% |
| 60 frames | 46.7 | 35.0 | −25% |

### Key Findings

- **Asymmetric Generalization**: Models trained on SHOW3D and evaluated on HOT3D achieve only 14.70 mm ADE, whereas the reverse (HOT3D→SHOW3D) yields 22.57 mm (+54%), confirming that in-the-wild data covers a substantially broader distribution.
- **Asymmetric Joint-Training Gains**: Adding SHOW3D to training improves HOT3D evaluation by 22% (11.29→8.80), whereas adding HOT3D improves SHOW3D evaluation by only 2% (13.82→13.50), suggesting SHOW3D already subsumes the studio distribution.
- Text conditioning yields the largest trajectory prediction improvement for the mustard object (72%) and 34% for the mug, demonstrating the genuine utility of semantic context in disambiguating similar trajectories.
- UMAP visualizations show that SHOW3D spans across the compact clusters formed by GigaHands, HOT3D, and ARCTIC in feature space.

## Highlights & Insights

- **Equal Emphasis on Engineering and Scientific Validation**: Beyond presenting a capture system, the paper extensively quantifies annotation accuracy — both hand and object annotations are compared against MoCap ground truth to sub-centimeter precision, which is exceptionally rare for in-the-wild dataset papers.
- **A Practical Solution that Breaks the Trade-off**: The combination of an 8 kg backpack and Meta Quest 3 makes genuinely outdoor capture operationally feasible (gardens, corridors, restaurants, outdoor seating areas, etc.) while maintaining 10-camera synchronization at 60 Hz.
- **Innovative Value of Text Annotations**: Diverse semantic descriptions are generated from manipulation instructions via an LLM. Text-conditioned trajectory prediction experiments confirm the practical utility of these annotations in downstream tasks, rather than merely enriching dataset metadata.

## Limitations & Future Work

- Only 21 everyday objects are included, which is limited compared to GigaHands' 417 object categories.
- High-end computing workstations (transported on a mobile cart following the user) are still required, making deployment costly.
- Grayscale images lack color information, which may be detrimental for appearance-dependent tasks such as object recognition.
- Personalized hand models require high-resolution hand scans, constraining large-scale participant recruitment.
- Future work may integrate tactile sensing and depth cameras to extend the data modalities.

## Related Work & Insights

- **vs. GigaHands**: 51 RGB cameras, studio setup, 3.7M frames, 417 objects. SHOW3D far surpasses GigaHands in environmental diversity but offers fewer objects and lacks RGB imagery.
- **vs. HOT3D**: Both use Meta Quest 3, but HOT3D relies on MoCap with markers, restricting collection to studio settings. SHOW3D achieves in-the-wild capture via a marker-free pipeline.
- **vs. Ego-Exo4D**: In-the-wild capture at scale, but with only sparse hand annotations and no object annotations. SHOW3D demonstrates that dense 3D annotation in the wild is achievable.

## Rating

- Novelty: ⭐⭐⭐⭐ — First in-the-wild 3D hand-object interaction dataset with a practically strong system design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three downstream task evaluations, quantitative annotation accuracy assessment, and cross-dataset generalization analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation, system design, annotation pipeline, and experiments are presented with clarity; an exemplary dataset paper.
- Value: ⭐⭐⭐⭐⭐ — Direct and significant contribution to the fields of egocentric vision and hand-object interaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EAG3R: Event-Augmented 3D Geometry Estimation for Dynamic and Extreme-Lighting Scenes](../../NeurIPS2025/video_understanding/eag3r_event-augmented_3d_geometry_estimation_for_dynamic_and_extreme-lighting_sc.md)
- [\[CVPR 2026\] Temporally Consistent Long-Term Memory for 3D Single Object Tracking](chronotrack_temporally_consistent_long_term_memory_for_3d_single_object_tracking.md)
- [\[ICML 2026\] AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](../../ICML2026/video_understanding/avtrack_audio-visual_tracking_in_human-centric_complex_scenes.md)
- [\[AAAI 2026\] Lifelong Domain Adaptive 3D Human Pose Estimation](../../AAAI2026/video_understanding/lifelong_domain_adaptive_3d_human_pose_estimation.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](../../ACL2026/video_understanding/gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)

</div>

<!-- RELATED:END -->
