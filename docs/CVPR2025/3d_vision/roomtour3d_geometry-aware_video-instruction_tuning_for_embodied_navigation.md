---
title: >-
  [Paper Note] RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation
description: >-
  [CVPR 2025][3D Vision][Vision-Language Navigation] RoomTour3D leverages online house tour videos to construct a geometry-aware video-instruction dataset. By obtaining geometry information of walking trajectories via 3D reconstruction and combining it with GPT-4 to generate open-vocabulary instructions, it significantly boots performance on multiple VLN benchmarks and supports zero-shot navigation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Vision-Language Navigation"
  - "Video-Instruction Tuning"
  - "Geometry-Aware"
  - "Indoor Navigation"
  - "Dataset"
date: 2026-05-08
content_hash: ed3f2b7e19fe2ca0
---

# RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation

**Conference**: CVPR 2025  
**arXiv**: [2412.08591](https://arxiv.org/abs/2412.08591)  
**Code**: [Project Page](https://roomtour3d.github.io)  
**Area**: 3D Vision / Embodied Navigation  
**Keywords**: Vision-Language Navigation, Video-Instruction Tuning, Geometry-Aware, Indoor Navigation, Dataset

## TL;DR

RoomTour3D leverages online house tour videos to construct a geometry-aware video-instruction dataset. By obtaining geometry information of walking trajectories via 3D reconstruction and combining it with GPT-4 to generate open-vocabulary instructions, it significantly boots performance on multiple VLN benchmarks and supports zero-shot navigation.

## Background & Motivation

The Vision-Language Navigation (VLN) task has long been limited by the diversity and scale of training data, relying primarily on human-curated simulator environments. Existing datasets, such as R2R and CVDN, are constructed in simulated environments, which lack scene diversity and cannot capture real-world complexity.

Prior attempts to utilize web data have their own limitations: AirBERT uses discrete Airbnb images and lacks continuity in indoor scenes; ScaleVLN relies on human-curated 3D scenes, which are costly and have poor scalability; YTB-VLN uses YouTube videos but ignores path geometry information and object diversity. No existing method simultaneously achieves the three critical attributes: scene diversity, object openness, and spatial geometry awareness.

The core mechanism of RoomTour3D is to leverage readily available house tour videos on the internet. These videos capture continuous motion from a first-person perspective, naturally embedding spatial geometric attributes. An automated pipeline is developed to extract geometry-aware navigation trajectories and spatially contextualized textual instructions.

## Method

### Overall Architecture

RoomTour3D consists of an automated data generation pipeline and two training tasks. Starting with house tour videos, the pipeline uses COLMAP for 3D reconstruction to acquire geometric information, combines RAM/Grounding-DINO/Depth-Anything to extract object labels/localization/depth, and finally uses GPT-4 to generate two types of trajectories: approx. 100K description-enhanced trajectories (for pre-training) and approx. 17K action-enhanced trajectories (for navigation fine-tuning).

### Key Designs

#### 1. Description-Enhanced Trajectory Generation

- **Function**: Provides rich physically/spatially-aware descriptions for model pre-training, enhancing the model's spatial understanding and object cognitive capabilities.
- **Mechanism**: Uniformly samples frames at a rate of 1 frame per 2 seconds to generate walking trajectories. For each frame, RAM is used for object categorization, Grounding-DINO for localization, and Depth-Anything for depth prediction, which are combined into a textual template: "There is a [object] to the [position] of current spot in [distance]". BLIP-2 is used to predict room types (from 16 predefined room categories), followed by temporal smoothing for denoising. Finally, frame-level descriptions are integrated into GPT-4 to generate controllable and open-vocabulary navigation instructions in a "Task instruction - In-context examples - Prediction" format.
- **Design Motivation**: The object diversity, spatial positions, and depth information provided by multi-source expert models enable GPT-4 to generate more accurate free-form descriptions than templated instructions.

#### 2. Action-Enhanced Trajectories and 3D Reconstruction

- **Function**: Provides trajectory data with navigation decision points and candidate actions to support navigation fine-tuning.
- **Mechanism**: Uses COLMAP for 3D scene reconstruction to obtain camera poses. "Decision frames" are sampled at the points of maximum yaw rotation, and sequential frames are sampled every ~1.5 meters to form trajectories. Utilizing 3D reconstruction to measure camera orientation differences and distances, DBSCAN is applied to cluster spatially adjacent frames. For each path, the nearest frame is selected as the positive sample, and the frame with the maximum angular difference is chosen as the negative sample. Videos are segmented into 100-second clips for parallel reconstruction, and sub-models are merged using depth-first search.
- **Design Motivation**: Unlike panoramic nodes, action-enhanced trajectories provide candidate viewpoints with different positions and orientations, closely mimicking the decision-making process in real-world navigation scenarios.

#### 3. NaviLLM-Based Dual-Task Training

- **Function**: Integrates the RoomTour3D dataset into general navigation models to achieve multi-task enhancement.
- **Mechanism**: 
  - (a) **Pre-training Stage**: Spatially-aware description-enhanced trajectories are used for visual-instruction summarization tasks, where frames are wrapped with `<cand>` tokens as candidate observations, and the model outputs trajectory summaries containing object progress and room locations.
  - (b) **Fine-tuning Stage**: Action-enhanced trajectories are used for action-instruction navigation tasks, where each frame is treated as a navigable step. The model selects the next action based on historical observations `<hist>`, and the final step requires summarizing the entire navigation path.
- **Design Motivation**: The two-stage strategy separately enhances spatial understanding (pre-training) and decision-making capabilities (fine-tuning), seamlessly interfacing with NaviLLM’s training paradigm.

### Loss & Training

Utilizes the standard next-token prediction loss, consistent with language model training methodologies.

## Key Experimental Results

### Main Results: Multi-Task VLN Performance Gains (SPL/GP Metrics)

| Method | CVDN(GP) | SOON(SPL) | R2R(SPL) | REVERIE(SPL) |
|------|---------|----------|---------|-------------|
| NaviLLM (Original) | 6.16 | 29.2 | 59 | 35.7 |
| NaviLLM (Reimplemented) | 6.09 | 28.0 | 56.7 | 31.4 |
| +RT3D Descr | Gain | Gain | Gain | Gain |
| **+RT3D Full** | **>6% Gain** | **9.8% Gain (SOTA)** | **Gain** | **Gain** |

### Key Data Scales

| Data Type | Quantity |
|---------|------|
| Description-Enhanced Trajectories | ~100K |
| Navigation Instructions | ~200K |
| Action-Enhanced Trajectories | ~17K |
| Number of House Environments | 1,847 |

### Key Findings

- Integrating RoomTour3D data improves baseline performance across all VLN tasks by more than 6% simultaneously.
- Achieves a 9.8% gain on the SOON task, setting a new SOTA.
- Zero-shot navigation performance exceeds all non-commercial methods and is comparable to commercial GPT-3.5-based methods.
- BLIP-2 room classification accuracy reaches 85% (verified by human annotation).

## Highlights & Insights

1. **Data-Driven Paradigm Shift**: Instead of designing better model architectures, this work utilizes the scale and diversity of online videos to address the data bottleneck in VLN.
2. **Scalability of Automated Pipeline**: The entire data generation process is automated, enabling continuous expansion with more house tour videos.
3. **Bridge from Video to Navigation**: 3D reconstruction translates continuous video frames into geometry-aware navigation trajectories.

## Limitations & Future Work

- COLMAP reconstruction quality is inconsistent, causing some video segments to fail to merge.
- The current dataset scale is still undergoing continuous expansion.
- Limiting room categories to 16 classes may not cover all scenarios.
- Future research can explore larger-scale video data and more diverse types of navigation tasks.

## Related Work & Insights

- **YTB-VLN**: Uses YouTube videos but ignores path geometry, utilizing templated instructions.
- **ScaleVLN**: Uses human-curated 3D scenes but incurs high costs.
- **NaviLLM**: The current SOTA LLM-based navigation model, whose performance is directly enhanced by RoomTour3D.
- **Insight**: The combination of web videos, 3D reconstruction, and LLMs can economically provide large-scale training data for embodied AI.

## Rating

⭐⭐⭐⭐ — Solid dataset construction methodology, well-designed automated pipeline, bringing consistent and significant improvements across multiple VLN benchmarks. The scalability of the dataset and its potential for open-world navigation are particularly noteworthy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness](../../ICCV2025/3d_vision/ross3d_reconstructive_visual_instruction_tuning_with_3d-awareness.md)
- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[ICLR 2026\] Towards Physically Executable 3D Gaussian for Embodied Navigation](../../ICLR2026/3d_vision/towards_physically_executable_3d_gaussian_for_embodied_navigation.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[CVPR 2025\] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video](4dgc_rate-aware_4d_gaussian_compression_for_efficient_streamable_free-viewpoint_.md)

</div>

<!-- RELATED:END -->
