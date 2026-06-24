---
title: >-
  [Paper Note] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics
description: >-
  [CVPR 2025][Multimodal VLM][Spatial Reasoning] RoboSpatial constructs a large-scale robotic spatial understanding dataset featuring 1M images, 5k 3D scans, and 3M spatial relation annotations. It leverages an automated pipeline to generate three categories of spatial QA pairs (spatial context, compatibility, and configuration) from existing 3D scene data and introduces three reference frames (ego-centric, world-centric, and object-centric). Training multiple 2D and 3D VLMs on…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Spatial Reasoning"
  - "Robotic Manipulation"
  - "VLM Fine-Tuning"
  - "3D Spatial Understanding"
  - "Reference Frame"
date: 2026-05-08
content_hash: 7cb18658cf446308
---

# RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics

**Conference**: CVPR 2025  
**arXiv**: [2411.16537](https://arxiv.org/abs/2411.16537)  
**Code**: [https://github.com/chanh-ee/RoboSpatial](https://github.com/chanh-ee/RoboSpatial)  
**Area**: Multimodal VLMs  
**Keywords**: Spatial Reasoning, Robotic Manipulation, VLM Fine-Tuning, 3D Spatial Understanding, Reference Frame

## TL;DR
RoboSpatial constructs a large-scale robotic spatial understanding dataset featuring 1M images, 5k 3D scans, and 3M spatial relation annotations. It leverages an automated pipeline to generate three categories of spatial QA pairs (spatial context, compatibility, and configuration) from existing 3D scene data and introduces three reference frames (ego-centric, world-centric, and object-centric). Training multiple 2D and 3D VLMs on this dataset significantly boosts spatial reasoning performance, with its effectiveness validated through real-world robotic manipulation experiments.

## Background & Motivation
While VLMs are increasingly applied in robotics, their spatial understanding capability remains a severe bottleneck—existing VLMs can describe "a bowl on the table" but fail to reason about complex spatial relationships such as the exact position of the bowl on the table or whether a new object can fit. **Key Challenge**: VLMs are typically trained on general image datasets and lack fine-grained spatial annotations in robotic contexts, particularly the capability to understand various reference frames (ego-centric, world-centric, or object-centric). Existing spatial reasoning datasets (such as SpatialVLM and BLINK) are either small-scale, lack multi-frame support, or are not well-suited for embodied scenarios. **Key Insight**: The bottleneck of spatial understanding is the lack of appropriate training data. Thus, this work constructs a large-scale, multi-frame, robot-oriented spatial QA dataset to bridge this gap.

## Method

### Overall Architecture
The core of RoboSpatial is an automated data generation pipeline: taking scene datasets annotated with 3D bounding boxes, camera poses, and semantic labels (e.g., ScanNet, Matterport3D, GraspNet-1B) as input, and outputting a spatial reasoning QA dataset containing $(I_i, q_i, a_i, l_i)$ (images, questions, answers, reference frame labels). The pipeline operates in two stages: first extracting spatial relations in 3D physical space, and then mapping them to 2D image coordinates to generate QA pairs.

### Key Designs
1. **Three Spatial Relation Categories**:
    - **Function**: Cover the core spatial reasoning requirements in robotic scenes.
    - **Mechanism**: Decompose spatial understanding into three levels— (1) Spatial Context: Identify open spaces in the environment and output 2D pixels of placeable positions; (2) Spatial Compatibility: Determine whether a target object can be placed into a specified region by simulating virtual bounding box placements and detecting collisions, outputting True/False; (3) Spatial Configuration: Determine the relative spatial relationship between two objects (left/right/above/below/front/behind), outputting True/False.
    - **Design Motivation**: Distance metrics are difficult to normalize across differing scenes, whereas these three relationships directly correspond to the robot's primary needs for path planning, object placement, and navigation.

2. **Three Reference Frames**:
    - **Function**: Enable the model to understand the variation of the same spatial relationship under different perspectives.
    - **Mechanism**: Each QA pair is generated from three distinct coordinates— (a) Ego-centric (centered on camera pose), (b) World-centric (global coordinate system), (c) Object-centric (centered on the anchor object’s orientation, e.g., "in front of the car" refers to the car's heading direction).
    - **Design Motivation**: Spatial descriptions in natural language implicitly contain a reference frame; "in front of the table" can be completely different depending on the viewpoint, and the model must learn to distinguish these differences.

3. **Two-Stage Data Generation Pipeline**:
    - **Function**: Automatically generate large-scale QA pairs from 3D annotated scenes.
    - **Mechanism**: Stage 1 calculates 3D physical relationships $r_i \in \{left, right, above, below, front, behind\}$ between objects in 3D space based on the position and orientation of oriented bounding boxes. Stage 2 samples clear points in 2D image space using top-down occupancy grids, filters occluded points via ray-casting, and determines compatibility using virtual collision detection (requiring at least a 10cm clearance along each axis).
    - **Design Motivation**: Leverage precise 3D geometry to prevent noisy annotations from perception models, while bridging 2D/3D modalities through camera projections.

### Loss & Training
Fine-tuning is conducted on existing VLMs (such as VILA-1.5-8B and LLaVA-NeXT-8B) using the RoboSpatial dataset jointly trained with an auxiliary object localization dataset (mapping object descriptions to 2D bounding boxes). The auxiliary dataset helps mitigate cascading failures caused by object reference parsing errors.

## Key Experimental Results

### Main Results (RoboSpatial-Val)

| Model | Indoor Avg | Tabletop Avg | Overall Avg | Gain |
|------|-----------|-------------|-------|------|
| VILA (baseline) | 43.1 | 37.4 | 40.2 | - |
| VILA + RoboSpatial | 64.8 | 62.9 | 63.9 | +23.7 |
| LLaVA-NeXT (baseline) | 31.4 | 29.2 | 30.3 | - |
| LLaVA-NeXT + RoboSpatial | 60.4 | 60.5 | 60.5 | +30.2 |
| LEO (3D, baseline) | 41.9 | 43.7 | 42.8 | - |
| LEO + RoboSpatial | 73.1 | 70.7 | 71.9 | +29.1 |
| GPT-4o (zero-shot) | 49.3 | 52.3 | 50.8 | - |

### Out-of-Distribution Generalization (RoboSpatial-Home / BLINK / SpatialBench)

| Model | Home Config | Home Compat | BLINK Acc | SpatialBench |
|------|------------|------------|-----------|-------------|
| LLaVA-NeXT | 68.3 | 70.5 | 71.3 | 55.9 |
| LLaVA-NeXT + RoboSpatial | 78.9 | 80.1 | 79.0 | 70.6 |
| SpaceLLaVA + RoboSpatial | 71.6 | 72.4 | 81.8 | 67.7 |
| GPT-4o | 77.2 | 58.1 | 76.2 | 70.6 |

### Robot Experiments

| Model | Success Rate (%) |
|------|-----------|
| LLaVA-NeXT | 23.7 |
| LLaVA-NeXT + RoboSpatial | **52.6** |
| RoboPoint | 44.7 |
| GPT-4o | 46.9 |

### Key Findings
- After fine-tuning on RoboSpatial, all 2D and 3D VLMs achieve massive performance improvements across all tasks ($\uparrow 20\text{-}30\%$).
- Models generalized well to unseen spatial prepositions during training (such as "next to" and "under") because the training dataset covered the six principal axes in 3D space.
- 3D VLMs (e.g., LEO) generally outperform 2D VLMs, though a fair comparison is limited by differences in pre-training data.
- Cross-environment transfer shows positive synergistic effects: training on indoor data also improves tabletop performance.

## Highlights & Insights
- **Data-driven spatial understanding**: Demonstrates that the bottleneck of spatial reasoning lies in the data rather than model architectures; general VLMs with high-quality spatial data can yield significant gains.
- **Introduction of reference frames** is a key innovation, enabling models to learn the difference between "in front of the car" and "in front of me".
- The automated pipeline can scale to new environments and spatial relationships, demonstrating solid extensibility.
- Fine-tuned LLaVA-NeXT surpasses GPT-4o in real-world robotic experiments, demonstrating the value of in-domain data.

## Limitations & Future Work
- The evaluation of spatial context tasks using convex hulls for correctness is overly strict, leading to lower-than-expected scores.
- 2D-to-3D projection errors (2 pixels $\rightarrow$ 5-10cm) remain a critical bottleneck in robotic manipulation.
- 3D VLMs currently require complete 3D scans as inputs, which are difficult to acquire in real-time in real-world scenes.
- Templating QA may restrict linguistic diversity; future iterations could incorporate LLM rephrasing.

## Related Work & Insights
- **vs SpatialVLM/SpatialRGPT**: They are based on web-crawled images and noisy perception model annotations, limiting generalization to embodied scenarios; RoboSpatial is built on real 3D scans and achieves higher precision.
- **vs RoboPoint/Molmo**: These pointing models lack understanding of reference frames and object compatibility, and they can only predict points rather than answering spatial relationships.
- **vs EmbSpatial-Bench**: Significantly smaller scale (4k QA vs 3M QA) and lacks support for multiple reference frames.
- **vs 3D-LLM/LEO**: 3D VLMs can directly utilize depth information but require full 3D scans as input; RoboSpatial supports both 2D and 3D modalities.
- **vs BLINK-Spatial**: BLINK only includes a small evaluation set of 286 samples, whereas RoboSpatial is a complete training-plus-evaluation solution.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The definition of reference frames and three spatial relation categories is insightful, but the overall methodology leans heavily toward data engineering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple VLMs, multiple benchmarks, out-of-distribution generalization, cross-environment transfer, and real-world robotic experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed pipeline descriptions.
- **Value**: ⭐⭐⭐⭐ Provides a direct impetus for VLM spatial reasoning in robotics, and the open-sourced dataset is highly valuable.

## Additional Notes
- Dataset Scale: 3M QA pairs derived from 5 source datasets (ScanNet, Matterport3D, 3RScan, HOPE, GraspNet-1B).
- The convex hull evaluation criteria are relatively strict; actual accuracy rates might be higher than reported.
- Cross-environment experiments show a positive transfer effect between indoor and tabletop setups, showing better performance when evaluated under joint training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](../../CVPR2026/multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] Abstract 3D Perception for Spatial Intelligence in Vision-Language Models](../../CVPR2026/multimodal_vlm/abstract_3d_perception_for_spatial_intelligence_in_vision-language_models.md)
- [\[CVPR 2025\] LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models](layoutvlm_differentiable_optimization_of_3d_layout_via_vision-language_models.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](../../NeurIPS2025/multimodal_vlm/sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)

</div>

<!-- RELATED:END -->
