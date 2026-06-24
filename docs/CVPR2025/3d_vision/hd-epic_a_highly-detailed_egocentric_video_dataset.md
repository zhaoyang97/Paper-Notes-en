---
title: >-
  [Paper Note] HD-EPIC: A Highly-Detailed Egocentric Video Dataset
description: >-
  [CVPR 2025][3D Vision][Egocentric video dataset] HD-EPIC provides 41 hours of unscripted egocentric kitchen videos with unprecedented annotation density (263 annotations per minute), covering recipe steps, fine-grained actions, nutritional information, 3D digital twins, object motion trajectories, and gaze directions. It also builds a VQA benchmark of 26K questions, on which the strongest Gemini Pro achieves only 37.6%.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Egocentric video dataset"
  - "video question answering"
  - "digital twin"
  - "fine-grained annotation"
  - "kitchen activity understanding"
date: 2026-05-08
content_hash: c41bf77cebc06b3d
---

# HD-EPIC: A Highly-Detailed Egocentric Video Dataset

**Conference**: CVPR 2025  
**arXiv**: [2502.04144](https://arxiv.org/abs/2502.04144)  
**Code**: [https://hd-epic.github.io](https://hd-epic.github.io)  
**Area**: 3D Vision  
**Keywords**: Egocentric video dataset, video question answering, digital twin, fine-grained annotation, kitchen activity understanding

## TL;DR
HD-EPIC provides 41 hours of unscripted egocentric kitchen videos with unprecedented annotation density (263 annotations per minute), covering recipe steps, fine-grained actions, nutritional information, 3D digital twins, object motion trajectories, and gaze directions. It also builds a VQA benchmark of 26K questions, on which the strongest Gemini Pro achieves only 37.6%.

## Background & Motivation

1. **Background**: Egocentric video understanding has advanced rapidly in recent years due to the development of large-scale datasets (such as Ego4D) and foundation models. However, while existing large-scale datasets are highly effective for training, their annotations are sparse. Conversely, densely annotated datasets typically originate from controlled laboratory environments and lack realism.

2. **Limitations of Prior Work**: There are two key gaps. First, there is a lack of datasets that cross-link various modality annotations such as actions, objects, 3D scenes, and gaze. Second, there is a lack of zero-shot benchmarks capable of comprehensively evaluating the diverse capabilities of video-language models.

3. **Key Challenge**: Real-world data is difficult to annotate with high precision (due to high costs and highly variable environments), whereas fine-grained annotations in controlled environments lack representativeness.

4. **Goal**: How to collect videos in real domestic environments while achieving laboratory-grade annotation density?

5. **Key Insight**: Recording consecutive kitchen activities over 3 days in participants' own homes using Project Aria glasses (multi-sensors: RGB, SLAM cameras, 7 microphones, and gaze tracking), then achieving ultra-dense annotations through a meticulously designed multi-layer annotation pipeline.

6. **Core Idea**: By leveraging multi-sensor recording devices and a systematic multi-layer annotation pipeline, this work builds the first egocentric video dataset in unscripted, real-world environments with laboratory-grade annotation density, and creates a multi-dimensional VQA benchmark based on it.

## Method

### Overall Architecture
Data collection $\rightarrow$ multi-layer annotation $\rightarrow$ benchmark construction. 9 domestic kitchens, 156 videos, 41.3 hours. Annotations are divided into six major layers: recipe steps & nutrition, fine-grained actions, audio events, digital twins (scene + object 3D), object motion trajectories (including 2D/3D), and gaze-object associations. Based on all annotations, 26,650 VQA questions across 30 question templates are automatically generated.

### Key Designs

1. **Recipe Step & Nutrition Annotation System**:

    - **Function**: Associating cooking activities in the videos with structured recipes, ingredient information, and nutritional values.
    - **Mechanism**: After recording three days of continuous kitchen activities, participants provide the recipes they cooked along with their sources. A "prep-step" pairing annotation is introduced: each cooking step is paired with its corresponding preparation phase (e.g., the preparation for "chopping tomatoes" includes retrieving the tomato, washing it, and grabbing the knife and cutting board). Participants record the weight of each ingredient using a scale, and annotate nutritional information via MyFitnessPal. The timing of ingredient addition is annotated to track nutritional changes throughout the dish preparation. The dataset contains 69 recipes and 558 ingredients in total.
    - **Design Motivation**: Preparation and execution are interleaved in real-world cooking (93.1% of cooking steps have a paired preparation step). Such fine-grained annotations are unprecedented and can validate model understanding of long-horizon, multi-step activities.

2. **3D Digital Twin & Object Motion Tracking**:

    - **Function**: Anchoring all annotations in 3D space.
    - **Mechanism**: Based on the multi-day SLAM point clouds from Aria MPS, complete digital twins of each kitchen are manually reconstructed in Blender, with an average of 45.9 annotated fixtures (such as cabinets, drawers, countertops, appliances, etc.). Object motion is annotated with 2D bounding boxes, and object masks are initialized using SAM2 followed by manual refinement (with a 74% correction rate). These masks are lifted to 3D using dense depth estimation and sparse 2D-3D correspondences. Object locations are linked to the closest fixtures to provide semantic localization (e.g., "which cabinet or countertop the object is in/on"). The average travel distance of objects is 61.4 cm.
    - **Design Motivation**: 3D anchoring enables questions that require spatial reasoning (e.g., "which cabinet was the object moved from, and to which countertop did it go?"), which pure 2D annotations cannot support. The combination of digital twins and object trajectories enables precise object trajectory tracking.

3. **Gaze-Driven Object Interaction Pre-Annotation (Gaze Priming)**:

    - **Function**: Utilizing gaze data to establish anticipatory annotations for object interaction.
    - **Mechanism**: By combining eye-tracking data and 3D object positions, "priming" is defined as the moment when the user's gaze focuses on an object's location prior to an action. For pickup actions, the system measures when the gaze begins focusing on the object (pre-pick priming). For placing actions, it measures when the gaze focuses on the target location (pre-place priming). Analysis shows that 94.8% of interactable objects are primed an average of 4.0 seconds before pickup, and 88.5% are primed an average of 2.6 seconds before placement.
    - **Design Motivation**: Gaze priming is a well-known phenomenon in cognitive science (where humans look at a target about 1 second before interaction). This work is the first to systematically annotate it, providing robust training and evaluation data for gaze-based action prediction.

### Loss & Training
HD-EPIC is a dataset paper and does not involve training losses. The VQA benchmark is formatted as a 5-way multiple-choice task. Negative distractors are sampled from similar annotations within the dataset (instead of being generated by LLMs) to ensure difficulty.

## Key Experimental Results

### Main Results

| Model | Recipe | Ingredient | Nutrition | Action | 3D | Motion | Gaze | Average |
|------|--------|------------|-----------|--------|----|--------|------|------|
| Llama 3.2 (Text-only) | 33.5 | 25.0 | 36.7 | 23.3 | 22.3 | 25.5 | 19.5 | 26.5 |
| VideoLlama 2 | 30.8 | 25.7 | 32.7 | 27.2 | 25.7 | 28.5 | 21.2 | 27.4 |
| LLaVA-Video | 36.3 | 33.5 | 38.7 | 43.0 | 27.3 | 18.9 | 29.3 | 32.4 |
| **Gemini Pro** | **60.5** | **46.2** | 34.7 | 39.6 | 32.5 | 20.8 | 28.7 | **37.6** |
| Human | 96.7 | 96.7 | 85.0 | 92.5 | 93.8 | 92.7 | 75.0 | **90.3** |

### Action Recognition Benchmark

| Model | Verb Acc | Noun Acc | Action Acc | Description |
|------|----------|----------|------------|------|
| SlowFast | 29.2 | 10.6 | 5.3 | Poor performance of traditional methods |
| VideoMAE-L | 47.5 | 29.4 | 17.9 | Moderate performance |
| TIM (A+V) | **51.3** | **36.1** | **23.4** | Best, but still substantial room for improvement |
| EPIC-100 TIM Comparison | 77.1 | 67.2 | 57.5 | Much higher on seen scenes |

### Key Findings
- Gemini Pro is the only model that significantly outperforms random guessing (37.6% vs 20% random), but a massive gap remains compared to human performance (90.3%), highlighting the extreme challenge of the VQA benchmark.
- Pure language models (Llama 3.2) perform comparably to video VLMs, indicating that many questions require visual reasoning as opposed to text prior knowledge.
- Object motion questions yield the poorest performance across all models ($\leq 28.5\%$), suggesting that long-horizon multi-hop object tracking is a major blind spot for modern models.
- Action recognition drops drastically when migrating from EPIC-100 to HD-EPIC (action accuracy from 57.5% down to 23.4%), revealing a lack of cross-scene generalization capability.
- Gemini performs best on recipe and ingredient questions (60.5% / 46.2%), likely benefiting from its external knowledge.

## Highlights & Insights
- **Unprecedented annotation density**: 263 annotations per minute, far exceeding any existing unscripted dataset. It approaches the density of synthetic datasets while preserving real-world complexity.
- **Cross-linked multi-layer annotations**: Annotations from different layers can be jointly utilized (e.g., gaze $\rightarrow$ object $\rightarrow$ 3D position $\rightarrow$ fixture $\rightarrow$ recipe step), enabling the construction of complex queries that require multi-hop reasoning.
- **Thoughtful dataset design**: Newly collected data ensures it has not been pre-trained on by current models. Distractors are sampled from real annotations rather than LLM generation, preventing trivial distribution-out negative options. Food weighing and nutrition tracking introduce an entirely new dimension of evaluation.

## Limitations & Future Work
- The scale of 9 kitchens is relatively small, which limits environmental diversity.
- Recipes are biased toward the participants' domestic cooking habits, which might lack broader diversity.
- Obtaining object masks using SAM2 automated segmentation with manual correction required a high correction rate of 74%, indicating that automated methods still fall short.
- The VQA benchmark currently only supports 5-way multiple choice; extending it to open-ended questions could further increase the challenge.
- The manual reconstruction of 3D digital twins is not easily scalable, which limits the scale-up of the dataset.

## Related Work & Insights
- **vs EPIC-KITCHENS-100**: Within the same family of datasets, HD-EPIC delivers orders of magnitude improvements in annotation density, and introduces new dimensions such as recipes, nutrition, digital twins, and gaze.
- **vs Ego4D**: While Ego4D is vastly larger, its annotations are sparse. HD-EPIC fills the evaluation benchmark gap with high-precision, small-scale annotation.
- **vs Aria Digital Twin**: ADT provides digital twins in controlled environments but only consists of 8.1 hours with limited annotations. HD-EPIC achieves real-world environments coupled with fine-grained annotation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Multiple firsts: fine-grained annotations in unscripted environments, nutrition tracking, and gaze priming annotations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation spanning VQA, action recognition, audio event detection, and object segmentation.
- Writing Quality: ⭐⭐⭐⭐⭐ Detailed description of the annotation pipeline, rich data statistics.
- Value: ⭐⭐⭐⭐⭐ Highly valuable as a zero-shot evaluation benchmark, exposing critical shortcomings of state-of-the-art models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EgoPressure: A Dataset for Hand Pressure and Pose Estimation in Egocentric Vision](egopressure_a_dataset_for_hand_pressure_and_pose_estimation_in_egocentric_vision.md)
- [\[NeurIPS 2025\] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants](../../NeurIPS2025/3d_vision/indego_a_dataset_of_industrial_scenarios_and_collaborative_work_for_egocentric_a.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](../../CVPR2026/3d_vision/ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[CVPR 2025\] Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset](digital_twin_catalog_a_large-scale_photorealistic_3d_object_digital_twin_dataset.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)

</div>

<!-- RELATED:END -->
