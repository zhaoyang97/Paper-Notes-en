---
title: >-
  [Paper Note] Lifting Unlabeled Internet-level Data for 3D Scene Understanding
description: >-
  [CVPR 2026][3D Vision][3D scene understanding] This paper presents SceneVerse++, an automated data engine that generates 3D scene understanding training data from 6…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D scene understanding"
  - "internet videos"
  - "automated data engine"
  - "vision-language navigation"
  - "spatial reasoning"
date: 2026-05-08
content_hash: 0a6254254dbdff0d
---

# Lifting Unlabeled Internet-level Data for 3D Scene Understanding

**Conference**: CVPR 2026
**arXiv**: [2604.01907](https://arxiv.org/abs/2604.01907)
**Code**: [Project Page](https://sv-pp.github.io/)
**Area**: 3D Vision
**Keywords**: 3D scene understanding, internet videos, automated data engine, vision-language navigation, spatial reasoning

## TL;DR

This paper presents SceneVerse++, an automated data engine that generates 3D scene understanding training data from 6,687 unlabeled internet videos. It demonstrates the feasibility of leveraging internet-scale data to advance 3D scene understanding across three tasks: 3D object detection (F1@.25 +20.6), spatial VQA (+14.9%), and vision-language navigation (+14% SR).

## Background & Motivation

3D scene understanding is a fundamental capability for both human cognition and embodied intelligence, spanning geometric perception (depth estimation, object detection), semantic understanding (segmentation, visual grounding), and high-level reasoning (spatial QA, navigation). Progress in this area via deep learning relies heavily on large-scale annotated real-world 3D datasets.

**Key Challenge**: Unlike 2D images that can be easily collected and annotated from the web, 3D scene data acquisition and annotation is prohibitively expensive — requiring specialized hardware (RGB-D/LiDAR), 3D mesh reconstruction, and dense manual semantic annotation. Since ScanNet, the field has seen virtually no order-of-magnitude growth in 3D data scale, while the internet hosts an abundance of unlabeled video data that naturally captures the 3D world.

**Key Insight**: The paper proposes an automated data engine that converts unlabeled internet videos into training data for 3D scene understanding. Rather than naively chaining existing sub-modules (reconstruction, segmentation, semantic annotation), the paper systematically analyzes the bottlenecks of automated data generation and provides guidelines for scaling end-to-end models across tasks of different perceptual granularity. **Core Idea**: Through a carefully designed data engine, internet videos can serve as a viable path to bridging the scarcity of annotated 3D data and improving end-to-end model capabilities.

## Method

### Overall Architecture

Starting from internet videos, the pipeline consists of three stages: (1) video filtering and structure-from-motion (SfM) to obtain camera poses and sparse 3D geometry; (2) a modular reconstruction and segmentation pipeline to produce dense 3D reconstructions and instance annotations; (3) task-specific data generation for downstream tasks (detection/segmentation, spatial VQA, VLN). The final dataset comprises 6,687 scenes sourced from 8,217 videos, including images, camera poses, dense reconstructions, instance segmentation, and high-level reasoning annotations.

### Key Designs

1. **Video Filtering and SfM Reconstruction Pipeline**:
   - **Function**: Extract high-quality camera poses and sparse 3D point clouds from raw internet videos.
   - **Mechanism**: TransNetV2 shot detection → filtering of low-quality/outdoor/portrait content → disparity-based keyframe selection (rather than uniform sampling) → dense pixel matching + global bundle adjustment → spatial coverage and SfM quality checks.
   - **Design Motivation**: Internet videos contain abundant irrelevant content; disparity-based frame selection ensures triangulation quality. Optimized pseudo-trajectory pixels are introduced to improve memory efficiency for long videos.

2. **Dense Reconstruction and Instance Segmentation Pipeline**:
   - **Function**: Produce complete 3D meshes and instance-level annotations from sparse SfM outputs.
   - **Mechanism**: For reconstruction, sparse SfM points are projected onto image planes to obtain sparse depth priors; PriorDA then predicts dense metric depth, and TSDF fusion generates watertight meshes. For segmentation, CropFormer produces per-frame segmentation masks, which are aggregated into 3D space via inter-frame view consensus and spatial consistency; VLMs then generate text descriptions and semantic labels.
   - **Design Motivation**: Neural rendering methods produce high-quality results but are too slow for per-scene optimization; end-to-end reconstruction methods are fast but suffer from memory constraints and geometric distortions on long videos. The metric depth + SfM approach strikes a balance between quality and efficiency (averaging 71 seconds for reconstruction and 96 seconds for segmentation per scene).

3. **Task-Specific Data Generation**:
   - **Function**: Transform 3D scenes into task-specific training data.
   - **Mechanism**: 3D detection/segmentation directly uses the reconstruction and instance annotations. Spatial VQA generates templated QA pairs (632K) via 3D scene graphs. VLN converts free-exploration room-tour trajectories into R2R-style navigation data through a three-stage pipeline (trajectory preprocessing → action encoding → instruction generation).
   - **Design Motivation**: The core challenge for VLN is bridging the gap between the irregular motion in room-tour videos and the goal-directed shortest paths in the R2R benchmark, necessitating dedicated trajectory refinement and action encoding.

### Loss & Training

- **3D Detection**: SpatialLM (MLLM-based) is pretrained on SceneVerse++ and fine-tuned on ScanNet.
- **3D Segmentation**: Mask3D is pretrained on SceneVerse++ and fine-tuned on ScanNet.
- **Spatial VQA**: Qwen2.5-VL-3B/7B is fine-tuned using LoRA on 202K training samples.
- **VLN**: LLaVA-Video serves as the base model, pretrained on SceneVerse++ and then fine-tuned on R2R.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | Ours (SceneVerse++) | Prev. SOTA | Gain |
|---|---|---|---|---|
| ScanNet 3D Detection | F1@.25 (pretrain+finetune) | 58.6 | 38.0 (SpatialLM orig.) | +20.6 |
| ScanNet 3D Detection | F1@.25 (zero-shot) | 30.9 | 29.0 (SpatialLM) | +1.9 |
| ARKitScenes 3D Detection | F1@.25 (zero-shot) | 35.8 | 35.1 (SpatialLM) | +0.7 |
| ScanNet 3D Segmentation | AP25 (pretrain+finetune) | 38.5 | 36.1 (from scratch) | +2.4 |
| VSI-Bench VQA (3B) | Avg Accuracy | 42.8 (SV++ zero-shot) | 27.9 (baseline) | +14.9 |
| VSI-Bench VQA (7B) | Avg Accuracy | 46.4 (SV++ zero-shot) | 36.6 (baseline) | +9.8 |
| R2R VLN | SR (pretrain+finetune) | 0.228 | 0.088 (R2R only) | +0.14 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Full SceneVerse++ pretrain + R2R finetune | SR 0.228 | Optimal strategy |
| Joint training (R2R + SV++) | SR 0.188 | Direct mixing underperforms pretrain-then-finetune |
| w/o Trajectory Refinement (w/o TR) | SR 0.036 → 0.177 (ft) | Raw trajectories are low-quality; refinement is critical |
| w/o Instruction Enhancement (w/o IE) | SR 0.022 → 0.074 (ft) | Language diversity has a large impact on performance |
| SV++ zero-shot VQA (ARKit subset) | 48.0 (3B) | Approaches annotated SN/SN++ training (49.0) |

### Key Findings

- On 3D detection, the real-world distribution prior from SceneVerse++ pretraining leads to large fine-tuning gains (F1@.25 from 38.0 to 58.6).
- On 3D segmentation, Mask3D's reliance on pipeline-specific graph cut results makes it sensitive to domain transfer; SceneVerse++ zero-shot performance is limited, though fine-tuning still yields improvements.
- On spatial VQA, SceneVerse++ yields the largest improvements on general spatial knowledge (relative distance, relative direction) but performs weaker on domain-specific knowledge (object count, room size), reflecting the domain gap.
- For VLN, trajectory refinement and instruction enhancement are both critical data quality factors; raw internet videos cannot be used directly.
- A clear overfitting inflection point exists: all evaluation metrics improve in early training, after which in-domain metrics continue to rise while out-of-domain metrics plateau or decline.

## Highlights & Insights

- The paper systematically analyzes the full pipeline from internet videos to 3D scene understanding, rather than naively assembling sub-modules.
- Three representative tasks spanning low-level perception (detection/segmentation) to high-level reasoning (VQA/VLN) provide comprehensive validation.
- The dataset scale is substantial: 6,687 scenes surpassing ARKitScenes, with an average of 49 objects and 21 categories per scene.
- The in-depth discussion of model scalability is valuable: models that depend on pre-computed segmentation (Mask3D) are harder to scale than those operating directly on raw modalities (SpatialLM).

## Limitations & Future Work

- The pipeline depends on multiple sub-modules (SfM, depth estimation, segmentation, VLM annotation), and errors from each module cascade through the pipeline.
- Video filtering still requires a small amount of human annotation (<10 seconds/scene) to ensure data quality.
- The 3D segmentation task demonstrates how domain-specific biases can limit model scalability, motivating the need for more robust model architectures.
- Internet videos are predominantly indoor room-tour style, resulting in limited coverage of outdoor or dynamic scenes.
- Sub-modules in the automated data generation pipeline are mostly trained on small-scale, task-specific benchmarks, limiting their generalization capability.

## Related Work & Insights

- **vs. ScanNet/ScanNet++**: High-quality manually collected 3D datasets, but scale is limited (~1.5k scenes for ScanNet); SceneVerse++ acquires 6.7k scenes from the internet via automation at a larger scale, with a trade-off in quality.
- **vs. RoomTour3D/NaVILA**: Also leverage internet videos, but are restricted to the single task of navigation; SceneVerse++ covers detection, segmentation, VQA, and VLN comprehensively.
- **vs. Miao et al.**: Uses 2D single-view datasets with estimated depth to generate 3D annotations, but is constrained to existing 2D datasets and only supports single-frame-level processing.
- **Insight**: Sub-module development should target "supporting robust in-the-wild 3D understanding," evaluating not only task-specific performance but also the contribution to automated data generation pipelines.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The systematic use of internet videos for comprehensive 3D scene understanding is innovative, and the bottleneck analysis is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three tasks, multiple training strategy comparisons, detailed ablations, and training dynamics analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, in-depth discussion, and honest analysis of the data engine's limitations.
- **Value**: ⭐⭐⭐⭐⭐ Provides a systematic roadmap and practical guidelines for scaling data in 3D scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)
- [\[CVPR 2026\] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)
- [\[CVPR 2026\] PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding](pointtpa_dynamic_network_parameter_adaptation_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](../../ICCV2025/3d_vision/towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)

</div>

<!-- RELATED:END -->
