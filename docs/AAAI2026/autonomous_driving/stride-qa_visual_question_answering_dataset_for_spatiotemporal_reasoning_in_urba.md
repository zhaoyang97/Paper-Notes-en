---
title: >-
  [Paper Note] STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes
description: >-
  [AAAI 2026 Oral][Autonomous Driving][VQA] This paper constructs STRIDE-QA, the largest spatiotemporal reasoning VQA dataset in autonomous driving (270K frames, 16M QA pairs), defines three categories of spatiotemporal reasoning tasks (object-centric spatial / ego-centric spatial / ego-centric spatiotemporal), and demonstrates that fine-tuning a VLM raises localization success rate from near zero to 55% and temporal localization consistency from 0 to 28.4%.
tags:
  - "AAAI 2026 Oral"
  - "Autonomous Driving"
  - "VQA"
  - "spatiotemporal reasoning"
  - "ego-centric perspective"
  - "3D annotation"
  - "VLM fine-tuning"
date: 2026-05-08
content_hash: d4c97a0b813b906f
---

# STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes

**Conference**: AAAI 2026 Oral  
**arXiv**: [2508.10427](https://arxiv.org/abs/2508.10427)  
**Code**: [turingmotors/stride-qa](https://turingmotors.github.io/stride-qa/)  
**Area**: Autonomous Driving / Visual Question Answering / Spatiotemporal Reasoning
**Keywords**: VQA, spatiotemporal reasoning, ego-centric perspective, 3D annotation, VLM fine-tuning

## TL;DR
This paper constructs STRIDE-QA, the largest spatiotemporal reasoning VQA dataset in autonomous driving (270K frames, 16M QA pairs), defines three categories of spatiotemporal reasoning tasks (object-centric spatial / ego-centric spatial / ego-centric spatiotemporal), and demonstrates that fine-tuning a VLM raises localization success rate from near zero to 55% and temporal localization consistency from 0 to 28.4%.

## Background & Motivation

**Background**: VLMs (e.g., GPT-4o, Qwen2.5-VL) have been applied to autonomous driving scene understanding and decision support. Existing driving VQA datasets include nuScenes-QA (460K QA pairs) and nuPlanQA (1M QA pairs), among others.

**Limitations of Prior Work**: Existing VLMs are trained on static web image–text pairs and lack precise spatiotemporal reasoning capabilities. Prior driving VQA datasets either support only object-centric perspectives, lack temporally aligned 3D annotations, or are limited in scale.

**Key Challenge**: No existing dataset simultaneously supports all three task types—object-centric spatial reasoning, ego-centric spatial reasoning, and ego-centric spatiotemporal reasoning (including future prediction). Existing methods score near zero on spatiotemporal prediction consistency.

**Key Insight**: The authors leverage 100 hours of driving data collected in Tokyo and construct a large-scale spatiotemporal reasoning VQA dataset of 16M QA pairs via a fully automated annotation pipeline (BEVFusion 3D detection + PubTracker tracking + SAM 2.1 segmentation + visibility filtering).

**Core Idea**: Provide VLMs with large-scale, physically grounded spatiotemporal supervision signals, enabling a capability leap from passive scene description to future motion prediction.

**Sensor Configuration**: 64-channel LiDAR + 6 cameras (front/rear 2880×1860, side 1920×1240, 60° FOV, 360° coverage) + IMU + RTK-GNSS, with 20-second clip segmentation.

## Method

### Overall Architecture
Multi-sensor driving data (6 cameras + 64-beam LiDAR, 2 Hz synchronization) → Fully automated annotation pipeline → QA pair generation → VLM fine-tuning. The annotation pipeline comprises seven modules: keyframe sampling, 3D object detection, multi-object tracking, attribute extraction, semantic segmentation, visibility filtering, and question generation.

Dataset statistics: the training set contains 5.30M qualitative QA pairs + 10.17M quantitative QA pairs; the validation set contains 0.28M qualitative + 0.69M quantitative. The evaluation set consists of 409 scene groups with 5,317 QA pairs covering 6 types of dynamic interaction scenarios.

### Key Design 1: Three VQA Task Categories
- **Object-centric Spatial QA**: Single-frame tasks judging spatial relationships between two non-ego agents (qualitative questions such as relative position; quantitative questions such as "distance 1.53 m").
- **Ego-centric Spatial QA**: Single-frame tasks describing an agent's distance, heading, and size relative to the ego vehicle.
- **Ego-centric Spatiotemporal QA**: Given 4 frames of context sampled at 2 Hz ($t \in \{-1.5, -1.0, -0.5, 0\}$ s), the model predicts the target agent's distance, heading angle, and velocity at future steps $t \in \{1, 2, 3\}$ s.

### Key Design 2: Fully Automated Annotation Pipeline
Seven modular components: (1) 1 Hz keyframe sampling (2–17 s window per clip); (2) BEVFusion LiDAR–camera fusion 3D detection (45 object classes, mAP = 0.701, position error only 13.6 cm); (3) PubTracker point-based 3D tracking (no appearance features required, AMOTA = 0.676); (4) attribute extractor (Euclidean distance, heading angle, and velocity in the ego coordinate frame); (5) SAM 2.1 Large segmentation (per-instance 2D masks); (6) visibility filtering (triple filter: IoU > 0.3, coverage > 0.8, SAM confidence > 0.8); (7) template-based QA generation (Region [X] reference identifiers).

### Key Design 3: Evaluation Metric Framework
- **LSR (Localization Success Rate)**: A prediction is deemed successful when distance error < ±25% of the GT value and heading error < ±10°.
- **MLSR (Mean LSR)**: Sequence-level mean of LSR, measuring temporal stability.
- **TLC (Temporal Localization Consistency)**: A strict consistency measure requiring success at all four time steps.

### Loss & Training
Fine-tuning uses LoRA (rank = 16, alpha = 32, dropout = 0.05), trained for 2 epochs on 16 H100 GPUs. Learning rates are 5e-5 (LLM) / 2e-6 (visual encoder) / 1e-5 (projection head), with a global batch size of 64. The optimizer is AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay = 0.1) with a cosine scheduler and 5% warmup. Inference temperature is set to 0 to ensure deterministic outputs.
Input consists of 4 front-view frames (532×336 resolution) annotated with Region ID masks generated via the Set-of-Mark method.

## Experiments

### Main Results: STRIDE-QA Bench Spatiotemporal Reasoning

| Model | LSR@0s | LSR@1s | LSR@2s | LSR@3s | MLSR | TLC |
|---|---|---|---|---|---|---|
| GPT-4o | 18.1 | 6.6 | 6.1 | 7.6 | 9.6 | 0.7 |
| GPT-4o mini | 4.6 | 2.0 | 0.7 | 0.7 | 2.0 | 0.0 |
| Qwen2.5-VL-7B | 1.0 | 3.4 | 4.4 | 1.0 | 2.4 | 0.0 |
| SpatialRGPT-8B | 0.5 | 0.2 | 0.2 | 0.0 | 0.2 | 0.0 |
| Cosmos-Reason1-7B | 1.5 | 3.2 | 2.0 | 1.5 | 2.0 | 0.0 |
| **STRIDE-Qwen2.5-VL-7B** | **96.3** | **46.2** | **38.4** | **38.9** | **55.0** | **28.4** |
| STRIDE-Cosmos-Reason1-7B | 96.8 | 43.5 | 37.4 | 36.2 | 53.5 | 25.4 |

### Ablation Study: SpatialRGPT-Bench Spatial Reasoning

| Model | Orig. Qual.↑ | Orig. Quant.↑ | Obj Spatial Quant.↑ | Ego Spatial Qual.↑ | Ego Spatial Quant.↑ |
|---|---|---|---|---|---|
| GPT-4o | 80.5 | 32.5 | 39.4 | 55.7 | 27.7 |
| Qwen2.5-VL-7B | 67.2 | 24.4 | 12.8 | 47.1 | 29.3 |
| STRIDE-Qwen2.5-VL-7B | 69.5 | 37.5 | **61.5** | **77.9** | **70.3** |
| STRIDE-Cosmos-Reason1-7B | 71.1 | 30.0 | 58.7 | 79.9 | 68.9 |

### Key Findings
1. **General-purpose VLMs almost completely fail at spatiotemporal reasoning**: All baseline models achieve TLC ≈ 0, indicating a complete lack of temporally consistent reasoning capability.
2. **Fine-tuning yields dramatic improvements**: STRIDE-Qwen2.5-VL-7B improves LSR@0s from 1.0% to 96.3% (a 96× gain), with MLSR reaching 55.0.
3. **Out-of-view prediction is the primary bottleneck**: LSR degrades gradually for the Maintain State scenario (target remains in view), but drops sharply for out-of-view (OOV) scenarios such as Oncoming Pass, which is the primary reason TLC reaches only 28.4%.
4. **Cross-domain transfer is effective**: On the external SpatialRGPT-Bench, ego-centric spatial quantitative accuracy improves from 29.3% to 70.3% (+41 pt).
5. **Baseline model behavior analysis**: Baseline VLMs produce sparse predictions with systematic biases, repeatedly generating similar erroneous guesses, indicating reliance on simple memorized behaviors rather than visual context-based reasoning.
6. **Significant variation across six dynamic scene types**: The LSR decay patterns of Oncoming Pass (OOV rate 100%) and Maintain State (OOV rate 5%) are sharply contrasting, confirming that OOV is the primary driver of performance degradation.
7. **Rich dataset statistics**: The vehicle class dominates (440K instances), with adequate coverage of pedestrians (139K) and large vehicles (119K).

## Highlights & Insights
- 16M QA pairs × 270K frames: currently the largest-scale dataset for spatiotemporal reasoning VQA in autonomous driving.
- The first unified framework supporting all three reasoning categories: object-centric spatial, ego-centric spatial, and ego-centric spatiotemporal.
- A fully automated annotation pipeline with 3D detection error of only 13.6 cm, scalable to larger datasets.
- A three-tier LSR/MLSR/TLC metric hierarchy that precisely quantifies each dimension of VLM spatiotemporal reasoning capability.
- Data collection covers diverse Tokyo scenarios: traffic congestion, construction zones, and pedestrian-dense intersections.
- Comprehensive privacy protection: Dashcam Anonymizer automatically blurs faces and license plates.

## Limitations & Future Work
1. **Single front-view evaluation**: Only the front-view camera (60° FOV) is used; prediction capability drops sharply when targets leave the field of view, and multi-camera fusion remains unexplored.
2. **LoRA-only fine-tuning**: Due to compute constraints, full-parameter fine-tuning performance upper bounds have not been investigated.
3. **No downstream task evaluation**: The benefit of spatiotemporal reasoning capability for safety-critical tasks such as motion planning and behavior prediction has not been verified.
4. **No cross-dataset generalization**: As the first spatiotemporal reasoning benchmark of its kind, no comparable public dataset is available for cross-validation.
5. **Limitations of template-based QA**: QA pairs are generated from templates and lack natural language diversity, potentially limiting generalization of the VLM's language understanding.
6. **Relatively large velocity estimation error**: The detection pipeline achieves mAVE = 1.28 m/s, introducing non-negligible noise into QA quality on the velocity dimension.

## Related Work & Insights
- **Spatially-aware VLMs**: SpatialVLM (CVPR 2024), Spatial-RGPT (NeurIPS 2024) — address spatial reasoning in static scenes, lacking the temporal dimension.
- **Driving VQA datasets**: nuScenes-QA (AAAI 2024, 460K pairs), nuPlanQA (2025, 1M pairs), TUMTraffic-VideoQA (ICML 2025, 87.3K) — inferior to STRIDE-QA in both scale and task coverage.
- **Driving VLMs**: DriveLM (ECCV 2024), Senna-VLM, Cosmos-Reason1-7B (NVIDIA 2025) — designed for scene understanding and high-level decision making.
- **ToD3Cap / NuPrompt**: Focus on ego-centric spatial description and object referring, but lack spatiotemporal prediction tasks.
- **Refer-KITTI**: Only 6 hours and 818 referring expressions — far insufficient in scale.

## Rating
⭐⭐⭐⭐ — The dataset scale and annotation pipeline quality are solid, and the three-task taxonomy is clearly defined with practical significance. The core findings—that general-purpose VLMs score near zero on spatiotemporal reasoning while fine-tuning yields substantial gains—provide important reference value. Limitations include the single front-view evaluation setup and the absence of validation on downstream planning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RoadSceneVQA: Benchmarking Visual Question Answering in Roadside Perception Systems for Intelligent Transportation System](roadscenevqa_benchmarking_visual_question_answering_in_roadside_perception_syste.md)
- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)
- [\[CVPR 2026\] HybridDriveVLA: Vision-Language-Action Model with Visual CoT reasoning and ToT Evaluation for Autonomous Driving](../../CVPR2026/autonomous_driving/hybriddrivevla_vision-language-action_model_with_visual_cot_reasoning.md)
- [\[CVPR 2026\] TopoHR: Hierarchical Centerline Representation for Cyclic Topology Reasoning in Driving Scenes with Point-to-Instance Relations](../../CVPR2026/autonomous_driving/topohr_hierarchical_centerline_representation_for_cyclic_topology_reasoning_in_d.md)
- [\[AAAI 2026\] Fine-Grained Representation for Lane Topology Reasoning](fine-grained_representation_for_lane_topology_reasoning.md)

</div>

<!-- RELATED:END -->
