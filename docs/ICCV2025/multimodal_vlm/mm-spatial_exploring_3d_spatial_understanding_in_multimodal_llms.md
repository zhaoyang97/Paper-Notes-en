---
title: >-
  [Paper Note] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs
description: >-
  [ICCV 2025][Multimodal VLM][3D spatial understanding] Apple proposes the CA-VQA dataset and MM-Spatial model, leveraging high-quality 3D scene data and open-set annotations to generate training/evaluation data covering spatial relation prediction, metric estimation, and 3D grounding. The resulting general-purpose MLLM achieves SOTA on 3D spatial understanding benchmarks while remaining competitive on other tasks.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "3D spatial understanding"
  - "multimodal LLM"
  - "depth estimation"
  - "multi-view"
  - "spatial reasoning"
date: 2026-05-08
content_hash: de06d857c549bcf9
---

# MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs

**Conference**: ICCV 2025
**arXiv**: [2503.13111](https://arxiv.org/abs/2503.13111)  
**Code**: [GitHub](https://github.com/apple/ml-cubifyanything)  
**Area**: Multimodal VLM
**Keywords**: 3D spatial understanding, multimodal LLM, depth estimation, multi-view, spatial reasoning

## TL;DR

Apple proposes the CA-VQA dataset and MM-Spatial model, leveraging high-quality 3D scene data and open-set annotations to generate training/evaluation data covering spatial relation prediction, metric estimation, and 3D grounding. The resulting general-purpose MLLM achieves SOTA on 3D spatial understanding benchmarks while remaining competitive on other tasks.

## Background & Motivation

MLLMs excel at 2D visual understanding, yet their 3D spatial reasoning capabilities remain limited:

**Critical gaps in 3D perception**: Existing MLLMs struggle with: (1) relative depth judgments ("in front of" vs. "behind"); (2) metric distance/size estimation ("A is 2.74m away"); (3) precise 3D bounding boxes.

**Limitations of Prior Work**: Existing 3D spatial datasets suffer from various constraints — they cover only a subset of tasks, lack high-quality 3D ground truth, omit depth maps, do not support multi-view inputs, and fail to provide both training and evaluation splits simultaneously.

**Underexplored depth and multi-view inputs**: Few works systematically evaluate the impact of different depth map types (sensor vs. monocular estimation vs. GT) and multi-view inputs on 3D spatial understanding.

The authors aim to systematically advance 3D spatial understanding research in MLLMs through a comprehensive dataset.

## Method

### Overall Architecture

1. **Data generation pipeline**: Built upon the Cubify Anything 1M (CA-1M) dataset (containing 7-DOF 3D bounding boxes per object in ARKitScenes with open-set semantic annotations), automatically generating templated QA pairs.
2. **CA-VQA dataset**: ~10M QA pairs covering 220K frames and 2K videos for training; ~62K QA pairs over 2.6K frames for evaluation.
3. **MM-Spatial model**: Based on the MM1.5-3B architecture (DFN-CLIP visual encoder + decoder-only LLM), acquiring spatial understanding capabilities via SFT.

### Key Designs

1. **CA-VQA Dataset and Benchmark**

   Covers six spatial task categories:
    - **Counting**: "How many chairs are in the scene?"
    - **Viewpoint-dependent relations**: "Is X behind Y?" (depends on camera pose)
    - **Metric regression**: "What is the distance from X to Y/the camera?" "How wide/tall is X?"
    - **2D/3D Referring & Grounding**
    - **Binary & multiple-choice questions**

   Distinguishing features:
    - 3D ground truth from high-precision FARO laser scanners (not pseudo-labels)
    - Three depth map types per frame: GT depth, ARKit depth (iPad LiDAR), monocular depth (DepthPro)
    - Multi-view support: up to 4 support frames per reference frame, with relative poses and camera intrinsics
    - **Blind filtering strategy**: 7 MLLMs serve as judges to remove samples answerable without visual input, reducing language prior bias

2. **Depth Utilization: CoT / Tool-Use**

   Two strategies for leveraging metric depth (without directly encoding depth maps):
    - **Tool-Use**: The model predicts 2D bounding boxes and issues a function call; the tool returns the median depth within the box as text, upon which the model reasons.
    - **CoT (Chain-of-Thought)**: During training, step-by-step reasoning examples with GT depth are provided; at test time, the model predicts depth values on its own.

   Design Motivation: Encoding full-image depth yields only normalized relative depth, whereas CoT/Tool-Use can exploit absolute metric depth. Moreover, CoT requires no external tools — the model learns to predict depth accurately through SFT.

3. **Multi-View Input**

   Leveraging MM1.5's multi-image input capability, support frames and the reference frame are concatenated as a sequence $I_{t-N}, ..., I_{t-1}, I_t$, with per-frame camera intrinsics and relative poses provided in JSON format. Image splitting is applied only to the reference frame.

   Design Motivation: Multi-view inputs provide additional geometric constraints that help resolve depth ambiguity inherent to single-view observations.

### Loss & Training

- Follows MM1.5's three-stage training: pretraining → continual pretraining → SFT.
- In the SFT stage, a Spatial category (CA-VQA data) is added to MM1.5's base data mixture (General VQA, Knowledge, Text-rich, 2D Ref./Grounding).
- Mixing ratios are tuned to ensure spatial task gains do not degrade other capabilities.
- Image resolution: 672×672, with 4 sub-images + 1 global image.
- Both the visual encoder and LLM are unfrozen.

## Key Experimental Results

### Main Results

**CA-VQA Benchmark results (average scores per task)**:

| Model | Binary | Count. | 2D AP@50 | 3D AP@15 | Multi-c. | Ego-Dist. | Obj-Size | Avg. |
|-------|--------|--------|----------|----------|----------|-----------|----------|------|
| GPT-4o | 44.2 | 69.0 | 0.0 | 0.0 | 36.6 | 11.7 | 11.0 | 22.8 |
| SpatialRGPT-8B | 53.6 | 68.8 | 5.5 | 0.0 | 37.2 | 10.5 | 7.0 | 23.9 |
| MM1.5-3B | 59.1 | 9.1 | 32.6 | 0.0 | 38.6 | 0.6 | 3.4 | 18.2 |
| **MM-Spatial-3B** | **68.8** | **75.8** | **53.2** | **20.7** | **74.2** | **40.0** | **24.4** | **47.0** |
| +CoT | 69.6 | 75.9 | 54.5 | 21.9 | 74.7 | 46.0 | 26.7 | 49.1 |
| +Multi-view+CoT | 69.2 | 76.1 | 55.0 | 23.6 | 75.3 | 46.1 | 28.2 | 49.7 |
| +Multi-view+Tool(GT) | 69.2 | 76.1 | 55.0 | 23.6 | 75.3 | **65.8** | 27.3 | **52.4** |

MM-Spatial-3B (only 3B parameters) substantially outperforms GPT-4o and SpatialRGPT-8B on all tasks.

**Cross-benchmark category results**:

| Model | Spatial | General | Knowledge | Text-rich | Ref./Ground | Avg. |
|-------|---------|---------|-----------|-----------|-------------|------|
| MM1.5-3B | 39.9 | 64.7 | 46.2 | 62.1 | 77.7 | 58.1 |
| **MM-Spatial-3B** | **70.1** | 65.0 | 46.2 | 62.1 | 79.1 | **64.5** |

Spatial capability improves substantially (+30.2), while other categories remain unchanged or improve marginally.

### Ablation Study

**Specialist model configuration comparison (CA-VQA)**:

| Configuration | Ego-Dist. @10% | Obj-Dist. @10% | Obj-Size @10% | Avg. |
|---------------|---------------|---------------|---------------|------|
| MM-Spatial | 47.3 | 24.4 | 24.3 | 49.4 |
| +CoT (self-predicted depth) | 49.5 | 27.9 | 26.7 | 50.8 |
| +Depth (Tool; Mon.) | 42.1 | 26.1 | 26.1 | 49.5 |
| +Depth (Tool; GT) | **74.0** | **32.4** | 27.4 | **54.5** |
| +Depth (Encoded; GT) | 48.3 | 25.4 | 24.5 | 49.9 |
| +Multi-view | 52.4 | 26.2 | 26.1 | 51.4 |
| +Multi-view+CoT | 55.2 | 29.7 | **28.6** | 52.7 |

The CoT model's self-predicted depth nearly matches the performance of GT depth tool-use, demonstrating that the model successfully learns monocular depth estimation.

### Key Findings

1. **Data-driven depth estimation**: Through SFT data alone, MM-Spatial achieves performance approaching that of dedicated monocular depth estimation models — a surprising finding.
2. **Multi-view consistency is beneficial**: Multi-view inputs provide consistent gains across all configurations, particularly in 3D grounding (AP@15: 24.2→27.5) and distance estimation.
3. **Encoded depth underperforms CoT**: Encoding depth maps through the visual encoder (yielding only relative depth) is inferior to CoT's textualized absolute depth.
4. **Monocular depth < GT depth**: Tool-use with DepthPro monocular depth underperforms GT depth, indicating that depth accuracy is a ceiling factor.
5. **Blind filtering is effective**: Removing samples answerable by blind models yields a more challenging and reliable benchmark.

## Highlights & Insights

- **Unmatched comprehensiveness**: CA-VQA is the first spatial understanding dataset to simultaneously provide high-quality 3D ground truth, three depth map types, multi-view inputs, diverse task categories, and train/evaluation splits.
- **General capability preserved**: MM-Spatial at only 3B parameters substantially surpasses GPT-4o on spatial tasks while maintaining performance on general, knowledge, and text-rich tasks.
- **Implications of CoT depth estimation**: Models can learn depth perception from data without explicit depth sensors — a significant insight for edge device deployment.
- **Blind filtering strategy is broadly applicable**: The 7-model joint filtering approach can be generalized to other benchmark construction efforts to reduce language prior bias.

## Limitations & Future Work

- The dataset is confined to indoor scenes (ARKitScenes); generalization to outdoor environments remains unverified.
- Only the 3B model is explored; the effects of larger-scale models (7B, 70B) are unknown.
- The absolute 3D grounding AP@15 remains low (maximum 27.5), leaving substantial room for improvement.
- Object-to-object distance estimation accuracy under the 10% relative error threshold is only ~30%, which is insufficient for practical applications.
- Whether the multi-view frame selection strategy (angle ≥15° or translation ≥30cm) is optimal has not been thoroughly ablated.

## Related Work & Insights

This work represents a comprehensive upgrade over prior efforts including SpatialRGPT, Cube-LLM, and SpatialBot. The key distinctions lie in data quality (precise 3D annotations vs. pseudo-labels) and input signal diversity (multi-view + three depth types). The success of CoT depth estimation suggests the broader possibility of "learning visual perception through language."

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dataset construction pipeline and CoT depth estimation are highlights, though the model architecture itself introduces no new design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive: multiple model variants, multiple benchmarks, ablation analysis, and blind filtering validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-organized, with rich tables and figures; the dataset comparison table is particularly clear.
- **Value**: ⭐⭐⭐⭐⭐ The dataset and benchmark will drive subsequent research on 3D spatial understanding in MLLMs; code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](../../CVPR2025/multimodal_vlm/robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](../../CVPR2026/multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[ICCV 2025\] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)
- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)

</div>

<!-- RELATED:END -->
