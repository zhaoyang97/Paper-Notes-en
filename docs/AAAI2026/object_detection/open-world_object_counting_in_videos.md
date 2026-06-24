---
title: >-
  [Paper Note] CountVid: Open-World Object Counting in Videos
description: >-
  [AAAI 2026][Object Detection][Open-world counting] This paper proposes CountVid, a model, and the VideoCount benchmark, presenting the first systematic study of open-world video object counting—given a text or image description specifying target objects, the system enumerates all unique instances in a video. By combining an image counting model with a promptable video segmentation and tracking model, CountVid addresses challenges such as occlusion and re-appearance…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "Open-world counting"
  - "video counting"
  - "tracking"
  - "video segmentation"
  - "multimodal query"
date: 2026-05-08
content_hash: 8a3e4668bf73b50f
---

# CountVid: Open-World Object Counting in Videos

**Conference**: AAAI 2026
**arXiv**: [2506.15368](https://arxiv.org/abs/2506.15368)  
**Code**: Available (publicly released)  
**Area**: Medical Imaging
**Keywords**: Open-world counting, video counting, tracking, video segmentation, multimodal query

## TL;DR
This paper proposes CountVid, a model, and the VideoCount benchmark, presenting the first systematic study of open-world video object counting—given a text or image description specifying target objects, the system enumerates all unique instances in a video. By combining an image counting model with a promptable video segmentation and tracking model, CountVid addresses challenges such as occlusion and re-appearance, achieving substantial improvements over strong baselines across diverse scenarios including TAO, MOT20, penguin colonies, and X-ray metal crystallization.

## Background & Motivation

**Background**: Object counting is a fundamental task in computer vision. Existing methods primarily focus on single-frame image counting (e.g., density map regression, few-shot counting), while video counting remains largely unexplored. Counting in videos presents unique challenges: objects appear repeatedly across frames, may reappear after occlusion, and are difficult to distinguish from visually similar neighbors in crowded scenes. Naively taking the per-frame maximum count leads to severe overcounting.

**Limitations of Prior Work**: (1) No systematic video counting methodology exists—existing literature typically approaches video counting via tracking, which depends on detectors constrained to categories seen during training. (2) No standardized evaluation benchmark or dataset is available. (3) Open-world requirements are unmet—users should be able to specify arbitrary target categories via natural language or example images rather than predefined class sets.

**Key Challenge**: Image counting models do not handle temporal consistency (i.e., they cannot determine whether the same object across two frames is the same instance), while tracking models require detection outputs and are restricted to closed category sets. A solution that combines the strengths of both paradigms is needed.

**Goal**: To formally define the open-world video counting task and propose an automated video counting system that supports target specification via text or image descriptions.

**Key Insight**: Leveraging two recent breakthroughs—powerful open-world image counting models (e.g., CounTR, GroundingDINO) and promptable video segmentation models (e.g., SAM 2)—and composing them into a unified pipeline.

**Core Idea**: First, an image counting model detects and localizes target objects on keyframes; then, a promptable video segmentation and tracking model propagates the detections temporally; finally, cross-frame instance association and deduplication yield a unique instance count.

## Method

### Overall Architecture
CountVid adopts a three-stage pipeline: (1) **Frame-level detection**: an open-world image counting/detection model localizes user-specified target instances on sampled keyframes; (2) **Video tracking**: keyframe detections are used as prompts for a promptable video segmentation model (e.g., SAM 2), which tracks each instance throughout the full video; (3) **Deduplication and counting**: tracking trajectories initiated from different keyframes are merged via IoU matching and feature similarity, and the total number of unique instances is output.

### Key Designs

1. **Open-World Frame-Level Detection**:

    - **Function**: Detect all target instances on selected keyframes.
    - **Mechanism**: Keyframes are uniformly sampled (e.g., every 30 frames), and an open-world detection/counting model—GroundingDINO for text queries or CounTR for image queries—produces instance-level bounding boxes per keyframe. Keyframe sampling density can be dynamically adjusted based on scene characteristics: sparse for static scenes, dense for dynamic ones.
    - **Design Motivation**: Per-frame detection is computationally prohibitive; the keyframe strategy balances efficiency and coverage. Using open-world detectors ensures support for arbitrary categories.

2. **Promptable Tracking**:

    - **Function**: Propagate keyframe detections temporally across the full video.
    - **Mechanism**: Bounding boxes from keyframes are provided as point/box prompts to SAM 2, which automatically segments and tracks each instance in subsequent frames—maintaining instance identity even through occlusion and reappearance. This prompted tracking eliminates the dependence on category-specific detectors required by traditional tracking methods.
    - **Design Motivation**: SAM 2's promptable tracking capability is complementary to open-world detection—the detector is responsible for *discovering* instances, while the tracker is responsible for *following* them.

3. **Cross-Frame Instance Deduplication and Merging**:

    - **Function**: Consolidate tracking trajectories from different keyframes into unique instances.
    - **Mechanism**: For trajectories originating from different keyframes, mask IoU and appearance feature cosine similarity are computed over temporally overlapping regions. When both metrics exceed predefined thresholds, the two trajectories are considered the same instance and merged. The final output is the total count of deduplicated unique instances.
    - **Design Motivation**: The same object may be independently detected across multiple keyframes, necessitating deduplication to prevent overcounting—particularly critical in crowded scenes such as human crowds or penguin colonies.

### VideoCount Dataset
Constructed from TAO (multi-object tracking), MOT20 (dense crowd tracking), penguin videos, and X-ray metal alloy crystallization videos, the dataset covers diverse scene complexities ranging from structured pedestrian tracking to unstructured natural environments. Each video is annotated with target category labels and ground-truth unique instance counts.

## Key Experimental Results

### Main Results: Comparison with Strong Baselines

| Method | TAO MAE ↓ | TAO Acc@1 ↑ | MOT20 MAE ↓ | MOT20 Acc@1 ↑ | Penguin MAE ↓ | Avg. MAE ↓ |
|---|---|---|---|---|---|---|
| Per-Frame Max Count | 8.4 | 24.3% | 42.7 | 5.1% | 15.2 | 22.1 |
| Track-then-Count | 5.1 | 38.6% | 28.3 | 12.4% | 9.7 | 14.4 |
| CLIP-Count + Merge | 6.8 | 31.2% | 35.2 | 8.3% | 12.1 | 18.0 |
| **CountVid (Ours)** | **2.3** | **62.8%** | **12.5** | **31.6%** | **4.1** | **6.3** |

### Comparison of Query Modalities

| Query Type | MAE ↓ | Acc@1 ↑ |
|---|---|---|
| Text Prompt | 7.1 | 48.3% |
| Image Exemplar | 5.8 | 55.2% |
| Text + Image | **4.9** | **59.1%** |

### Ablation Study

| Configuration | MAE ↓ | Notes |
|---|---|---|
| CountVid (full) | 6.3 | Full method |
| w/o keyframe sampling (per-frame) | 5.9 | Marginally better but 10× compute |
| w/o cross-frame deduplication | 15.8 | Severe overcounting |
| w/o SAM 2 tracking (frame-level only) | 12.4 | Temporal information lost |
| Sparse keyframes (every 120 frames) | 8.7 | Newly appearing objects missed |

### Key Findings
- **Cross-frame deduplication is critical**: Removing it causes MAE to increase from 6.3 to 15.8, demonstrating that overcounting is the primary challenge in video counting.
- **SAM 2 tracking substantially reduces missed detections**: Frame-level detection alone achieves MAE of 12.4; adding tracking reduces it to 6.3, confirming the importance of temporal propagation.
- **Crowded scenes remain challenging**: MOT20 (dense crowds) yields MAE of 12.5, far higher than TAO (2.3), indicating that occlusion and similar appearance remain bottlenecks.
- **Multimodal queries are complementary**: Text + image queries outperform either modality alone, reflecting the complementarity between category-level information from text and appearance details from images.

## Highlights & Insights
- **The new task definition is valuable**: Open-world video counting addresses an important gap in video understanding—existing video understanding research largely focuses on action recognition and temporal localization, while object counting is a fundamental yet neglected task.
- **The modular design is elegant**: Composing open-world detection and promptable tracking rather than pursuing end-to-end training leverages the strongest existing models while maintaining flexibility.
- **High dataset diversity**: Spanning pedestrian tracking, penguin colonies, and X-ray crystallization, the benchmark covers visually extreme and heterogeneous scenarios.

## Limitations & Future Work
- The method depends on keyframe detection quality—if a detector completely misses an instance in all sampled frames, tracking cannot recover it.
- Scalability to very long videos (e.g., hours in duration) has not been validated; the keyframe sampling strategy may require more intelligent adaptive schemes.
- Deduplication thresholds are set manually; adaptive thresholds may generalize better across diverse scenes.
- The method does not address temporally varying counts (e.g., 5 birds present at one moment, 2 of which fly away later).

## Related Work & Insights
- **vs. image counting methods (CounTR, etc.)**: These process only single frames and cannot deduplicate across time; CountVid adds temporal tracking to resolve overcounting.
- **vs. MOT methods (SORT, etc.)**: These rely on closed-set category detectors and cannot handle open-world queries; CountVid replaces them with open-world detectors.
- **vs. SAM 2**: Provides promptable tracking capability but does not perform counting; CountVid integrates it within a counting framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ New task definition and new dataset; the method is a modular composition rather than a novel end-to-end architecture.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple scenes and baselines evaluated; dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Task definition is clear; pipeline description is intuitive.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for video counting; dataset and code are publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](../../CVPR2026/object_detection/detecting_unknown_objects_via_energy-based_separation.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](../../CVPR2025/object_detection/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](../../CVPR2026/object_detection/ew-detr_evolving_world_object_detection_via_incremental_low-rank_detection_trans.md)

</div>

<!-- RELATED:END -->
