---
title: >-
  [Paper Note] MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps
description: >-
  [CVPR 2025][Segmentation][Camera traps] This paper proposes MammAlps—a multimodal, multi-view dataset for monitoring the behavior of wild mammals in the Swiss National Park (8.5 hours of dense annotation, 5 species, 11 activities + 19 actions), along with two benchmark tasks: multimodal species + hierarchical behavior recognition (B1) and the first multi-view long-term event understanding (B2), filling the gap in wildlife video behavior analysis regarding hierarchical behavio…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Camera traps"
  - "wildlife monitoring"
  - "multi-view video"
  - "hierarchical behavior recognition"
  - "multimodal"
date: 2026-05-08
content_hash: d9f5238255ab1528
---

# MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps

**Conference**: CVPR 2025  
**arXiv**: [2503.18223](https://arxiv.org/abs/2503.18223)  
**Code**: [https://github.com/eceo-epfl/MammAlps](https://github.com/eceo-epfl/MammAlps)  
**Area**: Image Segmentation  
**Keywords**: Camera traps, wildlife monitoring, multi-view video, hierarchical behavior recognition, multimodal

## TL;DR
This paper proposes MammAlps—a multimodal, multi-view dataset for monitoring the behavior of wild mammals in the Swiss National Park (8.5 hours of dense annotation, 5 species, 11 activities + 19 actions), along with two benchmark tasks: multimodal species + hierarchical behavior recognition (B1) and the first multi-view long-term event understanding (B2), filling the gap in wildlife video behavior analysis regarding hierarchical behavior annotation, multimodality, and multi-view.

## Background & Motivation

1. **Background**: Wildlife behavior monitoring is crucial for ecological conservation. Camera traps are the primary means of non-intrusive monitoring and have accumulated massive video data. Deep learning models are used for automating species identification and behavior classification.
2. **Limitations of Prior Work**: (1) Most existing wildlife video datasets feature only a single species, single view, or flat behavior annotations (independent categories without hierarchy); (2) Web-scraped datasets (e.g., MammalNet) have a large domain gap from real-world camera trap data; (3) No dataset provides multimodal (video + audio + segmentation maps) and multi-view information simultaneously; (4) Behavior annotations lack a hierarchical structure (activity vs. action hierarchy).
3. **Key Challenge**: The machine learning community needs wild datasets that are closer to real-world scenarios and more richly annotated to develop practical animal behavior recognition models, but the cost of collection and annotation is extremely high.
4. **Goal**: Build a camera trap video dataset that integrates multimodal, multi-view, and hierarchical behavior annotations, and propose two complementary benchmark tasks—short-clip behavior recognition and long-term event understanding.
5. **Key Insight**: Natural multi-view data is obtained by deploying nine camera traps (three at each of three sampling sites) in the Swiss National Park; annotation is conducted based on the hierarchical behavior classification theory of neuroethology (where an activity is composed of multiple actions).
6. **Core Idea**: Provide the first wild camera trap dataset that combines multimodal, multi-view, and hierarchical behavior annotations, paired with dual-task benchmarks for behavior recognition and long-term event understanding.

## Method

### Overall Architecture
The pipeline for constructing MammAlps is: (1) Data Collection—9 camera traps recorded for 6 weeks across 3 stations in the Swiss National Park; (2) Data Preprocessing—videos grouped into events, animal detection with MegaDetector, tracking with ByteTrack, and manual correction; (3) Annotation—individual-level species and two-layer behavior annotations (activity + action) + event-level individual counts and meteorological conditions; (4) Benchmark Tasks—B1 multimodal behavior recognition + B2 multi-view long-term event understanding.

### Key Designs

1. **Hierarchical Behavior Annotation System**:
    - **Function**: Provide more structured behavioral descriptions than existing datasets.
    - **Mechanism**: Based on ethological theory, behaviors are classified into two levels: (1) Actions (e.g., walking, grazing)—basic movements typically recognizable within a few frames; (2) Activities (e.g., foraging, courtship)—which require longer spatiotemporal context, can consist of multiple actions, or involve interactions between individuals. Each frame is annotated with one activity and one or two non-mutually exclusive actions.
    - **Design Motivation**: Existing datasets treat all behaviors as independent categories (flat taxonomy), ignoring the composition and hierarchical relationships of behaviors. Hierarchical annotation better aligns with behavioral ecology needs and allows models to make predictions at different levels of granularity.

2. **Multimodal Input Design (Video + Audio + Reference Scene Segmentation Map)**:
    - **Function**: Provide multiple complementary signals for behavior recognition.
    - **Mechanism**: (1) Video—cropped tracking clips of individual animals; (2) Audio—synchronized audio spectrograms processed through an AudioMAE tokenizer; (3) Reference scene segmentation map—each camera position is fixed, and a single scene segmentation map (10 classes, e.g., grass, water) is annotated, from which a segmentation map clip synchronized with the video is generated based on the animal's trajectory. The three modalities are fused using an improved VideoMAE architecture.
    - **Design Motivation**: Audio can capture behavioral signals like vocalization and walking; scene segmentation maps provide environmental context (e.g., an animal near a water source -> likely drinking), which might be missing in pure video inputs.

3. **Multi-view Long-term Event Understanding Benchmark (B2)**:
    - **Function**: Extract comprehensive event-level information from long-term recordings of multiple cameras.
    - **Mechanism**: (1) Video recordings from multiple cameras at the same station are temporally aligned and grouped into "events"; (2) Token Merging (ToME) is used to compress video tokens in both spatial and temporal dimensions—reducing patch tokens within frames first (reduction factor $r$), then merging them along the temporal dimension in fixed-duration blocks; (3) Three types of positional encodings are added: camera ID, relative time, and source frame/patch position; (4) All compressed tokens are input to a Transformer encoder, and four output heads predict species, activity, individual count, and meteorological conditions, respectively.
    - **Design Motivation**: Existing models cannot directly process multi-view + long-term (up to 12 minutes) ecological data. The offline token merging solution preserves key information within a manageable context length, serving as a practical workaround.

### Loss & Training
Task B1: Multi-task joint training based on VideoMAE, with a sampling strategy inversely weighted by label frequency to address class imbalance. During testing, sampling is performed 10 times per clip and averaged. The metric used is the mAP of each task. Task B2: Transformer encoder + four-head multi-task training, using a pretrained ViT-Base encoder.

## Key Experimental Results

### Main Results

**B1: Multimodal Behavior Recognition (VideoMAE, Joint Prediction)**:

| Modality | Species mAP | Activity mAP | Action mAP | Avg |
|------|------------|-------------|------------|-----|
| Video (V) | 0.495 | 0.436 | 0.452 | 0.453 |
| Audio (A) | 0.223 | 0.212 | 0.172 | 0.192 |
| V+A | 0.473 | 0.484 | 0.466 | 0.473 |
| V+A+S | 0.531 | 0.485 | 0.437 | 0.466 |

### Ablation Study

**B2: Multi-view Long-Term Event Understanding**:

| Configuration | Species | Activity | Met. Cond. | Individuals | Avg |
|------|---------|----------|-----------|-------------|-----|
| Joint, r=14 | 0.415 | 0.479 | 0.618 | 0.499 | 0.489 |
| Joint, r=11 | 0.446 | 0.481 | 0.594 | 0.543 | 0.500 |
| Single task (best) | 0.481 | 0.478 | 0.681 | 0.592 | - |

**Positional Encoding Ablation (B2, Individual Counting Task)**: After adding CamID + time + source frame positional encodings, mAP improved by up to $+0.109$, demonstrating that positional information is crucial for multi-view understanding.

### Key Findings
- **Video + audio fusion is most effective**: The V+A combination improves by $+0.020$ mAP compared to pure video. Although video + audio + segmentation map performs best on species (0.531), on average it is inferior to V+A.
- **Segmentation map modality is conditionally useful**: It yields a significant gain for pure audio ($+0.111$), but slightly decreases performance for the video + audio combination, likely because the current fusion strategy is not optimal.
- **Single task vs. Multi-task**: Joint training is not weaker than single-task training in most cases, and joint training in B1 improves performance across all tasks.
- **B2 task is highly challenging**: The longest event is 12 minutes, species and activities are severely imbalanced, and cross-camera counting requires multi-view fusion.

## Highlights & Insights
- **Hierarchical Behavior Annotation**: Introduces local activity-action two-layer annotation to a wild animal dataset for the first time, aligning with ethological theory. This hierarchical annotation method can be transferred to human action recognition—decomposing "cooking" into sub-actions like "chopping", "stirring", etc.
- **Reference Scene Segmentation Map as a Modality**: Leverages the fixed-position characteristic of the cameras to provide environmental context information for all videos with a single segmentation map, which is highly cost-effective yet rich in information. This idea is transferable to any surveillance scenario with fixed cameras.
- **Offline Token Merging for Long Video Processing**: Compresses video tokens in spatial + temporal dimensions to a manageable context length using ToME, representing a simple and effective engineering solution for long-form video understanding.

## Limitations & Future Work
- The data scale is limited (only 8.5 hours of annotated video, 5 species), showing a significant gap compared to MammalNet (394 hours, 173 species).
- The geographical scope is narrow (only the Swiss National Park), and generalizability remains to be verified.
- Nighttime videos captured using infrared have poorer quality, but the paper does not analyze nighttime data performance separately.
- Temporal drift between cameras can reach up to 1 minute, affecting multi-view alignment precision in the B2 task.
- Baseline models (improved VideoMAE + Transformer encoder) are relatively simple, and stronger multimodal/multi-view models (e.g., video-LLMs) could significantly improve performance.

## Related Work & Insights
- **vs MammalNet**: MammalNet is large scale but has diverse/noisy sources (documentaries, etc.), presenting a large domain gap; MammAlps is small scale but contains real camera trap data with finer annotations (individual-level, two-layer behavior, multimodal).
- **vs KABR/BaboonLand**: Both have individual-level behavior annotations, but they use UAVs instead of camera traps and offer only a single modality and flat behavior classification.
- **vs PanAf/LoTE**: These datasets feature camera trap data but behavior annotations are at the video level rather than the individual level; MammAlps offers finer annotation granularity.

## Rating
- Novelty: ⭐⭐⭐⭐ The first wild dataset combining multimodal + multi-view + hierarchical behavior annotations; the B2 task design is innovative.
- Experimental Thoroughness: ⭐⭐⭐ Baseline experiments are reasonably designed, but the models are relatively simple, lacking comparisons with more SOTA methods.
- Writing Quality: ⭐⭐⭐⭐ The dataset paper is well-structured with clear definitions of terminology (Table 2) and detailed statistical analysis.
- Value: ⭐⭐⭐⭐ Significant contribution to the field of wildlife behavior monitoring, though the scale of the dataset limits its immediate impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring](ripvis_rip_currents_video_instance_segmentation_benchmark_for_beach_monitoring_a.md)
- [\[CVPR 2025\] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation](m3-vos_multi-phase_multi-transition_and_multi-scenery_video_object_segmentation.md)
- [\[CVPR 2025\] Continuous Locomotive Crowd Behavior Generation](continuous_locomotive_crowd_behavior_generation.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)

</div>

<!-- RELATED:END -->
