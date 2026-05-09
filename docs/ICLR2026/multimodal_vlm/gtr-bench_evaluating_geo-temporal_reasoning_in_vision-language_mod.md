---
title: >-
  [Paper Note] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models
description: >-
  [ICLR 2026][Multimodal VLM][Geo-temporal reasoning] This paper introduces GTR-Bench, a novel benchmark for geo-temporal reasoning of moving targets in large-scale camera networks. Evaluation reveals that the strongest model, Gemini-2.5-Pro (34.9%), falls far short of human performance (78.61%), exposing three critical deficiencies in current VLMs: imbalanced utilization of spatial-temporal context, weak temporal prediction capability, and insufficient map-video alignment.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Geo-temporal reasoning
  - vision-language models
  - multi-camera networks
  - benchmark
  - spatial-temporal intelligence
date: 2026-05-08
content_hash: 68bcf7adb3362bf0
---

# GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.07791](https://arxiv.org/abs/2510.07791)
**Code**: [GitHub](https://github.com/X-Luffy/GTR-Bench)
**Area**: Spatial-Temporal Intelligence / Vision-Language Model Evaluation
**Keywords**: Geo-temporal reasoning, vision-language models, multi-camera networks, benchmark, spatial-temporal intelligence

## TL;DR

This paper introduces GTR-Bench, a novel benchmark for geo-temporal reasoning of moving targets in large-scale camera networks. Evaluation reveals that the strongest model, Gemini-2.5-Pro (34.9%), falls far short of human performance (78.61%), exposing three critical deficiencies in current VLMs: imbalanced utilization of spatial-temporal context, weak temporal prediction capability, and insufficient map-video alignment.

## Background & Motivation

**Spatial-temporal intelligence as a core capability**: Spatial intelligence underlies human interaction with the physical world. Its extension—spatial-temporal intelligence—is critical for autonomous driving, embodied AI, and related domains, encompassing spatial attributes (size, distance), temporal attributes (time intervals, speed), and reasoning over dynamic events.

**Limitations of existing benchmarks**: Current geographic reasoning benchmarks (e.g., ReasonMap) focus solely on static geometric tasks and graphical contexts (e.g., subway maps), while spatial-temporal reasoning benchmarks (e.g., VSI-Bench, STI-Bench) primarily adopt egocentric perspectives from single or few cameras, relying on image/video contexts.

**Absence of geo-level spatial-temporal reasoning evaluation**: No existing benchmark evaluates VLMs' ability to perform geo-temporal reasoning by jointly utilizing graphical context (maps) and multi-view video observations within large-scale camera networks.

**Pressing practical demand**: Real-world applications such as traffic management and emergency response require comprehensive spatial-temporal analysis across multiple camera views, including vehicle/pedestrian trajectory reasoning and traffic flow prediction.

**Uniqueness of the new challenge**: Geo-temporal reasoning (GTR) demands repeated perspective switching between maps and videos, joint reasoning across multiple non-overlapping fields of view, and inference over spatial-temporal regions unobserved in any video.

**Cognitive science perspective**: Traditional spatial-temporal intelligence covers only first-person (egocentric) and third-person (allocentric) perspectives, whereas the geographic perspective can provide VLMs with omniscient understanding of dynamic objects.

## Method

### Overall Architecture

GTR-Bench is a hierarchical geo-temporal reasoning benchmark comprising **3 basic reasoning tasks** and **4 combinatorial reasoning tasks**, with a total of 420 questions and 364 video clips. The benchmark covers two real-world scenarios: outdoor (CityFlow dataset, vehicles) and indoor (MTMMC dataset, pedestrians), each contributing 210 questions.

**Basic Tasks**:
- **Geo-Location (GL)**: Given start and end locations, infer the intermediate location (camera) that the target passes through.
- **Arrival Time-Interval (ATI)**: Given start/end points and an intermediate location, infer the time interval at which the target arrives at the intermediate location.
- **Motion-State (MS)**: Given start/end points and an intermediate location, infer the target's motion state (direction, speed, distance) at the intermediate location.

**Combinatorial Tasks**:
- **Causal Reordering (CR)**: Given unordered video clips and a map, determine the correct temporal order in which the target passes through cameras.
- **Next Spot Forecasting (NSF)**: Given the last observation and a map, predict the next camera location and the arrival time interval.
- **Trajectory Forecasting (TF)**: Based on multiple historical observations, predict the complete future trajectory (camera sequence and time intervals).
- **Multi-Target Trajectory Forecasting (MTTF)**: Predict the future meeting point (location and time) of two distinct targets.

### Key Designs

#### Evaluation Metric Design

- **Function**: Design appropriate evaluation metrics for different tasks.
- **Design Motivation**: Basic tasks and CR are standard multiple-choice questions, but forecasting tasks require simultaneous assessment of spatial correctness and temporal precision.
- **Mechanism**:
    - Basic tasks + CR: Standard MCQ accuracy.
    - Forecasting tasks (NSF/TF/MTTF): Propose **ST-IoU** (Spatial-Temporal IoU), integrating spatial accuracy (whether the Camera ID is correct) and temporal IoU (intersection-over-union of predicted and ground-truth time intervals):
    $$\text{ST-IoU} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(C_{p_i}=C_{gt_i}) \times \frac{|T_{p_i} \cap T_{gt_i}|}{|T_{p_i} \cup T_{gt_i}|}$$

#### Spatial-Temporal Complexity Grading

- **Function**: Categorize tasks into Long/Medium/Short levels according to geo-temporal complexity.
- **Design Motivation**: Ensure evaluation covers different spatial and temporal scales, prioritizing dynamic cues over static background.
- **Mechanism**: Levels are determined by physical thresholds on trajectory length ($track_d$) and duration ($track_t$), with distinct thresholds for indoor and outdoor settings (shorter time but longer distance outdoors due to driving scenarios), ensuring balanced distribution across the three complexity levels.

#### Benchmark Construction Pipeline

- **Function**: Automatically convert raw video data into standardized questions.
- **Design Motivation**: Accommodate the temporal, geographic, and formatting requirements of different tasks.
- **Mechanism**:
  1. **Data Preprocessing**: Video segmentation → camera calibration (homography matrix) → trajectory projection onto map → motion parameter computation (speed, direction) → data cleaning and validation → LLM-generated motion descriptions.
  2. **Task Construction**: Trajectory sampling → information integration (map + video + template) → question-answer formation → distractor generation (sampling from different building areas, algorithmically synthesizing spurious cameras, randomizing IDs).
  3. **Quality Control**: Two-stage human screening—Stage 1 ensures question diversity and removes instances with large trajectory errors; Stage 2 involves expert verification of answers and coverage of reasonable difficulty levels.

### Loss & Training

This paper presents a benchmark study and does not involve model training. Evaluation settings are as follows:
- Videos are uniformly sampled, with total frames across multiple videos capped at 20.
- Temperature = 0.1, max\_new\_token = 16384.
- Open-source models are deployed via LMDeploy on 8 NVIDIA V100 GPUs.
- Traditional ReID methods are included as comparison baselines.

## Key Experimental Results

### Main Results

| Model | Type | Rank | GL(Out/In) | ATI(Out/In) | MS(Out/In) | CR(Out/In) | NSF(Out/In) | TF(Out/In) | MTTF(Out/In) | Avg. |
|-------|------|------|------------|-------------|------------|------------|-------------|------------|--------------|------|
| Gemini-2.5-Pro | PM | 1 | 60.0/63.3 | 46.7/13.3 | 33.3/26.7 | 56.7/70.0 | 19.1/25.1 | 13.2/28.1 | 19.2/14.4 | **34.93** |
| GPT-5 | PM | 2 | 53.3/60.0 | 76.7/30.0 | 40.0/43.3 | 40.0/86.2 | 12.0/11.3 | 12.1/2.6 | 7.3/1.8 | 34.05 |
| Claude-4-Sonnet | PM | 3 | 73.3/66.7 | 50.0/33.3 | 50.0/43.3 | 63.3/58.6 | 8.1/2.6 | 6.2/4.0 | 16.9/0.0 | 34.03 |
| InternVL3-38B | OM | 5 | 40.0/50.0 | 73.3/56.7 | 30.0/26.7 | 53.3/37.9 | 8.3/11.1 | 8.2/4.4 | 20.6/10.2 | 30.76 |
| Qwen2.5-VL-32B | OM | 6 | 43.3/33.3 | 60.0/56.7 | 33.3/43.3 | 66.7/70.0 | 0.7/3.3 | 0.0/0.0 | 15.7/0.0 | 30.45 |
| **Human** | - | - | 90.0/98.2 | 84.3/90.8 | 90.9/89.5 | 89.8/97.4 | 68.3/74.6 | 51.2/57.4 | 55.8/62.5 | **78.61** |

### Ablation Study

**Spatial Reasoning vs. Spatial-Temporal Reasoning (MCQ Acc vs. ST-IoU)**:

| Model | NSF-MCQ/ST-IoU(Out) | TF-MCQ/ST-IoU(Out) | MTTF-MCQ/ST-IoU(Out) | NSF-MCQ/ST-IoU(In) |
|-------|---------------------|--------------------|-----------------------|---------------------|
| GPT-4o | 53.3/20.5 | 41.7/0.0 | 76.7/23.1 | 30.0/13.0 |
| Gemini-2.5-Pro | 38.5/19.1 | 45.5/13.2 | 51.7/19.2 | 43.3/25.1 |
| GPT-5 | 73.3/12.0 | 58.3/12.1 | 83.3/7.3 | 50.0/11.3 |
| GLM-4.1V-9B | 40.0/10.3 | 30.0/0.0 | 76.7/25.4 | 10.3/2.9 |

MCQ accuracy is generally much higher than ST-IoU, indicating that models can roughly localize spatial positions but fail to handle temporal constraints. GPT-5 achieves 83.3% MCQ accuracy on MTTF but only 7.3% ST-IoU, a gap of 76 percentage points.

### Key Findings

1. **Large human-machine gap**: The strongest model, Gemini-2.5-Pro (34.93%), lags behind humans (78.61%) by 43.68 percentage points; open-source models average only 23.82%.
2. **Sharp performance drop from basic to combinatorial tasks**: Models perform reasonably on basic tasks (GL and ATI reaching 60–76%), but ST-IoU on combinatorial forecasting tasks (NSF/TF/MTTF) is generally below 30%, with many open-source models approaching 0.
3. **Outdoor vs. indoor discrepancy**: Most models perform better outdoors (clearer spatial cues, more regular motion patterns), but Gemini-2.5-Pro anomalously excels indoors, possibly because advanced models leverage reasoning more effectively in complex scenes.
4. **Imbalanced spatial-temporal context utilization**: Top-tier models (e.g., Gemini-2.5-Pro) utilize spatial, temporal, and motion-state context in a balanced manner, whereas open-source models (e.g., InternVL3-38B) exhibit notably weaker temporal reasoning.
5. **Temporal prediction as the bottleneck**: All models demonstrate substantially stronger spatial localization than temporal prediction, with a large gap between MCQ Acc and ST-IoU (e.g., 76 percentage points for GPT-5).

## Highlights & Insights

- **Original task formulation**: This work is the first to extend spatial-temporal reasoning to geo-level large-scale camera networks, introducing joint reasoning over maps and multi-view videos—more representative of real-world applications than conventional single-video egocentric reasoning.
- **Elegant ST-IoU metric design**: The multiplicative combination of spatial accuracy and temporal IoU enables a single metric to assess joint spatial-temporal prediction quality.
- **Hierarchical task structure**: The basic-to-combinatorial progression precisely pinpoints capability bottlenecks in models.
- **In-depth three-deficiency analysis**: Beyond reporting performance numbers, the paper reveals fundamental shortcomings in current VLMs' spatial-temporal intelligence through context utilization analysis, MCQ vs. ST-IoU comparisons, and failure case studies.
- **Inclusion of ReID baselines**: Traditional Re-ID methods (45.72%) outperform most VLMs on forecasting tasks, indicating that current VLMs still fall short in leveraging visual feature matching.

## Limitations & Future Work

1. **Limited data scale**: Although carefully constructed, 420 questions may be insufficient to comprehensively assess model performance across more diverse scenarios.
2. **Video sampling constraints**: Capping total frames at 20 may discard important temporal information in videos, disadvantaging models that rely on dense frames.
3. **Only two scenario types**: Coverage is limited to outdoor vehicles and indoor pedestrians, excluding other settings (e.g., UAV perspectives, maritime scenarios).
4. **Lack of improvement proposals**: The paper identifies problems but does not propose targeted solutions or model improvement directions (e.g., fine-tuning, prompt engineering).
5. **Simplified map representation**: Maps are presented in simplified form, without involving more complex real-world map data (e.g., high-definition maps, 3D building models).
6. **Scalability**: Future work could extend to scenarios with more cameras (>31), longer time spans, and more diverse target types.

## Related Work & Insights

- **ReasonMap / SpatialLLM**: Static geographic reasoning benchmarks handling only graphical contexts—inspiring GTR to incorporate dynamic targets into geographic reasoning.
- **STI-Bench / VSI-Bench / ST-VLM**: Egocentric spatial-temporal reasoning benchmarks—demonstrating the necessity of extending from single-view to multi-camera network reasoning.
- **CityFlow / MTMMC**: Multi-camera tracking datasets—GTR-Bench repurposes their real-world trajectory data to construct higher-level reasoning tasks.
- **Implications for future research**: Promising directions include (1) designing dedicated temporal reasoning modules for VLMs, (2) developing map-video alignment pre-training strategies, and (3) leveraging graph structures to model camera network topology.

## Rating

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | ⭐⭐⭐⭐ | First to define the geo-temporal reasoning (GTR) task, extending VLM evaluation to multi-camera networks with a novel problem formulation. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Evaluates 13 mainstream VLMs plus human and ReID baselines, with rich analysis dimensions (context utilization, spatial-temporal comparison, failure cases). |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, well-defined tasks, rich tables and figures, though some analyses could be more in-depth. |
| Value | ⭐⭐⭐⭐ | Reveals critical bottlenecks in VLMs' spatial-temporal intelligence, with important reference value for autonomous driving and intelligent surveillance. |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[AAAI 2026\] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](../../AAAI2026/multimodal_vlm/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)
- [\[ICLR 2026\] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing](citylens_evaluating_large_vision-language_models_for_urban_socioeconomic_sensing.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)

<!-- RELATED:END -->
