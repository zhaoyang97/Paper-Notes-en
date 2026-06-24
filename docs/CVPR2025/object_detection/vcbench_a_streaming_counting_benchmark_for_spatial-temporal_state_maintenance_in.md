---
title: >-
  [Paper Note] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos
description: >-
  [CVPR 2025][Object Detection][Streaming video evaluation] VCBench repositions counting as a minimal probe to diagnose the "spatial-temporal state maintenance" capability of video models. It proposes 8 subcategories covering object counting (current state/identity tracking) and event counting (instantaneous events/periodic activities). By observing model prediction trajectories through streaming multi-point queries along the timeline, mainstream models are evaluated on 406 vid…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Streaming video evaluation"
  - "counting benchmark"
  - "spatial-temporal state maintenance"
  - "long video understanding"
  - "Video LLM"
date: 2026-05-08
content_hash: dea17ae4bc2aacde
---

# VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos

**Conference**: CVPR 2025  
**arXiv**: [2603.12703](https://arxiv.org/abs/2603.12703)  
**Code**: GitHub (see project page)  
**Area**: Video Understanding  
**Keywords**: Streaming video evaluation, counting benchmark, spatial-temporal state maintenance, long video understanding, Video LLM

## TL;DR
VCBench repositions counting as a minimal probe to diagnose the "spatial-temporal state maintenance" capability of video models. It proposes 8 subcategories covering object counting (current state/identity tracking) and event counting (instantaneous events/periodic activities). By observing model prediction trajectories through streaming multi-point queries along the timeline, mainstream models are evaluated on 406 videos and 4,576 query points, revealing that current models still exhibit significant deficiencies in spatial-temporal state maintenance.

## Background & Motivation
**Background**: Video understanding benchmarks (Video-MME, MLVU, LVBench, etc.) have covered various tasks, and streaming evaluation (OVO-Bench, StreamingBench) has introduced timestamp-based queries.

**Limitations of Prior Work**: (1) Most existing evaluations rely on single-turn answers (answering a question after watching the entire video), making it impossible to observe how models maintain internal states over time; (2) query time windows in streaming benchmarks are short, failing to impose long-term cumulative state maintenance pressure; (3) counting tasks do not differentiate between different tracking needs (currently visible vs. cumulative unique vs. instantaneous events vs. periodic activities).

**Key Challenge**: The core requirement of video understanding is to continuously update the world state, yet existing evaluations only check isolated answers, falling short of diagnosing specific deficiencies in the state maintenance process.

**Goal**: How to systematically evaluate the capability of models to maintain and update spatial-temporal states during video playback?

**Key Insight**: Counting serves as a minimal probe for state maintenance—numerical answers are highly deterministic, free from option bias, and cannot be bypassed through ambiguous semantics, while multi-point queries allow observation of the evolution patterns in prediction trajectories.

**Core Idea**: Counting as a probe + streaming multi-point querying + 8-category taxonomy + trajectory-level evaluation metrics.

## Method

### Overall Architecture
VCBench = A systematic taxonomy of counting capabilities + a frame-by-frame annotated video dataset + a streaming multi-point query design + three complementary evaluation metrics.

### Key Designs

1. **8-Subcategory Taxonomy**:

    - **Object Counting**: O1-Snap (currently visible quantity), O1-Delta (relative change), O2-Unique (cumulative unique individuals), O2-Gain (newly added individuals within a time window)
    - **Event Counting**: E1-Action (cumulative frequency of instantaneous actions), E1-Transit (frequency of scene transitions), E2-Periodic (complete cycles of periodic actions), E2-Episode (number of semantically complete activity segments)
    - Design Motivation: O1 measures real-time tracking (can increase or decrease), O2 measures identity de-duplication (should be monotonically non-decreasing), E1 measures instantaneous detection, while E2 measures tracking across complete lifecycles.

2. **Streaming Multi-point Queries**:

    - Function: Sets multiple query points along the video timeline, demanding the model to output the current answer for the same question at different timestamps.
    - Mechanism: Query points are selected at stable frames to avoid visual ambiguity, and ground-truth answers are computed based on annotations prior to that timestamp.
    - Design Motivation: Multi-point queries yield prediction trajectories rather than isolated answers, where the shape, consistency, and directional changes of trajectories can diagnose specific deficiencies.

3. **Three Complementary Metrics**:

    - **GPA (Gaussian Precision Accuracy)**: Measures the alignment between predictions and ground truths using a Gaussian kernel, where $\sigma = 0.05 \cdot \max(g_i, 1)$, and deviations $>15\%$ score near zero.
    - **MoC (Monotonicity Consistency)**: Verifies if the prediction trajectories of cumulative tasks (O2/E1/E2) are monotonically non-decreasing, where the score is determined by the position of the first violating point.
    - **UDA (Update Direction Accuracy)**: Evaluates whether the predicted change directions between adjacent query points match the ground truths.
    - Joint Diagnosis: High GPA + low UDA = accurate values but poor temporal awareness; low GPA + low MoC = comprehensive failure.

### Dataset Statistics
406 videos, 1,000 questions, 10,071 annotated event timestamps, and 4,576 query points. Video sources include YouTube, ARKitScenes, ScanNet, Ego4D, etc.

## Key Experimental Results

### Evaluated Models
- **Closed-source Models**: Gemini-3-Flash, Doubao-Seed-1.8, Kimi-K2.5
- **Open-source Offline Models**: Qwen3-VL-8B/30B, Qwen2.5-VL-7B, InternVL-3.5-8B, Molmo2-8B
- **Online Models**: StreamingVLM (based on Qwen2.5-VL-7B), Dispider
- **Baselines**: GPT-4-Turbo (text-only blind baseline), Human annotators

### Main Results (Overall + Object Counting)

| Model | Overall GPA | MoC | UDA | O1-Snap GPA | O2-Unique GPA/MoC/UDA |
|------|------------|-----|-----|-------------|----------------------|
| Human | 96.1 | 100.0 | 99.3 | 96.7 | 94.5 / 100.0 / 98.8 |
| Gemini-3-Flash | **37.0** | 73.7 | 73.8 | 44.3 | 36.5 / 77.8 / 79.8 |
| Doubao-Seed-1.8 | 36.2 | 77.2 | 76.8 | 43.4 | 32.5 / 75.9 / 79.8 |
| Qwen3-VL-8B | 31.0 | **84.3** | 55.1 | 21.1 | 34.2 / 87.1 / 58.0 |
| GPT-4-Turbo (blind) | 18.7 | 95.7 | 4.3 | 15.7 | 15.8 / 96.4 / 3.6 |

### Event Counting Results

| Model | E1-Action GPA | E1-Transit GPA | E2-Episode GPA | E2-Periodic GPA |
|------|--------------|----------------|----------------|-----------------|
| Human | 94.9 | 98.3 | 97.0 | 93.2 |
| Gemini-3-Flash | 28.5 | **49.8** | **41.7** | **3.9** |
| Doubao-Seed-1.8 | 24.0 | 38.2 | 40.5 | 0.8 |
| Qwen3-VL-8B | 22.5 | 36.9 | 35.6 | 0.0 |
| Dispider | 11.8 | 8.9 | 7.6 | 0.0 |

### Control Experiment: Visual-Temporal Alignment Bottleneck of E2-Periodic
- Overlaying explicit counting annotations on the top-left corner of the video (e.g., "Current count: 5") causes Gemini-3-Flash's E2-Periodic GPA to **skyrocket from 3.9 to 81.8** (a 21-fold increase), close to the human score of 93.2.
- MoC increases by 37.6 points, and monotonicity violations drop from 43.8% to 6.2%.
- **Key Insight**: Models possess adequate numerical reasoning capabilities, yet entirely lack the ability to detect periodic boundaries from raw visual inputs—the bottleneck is in "identifying when to increment" rather than "understanding that the count should increment".

### Control Experiment: Identity Persistence Degradation Under Camera Rotation
- Scenario: The camera rotates 360° in place for 60 rounds; there is only 1 bench in the scene repeatedly entering and leaving the field of view, so the ground-truth answer is always 1.
- First 10 rounds GPA: Gemini-3-Flash 76.2, Doubao-Seed-1.8 71.8, Qwen3-VL-8B 68.5.
- By the 60th round, GPAs drop to 28.3, 35.7, and 23.8, respectively (approaching random levels), while humans maintain a score of 100.
- **This indicates that models lack cross-temporal identity persistence representations**; as the number of re-occurrences increases, the internal state representations gradually collapse.

### Key Findings
- **Enormous Human-Machine Gap**: Humans score 92-100 in all subcategory GPAs, while the best model achieves an overall GPA of only around 37.
- **Comprehensive Collapse on E2-Periodic**: All models fail with a GPA < 4, whereas humans score 93.2—periodic event counting represents the greatest challenge.
- **Intriguing Closed-source vs. Open-source Divergence**: Closed-source models achieve higher GPAs (37 vs. 27-31), whereas open-source models demonstrate higher MoCs (84+ vs. 67-77)—open-source models tend to output smooth, monotonic sequences with inaccurate values.
- **High MoC of Blind Baseline**: GPT-4-Turbo, without visual input, achieves a MoC of 95.7, indicating that LLMs already understand the prior that "cumulative counts should increase", though its GPA is only 18.7 and UDA is 4.3.
- In O2 identity tracking, models frequently exhibit "decreasing counts" (low MoC), exposing logical contradictions in state maintenance mechanisms.
- Online models generally achieve lower GPAs than offline models of comparable parameter sizes, but exhibit higher MoCs in certain subcategories—the state compression of streaming architectures causes information loss but smoother prediction trajectories.

## Highlights & Insights
- **The positioning of "counting as a minimal probe" is highly ingenious**: Counting bypasses the evaluation noise of open-ended QA, directly quantifying state maintenance capability.
- **The MoC metric reveals logical consistency flaws**: A drop in cumulative count indicates that the model's internal state has been erroneously reset—a defect invisible in traditional accuracy metrics.
- **Orthogonal design of the taxonomy**: The 2D decomposition of Object vs. Event and Current vs. Cumulative provides independent diagnostic value for each quadrant.
- **Transferable methodology**: The approach of streaming multi-point querying + trajectory-level evaluation can be generalized to other tasks requiring state maintenance, such as causal reasoning in video QA.

## Limitations & Future Work
- Counting is only a proxy metric for state maintenance; more complex states (e.g., changes in object relations, causal chains) are not covered.
- E2-Periodic videos are generated through loop splicing, which may introduce distribution bias.
- High annotation cost (10,071 frame-by-frame annotations) hinders scalability.
- The improvement potential of models under counting-assisted prompts (e.g., Chain-of-Thought) has not been tested.

## Related Work & Insights
- **vs. Video-MME**: Video-MME is a comprehensive evaluation framework, whereas VCBench focuses specifically on diagnosing state maintenance, making them complementary.
- **vs. OVO-Bench**: OVO-Bench is broader (covering real-time perception, retrieval, and active response), whereas VCBench is more focused (counting probe + spatial-temporal state) with longer query time horizons.
- **vs. TOMATO**: TOMATO includes action counting but lacks streaming multi-point queries, preventing it from observing trajectory evolution.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of using "counting as a state maintenance probe" is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 subcategories, three complementary metrics, and diverse video sources.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear taxonomy, delicately designed metrics, and self-consistent diagnostic logic.
- Value: ⭐⭐⭐⭐ Fills the gap in the evaluation of video model state maintenance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark](rsar_restricted_state_angle_resolver_and_rotated_sar_benchmark.md)
- [\[CVPR 2025\] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)
- [\[AAAI 2026\] CountVid: Open-World Object Counting in Videos](../../AAAI2026/object_detection/open-world_object_counting_in_videos.md)
- [\[CVPR 2025\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)
- [\[NeurIPS 2025\] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](../../NeurIPS2025/object_detection/spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)

</div>

<!-- RELATED:END -->
