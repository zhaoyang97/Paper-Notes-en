---
title: >-
  [Paper Note] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding
description: >-
  [ICCV 2025][Video Understanding][Online Video Temporal Grounding] This paper addresses the Online Video Temporal Grounding (OnVTG) task by proposing a hierarchical event memory mechanism that stores historical event info…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Online Video Temporal Grounding"
  - "Hierarchical Event Memory"
  - "Event Proposal"
  - "Future Prediction"
  - "Low Latency"
date: 2026-05-08
content_hash: a54cca868860092e
---

# Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding

**Conference**: ICCV 2025
**arXiv**: [2508.04546](https://arxiv.org/abs/2508.04546)  
**Code**: [https://github.com/minghangz/OnVTG](https://github.com/minghangz/OnVTG)  
**Area**: Video Understanding
**Keywords**: Online Video Temporal Grounding, Hierarchical Event Memory, Event Proposal, Future Prediction, Low Latency

## TL;DR

This paper addresses the Online Video Temporal Grounding (OnVTG) task by proposing a hierarchical event memory mechanism that stores historical event information at multiple temporal scales. Combined with a segment-tree-based event proposal structure and a future prediction branch, the method achieves state-of-the-art grounding accuracy and low-latency prediction on TACoS, ActivityNet Captions, and MAD.

## Background & Motivation

Online Video Temporal Grounding (OnVTG) requires models to localize text-query-relevant events in a video stream in real time, with access only to frames observed up to the current moment. Existing methods suffer from two core issues:

**Lack of effective event modeling**: Prior methods (e.g., Gan et al.) employ frame-level memory to store recent frame features and predict whether each frame corresponds to the start or end of a target event. However, target events vary greatly in duration (e.g., "pushing a door" is brief, while "playing the saxophone" is prolonged), making frame-level prediction ill-suited for modeling events of diverse durations.

**Inability to retain long-term historical information**: Fixed-size frame-level memory fills up with redundant content when similar frames appear over extended periods, causing valuable event information to be discarded. For example, the query "the person plays the saxophone again" requires knowledge of when the first occurrence happened, which may be at the very beginning of the video.

## Method

### Overall Architecture

Streaming video frames are passed through a pretrained visual encoder to extract frame features. Within a short-term window, a segment-tree structure builds multi-scale event proposals. Hierarchical event memory provides long-term historical context to refine current proposals. Proposal features and text query features are fused via a Transformer decoder, after which a classifier determines whether each proposal is a positive sample; positive proposals are further processed by a regression head for temporal boundary prediction. A future prediction branch predicts whether the target event is about to begin and regresses its start time.

### Key Designs

1. **Hierarchical Event Construction**: A segment-tree structure generates multi-scale event proposals within a short-term window $[t-L_s, t]$. Scale-1 proposal features are obtained from frame features via 1D convolution: $P^1 = \{p_i^1\}_{i=t-L_s+1}^{t}$. Higher scales are produced by merging adjacent lower-scale proposals: $p_i^{j+1} = \text{MLP}([p_{2i}^j; p_{2i-1}^j])$. The duration of a scale-$j$ proposal is $2^j$. When duration exceeds the short-term window length, the most recent historical events from memory participate in the merging computation.

2. **Hierarchical Event Memory and Update**:

    - **Dynamic memory size allocation**: Given total memory size $K$, the capacity of each scale is dynamically allocated according to positive-sample frequency: $K_i = 1 + (K-L)\frac{w_i}{\sum_{j=1}^{L} w_j}$, guaranteeing at least one slot per scale and allocating more capacity to scales with higher positive-sample frequency.
    - **Adaptive memory update**: When a newly added event causes a scale to exceed capacity, the cosine similarity between adjacent events is computed. If any adjacent pair exceeds threshold $\delta$, they are merged via average pooling; otherwise, the oldest event is removed in a FIFO manner. This prevents redundant events from occupying memory.

3. **Memory-driven Event Refinement**: A Transformer encoder layer integrates historical memory into current event proposals: $P^j = E(P^j; M)$. Fine-grained recent information is provided by small-scale memory, while coarse-grained long-term information is retained in large-scale memory.

4. **Future Prediction Branch**: This branch addresses the start-time latency problem inherent in proposal-only approaches (where the model obtains a complete proposal only near the end of an event). At time $t$, an MLP predicts the probability $c_t^f$ that the target event will begin within a future window $[t+a, t+b]$ and regresses the start time offset $o_t^f$. This enables the model to predict the start time before the event actually begins.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{reg} + \mathcal{L}_{future}$

- **Classification loss**: Focal Loss supervises positive/negative classification of event proposals.
- **Regression loss**: DIoU Loss supervises temporal boundary regression for positive proposals.
- **Future prediction loss**: Focal Loss for imminence prediction + L1 loss for start time offset regression.

Hyperparameters: $a=-4$, $b=4$, $L=8$ (8 scales), short-term window length 8, total memory size 64. AdamW optimizer is used with learning rates of 0.002 / 0.0001 / 0.001 across the three datasets.

## Key Experimental Results

### Main Results (Tables)

Performance and latency comparison on TACoS:

| Method | R0.5@1 | R0.7@1 | R0.5@5 | R0.7@5 | SD (Start Delay) | ED (End Delay) |
|--------|--------|--------|--------|--------|------------------|----------------|
| Gan et al. (OnVTG baseline) | 29.74 | 19.07 | 48.11 | 31.19 | 1.42s | 1.38s |
| HAT (online action localization) | 34.15 | 14.53 | 51.16 | 34.98 | 20.07s | -1.98s |
| **Ours w/ FP** | **37.44** | **27.32** | **57.49** | **44.44** | **-1.28s** | **-3.78s** |
| **Ours w/o FP** | **44.19** | **30.87** | **68.96** | **52.69** | 22.38s | -4.26s |

ActivityNet Captions:

| Method | R0.5@1 | R0.7@1 | R0.5@5 | R0.7@5 | SD | ED |
|--------|--------|--------|--------|--------|----|----|
| Gan et al. | 25.48 | 12.56 | 53.77 | 33.70 | 2.13s | 1.49s |
| **Ours w/ FP** | **42.89** | **24.49** | **67.82** | **51.74** | **-1.58s** | -10.96s |
| **Ours w/o FP** | **45.29** | **26.25** | **76.24** | **62.14** | 41.10s | -10.89s |

MAD:

| Method | R0.3@5 | R0.5@5 | R0.3@50 | R0.5@50 | SD | ED |
|--------|--------|--------|---------|---------|----|----|
| Gan et al. | 4.71 | 2.00 | 16.34 | 7.80 | 0.13s | 1.52s |
| **Ours w/ FP** | **9.84** | **6.43** | 16.67 | **12.18** | 0.64s | **-1.10s** |
| **Ours w/o FP** | **15.76** | **11.07** | **37.84** | **29.21** | 3.60s | -1.45s |

### Ablation Study (Table)

Component ablation on TACoS:

| Event Memory | Dynamic Size | Adaptive Update | Future Prediction | R0.5@1 | R0.7@1 | SD | ED |
|---|---|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | ✗ | 34.88 | 23.47 | 18.36s | -3.17s |
| ✔ | | | | 41.61 | 29.84 | 20.59s | -4.18s |
| ✔ | ✔ | | | 43.97 | 30.54 | 22.41s | -4.16s |
| ✔ | ✔ | ✔ | | 44.19 | 30.87 | 22.38s | -4.26s |
| ✔ | ✔ | ✔ | ✔ | 37.44 | 27.32 | -1.28s | -3.78s |

### Key Findings

- **Event memory is the most critical component**: Introducing event memory improves R0.5@1 from 34.88 to 41.61 (+6.73%), as it enables the construction of proposals at diverse durations and the retrieval of long-term historical information.
- Dynamic size allocation contributes an additional +2.36%, and adaptive update contributes +0.22%.
- The future prediction branch significantly reduces start-time latency (from 22.38s to −1.28s, i.e., proactive prediction) at the cost of approximately 6.75% in R0.5@1—reflecting an inherent accuracy–latency tradeoff.
- Compared to Gan et al., the w/o FP variant achieves +14.45% R0.5@1 on TACoS and +11.05% R0.3@5 on MAD, representing substantial gains.
- Inference speed reaches 614.5 FPS on MAD, satisfying real-time requirements; it is only 17% slower than the baseline while achieving 3.3× its performance.
- Hierarchically structured memory theoretically captures event information spanning $2^K$ time steps with a memory of size $K$, far exceeding the $K$-frame coverage of frame-level memory.

## Highlights & Insights

- **Elegant hierarchical event memory design**: Small-scale memory stores recent fine-grained information while large-scale memory retains long-term coarse-grained information, naturally accommodating target events of varying durations.
- **Segment-tree structure** efficiently generates multi-scale event proposals, avoiding the quadratic growth in proposal count that afflicts traditional methods as video length increases.
- **Future prediction branch** provides a flexible accuracy–latency tradeoff, allowing practitioners to enable or disable it based on application requirements.
- Dynamic memory size allocation is a data-driven design—memory capacity is distributed according to positive-sample frequency, ensuring efficient utilization of limited memory.

## Limitations & Future Work

- Enabling the future prediction branch leads to a notable accuracy drop (−6.75% R0.5@1 on TACoS), indicating that low latency and high accuracy remain difficult to achieve simultaneously.
- The method relies on pre-extracted features from C3D/CLIP and does not consider the potential gains from end-to-end training.
- The similarity threshold $\delta$ in adaptive memory update is a hyperparameter that may require dataset-specific tuning.
- The method handles single-query localization only and does not address simultaneous multi-query grounding scenarios.
- Absolute performance on MAD remains low (R0.5@5 of only 11.07%), indicating that grounding in long movie settings remains highly challenging.

## Related Work & Insights

- Gan et al. is the sole prior work on OnVTG; the shift from frame-level to event-level prediction proposed in this paper represents a fundamental advancement.
- Online action localization methods such as OAT and HAT also employ proposals but rely on frame-level memory and do not account for start-time latency.
- The hierarchical memory concept is generalizable to other long-video understanding tasks, such as video question answering and video summarization.
- The future prediction idea parallels trajectory prediction in autonomous driving—estimating the onset of an event before it occurs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of hierarchical event memory, segment-tree-based event proposals, and future prediction constitutes a strong and well-integrated set of contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation spans three datasets with multiple baselines, ablation studies, accuracy–latency curves, and speed comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, method description is systematic, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ The paper provides a systematic solution for the emerging OnVTG task with substantial performance improvements over prior work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)
- [\[ICCV 2025\] OVG-HQ: Online Video Grounding with Hybrid-modal Queries](ovg-hq_online_video_grounding_with_hybrid-modal_queries.md)
- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
