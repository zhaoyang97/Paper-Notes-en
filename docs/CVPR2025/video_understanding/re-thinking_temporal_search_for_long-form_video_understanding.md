---
title: >-
  [Paper Note] T*: Re-thinking Temporal Search for Long-Form Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Long-form Video Understanding] Proposes T*, a lightweight temporal search framework that reformulates expensive temporal search as a spatial search problem. It iteratively localizes keyframes in both temporal and spatial dimensions via an adaptive zooming mechanism. Combined with LV-Haystack, the first large-scale benchmark for long-form video keyframe search, it significantly improves the performance of existing VLMs in long-form video unders…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Long-form Video Understanding"
  - "Keyframe Search"
  - "Temporal Localization"
  - "Adaptive Zooming"
  - "LV-Haystack"
date: 2026-05-08
content_hash: 68503753779fd7d3
---

# T*: Re-thinking Temporal Search for Long-Form Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2504.02259](https://arxiv.org/abs/2504.02259)  
**Code**: [https://github.com/longvideohaystack/tstar](https://github.com/longvideohaystack/tstar)  
**Area**: Video Understanding  
**Keywords**: Long-form Video Understanding, Keyframe Search, Temporal Localization, Adaptive Zooming, LV-Haystack

## TL;DR

Proposes T*, a lightweight temporal search framework that reformulates expensive temporal search as a spatial search problem. It iteratively localizes keyframes in both temporal and spatial dimensions via an adaptive zooming mechanism. Combined with LV-Haystack, the first large-scale benchmark for long-form video keyframe search, it significantly improves the performance of existing VLMs in long-form video understanding.

## Background & Motivation

**Background**: Long-form video understanding is a key challenge in computer vision. Current state-of-the-art long-context Vision-Language Models (VLMs), such as GPT-4o and LLaVA-OneVision, face frame limits (typically only 32-128 frames as input) when processing long-form videos, whereas real-world long videos can contain tens of thousands of frames. How to select a small subset of the most relevant keyframes from these frames directly impacts model performance.

**Limitations of Prior Work**: Existing temporal search methods perform poorly—on a subset of LongVideoBench, the temporal F1 score of the current SOTA keyframe selection method is only **2.1%**, indicating that finding the correct keyframes is nearly impossible. The main reasons are: (1) uniform sampling completely ignores query content; (2) existing search methods treat temporal search purely as a temporal-dimensional problem, failing to leverage the powerful capabilities of spatial image search.

**Key Challenge**: Needle-in-a-haystack keyframe search in long videos requires precise localization along the temporal dimension, which lacks efficient temporal localization leverage. In contrast, visual localization technologies in the spatial dimension (such as object detection and visual grounding) are already highly mature and efficient.

**Goal**: (1) Formulate the task definition and evaluation benchmark of keyframe search in long-form videos; (2) design an efficient keyframe search framework.

**Key Insight**: Reduce the dimensionality of temporal search to spatial search—first perform coarse sampling temporally, then detect query-relevant content in the spatial dimension of each frame. Once frames matching the content are found, zoom-in temporally and refine iteratively.

**Core Idea**: Reformulate temporal search as an iterative process of "spatial detection + temporal zooming-in", leveraging mature visual localization techniques to replace the weak temporal search.

## Method

### Overall Architecture

T* is a plug-and-play keyframe search framework located at the input stage of the VLM. Given a long video and a textual query, T* iteratively executes the following process: (1) uniformly sample frames within the current temporal window; (2) use a VLM to translate the query into a visual grounding description; (3) perform spatial search on the sampled frames using an object detector (e.g., YOLO-World); (4) localize high-response intervals based on detection results; (5) zoom-in temporally within the high-response intervals; (6) repeat until convergence. The final searched keyframes are fed into the VLM for question answering.

### Key Designs

1. **Formalization of the Long Video Haystack Problem**:

    - **Function**: Provides a rigorous problem definition and evaluation framework for long video keyframe search.
    - **Mechanism**: Defines temporal search as finding the minimal set of query-relevant frames (usually 1-5 frames) from tens of thousands of frames. It constructs the LV-Haystack dataset, containing 480 hours of video and 15,092 human-annotated instances, providing fine-grained temporal F1 and search efficiency evaluation metrics. Each instance is labeled with a query and corresponding reference keyframe timestamps.
    - **Design Motivation**: Previously, there was a lack of dedicated evaluation benchmarks for keyframe search quality. Existing long-form video understanding datasets only evaluate the final QA accuracy, failing to isolate search quality from reasoning performance.

2. **Temporal-Spatial Adaptive Zooming-in Mechanism**:

    - **Function**: Iteratively zooms in along both temporal and spatial dimensions to progressively lock onto keyframes and key regions.
    - **Mechanism**: Temporally, it ranks frames based on confidence scores from spatial detection, and selects temporal windows containing high-confidence frames to zoom-in (increasing sampling density in that window). Spatially, it crops out key regions using detection bounding boxes to reduce background distraction. The two dimensions alternate, narrowing down the search space with each iteration. The key to this mechanism is converting the temporal judgment of "whether this time interval contains relevant content" into a spatial judgment of "whether this frame contains relevant objects."
    - **Design Motivation**: Direct search in the temporal dimension is extremely inefficient (F1 is only 2.1%), whereas visual grounding in the spatial dimension (YOLO-World, etc.) is highly mature. This dimension conversion allows leveraging existing mature visual localization models.

3. **Query Grounding & Image Scoring Module**:

    - **Function**: Translates textual queries into descriptions suitable for visual detection and scores the relevance of each frame.
    - **Mechanism**: Uses a VLM (such as GPT-4o or LLaVA) to convert the user's textual query into concrete visual grounding descriptions (e.g., "find a red sofa"). Then, an open-vocabulary object detector (such as YOLO-World or OWL-ViT) is utilized to perform detection on the sampled frames, using the detection confidence as the frame's relevance score. Frames with high scores are considered more likely to be keyframes.
    - **Design Motivation**: Textual queries are often abstract ("What color is the sofa?"), making them unsuitable for direct visual detection. An intermediate step is required to translate them into concrete visual descriptions.

### Loss & Training

T* is a training-free inference framework that does not involve any loss function. All its components (VLMs, object detectors) use off-the-shelf pre-trained models.

## Key Experimental Results

### Main Results: Accuracy Improvement on LongVideoBench XL Subset (32-frame Budget)

| VLM Model | Without Search (Uniform) | + T* | Gain |
|----------|------------------|------|------|
| GPT-4o | 50.5% | **53.1%** | +2.6% |
| LLaVA-OneVision-72B | 56.5% | **62.4%** | +5.9% |
| Qwen-VL | Baseline | Improved Value | +Significant |

### Search Quality on the LV-Haystack Benchmark

| Search Method | Temporal F1 (%) ↑ | Search Cost |
|---------|-------------|---------|
| Uniform Sampling | ~1.0 | Lowest |
| SOTA Search Method | 2.1 | Medium |
| **T*** | **Significant Gain** | Lower |

### Key Findings

- Existing SOTA search methods nearly fail in keyframe localization (2.1% F1), revealing a huge research gap in this direction.
- T* yields significant improvements across different backend VLMs (GPT-4o, LLaVA-OV-72B, Qwen-VL), proving its generalizability.
- The performance gain of LLaVA-OV-72B (+5.9%) is greater than that of GPT-4o (+2.6%), possibly because open-source models are more sensitive to the quality of frame selection.
- Utilizing a stronger spatial detection backend (YOLO-World vs. OWL-ViT) can further improve search quality.
- The number of iterations for adaptive zooming typically converges within 2-4 steps.

## Highlights & Insights

- **Value of Problem Formalization**: Formalizing long-form video keyframe search as the "Long Video Haystack" problem provides a clear task definition and evaluation benchmark for this direction for the first time. The baseline F1 of 2.1% reveals massive room for improvement.
- **Ingenious Dimension Conversion**: Converting temporal search into spatial search is the core insight of this paper. Search in the temporal dimension is difficult to optimize end-to-end, whereas detection/localization technology in the spatial dimension is highly mature. This "dimension reduction" approach is widely inspiring.
- **Plug-and-play Design**: T* does not require fine-tuning of any model and can be directly used in tandem with any VLM, featuring extremely low barriers to practical application.

## Limitations & Future Work

- The current scale of the LV-Haystack dataset (480 hours) is still relatively small for long video research, and domain coverage may not be comprehensive enough.
- T* relies on the quality of the open-vocabulary object detector; when queries involve abstract concepts, actions, or events (rather than concrete objects), spatial detection may fail.
- The query translation step requires extra VLM calls, which increases inference latency.
- Lack of sufficient comparison with end-to-end learned temporal localization methods (such as moment retrieval models).
- The stopping criteria for iterative zooming might need to be adjusted for different scenarios.

## Related Work & Insights

- **vs. Uniform Sampling**: Uniform sampling entirely ignores query content, whereas T*'s query-aware search brings fundamental improvements.
- **vs. VideoAgent**: Methods like VideoAgent also attempt iterative search, but do not reformulate the temporal problem into a spatial one.
- **vs. Moment Retrieval**: Traditional moment retrieval methods require training specialized models, whereas T* is training-free and more generalizable.
- The dimension conversion concept of T* can be transferred to scenarios requiring localization in long sequences, such as audio search and document search.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The dimension conversion concept from temporal search to spatial search is highly novel, and the problem formalization is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple VLM backends with a well-designed LV-Haystack benchmark.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, and the method description is intuitive and easy to understand.
- Value: ⭐⭐⭐⭐⭐ Provides a highly practical solution to the efficiency problem of long video understanding, and the LV-Haystack benchmark will facilitate future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](../../ICCV2025/video_understanding/hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](../../ICCV2025/video_understanding/flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[CVPR 2025\] Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously](video_streaming_thinking_videollms_can_watch_and_think_simultaneously.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)

</div>

<!-- RELATED:END -->
