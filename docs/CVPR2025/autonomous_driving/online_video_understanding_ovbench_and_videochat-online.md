---
title: >-
  [Paper Note] Online Video Understanding: OVBench and VideoChat-Online
description: >-
  [CVPR 2025][Autonomous Driving][Online video understanding] This work advances online video understanding from three perspectives: evaluation benchmarks, model architectures, and training strategies. It proposes OVBench (an online video QA benchmark containing 16 subtasks across 6 task types), designs a Pyramid Memory Bank (PMB) to efficiently compress streaming video information, and builds the 4B-parameter VideoChat-Online model through progressive offline-to-online trainin…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Online video understanding"
  - "streaming video"
  - "pyramid memory bank"
  - "spatiotemporal awareness"
  - "benchmark"
date: 2026-05-08
content_hash: e011e58aa2a64873
---

# Online Video Understanding: OVBench and VideoChat-Online

**Conference**: CVPR 2025  
**arXiv**: [2501.00584](https://arxiv.org/abs/2501.00584)  
**Code**: [https://videochat-online.github.io/](https://videochat-online.github.io/)  
**Area**: Autonomous Driving  
**Keywords**: Online video understanding, streaming video, pyramid memory bank, spatiotemporal awareness, benchmark

## TL;DR
This work advances online video understanding from three perspectives: evaluation benchmarks, model architectures, and training strategies. It proposes OVBench (an online video QA benchmark containing 16 subtasks across 6 task types), designs a Pyramid Memory Bank (PMB) to efficiently compress streaming video information, and builds the 4B-parameter VideoChat-Online model through progressive offline-to-online training, outperforming a 7B offline model by 4.2% on OVBench.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have made significant progress in offline video understanding, but real-world applications (autonomous driving, AR glasses, human-computer interaction) require real-time processing of continuous online video streams. Existing models and benchmarks are mostly oriented towards offline scenarios.

**Limitations of Prior Work**: (1) Existing video benchmarks (such as MVBench, VideoMME) evaluate in offline modes, which fails to reflect the unique requirements of online scenarios—time-dependent context, multi-temporal reasoning across past/present/future, and real-time spatiotemporal interactions; (2) current online models (Flash-VStream, VideoLLM-Online) lack sound architectural designs to balance spatial details and temporal spans; (3) there are no dedicated training strategies for online video.

**Key Challenge**: Online video streaming generates an infinite amount of visual information. Models must retain crucial information and discard redundant information similarly to human cognition, while maintaining real-time responsiveness.

**Goal**: To build a comprehensive research framework for online video understanding, spanning evaluation benchmarks, model architectures, and training paradigms.

**Key Insight**: Divide the temporality of online videos into three dimensions: past, present, and future, and systematically design evaluation and training schemes based on six defined core capabilities (spatial awareness, temporal awareness, spatiotemporal awareness, past memory, temporal hallucination verification, and future prediction).

**Core Idea**: Implement progressive spatial-temporal abstraction using a pyramid-shaped multi-layered memory bank—retaining high-resolution spatial details for recent frames and compressing distant frames into low-resolution temporal summaries—coupled with an offline-to-online curriculum learning strategy.

## Method

### Overall Architecture
VideoChat-Online is built upon InternVL2-4B (InternViT-300M visual encoder + Phi-3 language model). Streaming video inputs are compressed by the Pyramid Memory Bank (PMB) before being fed into the LLM. PMB contains multi-layered queues, each with different sampling rates and resolutions: recent frames retain full spatial details, while distant frames progressively reduce spatial resolution while maintaining temporal coverage. When a layer is full, the evicted frames (retaining the least similar ones via adaptive frame eviction) are passed to the next layer after resolution downsampling. Training adopts a progressive "offline $\rightarrow$ online" paradigm.

### Key Designs

1. **Pyramid Memory Bank (PMB)**:

    - **Function**: Balances spatial and temporal information within a limited visual token budget.
    - **Mechanism**: Divides memory into $n$ layers $\{m_i\}$, each with a sampling rate $r_i$ (increasing layer-by-layer) and resolution $\text{Res}_i = \text{Res}_1 / \beta^{i-1}$ (decreasing layer-by-layer, $\beta=2$). Three operations: (1) Streaming write: receives frames at the sampling rate until capacity $C_i$ is full; (2) Frame eviction and propagation: identifies the adjacent joint frame pair with the highest cosine similarity, evicts the older frame, downsamples it via average pooling, and propagates it to $m_{i+1}$; (3) Readout: reads frames from all layers in chronological order. Practical configuration: 3-layer memory, sampling rates $\{1, 2, 8\}$, token counts per frame $\{256, 64, 16\}$.
    - **Design Motivation**: Spatial details of recent frames are crucial for current perception (requiring high resolution), while distant frames primarily provide temporal context (where low resolution suffices). The similar frame eviction strategy effectively removes redundancy.

2. **KVCache Compatibility Design**:

    - **Function**: Obviates the need for full recalculation during memory updates.
    - **Mechanism**: Frame tokens are written into the KVCache concurrently as they enter the memory bank. When a frame is evicted, all KVCache entries after the evicted frame's timestamp are removed: $\text{KVCache} \leftarrow \text{KVCache} \setminus \{t_i | t_i > \min(t_{f_a}, t_{f_b})\}$.
    - **Design Motivation**: Existing memory compression methods (e.g., MovieChat, FlashVStream) require re-processing the entire compressed memory at each update, creating computational bottlenecks. PMB synchronizes with the KVCache to enable highly efficient incremental updates.

3. **Progressive Offline-to-Online Training**:

    - **Function**: Progressively enhances the model's online spatiotemporal understanding capability.
    - **Mechanism**: Collects 96K high-quality spatiotemporal annotated data (covering dense captioning, step localization, object tracking, etc.) and converts them into an interleaved dialogue format—carefully placing questions along the timeline to distinguish past, present, and future tenses. Training first establishes foundational video understanding on offline video data, followed by joint fine-tuning with online data.
    - **Design Motivation**: Training directly on online data makes it difficult to simultaneously optimize spatiotemporal understanding and time/box prediction. A curriculum learning strategy is more stable.

### Loss & Training
Standard autoregressive language modeling loss. Training data mixture: offline data (VideoChat2-IT, STAR, PerceptionTest) + image data (ShareGPT4V/4o) + multi-image data (LLaVA-OneVision) + online spatiotemporal data (96K). Inputs are sampled at 1 fps, capped at 64 frames.

## Key Experimental Results

### Main Results

| Model | Params | Setup | FP | THV | PM | SP | STP | TP | Avg |
|------|--------|------|-----|-----|-----|-----|-----|-----|------|
| Qwen2-VL | 7B | Sliding Window | 49.5 | 52.5 | 57.2 | 35.3 | 49.4 | 35.8 | 49.7 |
| Flash-Vstream | 7B | Streaming | 29.5 | 47.3 | 28.3 | 24.7 | 21.4 | 27.4 | 31.2 |
| **VideoChat-Online** | **4B** | **Streaming** | **46.8** | **61.4** | **55.7** | **54.1** | **48.5** | **56.9** | **54.9** |

### Ablation Study

| Configuration | OVBench Avg | Description |
|------|-------------|------|
| No PMB (Fixed Sliding Window) | 47.2 | Lacks long-range memory |
| Single-layer memory | 49.8 | No spatial-temporal stratification |
| Offline training only | 48.5 | Lacks online spatiotemporal data |
| **Full Model** | **54.9** | PMB + progressive training |

### Key Findings
- VideoChat-Online (4B) in the streaming setup outperforms the 7B offline model Qwen2-VL (49.7%) with 54.9% while having fewer parameters.
- It outperforms the best streaming competitor Flash-Vstream (31.2%) by 23.7 percentage points, showing that current online models are severely deficient in architecture and training.
- PMB provides the largest gains in past memory (PM) and temporal hallucination verification (THV) tasks, as these tasks rely on long-range temporal information.
- In temporal perception (TP) tasks, the improvement in the Object Existence State subtask is the most significant (69.9% vs. the next best 46.9%), demonstrating that PMB's frame eviction mechanism effectively retains critical temporal information.

## Highlights & Insights
- **Systematic Online Video Research**: A complete system ranging from benchmarks and architectures to training, filling the gap in online video understanding research.
- **Progressive Spatial-Temporal Abstraction**: The design of the Pyramid Memory Bank aligns with human cognition—memorizing details for recent events and retaining outlines for distant ones—making it an intuitive and highly efficient solution.
- **4B Outperforms 7B**: Proves that targeted architectural design and training strategies are more important than blindly scaling up parameters.

## Limitations & Future Work
- OVBench is currently mainly adapted from existing datasets, which has limited scenario coverage (e.g., lacking conversational interactions, multimodal inputs, etc.).
- PMB's frame eviction strategy is based on adjacent frame similarity, potentially evicting key frames that are important but similar to neighboring frames.
- The 1 fps sampling rate may be insufficient for understanding fast-paced actions.
- Future work could explore fine-grained attention mechanisms to replace simple pooling-based resolution reduction.

## Related Work & Insights
- **vs Flash-Vstream**: Flash-Vstream uses a learnable memory module to compress streaming information but lacks a multi-layer design and dedicated training data, resulting in performance far below VideoChat-Online.
- **vs VideoLLM-Online**: This pioneering work almost fails on OVBench (9.6%) due to the limitation of single-frame visual token inputs.
- The hierarchical compression concept of the Pyramid Memory Bank can be extended to other scenarios requiring long-sequence processing (e.g., long document understanding).

## Rating
- Novelty: ⭐⭐⭐⭐ PMB is well-designed, and OVBench fills a critical gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive benchmark evaluations, offline/online dual comparisons, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and systematic definition of tasks.
- Value: ⭐⭐⭐⭐⭐ Provides a comprehensive research infrastructure for online video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](../../NeurIPS2025/autonomous_driving/streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)
- [\[CVPR 2025\] InteractionMap: Improving Online Vectorized HDMap Construction with Interaction](interactionmap_improving_online_vectorized_hdmap_construction_with_interaction.md)
- [\[CVPR 2025\] ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](recondreamer_crafting_world_models_for_driving_scene_reconstruction_via_online_r.md)
- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](../../ICCV2025/autonomous_driving/embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)

</div>

<!-- RELATED:END -->
