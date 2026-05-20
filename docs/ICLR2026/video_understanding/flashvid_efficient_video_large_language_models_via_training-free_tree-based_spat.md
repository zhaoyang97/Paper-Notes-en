---
title: >-
  [Paper Note] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging
description: >-
  [ICLR 2026][Video Understanding][visual token compression] This paper proposes FlashVID, a training-free inference acceleration framework for video large language models (VLLMs) that jointly models spatial and temporal r…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "visual token compression"
  - "spatiotemporal redundancy"
  - "token merging"
  - "video large language models"
  - "training-free acceleration"
date: 2026-05-08
content_hash: d3247dbeef3880c7
---

# FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging

**Conference**: ICLR 2026
**arXiv**: [2602.08024](https://arxiv.org/abs/2602.08024)  
**Code**: [https://github.com/Fanziyang-v/FlashVID](https://github.com/Fanziyang-v/FlashVID)  
**Area**: Video Understanding / LLM Efficiency / Multimodal VLM
**Keywords**: visual token compression, spatiotemporal redundancy, token merging, video large language models, training-free acceleration

## TL;DR
This paper proposes FlashVID, a training-free inference acceleration framework for video large language models (VLLMs) that jointly models spatial and temporal redundancy via Tree-based Spatiotemporal Token Merging (TSTM). Retaining only 10% of visual tokens, FlashVID preserves 99.1% of LLaVA-OneVision's performance and enables a 10× increase in input frames for Qwen2.5-VL.

## Background & Motivation

**Background**: VLLMs achieve strong performance on video understanding tasks, but require processing a large number of visual tokens (e.g., 32 frames × 196 tokens/frame = 6,272 tokens). The quadratic complexity of attention with respect to sequence length imposes substantial inference overhead.

**Limitations of Prior Work**: Existing acceleration methods (FastV, VisionZip, PruneVID) typically compress spatial and temporal redundancy independently, neglecting the intrinsic coupling between spatiotemporal relationships. In particular, Temporal Token Merging (TTM) assumes that semantically similar tokens in adjacent frames reside at the same spatial positions, which fails when objects move, deform, or scale across frames.

**Key Challenge**: The fixed spatial correspondence assumed by TTM does not hold in dynamic videos — the most relevant visual features across adjacent frames may not share the same spatial location, and forced merging under this assumption introduces noise and distorts video representations.

**Goal**: How can spatial and temporal redundancy be jointly modeled for efficient compression without any training, while remaining adaptive to the dynamic nature of video content?

**Key Insight**: The observation that spatial and temporal redundancy are coupled (redundant regions in one frame tend to persist across multiple frames), and that temporal redundancy is not tied to fixed spatial positions.

**Core Idea**: Replace fixed spatial position-based cross-frame token correspondence with a hierarchical spatiotemporal redundancy tree that matches the most similar — rather than co-located — tokens for merging.

## Method

### Overall Architecture
FlashVID comprises two complementary modules: (1) ADTS (Attention and Diversity-based Token Selection), which selects the most representative and diverse token subset from each frame; and (2) TSTM (Tree-based Spatiotemporal Token Merging), which constructs cross-frame redundancy trees over the remaining tokens and aggregates redundant ones. The final output consists of the union of retained important tokens and TSTM-aggregated tokens.

### Key Designs

1. **Tree-based Spatiotemporal Token Merging (TSTM)**:

    - **Function**: Constructs cross-frame redundancy trees to aggregate semantically similar tokens into a single representation.
    - **Mechanism**: A cosine similarity matrix is computed between tokens in adjacent frames. Each token is connected to its most similar counterpart in the previous frame (rather than its co-located one), provided the similarity exceeds a threshold. This progressively constructs cross-frame redundancy trees, and all tokens within each tree are aggregated via mean pooling.
    - **Design Motivation**: Unlike TTM's fixed spatial correspondence, TSTM permits free spatial matching, enabling it to capture positional shifts caused by object motion. Experiments show that, under the same threshold, TSTM merges more tokens than TTM and achieves higher average similarity at merge time.

2. **Attention and Diversity-based Token Selection (ADTS)**:

    - **Function**: Selects the most informative and feature-diverse token subset from each frame.
    - **Mechanism**: Token selection is formulated as a Max-Min Diversity Problem (MMDP) solved over a cosine distance matrix, calibrated by two terms: (a) CLS attention weights — identifying tokens most attended to by the encoder; and (b) event relevance — computed by global average pooling to obtain frame-level embeddings and measuring each token's correlation with the overall video event.
    - **Design Motivation**: Solving MMDP alone guarantees diversity but may omit the most important tokens. Incorporating attention and event relevance calibration ensures both diversity and informativeness.

3. **Two-Stage Compression Pipeline**:

    - Stage 1: ADTS selects important tokens from each frame (retained without modification).
    - Stage 2: Remaining tokens are passed to TSTM to construct redundancy trees and perform aggregation.
    - Final output: The union of aggregated tokens and retained tokens from all frames.

### Loss & Training
No training is required. FlashVID operates as a plug-and-play module that can be directly integrated into existing VLLMs.

## Key Experimental Results

### Main Results
Average performance across 5 video understanding benchmarks on LLaVA-OneVision (32 frames):

| Method | Retention Ratio | VideoMME | EgoSchema | LongVideoBench | MVBench | Avg. | Relative Acc. |
|--------|----------------|----------|-----------|----------------|---------|------|---------------|
| Vanilla | 100% | 58.5 | 60.3 | 56.6 | 58.3 | 58.4 | 100.0% |
| FastV | 10% | 51.5 | 51.2 | 52.3 | 52.3 | 51.8 | 88.7% |
| VisionZip | 10% | 51.6 | 55.6 | 50.1 | 50.3 | 51.9 | 88.9% |
| FastVID | 10% | 55.5 | 56.1 | 55.5 | 57.7 | 56.2 | 96.2% |
| **FlashVID** | **10%** | **57.2** | **59.5** | **56.0** | **57.7** | **57.9** | **99.1%** |

### Frame Scaling Experiment on Qwen2.5-VL

| Setting | Frames | VideoMME | MLVU | Relative Gain |
|---------|--------|----------|------|---------------|
| Vanilla | 16 | 65.7 | 67.6 | baseline |
| FlashVID | 160 | 69.9 | 74.5 | +8.6% |

### Ablation Study

| Configuration | VideoMME | EgoSchema | Avg. |
|---------------|----------|-----------|------|
| Full FlashVID (ADTS+TSTM) | 57.2 | 59.5 | 57.9 |
| w/o TSTM (ADTS only) | 56.2 | 58.0 | 56.7 |
| w/o ADTS (TSTM only) | 56.5 | 59.1 | 57.0 |
| TTM replacing TSTM | 55.5 | 57.8 | 56.5 |

### Key Findings
- TSTM contributes the most; adding TSTM on top of ADTS alone yields approximately 1.2 points of improvement.
- Replacing TSTM with TTM leads to a notable performance drop, validating the importance of dynamic spatial correspondence.
- At 10% token retention, FlashVID preserves 99.1% of performance, substantially outperforming FastV (88.7%) and VisionZip (88.9%).
- Extending the input to 10× more frames on Qwen2.5-VL under the same computational budget yields an 8.6% performance gain.

## Highlights & Insights
- **Tree-based Dynamic Matching**: The core insight is simple yet effective — relevant tokens in adjacent frames need not be co-located, and replacing fixed-position matching with globally optimal similarity-based matching is sufficient. This idea transfers readily to any task involving cross-frame correspondence.
- **Training-free Plug-and-Play**: No retraining is required, making FlashVID theoretically compatible with any VLLM and offering high practical engineering value.
- **Frame Scaling Application**: Computation saved through token compression is "exchanged" for additional input frames, elegantly converting efficiency gains into capability improvements.

## Limitations & Future Work
- The merging threshold is a hyperparameter; the optimal value may vary across videos, and adaptive threshold strategies warrant further exploration.
- For very long videos and high-resolution inputs, the cost of constructing the similarity matrix in TSTM is non-negligible.
- Compression is applied only at the inference stage; training-time efficiency is not addressed.
- The differential compression behavior between highly static scenes (extremely high redundancy) and highly dynamic scenes (very low redundancy) has not been thoroughly analyzed.

## Related Work & Insights
- **vs. FastV (Chen et al., 2024)**: FastV prunes tokens inside the LLM using text-to-visual attention, representing an Inner-LLM approach; FlashVID employs a hybrid strategy.
- **vs. PruneVID (Huang et al., 2025)**: PruneVID also performs spatiotemporal merging but relies on fixed spatial position correspondence via TTM; FlashVID uses dynamic matching instead.
- **vs. ToMe (Bolya et al., 2023)**: ToMe is the seminal work on image token merging; FlashVID extends this paradigm to the video spatiotemporal domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The tree-based spatiotemporal merging concept is concise and effective, though the overall framework combines existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluations span 3 VLLMs × 5 benchmarks × multiple compression ratios, yielding a comprehensive assessment.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, figures are intuitive, and algorithm descriptions are complete.
- Value: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, and negligible performance loss make this highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](../../CVPR2026/video_understanding/token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[CVPR 2026\] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention](../../CVPR2026/video_understanding/unified_spatiotemporal_token_compression_for_video-llms_at_ultra-low_retention.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[AAAI 2026\] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](../../AAAI2026/video_understanding/listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)
- [\[ICLR 2026\] Video-KTR: Enhancing Video Reasoning via Key Token Attribution](video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)

</div>

<!-- RELATED:END -->
