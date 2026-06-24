---
title: >-
  [Paper Note] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs
description: >-
  [ICCV 2025][Video Understanding][Video Frame Selection] This paper proposes Q-Frame, a training-free plug-and-play framework for video frame selection and multi-resolution adaptation. By leveraging CLIP cross-modal matching and the Gumbel-Max trick, Q-Frame achieves query-aware frame selection, enabling Video-LLMs to process more informative frames under the same computational budget. It achieves significant performance gains on three benchmarks: MLVU, LongVideoBench…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Video Frame Selection"
  - "Multi-Resolution Adaptation"
  - "Video-LLM"
  - "CLIP"
  - "Gumbel-Max Sampling"
date: 2026-05-08
content_hash: a9396dacf104731f
---

# Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs

**Conference**: ICCV 2025
**arXiv**: [2506.22139](https://arxiv.org/abs/2506.22139)  
**Code**: N/A  
**Area**: Video Understanding / Video Large Language Models
**Keywords**: Video Frame Selection, Multi-Resolution Adaptation, Video-LLM, CLIP, Gumbel-Max Sampling

## TL;DR

This paper proposes Q-Frame, a training-free plug-and-play framework for video frame selection and multi-resolution adaptation. By leveraging CLIP cross-modal matching and the Gumbel-Max trick, Q-Frame achieves query-aware frame selection, enabling Video-LLMs to process more informative frames under the same computational budget. It achieves significant performance gains on three benchmarks: MLVU, LongVideoBench, and Video-MME.

## Background & Motivation

Video-LLMs face a fundamental tension between the large number of video frames and the limited context length. A 3-minute video at 24 fps contains approximately 4,320 frames, yet VideoLLaMA2 supports only 2,000 tokens and VILA-V1.5 approximately 4,000 tokens.

Uniform frame sampling suffers from three key issues:

**Temporal Sparsity**: A fixed number of frames becomes increasingly sparse in long videos, disrupting temporal continuity and causing critical transitions to be missed.

**Query Agnosticism**: The same set of frames is used for every question, regardless of query-specific requirements.

**One-Size-Fits-All Resolution**: All frames are processed at the same resolution, causing detail loss in high-information-density frames through downsampling, while wasting computation on low-density frames.

Existing improvements—such as Top-K semantic retrieval and frame ranking—still fail to capture complex temporal dependencies and do not explore dynamic resolution adaptation.

## Method

### Overall Architecture

Q-Frame consists of three components:
1. **CQR (Cross-modal Query Retrieval)**
2. **QFS (Query-aware Frame Selection)**
3. **MRA (Multi-Resolution Adaptation)**

### Key Designs

1. **Cross-modal Query Retrieval (CQR)**:

    - Uniformly subsamples $T$ frames from the original video as candidate frame sequence $\mathcal{F}$
    - Maps video frames and text queries to a shared semantic space using pretrained CLIP/Long-CLIP
    - Computes per-frame similarity to the query: $I = QF^T \in \mathbb{R}^{1 \times T}$
    - Adopts Long-CLIP to overcome the 77-token text encoder limit of the original CLIP

2. **Query-aware Frame Selection (QFS)**:

    - Since CLIP is primarily trained on image-text pairs, it lacks temporal relationship modeling for video
    - Introduces a probability-guided sampling strategy based on the **Gumbel-Max trick**
    - Converts matching scores to a probability distribution: $\pi = \text{Softmax}(I/\tau)$, where $\tau$ is the temperature parameter
    - Injects independent Gumbel noise to perturb log-probabilities: $p = \log\pi + g$, where $g = -\log(-\log\epsilon)$
    - Selects Top-K frames: $\text{idx}^{\text{select}} = \{i | \text{rank}(i) \leq K\}$
    - Core advantage: **exploration–exploitation balance**—highly relevant frames are selected with higher probability, while random noise ensures diversity

3. **Multi-Resolution Adaptation (MRA)**:

    - Allocates one of three resolution levels based on query relevance
    - High-relevance frames ($\text{rank} \leq K$) → high resolution $r^{(3)}$
    - Mid-relevance frames ($K < \text{rank} \leq M$) → medium resolution $r^{(2)}$
    - Low-relevance frames ($M < \text{rank} \leq N$) → low resolution $r^{(1)}$
    - Resolution relationship: $r^{(1)} = 4r^{(2)} = 16r^{(3)}$ (high-resolution frames produce 16× more visual tokens)
    - Token budget constraint: $K + M/4 + N/16 = 8$ (equivalent to the computation of 8 high-resolution frames)

### Loss & Training

**No training is required.** Q-Frame operates entirely at inference time:
- Similarity scores are computed using pretrained Long-CLIP
- Gumbel-Max sampling requires no gradient computation
- The framework is plug-and-play and compatible with any Video-LLM, including both open-source models and closed-source APIs

## Key Experimental Results

### Main Results

Evaluated on MLVU, LongVideoBench, and Video-MME. Frames are selected from 128 candidates, with a budget of 8 frames (or equivalent token budget).

| Model | #Frames | MLVU | LongVideoBench | Video-MME (wo/w sub) |
|-------|---------|------|----------------|----------------------|
| VILA-V1.5 | 8 | 46.3 | 47.1 | 47.5 / 50.0 |
| +Frame-Voyager | 8 | 49.8 | - | 50.5 / 53.6 |
| **+Q-Frame** | 8 | **54.4** | **51.6** | **50.7 / 55.0** |
| Qwen2-VL | 8 | 56.9 | 53.5 | 53.7 / 59.4 |
| **+Q-Frame** | 4+8+32 | **65.4** | **58.4** | **58.3 / 61.8** |
| GPT-4o | 8 | 28.6 | 53.3 | 61.9 / 64.5 |
| **+Q-Frame** | 8 | 29.3 | **58.6** | **63.8 / 66.5** |

Q-Frame consistently improves performance across all models and benchmarks. Qwen2-VL + Q-Frame achieves state-of-the-art on MLVU (65.4%), while GPT-4o + Q-Frame achieves state-of-the-art on the other two benchmarks.

### Ablation Study

| Sampling Strategy | Resolution | Acc (%) |
|------------------|------------|---------|
| Uniform + Fixed | - | 53.5 |
| CLIP Top-K + Fixed | - | 56.0 |
| QFS + Fixed | - | 57.6 |
| QFS + MRA | - | **58.4** |

Ablation on frame resolution allocation (token budget equivalent to 8 high-resolution frames):

| K (High) | M (Mid) | N (Low) | Tokens/Video | Acc (%) |
|----------|---------|---------|-------------|---------|
| 8 | 0 | 0 | 2265 | 57.6 |
| 6 | 4 | 16 | 2313 (+2%) | 58.3 |
| 4 | 8 | 32 | 2345 (+3.5%) | **58.4** |
| 4 | 4 | 48 | 2370 (+4.6%) | 57.4 |

### Key Findings

- Q-Frame yields larger gains on longer videos: Qwen2-VL improves by +10.5% on 15m–60m videos
- Among 6 video task categories, Reasoning, Recognition, and Counting benefit the most
- Temperature $\tau=0.8$ is optimal; lower values lead to over-exploitation, higher values introduce excessive randomness
- The optimal resolution configuration is 4 high + 8 medium + 32 low resolution frames
- Increasing low-resolution frames to 48 degrades performance, indicating that more frames do not always help
- Q-Frame adds only 3–5% token overhead, with negligible impact on overall inference time

## Highlights & Insights

- **Elegant training-free design**: The Gumbel-Max trick converts deterministic ranking into probabilistic sampling, balancing diversity and relevance
- **Model agnosticism**: Effective for both open-source (VILA-V1.5, Qwen2-VL) and closed-source (GPT-4o) models
- **Multi-resolution innovation**: The first work to introduce dynamic resolution adaptation for Video-LLMs, retaining more information under the same computational budget
- **Rigorous experimental design**: Both fixed-frame-count and fixed-token-budget settings provide comprehensive and fair comparisons

## Limitations & Future Work

- Relies on CLIP's image-text matching capability; frame selection may be inaccurate for queries requiring complex temporal reasoning (e.g., event ordering)
- Gumbel-Max sampling introduces stochasticity, making results non-deterministic (though this is generally an advantage)
- MRA is only applicable to models supporting dynamic resolution input (e.g., Qwen2-VL), not all Video-LLMs
- The candidate frame count $T=128$ is fixed, which may be insufficient for very long videos
- Explicit modeling of temporal causal relationships is absent; case analysis reveals persistent difficulties on temporal reasoning tasks

## Related Work & Insights

- KeyVideoLLM's Top-K semantic retrieval and Frame-Voyager's loss-driven optimization serve as comparison baselines
- Qwen2-VL's native dynamic resolution framework provides the underlying support for MRA
- The Gumbel-Max trick originates from discrete probabilistic sampling theory; its application to frame selection represents an elegant cross-domain transfer
- Takeaway: Intelligent input-side filtering during large model inference may be more efficient than model-side improvements

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of Gumbel-Max frame selection and multi-resolution adaptation is original, though individual components are not entirely novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, three baseline models, multi-dimensional ablations, cross-video-length analysis, and sub-task analysis—very comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ Clear exposition with intuitive figures and tables
- **Value**: ⭐⭐⭐⭐ The training-free plug-and-play design is highly practical and directly applicable to Video-LLM deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](../../CVPR2025/video_understanding/m-llm_based_video_frame_selection_for_efficient_video_understanding.md)
- [\[CVPR 2025\] Progress-Aware Video Frame Captioning](../../CVPR2025/video_understanding/progress-aware_video_frame_captioning.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](../../CVPR2026/video_understanding/divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [\[ICCV 2025\] Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos](beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)

</div>

<!-- RELATED:END -->
