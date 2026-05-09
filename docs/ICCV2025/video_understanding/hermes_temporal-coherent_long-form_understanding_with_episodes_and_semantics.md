---
title: >-
  [Paper Note] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics
description: >-
  [ICCV 2025][Video Understanding][long-form video understanding] This paper proposes HERMES, a framework comprising two general-purpose modules — the Episodic COmpressor (ECO) and the Semantics reTRiever (SeTR) — that capture episodic memory and semantic information from video respectively. HERMES can serve as a standalone system achieving state-of-the-art performance, or be integrated as plug-and-play components into existing video-language models, simultaneously reducing inference latency by up to 43% and memory consumption by up to 46%.
tags:
  - ICCV 2025
  - Video Understanding
  - long-form video understanding
  - episodic compression
  - semantic retrieval
  - video question answering
  - plug-and-play module
date: 2026-05-08
content_hash: ff348ccdfeea9acd
---

# HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics

**Conference**: ICCV 2025
**arXiv**: [2408.17443](https://arxiv.org/abs/2408.17443)
**Code**: [GitHub](https://github.com/joslefaure/HERMES)
**Area**: Video Understanding
**Keywords**: long-form video understanding, episodic compression, semantic retrieval, video question answering, plug-and-play module

## TL;DR

This paper proposes HERMES, a framework comprising two general-purpose modules — the Episodic COmpressor (ECO) and the Semantics reTRiever (SeTR) — that capture episodic memory and semantic information from video respectively. HERMES can serve as a standalone system achieving state-of-the-art performance, or be integrated as plug-and-play components into existing video-language models, simultaneously reducing inference latency by up to 43% and memory consumption by up to 46%.

## Background & Motivation

Long-form video understanding (spanning minutes to hours) faces three core challenges:

**Temporal complexity**: Processing thousands of frames incurs prohibitive computational costs for existing methods.

**Semantic understanding**: Beyond frame-level events, models must comprehend high-level narrative structures and thematic concepts.

**Memory constraints**: Addressing both challenges simultaneously under limited computational resources is extremely difficult.

Most existing long-video methods are simple extensions of short-video approaches (pooling, 3D convolutions, etc.) and fail to account for the unique properties of long-form content. Inspired by **human cognition**, the authors distinguish two types of information in video:

- **Episodic information**: Concrete, temporally ordered sequences of events (e.g., "teenagers interacting on a baseball field")
- **Semantic information**: High-level themes and concepts that transcend specific moments (e.g., "youth sports culture")

Existing methods typically focus on only one of these two types. The motivation behind HERMES is to **capture both simultaneously**, enabling more comprehensive long-form video understanding.

## Method

### Overall Architecture

HERMES adopts a streaming processing architecture: video frames are fed incrementally in windows into a frozen ViT-G/14 encoder for feature extraction, then processed by ECO (episodic compression) and SeTR (semantic retrieval) respectively. The two representations are concatenated and passed to a frozen LLM (Vicuna-7B) to generate text output.

Formally: $U = G(ECO(V, I), SeTR(V))$, where $U$ denotes the comprehensive understanding and $G$ is an integration function.

### Key Designs

1. **ECO: Episodic COmpressor**

   Maintains a memory buffer with maximum capacity $E$. Upon receiving new window features $\mathcal{W}_k$:

    $\mathcal{M} = \begin{cases} \mathcal{M} \oplus \mathcal{W}_k & \text{if } \|\mathcal{M}\| + \|\mathcal{W}_k\| \leq E \\ \text{ECO}(\mathcal{M}, \mathcal{W}_k) & \text{otherwise} \end{cases}$

   Compression algorithm: (1) concatenate buffer and new features → (2) identify the frame pair $(i^*, j^*)$ with highest cosine similarity → (3) merge by averaging → (4) remove merged frames → repeat until frame count $\leq E$.

   Core Idea: The most similar frames carry redundant information; merging them reduces data volume while preserving episodic structure. The optimal number of episodes is 20.

2. **SeTR: Semantics reTRiever**

   Captures high-level semantic information distributed across the entire video. Given feature tensor $F \in \mathbb{R}^{B \times N \times T \times C}$:

    - Normalize features
    - Partition into two groups at stride $k$: $X$ (every $k$-th frame) and $Y$ (remaining frames)
    - Compute dot-product similarity between $X$ and $Y$
    - Merge each frame in $Y$ into its most similar frame in $X$

   Effect: Frame count reduces from $N$ to $N/k$; the optimal keep ratio is 20% (i.e., $k=5$, discarding 80% of frames).

3. **Hierarchical Q-Former**

    - **Episodic Q-Former**: Applies self-attention and cross-attention to ECO's episodic memory output, then compresses queries via a merging process similar to ECO.
    - **Hierarchical Frame-to-Sequence Q-Former**: A frame-level Q-Former first independently enhances the semantics of each frame; a video-level Q-Former then aggregates representations across frames.

   The two sets of queries are concatenated and projected into the LLM via a linear layer: $U = W[Q; Q_{sem}] + b$

### Loss & Training

Standard language modeling loss (cross-entropy) is used, supporting both zero-shot and fully supervised evaluation settings. ECO and SeTR require no additional learning and can serve as plug-and-play modules; only minimal adaptation is needed when integrating them into existing models.

## Key Experimental Results

### Main Results

**MovieChat-1k Zero-shot VQA**

| Model | Global Acc. | Global Score | Breakpoint Acc. |
|-------|-------------|--------------|-----------------|
| MovieChat | 63.7 | 3.15 | 48.1 |
| Video-ChatGPT | 58.7 | 2.89 | 47.8 |
| Video-LLaMA | 56.3 | 2.72 | 45.8 |
| **HERMES** | **78.6** | **4.23** | **57.3** |
| **HERMES (fully supervised)** | **84.9** | **4.40** | **65.8** |

Gain of **+14.9%** in Global Acc. over the previous state-of-the-art.

**LVU + Breakfast + COIN Classification**

| Model | LVU Avg. | Breakfast | COIN |
|-------|----------|-----------|------|
| MA-LMM | 63.0 | 93.0 | 93.2 |
| S5 | 59.2 | 90.7 | 90.8 |
| **HERMES** | **70.3** | **95.2** | **93.5** |

Gain of **+7.3%** on LVU over the previous state-of-the-art.

### Ablation Study

**ECO Memory Update Strategy Comparison**

| Strategy | Acc. | Score |
|----------|------|-------|
| w/o ECO | 55.1 | 3.55 |
| Random retention | 76.9 | 4.13 |
| FIFO | 77.1 | 4.15 |
| **ECO** | **78.6** | **4.23** |

**Semantic Compression Method Comparison**

| Method | Acc. | Score |
|--------|------|-------|
| w/o SeTR | 73.3 | 4.09 |
| MaxPool | 70.4 | 3.99 |
| AvgPool | 73.3 | 4.04 |
| K-Means | 75.7 | 4.11 |
| **SeTR** | **78.6** | **4.23** |

**Plug-and-Play Effect (Integration into Existing SOTA Models)**

| Base Model | +ECO Acc. Δ | +ECO Latency Δ | +SeTR Acc. Δ |
|------------|-------------|----------------|--------------|
| MA-LMM | +3.4% | −43% | +3.8% |
| LongVA | +0.08% | −30% | +0.45% |
| LLaVA-OV | +0.67% | −35% | +1.04% |

### Key Findings

- ECO and SeTR each independently improve performance, and their combination yields additive gains — confirming that episodic and semantic information are genuinely complementary.
- ECO's cosine-similarity-based merging strategy outperforms random retention and FIFO, validating the intuition of "merging the most redundant frames."
- SeTR substantially outperforms naive approaches such as MaxPool and AvgPool, as it performs similarity-based selective merging that preserves the most representative frames.
- The hierarchical Q-Former (95.2%) significantly outperforms standalone frame-level (93.2%) and video-level (94.1%) Q-Formers.
- The optimal episode count of 20 and keep ratio of 20% are consistent across different datasets, indicating good hyperparameter robustness.
- HERMES processes only 100 frames out of 14k (0.7%), whereas MovieChat processes 2,048 frames, yet HERMES achieves higher accuracy.

## Highlights & Insights

- The **cognitive dual-pathway design of episodic vs. semantic processing** is the paper's most significant contribution — this cognitive-science-inspired framework elegantly decouples two complementary types of information.
- The **plug-and-play generality of ECO** is impressive: integrating it into MA-LMM yields a 3.4% accuracy gain alongside a 43% inference speedup, with no memory overhead.
- SeTR's keep ratio can be as low as 20% (discarding 80% of frames), confirming the substantial redundancy present in long-form video.
- Qualitative analysis shows that HERMES can honestly express uncertainty (e.g., "I'm not sure") rather than hallucinating answers as other models tend to do.

## Limitations & Future Work

- ECO and SeTR rely on heuristic rules (cosine-similarity merging and stride-based sampling), which may fail on subtle temporal details in certain scenarios.
- The two modules operate independently without joint optimization, potentially introducing redundancy.
- Due to computational constraints, no large-scale video pretraining was conducted, limiting direct comparisons with models such as LLaVA-OneVision on VideoMME.
- Only Vicuna-7B is used as the LLM backend; stronger language models remain unexplored.

## Related Work & Insights

- MA-LMM's memory mechanism can be directly replaced by ECO with improved results — suggesting that task-agnostic general-purpose compression may outperform task-specific designs.
- ToMe (Token Merging) performs token merging within ViT layers, whereas SeTR performs similarity-based semantic aggregation at the cross-frame level; the two differ fundamentally in objective and granularity.
- HERMES's modular design provides a blueprint for future "LEGO-style" video understanding systems, where distinct modules handle distinct functions and can be freely combined.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The cognitive-inspired episodic+semantic dual-pathway design is novel, and the plug-and-play integration experiments are convincing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets, integration with three SOTA models, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Fluent exposition with clearly articulated cognitive-science motivation.
- **Value**: ⭐⭐⭐⭐ ECO and SeTR have practical applicability as plug-and-play modules.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](../../CVPR2026/video_understanding/divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)

<!-- RELATED:END -->
