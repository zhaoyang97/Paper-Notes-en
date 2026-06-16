---
title: >-
  [Paper Note] Causality Matters: How Temporal Information Emerges in Video Language Models
description: >-
  [AAAI 2026][Video Understanding][Video Language Models] Through systematic ablation experiments, this work demonstrates that the temporal understanding capability of VideoLMs does not originate from positional encoding (…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Video Language Models"
  - "Temporal Understanding"
  - "Causal Attention"
  - "Positional Encoding"
  - "Model Interpretability"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: ea8f3a04525c7af5
---

# Causality Matters: How Temporal Information Emerges in Video Language Models

**Conference**: AAAI 2026
**arXiv**: [2508.11576](https://arxiv.org/abs/2508.11576)  
**Authors**: Yumeng Shi, Quanyu Long, Yin Wu, Wenya Wang (NTU)
**Code**: [github.com/ANDgate99/Causality-Matters](https://github.com/ANDgate99/Causality-Matters)  
**Area**: Video Understanding
**Keywords**: Video Language Models, Temporal Understanding, Causal Attention, Positional Encoding, Model Interpretability, Inference Acceleration

## TL;DR

Through systematic ablation experiments, this work demonstrates that the temporal understanding capability of VideoLMs does not originate from positional encoding (PE), but rather emerges from the sequence sensitivity of causal attention masks. Temporal information is constructed layer by layer along a causal pathway of "inter-frame interaction → last-frame aggregation → query integration," based on which two lossless inference acceleration strategies are proposed.

## Background & Motivation

### State of the Field

Video language models (VideoLMs) have achieved remarkable progress on tasks such as video question answering, caption generation, and temporal reasoning. **Temporal understanding**—recognizing event order, duration, and cross-temporal relationships—is a core challenge in video comprehension. From causal inference to narrative coherence in long videos, temporal understanding is essential for intelligent video-language interaction.

### Limitations of Prior Work

The prevailing assumption in the community is that **positional encoding (PE) is the central mechanism for temporal modeling in VideoLMs**, giving rise to a large body of PE-focused research: Qwen2.5-VL adopts 3D PE, V2PE designs variable-rate encoding, and VideoRoPE extends PE to higher-dimensional spaces. However, this assumption has never been rigorously validated—to what extent does PE actually support temporal understanding? And if PE is not the key, what is the true source of temporal information?

### Root Cause

The starting point of this paper is a counterintuitive observation: **removing or modifying PE of video inputs has minimal impact on temporal understanding performance, whereas reversing the frame sequence (while keeping PE unchanged) leads to substantial performance degradation**. This suggests that the model is highly sensitive to the physical order of frames, yet does not rely on PE to perceive that order. The paper therefore proposes the core hypothesis: temporal understanding is an **emergent phenomenon** of the causal attention mechanism, arising from sequence-sensitive interactions among tokens rather than from explicit positional signals.

### Paper Goals

1. Quantify the actual contribution of PE to temporal understanding
2. Trace the generation, propagation, and aggregation of temporal information within the model
3. Verify the causal nature of this pathway (rather than mere correlation)
4. Propose practical inference acceleration strategies based on mechanistic insights

## Method

### Experimental Framework

The study uses Qwen2.5-VL-7B and LLaVA-OneVision-7B as primary subjects of analysis, evaluated on the TempCompass benchmark (covering temporal dimensions including action, speed, direction, attribute change, and event order). The evaluation metric is the change in ground-truth token probability after perturbation, $P_C = \tilde{P}_\text{next}(t_\text{gt}) - P_\text{next}(t_\text{gt})$, directly measuring the impact of perturbations on temporal reasoning.

### Stage 1: Decomposing the Role of Positional Encoding

**Layer-wise PE ablation**: PE is removed individually at each layer to examine the degree of PE dependency per layer. Results show that removing PE only at the first layer causes a substantial drop in correct-answer probability (up to 60%), while removing PE at intermediate or deep layers has negligible effect (typically within ±2%). This indicates that the role of PE is strictly confined to the first layer of the model.

**Modality-specific analysis**: Position IDs of video tokens and query tokens are separately shuffled at the first layer. Shuffling query position IDs leads to significant performance degradation (−53% on Multiple Choice), whereas shuffling video position IDs has an effect of less than 0.1%. This demonstrates that even in the first layer where PE has an effect, its impact is concentrated on the text modality rather than the video modality.

**PE vs. frame sequence comparison**: Reversing position IDs across all layers (while preserving frame order) causes only 1–2% performance loss, whereas reversing the frame sequence (while preserving position IDs) causes a 15.45% drop on the Multiple Choice task. This constitutes the paper's most critical empirical finding: **the physical order of frames is far more important than positional encoding**.

### Stage 2: Tracing the Causal Information Pathway

The paper employs **attention knockout** to systematically trace the flow of temporal information. Specifically, an additional mask $M$ is introduced into the attention computation to prevent a target token set $\mathcal{T}$ from attending to a source token set $\mathcal{S}$, scanning progressively in sliding windows of $k=5$ layers.

**Finding 1: Query-driven final prediction.** Blocking the final output token from attending to video tokens has almost no effect on performance, whereas blocking its attention to query tokens causes over 40% performance degradation. This indicates that at the final decoding stage, the model primarily retrieves temporal information from the query rather than directly from the video—the query has already absorbed temporal information in preceding layers.

**Finding 2: Two-stage temporal processing.** Separately blocking inter-frame attention and frame-to-query attention reveals that the two types of blocking cause performance degradation at different layers: inter-frame attention blocking peaks around layer 10 (4–6% degradation), while frame-to-query blocking peaks around layer 15 (over 10% degradation). This reveals a clear two-stage pipeline—**early layers build inter-frame temporal relations, while intermediate layers integrate temporal information into query representations**.

**Finding 3: Last-frame aggregation effect.** When the query is restricted to attending to a single frame, restricting attention to Frame 1 causes over 8% performance drop on the Multiple Choice task, whereas restricting attention to Frame 4 (the last frame) results in less than 2% loss. This demonstrates that temporal information accumulates progressively along the forward direction across frames, with the final frame serving as an aggregation node that consolidates all temporal cues.

**Finding 4: Causal verification.** After reversing the frame sequence in Yes/No tasks, the attribution pattern is completely inverted: the last frame, which previously contributed most positively, becomes the most harmful, while the first frame, previously the most harmful, instead improves performance. This reversal persists even when position IDs are simultaneously reversed (eliminating PE as a confounding factor), proving that the model performs **sequence-sensitive causal reasoning** rather than simple visual information accumulation.

**Finding 5: Spatiotemporal assembly mechanism.** In the spatial dimension of inter-frame attention, sparse long-range correspondence attention (each frame attending only to the corresponding spatial positions in preceding frames) outperforms dense short-range attention (each frame attending to all tokens in the immediately preceding frame). Early layers favor global spatial context, while deep layers shift toward local detail processing.

### Stage 3: Inference Acceleration Strategies

Based on the mechanistic insights above, two acceleration schemes are proposed:

1. **Stage-wise sparse attention**: In intermediate layers (10–20), the query is restricted to attending only to the last frame's video tokens; in deep layers (20–28), inter-frame attention is disabled, reducing FLOPs.
2. **Temporal Exit mechanism**: After layer 20, all frame tokens are removed from the KV cache, reducing GPU memory consumption.

## Key Experimental Results

### Table 1: Effect of PE Reversal vs. Frame Sequence Reversal on Temporal Understanding (Qwen2.5-VL-7B, TempCompass)

| Perturbation | Yes/No | Multiple Choice | Captioning | Caption Matching |
|---|---|---|---|---|
| Reverse Position IDs | −1.13% | −1.42% | −1.76% | −1.55% |
| **Reverse Frame Sequence** | **−10.28%** | **−15.45%** | **−8.73%** | **−11.82%** |

Performance degradation from frame sequence reversal is 7–10× that of PE reversal, strongly demonstrating that temporal understanding relies on the physical order of frames rather than positional encoding.

### Table 2: Accuracy of Inference Acceleration Strategies on TempCompass and NExT-QA

| Strategy | Yes/No | MCQ | Caption | Match | NExT-QA |
|---|---|---|---|---|---|
| Baseline | 70.8 | 66.5 | 59.0 | 57.4 | 75.1 |
| (1) Query attends only to last frame (L10–20) | 67.9 | 64.1 | 57.6 | 55.0 | 71.9 |
| (2) Disable inter-frame attention (L20–28) | 70.5 | 66.5 | 59.3 | 57.4 | 75.2 |
| (3) Remove frame KV cache (L20+) | 70.7 | 66.6 | 59.0 | 57.2 | 75.1 |

Strategies (2) and (3) are nearly indistinguishable from the baseline across all tasks (difference <0.3%), with strategy (3) even marginally exceeding the baseline on MCQ. This validates that inter-frame interactions in deep layers and frame token storage can indeed be safely pruned.

## Highlights & Insights

- **Paradigm-shifting finding**: Overturns the community consensus that "PE is central to temporal understanding," demonstrating that the causal attention mask is the true source of temporal modeling, providing an important corrective signal for the PE research direction.
- **Complete mechanistic picture**: For the first time, systematically characterizes the full causal pathway of temporal information within VideoLMs—inter-frame construction → last-frame aggregation → query absorption → independent decoding—with clear hierarchical structure and verified causal relationships.
- **Cross-model consistency**: Findings are consistent across three models with significantly different architectures (Qwen2.5-VL-7B/3B and LLaVA-OneVision), enhancing generalizability.
- **Mechanism-driven application**: The inference acceleration strategies are not heuristically designed but are directly derived from mechanistic analysis, forming a logically complete loop.
- **Methodological value**: The analysis framework combining attention knockout and backward tracing is generalizable to mechanistic studies of other multimodal understanding problems.

## Limitations & Future Work

- **Limited model coverage**: Only two model families are analyzed; non-causal architectures (e.g., bidirectional attention or encoder-decoder structures) are not examined, and applicability to non-autoregressive models is unknown.
- **Restricted frame count**: Experiments use only 4–8 frames as input; whether the last-frame aggregation mechanism holds for long videos (with tens to hundreds of frames) remains to be verified.
- **Preliminary acceleration strategies**: The layer intervals for the proposed acceleration schemes (e.g., L10–20, L20–28) are manually specified without adaptive mechanisms, and strategy (1) incurs approximately 3% performance loss.
- **Training-side not addressed**: The analysis focuses on attention mechanisms during inference and does not explore the effect of PE on learning temporal representations during training; the paper acknowledges that PE may still be important at training time.
- **Task type limitations**: Validation is primarily conducted on judgment/selection tasks; on open-ended generation tasks (ActivityNet-QA), strategy (3) shows a larger performance drop (42.7% vs. 46.9%), indicating that the generalizability of findings to generation tasks requires further investigation.

## Related Work & Insights

- **PE design methods (V2PE, VideoRoPE, Qwen2.5-VL 3D-PE)**: This paper fundamentally questions the effectiveness of this technical direction—if PE contributes minimally to video temporal understanding, carefully designing PE may not be the right path forward.
- **T3 (Li et al.)**: Transfers temporal skills to VideoLMs via synthetic text data, representing a data-driven approach; this paper instead reveals the intrinsic source of temporal capability from a mechanistic perspective.
- **Basu et al. (2024)**: Applies causal tracing to analyze visual information flow in multimodal models, but limited to single-image scenarios; this paper is the first to extend such analysis to the video temporal dimension.
- **Zhang et al. (2025)**: Work analyzing cross-modal information flow, from which this paper adapts the backward tracing idea into an attention knockout scheme.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic revelation of the internal mechanism of temporal understanding in VideoLMs, overturning PE-centrism
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-model, multi-task validation with well-designed ablations, but limited coverage of frame count and video duration
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear and progressive logical chain, with a coherent arc from phenomena to mechanism to application
- Value: ⭐⭐⭐⭐ — Provides a new cognitive framework for temporal modeling in VideoLMs; acceleration strategies are practical but require further refinement

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[AAAI 2026\] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](../../CVPR2026/video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](../../CVPR2026/video_understanding/lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)

</div>

<!-- RELATED:END -->
